#include "BreakdownAnalyzer.hpp"
#include "SimdHelper.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

float BreakdownAnalyzer::calculateGasBreakdownField(float pressurePa, float bubbleDiameterM) {
    // Correct absolute pressure and set safe lower bound
    float P = pressurePa + 101325.0f; 
    float d = std::max(bubbleDiameterM, 1.0e-7f); // Avoid division by zero
    
    // 1. Classic Paschen's Law electric field calculation
    float Pd = P * d;
    float lnTerm = A * Pd;
    float TownsendLimit = std::log(1.0f + 1.0f / gammaSE);
    
    float ePaschen = 1.0e9f; // default large breakdown field if avalanche is impossible
    if (lnTerm > TownsendLimit * 1.001f) {
        float denomTerm = std::log(lnTerm) - std::log(TownsendLimit);
        float vBreakdown = (B * Pd) / denomTerm;
        ePaschen = vBreakdown / d;
    }
    
    // 2. Fowler-Nordheim Vacuum Breakdown / Field Emission limit
    // Below ~5 um, electrons cross without collisions. Breakdown is dominated by field-emission, 
    // limiting the local breakdown field to a finite threshold (typically 1.0e8 V/m for air micro-gaps)
    float eVacuum = 1.0e8f; 
    
    // 3. Smooth Physical Blending based on electron mean free path and gap size
    // Sigmoid transition around critical gap dc = 5.0 micrometers (k = 1.0 micrometer)
    float dc = 5.0e-6f;
    float kWidth = 1.0e-6f;
    float w = 1.0f / (1.0f + std::exp((d - dc) / kWidth));
    
    float eBlended = (1.0f - w) * ePaschen + w * eVacuum;
    
    // Ensure absolute physical lower bound corresponding to air's macroscopic dielectric strength (3.0e6 V/m)
    return std::max(eBlended, 3.0e6f);
}
 
BreakdownAnalyzer::BreakdownResult BreakdownAnalyzer::analyzeBreakdown(GPUParticle* particles, int numParticles) {
    BreakdownResult result = { false, 0.0f, simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f), -1 };
    
    // 1. Dynamic 3D bubble size tracking per BubbleID using the cached contiguous vector
    // Find the maximum gasBubbleID to allocate a contiguous vector across all particles safely
    int maxBubbleID = -1;
    for (int i = 0; i < numParticles; ++i) {
        if (particles[i].gasBubbleID > maxBubbleID) {
            maxBubbleID = particles[i].gasBubbleID;
        }
    }
    
    int requiredSize = (maxBubbleID >= 0) ? (maxBubbleID + 1) : 0;
    if (static_cast<int>(m_bubbleBoundsCache.size()) < requiredSize) {
        m_bubbleBoundsCache.resize(requiredSize);
    }
    
    // In-place initialization of BubbleBounds cache to prevent dynamic allocations
    for (int i = 0; i < requiredSize; ++i) {
        m_bubbleBoundsCache[i].active = false;
        m_bubbleBoundsCache[i].minX = std::numeric_limits<float>::max();
        m_bubbleBoundsCache[i].maxX = std::numeric_limits<float>::lowest();
        m_bubbleBoundsCache[i].minY = std::numeric_limits<float>::max();
        m_bubbleBoundsCache[i].maxY = std::numeric_limits<float>::lowest();
        m_bubbleBoundsCache[i].minZ = std::numeric_limits<float>::max();
        m_bubbleBoundsCache[i].maxZ = std::numeric_limits<float>::lowest();
    }
    
    for (int i = 0; i < numParticles; ++i) {
        const auto& p = particles[i];
        if (p.phase == 1 && p.gasBubbleID >= 0) { // Gas phase
            int32_t bid = p.gasBubbleID;
            if (bid >= 0 && bid < static_cast<int32_t>(m_bubbleBoundsCache.size())) {
                auto& bounds = m_bubbleBoundsCache[bid];
                bounds.active = true;
                
                // NaN safety protection: prevent coordinate explosion from corrupting diagnostic bounds
                if (!std::isnan(p.position.x) && !std::isnan(p.position.y) && !std::isnan(p.position.z)) {
                    bounds.minX = std::min(bounds.minX, p.position.x);
                    bounds.maxX = std::max(bounds.maxX, p.position.x);
                    bounds.minY = std::min(bounds.minY, p.position.y);
                    bounds.maxY = std::max(bounds.maxY, p.position.y);
                    bounds.minZ = std::min(bounds.minZ, p.position.z);
                    bounds.maxZ = std::max(bounds.maxZ, p.position.z);
                }
            }
        }
    }
    
    if (static_cast<int>(m_bubbleDiametersCache.size()) < requiredSize) {
        m_bubbleDiametersCache.resize(requiredSize);
    }
    std::fill(m_bubbleDiametersCache.begin(), m_bubbleDiametersCache.begin() + requiredSize, 0.001f);
    
    for (int bid = 0; bid < requiredSize; ++bid) {
        const auto& bounds = m_bubbleBoundsCache[bid];
        if (bounds.active) {
            float dx = bounds.maxX - bounds.minX;
            float dy = bounds.maxY - bounds.minY;
            float dz = bounds.maxZ - bounds.minZ;
            float diameter = std::sqrt(dx * dx + dy * dy + dz * dz);
            m_bubbleDiametersCache[bid] = std::max(diameter, 0.0005f); // Minimum diameter of 0.5mm
        }
    }
    
        // 2. Calculate DBI per particle in 3D with Grid-to-Micro-Gap Singularity Concentration
    for (int i = 0; i < numParticles; ++i) {
        auto& p = particles[i];
        float eFieldMagnitude = length3(p.electricField);
        
        float eth = liquidBreakdownThreshold;
        
        // Dynamic Analytical Singularity Concentration Factor based on dielectric contrast
        // and local potential gradient curvature (Frobenius norm of E-field gradient ||grad(E)||_F)
        float singularityFactor = 1.0f;
        float eMag = eFieldMagnitude;
        float G = p.eFieldGradientNorm;
        float d_gap = 12.0e-6f; // Characteristic micro-gap thickness (12 um)
        
        if (p.phase == 1) { // Gas phase
            int32_t bid = p.gasBubbleID;
            float d = (bid >= 0 && bid <= maxBubbleID) ? m_bubbleDiametersCache[bid] : 0.001f;
            eth = calculateGasBreakdownField(p.pressure, d);
            
            // A. Dielectric polarization concentration factor (spherical gas bubble in oil)
            float eps_L = liquidPermittivity;
            float beta_polarization = (3.0f * eps_L) / (2.0f * eps_L + 1.0f);
            
            // B. Local gradient-based field concentration (Hessian-driven potential curvature)
            float beta_curvature = 1.0f + 1.5f * (G * d_gap) / (eMag + 1.0e4f);
            singularityFactor = beta_polarization * beta_curvature * 27.5f; // calibrated scale
        } else if (p.phase == 2 || p.phase == 3) { // Solid membrane / Electrode
            eth = membraneBreakdownThreshold; // Dielectric strength of solid membrane
            
            // Dynamic zipping edge curvature singularity concentration driven by local field gradient
            float beta_curvature = 1.0f + 5.0f * (G * d_gap) / (eMag + 1.0e4f);
            singularityFactor = beta_curvature * 95.0f; // calibrated scale
        }
        
        // DBI = (E * SingularityFactor) / Eth
        float dbi = (eFieldMagnitude * singularityFactor) / std::max(eth, 1e-9f);
        p.dbi = dbi;
        
        if (dbi > result.maxDBI) {
            result.maxDBI = dbi;
            result.breakdownLocation = p.position;
            result.targetBubbleID = p.gasBubbleID;
        }
        
        if (dbi >= 1.0f) {
            result.hasBreakdown = true;
        }
    }
    
    return result;
}

std::string BreakdownAnalyzer::generateBreakdownReport(const GPUParticle* particles, int numParticles) {
    std::stringstream ss;
    ss << "ParticleID,Phase,X_m,Y_m,Z_m,Pressure_Pa,E_Field_V_m,E_Potential_V,DBI,Status\n";
    
    ss << std::fixed << std::setprecision(6);
    
    for (int i = 0; i < numParticles; ++i) {
        const auto& p = particles[i];
        float eMag = length3(p.electricField);
        std::string status = (p.dbi >= 1.0f) ? "BREAKDOWN_RISK" : ((p.dbi >= 0.8f) ? "WARNING" : "SAFE");
        
        ss << i << ","
           << p.phase << ","
           << p.position.x << ","
           << p.position.y << ","
           << p.position.z << ","
           << p.pressure << ","
           << eMag << ","
           << p.electricPotential << ","
           << p.dbi << ","
           << status << "\n";
    }
    return ss.str();
}
