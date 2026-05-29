#ifndef BREAKDOWN_ANALYZER_HPP
#define BREAKDOWN_ANALYZER_HPP

#include <vector>
#include <string>
#include <limits>
#include "Particle.hpp"

class BreakdownAnalyzer {
private:
    struct BubbleBounds {
        float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
        float minY = std::numeric_limits<float>::max(), maxY = std::numeric_limits<float>::lowest();
        float minZ = std::numeric_limits<float>::max(), maxZ = std::numeric_limits<float>::lowest();
        bool active = false;
    };
    std::vector<BubbleBounds> m_bubbleBoundsCache;
    std::vector<float> m_bubbleDiametersCache;

public:
    float A = 11.25f;             // Pa^-1 * m^-1
    float B = 273.75f;            // V * Pa^-1 * m^-1
    float gammaSE = 0.01f;        // Second Electron Emission Coefficient
    float liquidBreakdownThreshold = 6.0e7f; // 60 kV/mm
    float membraneBreakdownThreshold = 9.0e7f;     // Default 90 kV/mm (PDMS-like)
    float liquidPermittivity = 2.7f;         // Relative permittivity of liquid oil
    
    BreakdownAnalyzer() = default;
    
    float calculateGasBreakdownField(float pressurePa, float bubbleDiameterM);
    
    struct BreakdownResult {
        bool hasBreakdown;
        float maxDBI;
        simd_float4 breakdownLocation; // Upgraded to 3D float4
        int32_t targetBubbleID;
    };
    
    BreakdownResult analyzeBreakdown(GPUParticle* particles, int numParticles);
    
    std::string generateBreakdownReport(const GPUParticle* particles, int numParticles);
};

#endif // BREAKDOWN_ANALYZER_HPP
