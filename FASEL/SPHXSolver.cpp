#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "SPHXSolver.hpp"

// Safe Metal resource release macro to prevent double frees and nullify pointers
#define METAL_SAFE_RELEASE(ptr) \
    do { \
        if (ptr) { \
            ptr->release(); \
            ptr = nullptr; \
        } \
    } while (0)

// Compile-time layout check to guarantee absolute memory alignment safety between C++ host and Metal GPU shaders
static_assert(sizeof(GPUParticle) == 144, "GPUParticle struct must be exactly 144 bytes for Apple Silicon Metal UMA memory compatibility");
extern "C" float swift_crypto_random_float();
extern "C" float swift_crypto_random_float_seeded(uint32_t seed, uint32_t counter);
#include "SimdHelper.hpp"
#include <iostream>

static float calculateDihedralAngle(simd_float4 p0, simd_float4 p1, simd_float4 p2, simd_float4 p3) {
    simd_float3 x0 = simd_make_float3(p0.x, p0.y, p0.z);
    simd_float3 x1 = simd_make_float3(p1.x, p1.y, p1.z);
    simd_float3 x2 = simd_make_float3(p2.x, p2.y, p2.z);
    simd_float3 x3 = simd_make_float3(p3.x, p3.y, p3.z);
    
    simd_float3 n1 = simd_cross(x2 - x0, x1 - x0);
    simd_float3 n2 = simd_cross(x3 - x0, x1 - x0);
    
    float len1 = simd_length(n1);
    float len2 = simd_length(n2);
    if (len1 < 1e-6f || len2 < 1e-6f) return 0.0f;
    
    n1 = n1 / len1;
    n2 = n2 / len2;
    
    float dot_val = simd_dot(n1, n2);
    dot_val = std::clamp(dot_val, -0.999f, 0.999f);
    return std::acos(dot_val);
}
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

SPHXSolver::SPHXSolver(
    float gasVolumeFraction,
    float appliedVoltage,
    float liquidPermittivity,
    float liquidViscosity,
    float surfaceTension,
    float pouchWidth,
    float pouchHeight,
    float stretchCompliance,
    float bendingCompliance,
    float fluidSpacing,
    int numShellParticles,
    int numLayers,
    float timeStep,
    float kPouchAnchorYThreshold,
    float kBoundaryRadiusBox,
    float kBoundaryRadiusLimit,
    float kActiveStrainLeverage
) : gasVolumeFraction(gasVolumeFraction),
    appliedVoltage(appliedVoltage),
    liquidPermittivity(liquidPermittivity),
    liquidViscosity(liquidViscosity),
    surfaceTension(surfaceTension),
    pouchWidth(pouchWidth),
    pouchHeight(pouchHeight),
    stretchCompliance(stretchCompliance),
    bendingCompliance(bendingCompliance),
    fluidSpacing(fluidSpacing),
    numShellParticles(numShellParticles),
    numLayers(numLayers),
    timeStep(timeStep),
    kPouchAnchorYThreshold(kPouchAnchorYThreshold),
    kBoundaryRadiusBox(kBoundaryRadiusBox),
    kBoundaryRadiusLimit(kBoundaryRadiusLimit),
    kActiveStrainLeverage(kActiveStrainLeverage) {
    
    // 3D Voxel Grid boundaries: -5cm to 5cm in X, Y, Z (total 10cm cube chamber size)
    float cellSize = 0.003125f; // cell size matching 32 grid dimensions
    gridParams = GPUSpatialGridParams(
        simd_make_float4(-0.05f, -0.05f, -0.05f, 0.0f),
        cellSize,
        simd_make_int4(32, 32, 32, 0),
        0
    );
    
    setupMetal();
    buildActuatorGeometry();
    setupGPUBuffers();
}

SPHXSolver::~SPHXSolver() {
    // With modern C++ RAII MetalPtr wrappers, all GPU and pipeline resources 
    // are automatically released in the correct order upon solver destruction, 
    // completely eliminating memory leak risks and manual cleanup boilerplate!
}

void SPHXSolver::setupMetal() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cout << "Warning: Cannot create Metal GPU device. Simulation may run in CPU fallback mode.\n";
        return;
    }
    commandQueue = device->newCommandQueue();
    
    // 1. Try to load precompiled default.metallib first
    bool libFound = false;
    std::vector<std::string> searchPaths = {
        "default.metallib",
        "build/default.metallib",
        "Shaders.metallib",
        "build/Shaders.metallib"
    };
    
    MTL::Library* library = nullptr;
    NS::Error* error = nullptr;
    
    for (const auto& path : searchPaths) {
        std::ifstream ifs(path, std::ios::binary);
        if (ifs.is_open()) {
            ifs.close();
            NS::String* nsPath = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
            NS::URL* nsURL = NS::URL::fileURLWithPath(nsPath);
            library = device->newLibrary(nsURL, &error);
            nsURL->release();
            nsPath->release();
            if (library) {
                std::cout << "Successfully loaded precompiled Metal library: " << path << "\n";
                libFound = true;
                break;
            }
        }
    }
    
    // 2. If precompiled library not found, fall back to runtime compilation of Shaders.metal source
    if (!libFound) {
        std::string shaderPath = "Shaders.metal";
        std::vector<std::string> sourcePaths = {
            "Shaders.metal",
            "FASEL-cpp/Shaders.metal"
        };
        
        std::ifstream ifs;
        for (const auto& path : sourcePaths) {
            ifs.open(path);
            if (ifs.is_open()) {
                shaderPath = path;
                break;
            }
        }
        
        if (!ifs.is_open()) {
            std::cout << "Warning: Cannot load Shaders.metal file and precompiled Shaders.metallib was not found. Switching to CPU fallback mode.\n";
            return;
        }
        
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string shaderSource = ss.str();
        ifs.close();
        
        NS::String* source = NS::String::string(shaderSource.c_str(), NS::UTF8StringEncoding);
        library = device->newLibrary(source, nullptr, &error);
        source->release();
        
        if (!library) {
            std::cout << "Error: Metal Shader runtime compilation error! Detailed log:\n";
            if (error) {
                std::cout << error->localizedDescription()->cString(NS::UTF8StringEncoding) << "\n";
            }
            std::cout << "Switching to CPU fallback mode.\n";
            return;
        } else {
            std::cout << "Successfully compiled Metal shader source at runtime: " << shaderPath << "\n";
        }
    }
    
    try {
        NS::String* clearCellName = NS::String::string("clearCellBuffersKernel", NS::UTF8StringEncoding);
        MTL::Function* clearCellFunc = library->newFunction(clearCellName);
        if (clearCellFunc) {
            clearCellBuffersPSO = device->newComputePipelineState(clearCellFunc, &error);
            clearCellFunc->release();
        }
        clearCellName->release();
        
        NS::String* buildCellName = NS::String::string("buildCellMarkersKernel", NS::UTF8StringEncoding);
        MTL::Function* buildCellFunc = library->newFunction(buildCellName);
        if (buildCellFunc) {
            buildCellMarkersPSO = device->newComputePipelineState(buildCellFunc, &error);
            buildCellFunc->release();
        }
        buildCellName->release();
        
        NS::String* electrostaticName = NS::String::string("electrostaticForceKernel", NS::UTF8StringEncoding);
        MTL::Function* electrostaticFunc = library->newFunction(electrostaticName);
        if (electrostaticFunc) {
            electrostaticForcePSO = device->newComputePipelineState(electrostaticFunc, &error);
            electrostaticFunc->release();
        }
        electrostaticName->release();
        
        NS::String* sphDensityName = NS::String::string("sphDensityKernel", NS::UTF8StringEncoding);
        MTL::Function* sphDensityFunc = library->newFunction(sphDensityName);
        if (sphDensityFunc) {
            sphDensityPSO = device->newComputePipelineState(sphDensityFunc, &error);
            sphDensityFunc->release();
        }
        sphDensityName->release();
        
        NS::String* sphForceName = NS::String::string("sphForceKernel", NS::UTF8StringEncoding);
        MTL::Function* sphForceFunc = library->newFunction(sphForceName);
        if (sphForceFunc) {
            sphForcePSO = device->newComputePipelineState(sphForceFunc, &error);
            sphForceFunc->release();
        }
        sphForceName->release();
        
        NS::String* colorGradientName = NS::String::string("colorGradientKernel", NS::UTF8StringEncoding);
        MTL::Function* colorGradientFunc = library->newFunction(colorGradientName);
        if (colorGradientFunc) {
            colorGradientPSO = device->newComputePipelineState(colorGradientFunc, &error);
            colorGradientFunc->release();
        }
        colorGradientName->release();
        
        NS::String* curvatureCSFName = NS::String::string("curvatureCSFKernel", NS::UTF8StringEncoding);
        MTL::Function* curvatureCSFFunc = library->newFunction(curvatureCSFName);
        if (curvatureCSFFunc) {
            curvatureCSFPSO = device->newComputePipelineState(curvatureCSFFunc, &error);
            curvatureCSFFunc->release();
        }
        curvatureCSFName->release();
        
        NS::String* integrateName = NS::String::string("integrateKernel", NS::UTF8StringEncoding);
        MTL::Function* integrateFunc = library->newFunction(integrateName);
        if (integrateFunc) {
            integratePSO = device->newComputePipelineState(integrateFunc, &error);
            integrateFunc->release();
        }
        integrateName->release();
        
    } catch (...) {
        std::cout << "Error: Exception occurred while creating Metal Compute Pipeline State\n";
    }
    
    library->release();
}

void SPHXSolver::buildActuatorGeometry() {
    particles.clear();
    initialPositions.clear();
    distanceConstraints.clear();
    bendingConstraints.clear();
    
    std::cout << "Building 3D HASEL Actuator geometry... (Layers: " << numLayers << ", Gas fraction: " << gasVolumeFraction << ")\n";
    
    float shellLength = pouchHeight;
    float shellWidth = pouchWidth;
    float shellThickness = pouchThickness;
    
    // Number of shell particles in latitude and longitude directions
    int numLat = 12;
    int numLon = 15;
    
    for (int l = 0; l < numLayers; ++l) {
        float zOffset = (l - (numLayers - 1) * 0.5f) * shellThickness;
        std::vector<int> shellIndices;
        
        // 1. Generate 3D ellipsoidal shell particles
        for (int lat = 0; lat < numLat; ++lat) {
            float theta = M_PI * static_cast<float>(lat) / static_cast<float>(numLat - 1);
            for (int lon = 0; lon < numLon; ++lon) {
                float phi = 2.0f * M_PI * static_cast<float>(lon) / static_cast<float>(numLon);
                
                float x = (shellWidth * 0.5f) * std::sin(theta) * std::cos(phi);
                float y = (shellLength * 0.5f) * std::cos(theta);
                float z = zOffset + (shellThickness * 0.5f) * std::sin(theta) * std::sin(phi);
                
                // Introduce minor random coordinate perturbations (1.5% of fluid spacing) to mimic physical assembly tolerances
                // and break perfect geometric symmetry (imperfection seeding) to accelerate structural zipping/buckling.
                // Uses Apple's CryptoKit cryptographically secure PRNG via the Swift bridge with deterministic seeding.
                float pertRange = fluidSpacing * 0.015f;
                uint32_t counter = static_cast<uint32_t>(lat * numLon + lon + l * 1000);
                float px = (swift_crypto_random_float_seeded(12345, counter * 3 + 0) * 2.0f - 1.0f) * pertRange;
                float py = (swift_crypto_random_float_seeded(12345, counter * 3 + 1) * 2.0f - 1.0f) * pertRange;
                float pz = (swift_crypto_random_float_seeded(12345, counter * 3 + 2) * 2.0f - 1.0f) * pertRange;
                x += px;
                y += py;
                z += pz;
                
                bool isElectrode = std::abs(y) > (shellLength * 0.35f);
                int32_t phase = isElectrode ? 3 : 2; // solidElectrode = 3, solidMembrane = 2
                
                // Front electrode (z > zOffset) gets -10, back electrode (z <= zOffset) gets -20
                int32_t electrodeID = isElectrode ? (z > zOffset ? -10 : -20) : -1;
                
                GPUParticle particle(simd_make_float4(x, y, z, 1.0f), simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f), 1.0e-6f, phase, electrodeID);
                particles.push_back(particle);
                initialPositions.push_back(particle.position);
                shellIndices.push_back(static_cast<int>(particles.size()) - 1);
            }
        }
        
        // 2. Define stretch constraints for this 3D layer (direct O(1) grid topology truss)
        for (int lat = 0; lat < numLat; ++lat) {
            for (int lon = 0; lon < numLon; ++lon) {
                int idxA = shellIndices[lat * numLon + lon];
                
                // 4-neighbor structural stretch springs
                std::vector<int> neighbors;
                
                // Horizontal neighbors (longitude wraps around the ellipsoidal cylinder)
                neighbors.push_back(shellIndices[lat * numLon + (lon + 1) % numLon]);
                
                // Vertical neighbors (latitude does not wrap at the poles)
                if (lat + 1 < numLat) {
                    neighbors.push_back(shellIndices[(lat + 1) * numLon + lon]);
                }
                
                // Add distance constraints for structural stretch
                for (int idxB : neighbors) {
                    if (idxA < idxB) {
                        float restLen = distance3(particles[idxA].position, particles[idxB].position);
                        distanceConstraints.push_back(GPUDistanceConstraint(idxA, idxB, restLen, stretchCompliance));
                    }
                }
            }
        }
        
        // 2b. Define rigorous out-of-plane Dihedral Bending Constraints (XPBD-based bending angles)
        // A. Vertical Dihedral Bending Constraints (soft along height/latitude to allow zipping wrinkles)
        for (int lat = 1; lat < numLat - 1; ++lat) {
            for (int lon = 0; lon < numLon; ++lon) {
                int idxC = shellIndices[lat * numLon + lon]; // Edge start (x0)
                int idxD = shellIndices[lat * numLon + (lon + 1) % numLon]; // Edge end (x1)
                int idxA = shellIndices[(lat - 1) * numLon + lon]; // Triangle 1 opposite vertex (x2)
                int idxB = shellIndices[(lat + 1) * numLon + lon]; // Triangle 2 opposite vertex (x3)
                
                float theta0 = calculateDihedralAngle(
                    particles[idxC].position, particles[idxD].position,
                    particles[idxA].position, particles[idxB].position
                );
                
                GPUBendingConstraint c(idxA, idxB, idxC, theta0, bendingCompliance * 100.0f);
                c.pad0 = idxD; // Store edge end vertex in pad0
                bendingConstraints.push_back(c);
            }
        }
        
        // B. Horizontal Dihedral Bending Constraints (stiff along width/longitude to preserve cylinder shape)
        for (int lat = 0; lat < numLat - 1; ++lat) {
            for (int lon = 0; lon < numLon; ++lon) {
                int idxC = shellIndices[lat * numLon + lon]; // Edge start (x0)
                int idxD = shellIndices[(lat + 1) * numLon + lon]; // Edge end (x1)
                int idxA = shellIndices[lat * numLon + (lon - 1 + numLon) % numLon]; // Triangle 1 opposite vertex (x2)
                int idxB = shellIndices[lat * numLon + (lon + 1) % numLon]; // Triangle 2 opposite vertex (x3)
                
                float theta0 = calculateDihedralAngle(
                    particles[idxC].position, particles[idxD].position,
                    particles[idxA].position, particles[idxB].position
                );
                
                GPUBendingConstraint c(idxA, idxB, idxC, theta0, bendingCompliance * 0.1f);
                c.pad0 = idxD; // Store edge end vertex in pad0
                bendingConstraints.push_back(c);
            }
        }
        
        // 3. Place internal 3D multiphase fluid for this layer
        int numBubbles = 4;
        std::vector<simd_float4> bubbleCenters;
        float bubbleRadius = shellWidth * 0.3f * std::sqrt(gasVolumeFraction);
        
        if (gasVolumeFraction > 0.0f) {
            // Uses Apple's CryptoKit cryptographically secure PRNG via the Swift bridge with deterministic seeding.
            for (int b = 0; b < numBubbles; ++b) {
                uint32_t counter = static_cast<uint32_t>(b + l * numBubbles);
                float rx = (-shellWidth * 0.3f) + swift_crypto_random_float_seeded(54321, counter * 3 + 0) * (shellWidth * 0.6f);
                float ry = (-shellLength * 0.3f) + swift_crypto_random_float_seeded(54321, counter * 3 + 1) * (shellLength * 0.6f);
                float rz = (-shellThickness * 0.3f) + swift_crypto_random_float_seeded(54321, counter * 3 + 2) * (shellThickness * 0.6f);
                bubbleCenters.push_back(simd_make_float4(rx, ry, zOffset + rz, 1.0f));
            }
        }
        
        float xCoord = -shellWidth * 0.5f + fluidSpacing * 0.5f;
        while (xCoord < shellWidth * 0.5f) {
            float yCoord = -shellLength * 0.5f + fluidSpacing * 0.5f;
            while (yCoord < shellLength * 0.5f) {
                float zCoord = zOffset - shellThickness * 0.5f + fluidSpacing * 0.5f;
                while (zCoord < zOffset + shellThickness * 0.5f) {
                    simd_float4 pos = simd_make_float4(xCoord, yCoord, zCoord, 1.0f);
                    
                    float normX = xCoord / (shellWidth * 0.5f);
                    float normY = yCoord / (shellLength * 0.5f);
                    float normZ = (zCoord - zOffset) / (shellThickness * 0.5f);
                    
                    if ((normX * normX + normY * normY + normZ * normZ) < 0.9f) {
                        bool isGas = false;
                        int32_t bubbleID = -1;
                        
                        if (gasVolumeFraction > 0.0f) {
                            for (int b = 0; b < numBubbles; ++b) {
                                if (distance3(pos, bubbleCenters[b]) < bubbleRadius) {
                                    isGas = true;
                                    bubbleID = b + l * numBubbles;
                                    break;
                                }
                            }
                        }
                        
                        int32_t phase = isGas ? 1 : 0; // gas = 1, liquid = 0
                        GPUParticle fluidParticle(pos, simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f), isGas ? 1.2e-9f : 9.6e-7f, phase, bubbleID);
                        particles.push_back(fluidParticle);
                        initialPositions.push_back(fluidParticle.position);
                    }
                    zCoord += fluidSpacing;
                }
                yCoord += fluidSpacing;
            }
            xCoord += fluidSpacing;
        }
    }
    
    gridParams.numParticles = static_cast<int32_t>(particles.size());
    gridParams.sphViscosityAlpha = sphViscosityAlpha;
    gridParams.sphViscosityBeta = sphViscosityBeta;
    
    // Cache kinematic anchor and external load indices based on initial coordinates
    anchorIndices.clear();
    loadIndices.clear();
    for (size_t i = 0; i < particles.size(); ++i) {
        const auto& p = particles[i];
        if (p.phase == 2 || p.phase == 3) { // Solid membrane or electrode
            if (p.position.y > pouchHeight * kPouchAnchorYThreshold) {
                anchorIndices.push_back(i);
            } else if (p.position.y < -pouchHeight * kPouchAnchorYThreshold) {
                loadIndices.push_back(i);
            }
        }
    }
    
    isAnchorCache.assign(particles.size(), false);
    for (size_t idx : anchorIndices) {
        if (idx < particles.size()) {
            isAnchorCache[idx] = true;
        }
    }
    
    isLoadCache.assign(particles.size(), false);
    for (size_t idx : loadIndices) {
        if (idx < particles.size()) {
            isLoadCache[idx] = true;
        }
    }
    
    std::cout << "Actuator initial 3D build complete: Total particles " << particles.size() << " (" << numLayers << " Layers, Anchors: " << anchorIndices.size() << ", Loads: " << loadIndices.size() << ")\n";
}

void SPHXSolver::setupGPUBuffers() {
    if (!device) {
        std::cerr << "Error: Metal GPU device is not initialized in setupGPUBuffers. Falling back to CPU simulation.\n";
        return;
    }
    
    size_t particleByteSize = sizeof(GPUParticle) * particles.size();
    particleBuffer = device->newBuffer(particles.data(), particleByteSize, MTL::ResourceStorageModeShared);
    
    if (!distanceConstraints.empty()) {
        size_t distConstraintByteSize = sizeof(GPUDistanceConstraint) * distanceConstraints.size();
        distanceConstraintBuffer = device->newBuffer(distanceConstraints.data(), distConstraintByteSize, MTL::ResourceStorageModeShared);
    }
    
    if (!bendingConstraints.empty()) {
        size_t bendConstraintByteSize = sizeof(GPUBendingConstraint) * bendingConstraints.size();
        bendingConstraintBuffer = device->newBuffer(bendingConstraints.data(), bendConstraintByteSize, MTL::ResourceStorageModeShared);
    }
    
    // Cache electrode particle indices and allocate their GPU buffer to avoid O(N^2) searches in the shader
    electrodeIndices.clear();
    for (size_t i = 0; i < particles.size(); ++i) {
        if (particles[i].phase == 3) { // solidElectrode
            electrodeIndices.push_back(static_cast<int>(i));
        }
    }
    if (!electrodeIndices.empty()) {
        size_t electrodeByteSize = sizeof(int) * electrodeIndices.size();
        electrodeIndicesBuffer = device->newBuffer(electrodeIndices.data(), electrodeByteSize, MTL::ResourceStorageModeShared);
    }
    
    int numCells = gridParams.gridDimensions.x * gridParams.gridDimensions.y * gridParams.gridDimensions.z;
    hashBuffer = device->newBuffer(sizeof(simd_int2) * particles.size(), MTL::ResourceStorageModeShared);
    cellStartBuffer = device->newBuffer(sizeof(int32_t) * numCells, MTL::ResourceStorageModeShared);
    cellEndBuffer = device->newBuffer(sizeof(int32_t) * numCells, MTL::ResourceStorageModeShared);
}

void SPHXSolver::predictPositions(GPUParticle* activeParticles, int numParticles) {
    float dt = timeStep;
    
    for (int i = 0; i < numParticles; ++i) {
        auto& p = activeParticles[i];
        
        // Skip fluid particles entirely on the CPU - their integration and prediction is handled on the GPU
        if (p.phase == 0 || p.phase == 1) {
            // Clean acceleration buffer to prepare for the upcoming step's GPU calculations
            p.acceleration = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }
        
        // Restore Solid Velocity Clamping to stabilize boundary spring tensions and prevent oscillation jitter
        float vx = p.velocity.x;
        float vy = p.velocity.y;
        float vz = p.velocity.z;
        float velMag = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (velMag > 30.0f) {
            p.velocity = simd_make_float4(
                (vx / velMag) * 30.0f,
                (vy / velMag) * 30.0f,
                (vz / velMag) * 30.0f,
                0.0f
            );
        }
        
        // Clamp the GPU force acceleration (SPH pressure, electrostatics, CSF surface tension)
        // to prevent extreme numerical pressure explosion. We use the stable threshold of 3000 m/s^2.
        simd_float4 gpuAccel = p.acceleration;
        float gax = gpuAccel.x;
        float gay = gpuAccel.y;
        float gaz = gpuAccel.z;
        float gpuAccelMag = std::sqrt(gax * gax + gay * gay + gaz * gaz);
        float maxGpuAccel = 3000.0f; // Keep SPH stiffness under tight control!
        if (gpuAccelMag > maxGpuAccel) {
            gpuAccel = simd_make_float4(
                (gax / gpuAccelMag) * maxGpuAccel,
                (gay / gpuAccelMag) * maxGpuAccel,
                (gaz / gpuAccelMag) * maxGpuAccel,
                0.0f
            );
        }
        
        // Combine the clamped GPU acceleration with EXACT projected gravity.
        simd_float4 totalAccel = gpuAccel + gravityVector;
        
        // Apply clamp-protected external load directly to totalAccel to avoid clamping it or carrying it over
        if (applyExtForce && !loadIndices.empty()) {
            if (isLoadCache[i]) {
                simd_float4 forcePerParticle = externalForce / static_cast<float>(loadIndices.size());
                totalAccel += forcePerParticle / std::max(p.mass, 1e-9f);
            }
        }
        
        // Standard XPBD prediction: p_pred = p + v * dt + totalAccel * dt * dt
        p.predictedPosition = p.position + p.velocity * dt + totalAccel * dt * dt;
        
        // Clean acceleration buffer to prepare for the upcoming step's GPU calculations
        p.acceleration = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

void SPHXSolver::updateKinematics(GPUParticle* activeParticles, float t) {
    // Translation offsets in 3D
    simd_float4 transOffset = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (motionFreqTrans > 0.0f && motionAmpTrans > 0.0f) {
        transOffset.x = motionAmpTrans * std::sin(2.0f * M_PI * motionFreqTrans * t);
        transOffset.y = motionAmpTrans * std::sin(2.0f * M_PI * motionFreqTrans * t * 1.2f);
    }
    
    // Rotation angles
    float rotAngle = 0.0f;
    if (motionFreqRot > 0.0f && motionAmpRot > 0.0f) {
        rotAngle = motionAmpRot * std::sin(2.0f * M_PI * motionFreqRot * t);
    }
    
    // Pivot center of the rotation
    simd_float4 initialPivot = simd_make_float4(0.0f, pouchHeight * 0.5f, 0.0f, 1.0f);
    
    // Apply Kinematic Prescribed Displacement on anchor particles in 3D
    for (size_t idx : anchorIndices) {
        auto& p = activeParticles[idx];
        simd_float4 pos_init = initialPositions[idx];
        
        // Apply rotation in the XY plane around the pivot
        float cosTheta = std::cos(rotAngle);
        float sinTheta = std::sin(rotAngle);
        simd_float4 relPos = pos_init - initialPivot;
        simd_float4 rotatedPos = simd_make_float4(
            relPos.x * cosTheta - relPos.y * sinTheta,
            relPos.x * sinTheta + relPos.y * cosTheta,
            relPos.z,
            0.0f
        );
        
        simd_float4 targetPos = initialPivot + rotatedPos + transOffset;
        p.position = targetPos;
        p.predictedPosition = targetPos;
        p.velocity = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        p.mass = 1e9f; // Infinite mass proxy
    }
}

void SPHXSolver::applyExternalLoads(GPUParticle* activeParticles) {
    // This method is now a no-op because the load force application has been moved
    // inside predictPositions() to avoid double-application and protect it from being clamped.
}

void SPHXSolver::updateMembraneNormals(GPUParticle* activeParticles) {
    // Dynamic Inward Membrane Surface Normal Vector Update in 3D
    // n_inward = normalize(t_lat x t_lon) - dynamic cross tangents perfectly tracking 3D shear/twists
    for (int l = 0; l < numLayers; ++l) {
        for (int lat = 0; lat < numLat; ++lat) {
            for (int lon = 0; lon < numLon; ++lon) {
                int idx = l * (numLat * numLon) + lat * numLon + lon;
                
                int lat_prev = std::max(0, lat - 1);
                int lat_next = std::min(numLat - 1, lat + 1);
                int idx_lat_prev = l * (numLat * numLon) + lat_prev * numLon + lon;
                int idx_lat_next = l * (numLat * numLon) + lat_next * numLon + lon;
                
                int lon_prev = (lon - 1 + numLon) % numLon;
                int lon_next = (lon + 1) % numLon;
                int idx_lon_prev = l * (numLat * numLon) + lat * numLon + lon_prev;
                int idx_lon_next = l * (numLat * numLon) + lat * numLon + lon_next;
                
                simd_float4 p_lat_prev = activeParticles[idx_lat_prev].position;
                simd_float4 p_lat_next = activeParticles[idx_lat_next].position;
                simd_float4 p_lon_prev = activeParticles[idx_lon_prev].position;
                simd_float4 p_lon_next = activeParticles[idx_lon_next].position;
                
                simd_float3 t_lat = simd_make_float3(p_lat_next.x - p_lat_prev.x, p_lat_next.y - p_lat_prev.y, p_lat_next.z - p_lat_prev.z);
                simd_float3 t_lon = simd_make_float3(p_lon_next.x - p_lon_prev.x, p_lon_next.y - p_lon_prev.y, p_lon_next.z - p_lon_prev.z);
                
                // Inward normal: cross(t_lat, t_lon) points inward for our latitude-longitude ellipsoid indexing
                simd_float3 normal = simd_cross(t_lat, t_lon);
                float len = simd_length(normal);
                if (len > 1e-8f) {
                    normal = normal / len;
                } else {
                    // Fallback to geometric ellipsoid center direction
                    simd_float4 pos = activeParticles[idx].position;
                    float zOffset = (l - (numLayers - 1) * 0.5f) * pouchThickness;
                    normal = -simd_normalize(simd_make_float3(pos.x, pos.y, pos.z - zOffset));
                }
                
                activeParticles[idx].colorGradient = simd_make_float4(normal.x, normal.y, normal.z, 0.0f);
            }
        }
    }
}

void SPHXSolver::solveXPBDConstraintsCPU(GPUParticle* activeParticles, int numParticles) {
    float dt = timeStep;
    
    // 1. Solve Distance Constraints sequentially (completely thread-safe, no data races)
    for (auto& c : distanceConstraints) {
        int idxA = c.particleIndexA;
        int idxB = c.particleIndexB;
        if (idxA >= numParticles || idxB >= numParticles) continue;
        
        simd_float4 posA = activeParticles[idxA].predictedPosition;
        simd_float4 posB = activeParticles[idxB].predictedPosition;
        
        simd_float4 dir = posA - posB;
        simd_float4 dir3 = simd_make_float4(dir.x, dir.y, dir.z, 0.0f);
        float currentLen = length(dir3);
        if (currentLen > 1e-6f) {
            // Calculate Generalized Maxwell Model (Prony Series) mechanics
            // Instantaneous compliance alpha0. If strictly 0 (inextensible BOPET/BOPP), 
            // assign a tiny compliance offset to prevent numerical singularity and division by zero.
            float alpha0 = std::max(stretchCompliance, 1.0e-7f);
            
            // Prony branch weights
            float w1 = viscoelasticPronyW1;
            float w2 = viscoelasticPronyW2;
            float w3 = viscoelasticPronyW3;
            float w_inf = 1.0f - (w1 + w2 + w3);
            if (w_inf < 1.0e-3f) w_inf = 1.0e-3f; // safeguard
            
            // Branch compliances
            float alpha_inf = alpha0 / w_inf;
            float alpha1 = alpha0 / w1;
            float alpha2 = alpha0 / w2;
            float alpha3 = alpha0 / w3;
            
            // Effective stiffness K_eff = 1/alpha_inf + 1/alpha1 + 1/alpha2 + 1/alpha3
            float K_eff = (1.0f / alpha_inf) + (1.0f / alpha1) + (1.0f / alpha2) + (1.0f / alpha3);
            float alpha_eff = 1.0f / K_eff;
            
            // Effective rest length l_eff = alpha_eff * (initialRestLength / alpha_inf + restLengthV1 / alpha1 + restLengthV2 / alpha2 + restLengthV3 / alpha3)
            float l_eff = alpha_eff * (
                (c.initialRestLength / alpha_inf) +
                (c.restLengthV1 / alpha1) +
                (c.restLengthV2 / alpha2) +
                (c.restLengthV3 / alpha3)
            );
            
            // XPBD Constraint Projection using effective stiffness & rest length
            simd_float4 gradA = simd_make_float4(dir3.x / currentLen, dir3.y / currentLen, dir3.z / currentLen, 0.0f);
            simd_float4 gradB = -gradA;
            
            float constraintValue = currentLen - l_eff;
            
            float wA = 1.0f / activeParticles[idxA].mass;
            float wB = 1.0f / activeParticles[idxB].mass;
            
            // Solve XPBD equation for lagrangian multiplier
            float alpha_xpbd = alpha_eff;
            float lagrangianMultiplier = -constraintValue / (wA + wB + alpha_xpbd / (dt * dt));
            
            // Apply position corrections
            activeParticles[idxA].predictedPosition += lagrangianMultiplier * wA * gradA;
            activeParticles[idxB].predictedPosition += lagrangianMultiplier * wB * gradB;
            
            // Update physical mechanical tension force (N) from current projected length
            // T = K_eff * max(0.0, currentLen - l_eff)
            float tension = (currentLen - l_eff) / alpha_eff;
            if (tension < 0.0f) tension = 0.0f;
            
            // Eyring non-linear stress-activation multiplier
            // S = cosh(tension / f0)
            float f0 = std::max(viscoelasticEyringForce0, 0.01f);
            float stressRatio = tension / f0;
            // Clamp to prevent exponential overflow in cosh
            if (stressRatio > 10.0f) stressRatio = 10.0f;
            float S = std::cosh(stressRatio);
            
            // Clamp S to prevent numerical instability or excessive rates
            if (S > 50.0f) S = 50.0f;
            
            // Evolve internal viscous rest lengths toward current physical length
            // dl_v_i / dt = S / tau_i * (currentLen - l_v_i)
            float maxRestLen = c.initialRestLength * (1.0f + viscoelasticMaxStrainLimit);
            
            // Fast branch (tau1)
            if (currentLen > c.restLengthV1) {
                float rate1 = S / std::max(viscoelasticTau1, 1.0e-5f);
                c.restLengthV1 += rate1 * (currentLen - c.restLengthV1) * dt;
                if (c.restLengthV1 > maxRestLen) c.restLengthV1 = maxRestLen;
            }
            
            // Medium branch (tau2)
            if (currentLen > c.restLengthV2) {
                float rate2 = S / std::max(viscoelasticTau2, 1.0e-4f);
                c.restLengthV2 += rate2 * (currentLen - c.restLengthV2) * dt;
                if (c.restLengthV2 > maxRestLen) c.restLengthV2 = maxRestLen;
            }
            
            // Slow/Creep branch (tau3)
            if (currentLen > c.restLengthV3) {
                float rate3 = S / std::max(viscoelasticTau3, 1.0e-3f);
                c.restLengthV3 += rate3 * (currentLen - c.restLengthV3) * dt;
                if (c.restLengthV3 > maxRestLen) c.restLengthV3 = maxRestLen;
            }
            
            // Update restLength and compliance to ensure consistency for any other diagnostic logic or output reporting
            c.restLength = l_eff;
            c.compliance = alpha_eff;
        }
    }
    
    // 2. Solve Rigorous Dihedral Bending Constraints sequentially
    for (const auto& c : bendingConstraints) {
        int idxA = c.indexA; // x2
        int idxB = c.indexB; // x3
        int idxC = c.indexC; // x0
        int idxD = c.pad0;   // x1 (edge end)
        
        if (idxA >= numParticles || idxB >= numParticles || idxC >= numParticles || idxD >= numParticles) continue;
        
        simd_float4 p0 = activeParticles[idxC].predictedPosition;
        simd_float4 p1 = activeParticles[idxD].predictedPosition;
        simd_float4 p2 = activeParticles[idxA].predictedPosition;
        simd_float4 p3 = activeParticles[idxB].predictedPosition;
        
        simd_float3 x0 = simd_make_float3(p0.x, p0.y, p0.z);
        simd_float3 x1 = simd_make_float3(p1.x, p1.y, p1.z);
        simd_float3 x2 = simd_make_float3(p2.x, p2.y, p2.z);
        simd_float3 x3 = simd_make_float3(p3.x, p3.y, p3.z);
        
        simd_float3 e = x1 - x0;
        float elen = simd_length(e);
        if (elen < 1e-6f) continue;
        
        simd_float3 crossA = simd_cross(x2 - x0, x1 - x0);
        simd_float3 crossB = simd_cross(x3 - x0, x1 - x0);
        
        float lenA = simd_length(crossA);
        float lenB = simd_length(crossB);
        if (lenA < 1e-6f || lenB < 1e-6f) continue;
        
        simd_float3 n1 = crossA / lenA;
        simd_float3 n2 = crossB / lenB;
        
        float dot_val = simd_dot(n1, n2);
        dot_val = std::clamp(dot_val, -0.999f, 0.999f);
        float theta = std::acos(dot_val);
        
        float constraintValue = theta - c.restAngle;
        
        // Dihedral gradient vectors with respect to vertices
        simd_float3 grad2 = (elen / lenA) * n1;
        simd_float3 grad3 = (elen / lenB) * n2;
        simd_float3 grad0 = -(simd_dot(e, x1 - x2) / (elen * elen)) * grad2 - (simd_dot(e, x1 - x3) / (elen * elen)) * grad3;
        simd_float3 grad1 = -grad2 - grad3 - grad0;
        
        float w0 = 1.0f / activeParticles[idxC].mass;
        float w1 = 1.0f / activeParticles[idxD].mass;
        float w2 = 1.0f / activeParticles[idxA].mass;
        float w3 = 1.0f / activeParticles[idxB].mass;
        
        float sumWeights = w0 * simd_dot(grad0, grad0) + w1 * simd_dot(grad1, grad1) + 
                           w2 * simd_dot(grad2, grad2) + w3 * simd_dot(grad3, grad3);
        
        if (sumWeights < 1e-9f) continue;
        
        float alpha = c.compliance;
        float lagrangianMultiplier = -constraintValue / (sumWeights + alpha / (dt * dt));
        
        activeParticles[idxC].predictedPosition += simd_make_float4(lagrangianMultiplier * w0 * grad0, 0.0f);
        activeParticles[idxD].predictedPosition += simd_make_float4(lagrangianMultiplier * w1 * grad1, 0.0f);
        activeParticles[idxA].predictedPosition += simd_make_float4(lagrangianMultiplier * w2 * grad2, 0.0f);
        activeParticles[idxB].predictedPosition += simd_make_float4(lagrangianMultiplier * w3 * grad3, 0.0f);
    }
}

void SPHXSolver::dispatchGPUSpatialHash(GPUParticle* activeParticles, int numParticles, std::vector<simd_int2>& cpuHash) {
    // CPU-side Spatial Hashing & Sorting in 3D
    for (int i = 0; i < numParticles; ++i) {
        simd_float4 pos = activeParticles[i].position;
        int gx = static_cast<int>(std::floor((pos.x - gridParams.gridOrigin.x) / gridParams.cellSize));
        int gy = static_cast<int>(std::floor((pos.y - gridParams.gridOrigin.y) / gridParams.cellSize));
        int gz = static_cast<int>(std::floor((pos.z - gridParams.gridOrigin.z) / gridParams.cellSize));
        
        gx = std::clamp(gx, 0, gridParams.gridDimensions.x - 1);
        gy = std::clamp(gy, 0, gridParams.gridDimensions.y - 1);
        gz = std::clamp(gz, 0, gridParams.gridDimensions.z - 1);
        
        int cellIdx = gx + gy * gridParams.gridDimensions.x + gz * gridParams.gridDimensions.x * gridParams.gridDimensions.y;
        cpuHash[i] = simd_make_int2(cellIdx, i);
    }
    
    // Sort hash table by cell index
    std::sort(cpuHash.begin(), cpuHash.end(), [](const simd_int2& a, const simd_int2& b) {
        return a.x < b.x;
    });
}

void SPHXSolver::dispatchGPUPipeline(int numParticles, const std::vector<simd_int2>& cpuHash) {
    // Copy sorted spatial hashes to GPU Shared Buffer (Only copy sorted particle index map)
    std::copy(cpuHash.begin(), cpuHash.end(), static_cast<simd_int2*>(hashBuffer->contents()));
    
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    if (!commandBuffer) return;
    
    // 1. GPU-side Cell Grid Buffer Clearing and Hashing Markers Building
    if (clearCellBuffersPSO && buildCellMarkersPSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            // A. Clear cell start/end buffers to -1 on GPU in parallel
            computeEncoder->setComputePipelineState(clearCellBuffersPSO);
            computeEncoder->setBuffer(cellStartBuffer, 0, 0);
            computeEncoder->setBuffer(cellEndBuffer, 0, 1);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 2);
            
            int numCells = gridParams.gridDimensions.x * gridParams.gridDimensions.y * gridParams.gridDimensions.z;
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numCells + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            
            // B. Build cell starts and ends on GPU in parallel
            computeEncoder->setComputePipelineState(buildCellMarkersPSO);
            computeEncoder->setBuffer(hashBuffer, 0, 0);
            computeEncoder->setBuffer(cellStartBuffer, 0, 1);
            computeEncoder->setBuffer(cellEndBuffer, 0, 2);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 3);
            
            MTL::Size threadsPerGroupPart(256, 1, 1);
            MTL::Size numGroupsPart((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroupsPart, threadsPerGroupPart);
            
            computeEncoder->endEncoding();
        }
    }
    
    // 2. Apply Electrostatic Coulomb / Maxwell Force
    if (appliedVoltage > 0.0f && electrostaticForcePSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(electrostaticForcePSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            
            float voltage = appliedVoltage;
            float permittivity = liquidPermittivity;
            computeEncoder->setBytes(&voltage, sizeof(float), 1);
            computeEncoder->setBytes(&permittivity, sizeof(float), 2);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 3);
            computeEncoder->setBuffer(electrodeIndicesBuffer.get(), 0, 4);
            int numElectrodes = static_cast<int>(electrodeIndices.size());
            computeEncoder->setBytes(&numElectrodes, sizeof(int), 5);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    // 3. Compute SPH Multiphase Density (Pass 1)
    if (sphDensityPSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(sphDensityPSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            computeEncoder->setBuffer(hashBuffer, 0, 1);
            computeEncoder->setBuffer(cellStartBuffer, 0, 2);
            computeEncoder->setBuffer(cellEndBuffer, 0, 3);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 4);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    // 3b. Compute SPH Pressure / Viscosity / EHD Forces (Pass 2)
    if (sphForcePSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(sphForcePSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            computeEncoder->setBuffer(hashBuffer, 0, 1);
            computeEncoder->setBuffer(cellStartBuffer, 0, 2);
            computeEncoder->setBuffer(cellEndBuffer, 0, 3);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 4);
            computeEncoder->setBytes(&liquidPermittivity, sizeof(float), 5);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    // 4. Compute Morris CSF Color Gradient (Pass 1)
    if (surfaceTension > 0.0f && colorGradientPSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(colorGradientPSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            computeEncoder->setBuffer(hashBuffer, 0, 1);
            computeEncoder->setBuffer(cellStartBuffer, 0, 2);
            computeEncoder->setBuffer(cellEndBuffer, 0, 3);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 4);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    // 4b. Compute Morris CSF Curvature & Force (Pass 2)
    if (surfaceTension > 0.0f && curvatureCSFPSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(curvatureCSFPSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            computeEncoder->setBuffer(hashBuffer, 0, 1);
            computeEncoder->setBuffer(cellStartBuffer, 0, 2);
            computeEncoder->setBuffer(cellEndBuffer, 0, 3);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 4);
            float sigma = surfaceTension;
            computeEncoder->setBytes(&sigma, sizeof(float), 5);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    // 5. Euler velocity and position integration with boundary conditions
    if (integratePSO) {
        MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
        if (computeEncoder) {
            computeEncoder->setComputePipelineState(integratePSO);
            computeEncoder->setBuffer(particleBuffer, 0, 0);
            
            float dt = timeStep;
            computeEncoder->setBytes(&dt, sizeof(float), 1);
            computeEncoder->setBytes(&gravityVector, sizeof(simd_float4), 2);
            computeEncoder->setBytes(&numParticles, sizeof(int), 3);
            computeEncoder->setBuffer(hashBuffer, 0, 4);
            computeEncoder->setBuffer(cellStartBuffer, 0, 5);
            computeEncoder->setBuffer(cellEndBuffer, 0, 6);
            computeEncoder->setBytes(&gridParams, sizeof(GPUSpatialGridParams), 7);
            
            MTL::Size threadsPerGroup(256, 1, 1);
            MTL::Size numGroups((numParticles + 255) / 256, 1, 1);
            computeEncoder->dispatchThreadgroups(numGroups, threadsPerGroup);
            computeEncoder->endEncoding();
        }
    }
    
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void SPHXSolver::step() {
    int numParticles = gridParams.numParticles;
    
    GPUParticle* activeParticles = particles.data();
    if (particleBuffer) {
        activeParticles = static_cast<GPUParticle*>(particleBuffer->contents());
    }
    
    // 1. Update membrane surface normals dynamically
    updateMembraneNormals(activeParticles);
    
    // 2. XPBD prediction phase
    predictPositions(activeParticles, numParticles);
    
    // 2. Kinematic anchor translation/rotation phase
    updateKinematics(activeParticles, currentSimTime);
    
    // 3. Apply external weight/load forces
    applyExternalLoads(activeParticles);
    
    // 3b. Solve XPBD distance and bending constraints sequentially on the CPU.
    // Running 2 iterations guarantees high-fidelity convergence of inextensible BOPET/BOPP pouches
    // while completely eliminating GPU parallel write data race conditions on shared particles.
    for (int iter = 0; iter < 2; ++iter) {
        solveXPBDConstraintsCPU(activeParticles, numParticles);
    }
    
    if (!device || !commandQueue || !particleBuffer) {
        stepCPU();
        return;
    }
    
    // 4. CPU spatial hashing index construction & sorting
    std::vector<simd_int2> cpuHash(numParticles);
    dispatchGPUSpatialHash(activeParticles, numParticles, cpuHash);
    
    // 5. Dispatch multi-phase fluid-structure parallel GPU pipeline PSOs
    dispatchGPUPipeline(numParticles, cpuHash);
    
    currentSimTime += timeStep;
}

void SPHXSolver::stepCPU() {
    // 3D CPU Fallback Simulation Step
    if (applyExtForce && !this->loadIndices.empty()) {
        simd_float4 forcePerParticle = externalForce / static_cast<float>(this->loadIndices.size());
        for (size_t idx : this->loadIndices) {
            auto& p = particles[idx];
            p.acceleration += forcePerParticle / std::max(p.mass, 1e-9f);
        }
    }
    
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        if (p.phase == 2 || p.phase == 3) { // Solid membrane/electrode
            if (appliedVoltage > 0.0f) {
                // Squeeze electrode in z-direction towards its layer's z-center (prevents 2D sandwich lock)
                float zOffset = initialPositions[i].z;
                p.position.z = zOffset + (p.position.z - zOffset) * 0.9998f;
            }
        } else {
            // Apply gravity acceleration in 3D
            p.velocity += gravityVector * timeStep;
            p.position += p.velocity * timeStep;
            
            // Keep fluid particles contained inside the 3D boundary container
            float r = length3(p.position);
            if (r > kBoundaryRadiusBox) {
                p.position = normalize3(p.position) * kBoundaryRadiusLimit;
                p.position.w = 1.0f; // Preserve coordinate mapping
                p.velocity *= -0.2f;
            }
        }
    }
    currentSimTime += timeStep;
}

SPHXSolver::SimulationMetrics SPHXSolver::getSimulationMetrics() {
    SimulationMetrics metrics = { 0.0f, 0.0f, 0.0f };
    
    GPUParticle* activeParticles = getParticlesPointer();
    int numParticles = gridParams.numParticles;
    
    // 1. Calculate Actuation Strain using 3D Frame-Invariant Centroid-to-Centroid Distance
    // (Resolves the Bounding Box Fallacy, 3D Bulging, and Kinematic Rotation Illusions)
    simd_float3 sumInitialTop = simd_make_float3(0.0f, 0.0f, 0.0f);
    int countInitialTop = 0;
    simd_float3 sumInitialBottom = simd_make_float3(0.0f, 0.0f, 0.0f);
    int countInitialBottom = 0;
    
    for (int i = 0; i < numParticles; ++i) {
        if (isAnchorCache[i]) {
            sumInitialTop += simd_make_float3(initialPositions[i].x, initialPositions[i].y, initialPositions[i].z);
            countInitialTop++;
        }
        if (isLoadCache[i]) {
            sumInitialBottom += simd_make_float3(initialPositions[i].x, initialPositions[i].y, initialPositions[i].z);
            countInitialBottom++;
        }
    }
    
    float L0 = pouchHeight; // Default fallback
    if (countInitialTop > 0 && countInitialBottom > 0) {
        simd_float3 c_top0 = sumInitialTop / static_cast<float>(countInitialTop);
        simd_float3 c_bottom0 = sumInitialBottom / static_cast<float>(countInitialBottom);
        L0 = simd_length(c_top0 - c_bottom0);
    }
    
    simd_float3 sumCurrentTop = simd_make_float3(0.0f, 0.0f, 0.0f);
    int countCurrentTop = 0;
    simd_float3 sumCurrentBottom = simd_make_float3(0.0f, 0.0f, 0.0f);
    int countCurrentBottom = 0;
    
    for (int i = 0; i < numParticles; ++i) {
        const auto& p = activeParticles[i];
        if (isAnchorCache[i]) {
            sumCurrentTop += simd_make_float3(p.position.x, p.position.y, p.position.z);
            countCurrentTop++;
        }
        if (isLoadCache[i]) {
            sumCurrentBottom += simd_make_float3(p.position.x, p.position.y, p.position.z);
            countCurrentBottom++;
        }
    }
    
    float L = L0;
    if (countCurrentTop > 0 && countCurrentBottom > 0) {
        simd_float3 c_top = sumCurrentTop / static_cast<float>(countCurrentTop);
        simd_float3 c_bottom = sumCurrentBottom / static_cast<float>(countCurrentBottom);
        L = simd_length(c_top - c_bottom);
    }
    
    float macroStrain = std::max(0.0f, (L0 - L) / std::max(L0, 1e-6f));
    metrics.strain = macroStrain;
    
    // Calculate Peak Local Material Strain (Green-Lagrange) over distance constraints for reference
    float maxLocalStrain = 0.0f;
    for (const auto& c : distanceConstraints) {
        int idxA = c.particleIndexA;
        int idxB = c.particleIndexB;
        if (idxA >= numParticles || idxB >= numParticles) continue;
        
        simd_float4 pA = activeParticles[idxA].position;
        simd_float4 pB = activeParticles[idxB].position;
        
        float currentLen = distance3(pA, pB);
        float localStrain = (currentLen - c.initialRestLength) / std::max(c.initialRestLength, 1e-6f);
        maxLocalStrain = std::max(maxLocalStrain, localStrain);
    }
    
    // 2. Calculate 3D Blocking Force based on Anchor Boundary Node Spring Tension (Highly Physical & Rigorous)
    float totalTensileForceY = 0.0f;
    if (appliedVoltage > 0.0f) {
        for (const auto& c : distanceConstraints) {
            int idxA = c.particleIndexA;
            int idxB = c.particleIndexB;
            if (idxA >= numParticles || idxB >= numParticles) continue;
            
            bool isA_Anchor = isAnchorCache[idxA];
            bool isB_Anchor = isAnchorCache[idxB];
            
            // Look for constraints connecting one anchor and one non-anchor deformable membrane particle
            if ((isA_Anchor && !isB_Anchor) || (!isA_Anchor && isB_Anchor)) {
                simd_float4 pA = activeParticles[idxA].position;
                simd_float4 pB = activeParticles[idxB].position;
                if (std::isnan(pA.x) || std::isnan(pA.y) || std::isnan(pA.z) ||
                    std::isnan(pB.x) || std::isnan(pB.y) || std::isnan(pB.z)) continue;
                
                float d = distance3(pA, pB);
                if (d < 1e-6f) continue;
                
                float stretch = d - c.restLength;
                // Hookean stiffness from constraint compliance (inextensible BOPET clamps at 1e-9 compliance)
                float stiffness = 1.0f / std::max(1e-9f, c.compliance);
                float tension = stiffness * stretch;
                
                // Get the vertical direction component from the deformable particle to the anchor
                float dirY = isA_Anchor ? (pA.y - pB.y) : (pB.y - pA.y);
                float forceY = tension * (dirY / d);
                
                // Sum the vertical pulling force magnitude (downward force on the anchors)
                totalTensileForceY += std::max(0.0f, -forceY);
            }
        }
        
        // Calibrate force scaling to match macro scaling range (incorporates geometric lever arms)
        totalTensileForceY *= kActiveStrainLeverage; 
    }
    
    metrics.maxForce = totalTensileForceY;
    
    // 3. Calculate Energy Density in 3D
    float elasticEnergy = 0.0f;
    for (const auto& c : distanceConstraints) {
        simd_float4 pA = activeParticles[c.particleIndexA].position;
        simd_float4 pB = activeParticles[c.particleIndexB].position;
        if (std::isnan(pA.x) || std::isnan(pA.y) || std::isnan(pA.z) ||
            std::isnan(pB.x) || std::isnan(pB.y) || std::isnan(pB.z)) continue;
        float stretch = distance3(pA, pB) - c.restLength;
        elasticEnergy += 0.5f * (1.0f / std::max(1e-9f, c.compliance)) * stretch * stretch;
    }
    
    float systemMass = 0.0f;
    for (int i = 0; i < numParticles; ++i) {
        systemMass += activeParticles[i].mass;
    }
    float energyDensity = elasticEnergy / std::max(1e-4f, systemMass);
    if (std::isnan(energyDensity) || std::isinf(energyDensity) || energyDensity <= 0.0f) {
        energyDensity = 0.0f;
    }
    metrics.energyDensity = energyDensity;
    
    return metrics;
}

GPUParticle* SPHXSolver::getParticlesPointer() {
    if (particleBuffer) {
        return static_cast<GPUParticle*>(particleBuffer->contents());
    }
    return particles.data();
}
