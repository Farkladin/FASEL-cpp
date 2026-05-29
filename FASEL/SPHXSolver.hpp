#ifndef SPHX_SOLVER_HPP
#define SPHX_SOLVER_HPP

#include <vector>
#include "Particle.hpp"

// Forward declarations of metal-cpp types
namespace MTL {
    class Device;
    class CommandQueue;
    class ComputePipelineState;
    class Buffer;
}

// Modern C++ RAII wrapper class to manage the lifecycle of metal-cpp resources automatically
template<typename T>
class MetalPtr {
private:
    T* m_ptr = nullptr;
public:
    MetalPtr() = default;
    explicit MetalPtr(T* ptr) : m_ptr(ptr) {}
    ~MetalPtr() {
        if (m_ptr) {
            m_ptr->release();
            m_ptr = nullptr;
        }
    }
    
    // Disable copy
    MetalPtr(const MetalPtr&) = delete;
    MetalPtr& operator=(const MetalPtr&) = delete;
    
    // Enable move
    MetalPtr(MetalPtr&& other) noexcept : m_ptr(other.m_ptr) {
        other.m_ptr = nullptr;
    }
    MetalPtr& operator=(MetalPtr&& other) noexcept {
        if (this != &other) {
            if (m_ptr) m_ptr->release();
            m_ptr = other.m_ptr;
            other.m_ptr = nullptr;
        }
        return *this;
    }
    
    void reset(T* ptr = nullptr) {
        if (m_ptr) m_ptr->release();
        m_ptr = ptr;
    }
    
    T* get() const { return m_ptr; }
    T* operator->() const { return m_ptr; }
    explicit operator bool() const { return m_ptr != nullptr; }
    operator T*() const { return m_ptr; }
    
    MetalPtr& operator=(T* ptr) {
        reset(ptr);
        return *this;
    }
};


class SPHXSolver {
public:
    // Physical & Numerical Simulation Constants
    float kPouchAnchorYThreshold = 0.43f;   // top/bottom anchor/load height threshold
    float kBoundaryRadiusBox = 0.025f;       // boundary container box limits
    float kBoundaryRadiusLimit = 0.024f;
    float kActiveStrainLeverage = 0.01f;     // macroscopic force leverage factor
    
    // Physical system data
    std::vector<GPUParticle> particles;
    std::vector<simd_float4> initialPositions; // Stores initial coordinates for reference
    std::vector<GPUDistanceConstraint> distanceConstraints;
    std::vector<GPUBendingConstraint> bendingConstraints;
    
    GPUSpatialGridParams gridParams;
    simd_float4 gravityVector = {0.0f, -9.80665f, 0.0f, 0.0f}; // Upgraded to 3D float4
    float timeStep = 0.000025f; // Optimized from 40us to 25us for extreme numerical stability
    float currentSimTime = 0.0f;
    
    // Kinematic motion options
    float motionFreqTrans = 0.0f; // Hz
    float motionAmpTrans = 0.0f;  // meters
    float motionFreqRot = 0.0f;    // Hz
    float motionAmpRot = 0.0f;     // radians
    
    // External force options
    simd_float4 externalForce = {0.0f, 0.0f, 0.0f, 0.0f}; // Upgraded to 3D float4
    bool applyExtForce = false;
    
    // Simulator options
    float gasVolumeFraction;
    float appliedVoltage;
    float liquidPermittivity;
    float liquidViscosity;
    float surfaceTension;
    
    // SPH viscosity options
    float sphViscosityAlpha = 0.15f;
    float sphViscosityBeta = 0.30f;
    
    // Pouch dimensions and material properties
    float pouchWidth = 0.02f;
    float pouchHeight = 0.04f;
    float pouchThickness = 0.004f; // Added 3D thickness (4mm)
    float stretchCompliance = 1e-6f;
    float bendingCompliance = 1e-5f;
    float fluidSpacing = 0.001f;
    int numShellParticles = 120;
    int numLayers = 1;
    
    static constexpr int numLat = 12;
    static constexpr int numLon = 15;
    
    // Viscoelasticity options (Generalized Maxwell Prony Series + Eyring model)
    float viscoelasticRelaxationRate = 0.05f;   // s^-1 (Legacy fallback)
    float viscoelasticMaxStrainLimit = 0.12f;   // 12% max creep
    float viscoelasticPronyW1 = 0.30f;          // Prony branch 1 stiffness weight
    float viscoelasticPronyW2 = 0.30f;          // Prony branch 2 stiffness weight
    float viscoelasticPronyW3 = 0.20f;          // Prony branch 3 stiffness weight
    float viscoelasticTau1 = 0.001f;            // Prony branch 1 relaxation time (seconds)
    float viscoelasticTau2 = 0.010f;            // Prony branch 2 relaxation time (seconds)
    float viscoelasticTau3 = 0.100f;            // Prony branch 3 relaxation time (seconds)
    float viscoelasticEyringForce0 = 0.50f;      // Eyring activation reference force (N)
    
    SPHXSolver(
        float gasVolumeFraction,
        float appliedVoltage,
        float liquidPermittivity = 4.5f,
        float liquidViscosity = 0.05f,
        float surfaceTension = 0.03f,
        float pouchWidth = 0.02f,
        float pouchHeight = 0.04f,
        float stretchCompliance = 1e-6f,
        float bendingCompliance = 1e-5f,
        float fluidSpacing = 0.001f,
        int numShellParticles = 120,
        int numLayers = 1,
        float timeStep = 0.000025f,
        float kPouchAnchorYThreshold = 0.43f,
        float kBoundaryRadiusBox = 0.025f,
        float kBoundaryRadiusLimit = 0.024f,
        float kActiveStrainLeverage = 12.0f
    );
    
    ~SPHXSolver();
    
    void step();
    
    struct SimulationMetrics {
        float strain;
        float maxForce;
        float energyDensity;
    };
    
    SimulationMetrics getSimulationMetrics();
    
    GPUParticle* getParticlesPointer();
    
private:
    void setupMetal();
    void buildActuatorGeometry();
    void setupGPUBuffers();
    void stepCPU();
    
    void predictPositions(GPUParticle* activeParticles, int numParticles);
    void updateKinematics(GPUParticle* activeParticles, float t);
    void updateMembraneNormals(GPUParticle* activeParticles);
    void applyExternalLoads(GPUParticle* activeParticles);
    void solveXPBDConstraintsCPU(GPUParticle* activeParticles, int numParticles);
    void dispatchGPUSpatialHash(GPUParticle* activeParticles, int numParticles, std::vector<simd_int2>& cpuHash);
    void dispatchGPUPipeline(int numParticles, const std::vector<simd_int2>& cpuHash);
    
    // Cached kinematic anchor and external load particle indices
    std::vector<size_t> anchorIndices;
    std::vector<size_t> loadIndices;
    std::vector<bool> isAnchorCache;
    std::vector<bool> isLoadCache;
    
    // GPU hardware resources
    MetalPtr<MTL::Device> device;
    MetalPtr<MTL::CommandQueue> commandQueue;
    
    // Compute Pipeline States
    MetalPtr<MTL::ComputePipelineState> clearCellBuffersPSO;
    MetalPtr<MTL::ComputePipelineState> buildCellMarkersPSO;
    MetalPtr<MTL::ComputePipelineState> electrostaticForcePSO;
    MetalPtr<MTL::ComputePipelineState> sphDensityPSO;
    MetalPtr<MTL::ComputePipelineState> sphForcePSO;
    MetalPtr<MTL::ComputePipelineState> colorGradientPSO;
    MetalPtr<MTL::ComputePipelineState> curvatureCSFPSO;
    MetalPtr<MTL::ComputePipelineState> integratePSO;
    
    // GPU Buffers
    MetalPtr<MTL::Buffer> particleBuffer;
    MetalPtr<MTL::Buffer> distanceConstraintBuffer;
    MetalPtr<MTL::Buffer> bendingConstraintBuffer;
    
    // Spatial hashing helper GPU buffers
    MetalPtr<MTL::Buffer> hashBuffer;
    MetalPtr<MTL::Buffer> cellStartBuffer;
    MetalPtr<MTL::Buffer> cellEndBuffer;
    
    // Cached electrode indices for highly optimized GPU Coulomb force calculations
    std::vector<int> electrodeIndices;
    MetalPtr<MTL::Buffer> electrodeIndicesBuffer;
};

#endif // SPHX_SOLVER_HPP
