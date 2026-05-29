#ifndef PARTICLE_HPP
#define PARTICLE_HPP

#include <simd/simd.h>

/// Constant values representing particle physical phases
enum class ParticlePhase : int32_t {
    liquid = 0,
    gas = 1,
    solidMembrane = 2,
    solidElectrode = 3
};

/// Particle physical state structure mapping 1:1 with the Metal Shader in 3D
struct alignas(16) GPUParticle {
    simd_float4 position;          // 16 bytes (x, y, z, 1.0)
    simd_float4 velocity;          // 16 bytes (vx, vy, vz, 0.0)
    simd_float4 acceleration;      // 16 bytes (ax, ay, az, 0.0)
    simd_float4 predictedPosition; // 16 bytes (px, py, pz, 1.0)
    
    float density;            // 4 bytes
    float pressure;           // 4 bytes
    float mass;               // 4 bytes
    int32_t phase;            // 4 bytes (0: Liquid, 1: Gas, 2: Membrane, 3: Electrode)
    
    float dbi;                // 4 bytes (Dielectric Breakdown Index)
    float color;              // 4 bytes (Liquid: 1.0, Gas: 0.0)
    float pad0;               // 4 bytes padding
    float pad1;               // 4 bytes padding
    
    simd_float4 colorGradient; // 16 bytes
    
    float curvature;          // 4 bytes
    int32_t gasBubbleID;      // 4 bytes (for identifying gas bubble ID)
    float electricPotential;  // 4 bytes (stores Poisson Solver potential results)
    float eFieldGradientNorm; // 4 bytes (Frobenius norm of local electric field gradient tensor)
    
    simd_float4 electricField; // 16 bytes
    
    GPUParticle() = default;
    GPUParticle(
        simd_float4 pos,
        simd_float4 vel = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f),
        float m = 1.0f,
        int32_t ph = 0,
        int32_t bubbleID = -1
    ) : position(pos), velocity(vel), acceleration(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f)), predictedPosition(pos),
        density(0.0f), pressure(0.0f), mass(m), phase(ph), dbi(0.0f),
        color((ph == 0) ? 1.0f : 0.0f), pad0(0.0f), pad1(0.0f),
        colorGradient(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f)), curvature(0.0f),
        gasBubbleID(bubbleID), electricPotential(0.0f), eFieldGradientNorm(0.0f),
        electricField(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f)) {}
};

/// 3D Grid domain and tuning parameters structure for spatial hashing
struct alignas(16) GPUSpatialGridParams {
    simd_float4 gridOrigin;       // 16 bytes (minX, minY, minZ, 0)
    float cellSize;               // 4 bytes
    simd_int4 gridDimensions;     // 16 bytes (nx, ny, nz, 0)
    int32_t numParticles;         // 4 bytes
    float sphViscosityAlpha;      // 4 bytes (replaces pad0)
    float sphViscosityBeta;       // 4 bytes (replaces pad1)
    
    GPUSpatialGridParams() = default;
    GPUSpatialGridParams(simd_float4 origin, float size, simd_int4 dims, int32_t num)
        : gridOrigin(origin), cellSize(size), gridDimensions(dims), numParticles(num), sphViscosityAlpha(0.15f), sphViscosityBeta(0.30f) {}
};

/// Distance constraint for XPBD outer membrane in 3D
struct GPUDistanceConstraint {
    int32_t particleIndexA;
    int32_t particleIndexB;
    float restLength;
    float compliance;
    float initialRestLength;
    float restLengthV1;
    float restLengthV2;
    float restLengthV3;
    
    GPUDistanceConstraint() = default;
    GPUDistanceConstraint(int32_t indexA, int32_t indexB, float length, float comp = 0.0f)
        : particleIndexA(indexA), particleIndexB(indexB), restLength(length), compliance(comp), initialRestLength(length),
          restLengthV1(length), restLengthV2(length), restLengthV3(length) {}
};

/// Bending constraint for XPBD outer membrane in 3D (32 bytes aligned)
struct GPUBendingConstraint {
    int32_t indexA;
    int32_t indexB;
    int32_t indexC; // center bending node
    float restAngle;
    float compliance;
    int32_t pad0;
    int32_t pad1;
    int32_t pad2;
    
    GPUBendingConstraint() = default;
    GPUBendingConstraint(int32_t idxA, int32_t idxB, int32_t idxC, float angle, float comp = 0.0f)
        : indexA(idxA), indexB(idxB), indexC(idxC), restAngle(angle), compliance(comp), pad0(0), pad1(0), pad2(0) {}
};

static_assert(sizeof(GPUParticle) == 144, "GPUParticle layout size must be exactly 144 bytes for UMA compatibility!");
static_assert(sizeof(GPUSpatialGridParams) == 64, "GPUSpatialGridParams layout size must be exactly 64 bytes for UMA compatibility!");

#endif // PARTICLE_HPP
