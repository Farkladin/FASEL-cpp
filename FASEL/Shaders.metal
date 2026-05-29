//
//  Shaders.metal
//  FASEL
//
//  Created by Antigravity on 5/27/26.
//

#include <metal_stdlib>
using namespace metal;

// MARK: - Swift compatible physical data structure definitions in 3D
struct GPUParticle {
    float4 position;          // 16 bytes (x, y, z, 1.0)
    float4 velocity;          // 16 bytes (vx, vy, vz, 0.0)
    float4 acceleration;      // 16 bytes (ax, ay, az, 0.0)
    float4 predictedPosition; // 16 bytes (px, py, pz, 1.0)
    
    float density;            // 4 bytes
    float pressure;           // 4 bytes
    float mass;               // 4 bytes
    int phase;                // 4 bytes (0: Liquid, 1: Gas, 2: Membrane, 3: Electrode)
    
    float dbi;                // 4 bytes (Dielectric Breakdown Index)
    float color;              // 4 bytes (Liquid: 1.0, Gas: 0.0)
    float pad0;               // 4 bytes padding
    float pad1;               // 4 bytes padding
    
    float4 colorGradient;     // 16 bytes
    
    float curvature;          // 4 bytes
    int gasBubbleID;          // 4 bytes
    float electricPotential;  // 4 bytes
    float pad2;               // 4 bytes padding
    
    float4 electricField;     // 16 bytes
};

struct GPUSpatialGridParams {
    float4 gridOrigin;       // 16 bytes
    float cellSize;          // 4 bytes
    int4 gridDimensions;     // 16 bytes (nx, ny, nz, 0)
    int numParticles;        // 4 bytes
    float sphViscosityAlpha;
    float sphViscosityBeta;
};

struct GPUDistanceConstraint {
    int particleIndexA;
    int particleIndexB;
    float restLength;
    float compliance;
    float initialRestLength;
    float restLengthV1;
    float restLengthV2;
    float restLengthV3;
};

struct GPUBendingConstraint {
    int indexA;
    int indexB;
    int indexC;
    float restAngle;
    float compliance;
    int pad0;
    int pad1;
    int pad2;
};

// MARK: - 3D SPH Kernel Interpolation Functions
static float SPH_Kernel_W(float4 r, float h) {
    float d = length(r.xyz);
    if (d >= h) return 0.0f;
    
    float q = d / h;
    float factor = 8.0f / (M_PI_F * h * h * h); // 3D cubic spline normalization factor
    
    if (q < 0.5f) {
        return factor * (6.0f * (q * q * q - q * q) + 1.0f);
    } else {
        float one_minus_q = 1.0f - q;
        return factor * 2.0f * one_minus_q * one_minus_q * one_minus_q;
    }
}

static float4 SPH_Kernel_GradW(float4 r, float h) {
    float d = length(r.xyz);
    if (d == 0.0f || d >= h) return float4(0.0f);
    
    float q = d / h;
    float factor = 8.0f / (M_PI_F * h * h * h);
    float4 dir = float4(r.xyz / d, 0.0f);
    
    if (q < 0.5f) {
        return factor * (18.0f * q * q - 12.0f * q) / h * dir;
    } else {
        float one_minus_q = 1.0f - q;
        return factor * (-6.0f * one_minus_q * one_minus_q) / h * dir;
    }
}

// MARK: - 1. Spatial Hash Kernel (3D spatial grid neighbor search helper)
kernel void clearCellBuffersKernel(
    device int* cellStartBuffer             [[buffer(0)]],
    device int* cellEndBuffer               [[buffer(1)]],
    constant GPUSpatialGridParams& grid     [[buffer(2)]],
    uint id                                 [[thread_position_in_grid]])
{
    int totalCells = grid.gridDimensions.x * grid.gridDimensions.y * grid.gridDimensions.z;
    if (id >= (uint)totalCells) return;
    cellStartBuffer[id] = -1;
    cellEndBuffer[id] = -1;
}

kernel void spatialHashKernel(
    device GPUParticle* particles           [[buffer(0)]],
    device int2* hashBuffer                 [[buffer(1)]], // (cellIndex, particleIndex)
    constant GPUSpatialGridParams& grid     [[buffer(2)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    
    float4 pos = particles[id].position;
    
    // Get 3D cell coordinates in space
    int3 cellCoord = int3(floor((pos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
    
    // Safe boundary handling
    cellCoord = clamp(cellCoord, int3(0), grid.gridDimensions.xyz - 1);
    
    int cellIndex = cellCoord.x + cellCoord.y * grid.gridDimensions.x + cellCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
    
    hashBuffer[id] = int2(cellIndex, (int)id);
}

kernel void buildCellMarkersKernel(
    device int2* hashBuffer                 [[buffer(0)]], // (cellIndex, particleIndex)
    device int* cellStartBuffer             [[buffer(1)]],
    device int* cellEndBuffer               [[buffer(2)]],
    constant GPUSpatialGridParams& grid     [[buffer(3)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    
    int cellIdx = hashBuffer[id].x;
    
    // Check if first particle in this cell
    if (id == 0 || cellIdx != hashBuffer[id - 1].x) {
        cellStartBuffer[cellIdx] = (int)id;
    }
    
    // Check if last particle in this cell
    if (id == (uint)(grid.numParticles - 1) || cellIdx != hashBuffer[id + 1].x) {
        cellEndBuffer[cellIdx] = (int)id;
    }
}

// MARK: - 2. Electrostatic Maxwell Force Kernel (3D Coulomb Zipping)
kernel void electrostaticForceKernel(
    device GPUParticle* particles           [[buffer(0)]],
    constant float& voltage                 [[buffer(1)]],
    constant float& permittivity            [[buffer(2)]],
    constant GPUSpatialGridParams& grid     [[buffer(3)]],
    device int* electrodeIndices            [[buffer(4)]],
    constant int& numElectrodes             [[buffer(5)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    if (particles[id].phase != 3) return; 
    
    float4 E = particles[id].electricField;
    float E_mag = length(E.xyz);
    
    if (E_mag > 1.0e-3f) {
        float eps_0 = 8.854187817e-12f;
        float eps_r = permittivity;
        
        // Local inward surface normal vector of the membrane (precomputed dynamically in C++)
        float4 n = particles[id].colorGradient;
        
        // Maxwell Stress Tensor traction force: f = eps_0 * eps_r * ( E * (E . n) - 0.5 * |E|^2 * n )
        float E_dot_n = dot(E.xyz, n.xyz);
        float4 traction = (E * E_dot_n - 0.5f * (E_mag * E_mag) * n);
        traction.w = 0.0f;
        
        // Calculate discrete particle area based on active electrode area (~35% of pouch)
        // pouch active area = 0.02m * 0.04m * 0.35 = 0.00028 m^2
        float totalActiveArea = 0.00028f;
        float particleArea = totalActiveArea / max(1.0f, (float)numElectrodes);
        
        float4 totalForce = traction * (eps_r * eps_0) * particleArea;
        
        // Apply electrostatic acceleration
        particles[id].acceleration += totalForce / particles[id].mass;
    }
}

static float hash3d(float3 p) {
    return fract(sin(dot(p, float3(127.1, 311.7, 74.7))) * 43758.5453123);
}

// MARK: - 3. SPH Density & Tait EOS Pressure Kernel (Pass 1)
kernel void sphDensityKernel(
    device GPUParticle* particles           [[buffer(0)]],
    device int2* hashBuffer                 [[buffer(1)]],
    device int* cellStartBuffer             [[buffer(2)]],
    device int* cellEndBuffer               [[buffer(3)]],
    constant GPUSpatialGridParams& grid     [[buffer(4)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    if (particles[id].phase == 2 || particles[id].phase == 3) return; // solids excluded from SPH
    
    float4 myPos = particles[id].position;
    float densitySum = 0.0f;
    float h = grid.cellSize;
    
    // 3D 27-cell spatial hashing neighbor search
    int3 cellCoord = int3(floor((myPos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
    
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 neighborCoord = cellCoord + int3(dx, dy, dz);
                if (neighborCoord.x < 0 || neighborCoord.x >= grid.gridDimensions.x ||
                    neighborCoord.y < 0 || neighborCoord.y >= grid.gridDimensions.y ||
                    neighborCoord.z < 0 || neighborCoord.z >= grid.gridDimensions.z) continue;
                    
                int cellIdx = neighborCoord.x + neighborCoord.y * grid.gridDimensions.x + neighborCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
                int start = cellStartBuffer[cellIdx];
                int end = cellEndBuffer[cellIdx];
                
                if (start == -1) continue;
                
                for (int i = start; i <= end; ++i) {
                    int j = hashBuffer[i].y;
                    if (particles[j].phase == 2 || particles[j].phase == 3) continue;
                    
                    float d = distance(myPos.xyz, particles[j].position.xyz);
                    if (d < h) {
                        densitySum += SPH_Kernel_W(myPos - particles[j].position, h);
                    }
                }
            }
        }
    }
    
    particles[id].density = max(particles[id].mass * densitySum, 0.01f);
    
    // Calculate pressure (Polytropic adiabatic EoS for gas, Tait EoS for liquid)
    if (particles[id].phase == 1) {
        float gas_rho0 = 1.2f;
        float p0 = 150.0f; // Reference pressure scale for gas SPH sound speed
        float gamma_gas = 1.4f; // Polytropic index for gas (adiabatic air)
        float P = p0 * (pow(max(particles[id].density, 0.01f) / gas_rho0, gamma_gas) - 1.0f);
        particles[id].pressure = max(0.0f, P); // Clamp to prevent unphysical gas clumping under negative pressure
    } else {
        float liquid_rho0 = 960.0f;
        float B = 2000.0f;
        float gamma = 7.0f;
        float P = B * (pow(particles[id].density / liquid_rho0, gamma) - 1.0f);
        particles[id].pressure = max(-200.0f, P); // Prevent extreme negative pressure and SPH tensile clumping instability
    }
}

// MARK: - 3b. SPH Force Calculation Kernel (Pass 2)
kernel void sphForceKernel(
    device GPUParticle* particles           [[buffer(0)]],
    device int2* hashBuffer                 [[buffer(1)]],
    device int* cellStartBuffer             [[buffer(2)]],
    device int* cellEndBuffer               [[buffer(3)]],
    constant GPUSpatialGridParams& grid     [[buffer(4)]],
    constant float& permittivity            [[buffer(5)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    if (particles[id].phase == 2 || particles[id].phase == 3) return; // solids excluded from SPH
    
    float4 myPos = particles[id].position;
    float h = grid.cellSize;
    int3 cellCoord = int3(floor((myPos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
    
    // SPH Pressure gradient and viscosity forces
    float4 pressForce = float4(0.0f);
    float4 viscForce = float4(0.0f);
    
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 neighborCoord = cellCoord + int3(dx, dy, dz);
                if (neighborCoord.x < 0 || neighborCoord.x >= grid.gridDimensions.x ||
                    neighborCoord.y < 0 || neighborCoord.y >= grid.gridDimensions.y ||
                    neighborCoord.z < 0 || neighborCoord.z >= grid.gridDimensions.z) continue;
                    
                int cellIdx = neighborCoord.x + neighborCoord.y * grid.gridDimensions.x + neighborCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
                int start = cellStartBuffer[cellIdx];
                int end = cellEndBuffer[cellIdx];
                
                if (start == -1) continue;
                
                for (int i = start; i <= end; ++i) {
                    int j = hashBuffer[i].y;
                    if (j == (int)id) continue;
                    if (particles[j].phase == 2 || particles[j].phase == 3) continue;
                    
                    float4 otherPos = particles[j].position;
                    float d = distance(myPos.xyz, otherPos.xyz);
                    
                    if (d < h && d > 0.0f) {
                        float4 gradW = SPH_Kernel_GradW(myPos - otherPos, h);
                        
                        // A. SPH Symmetric Pressure Force (incorporating Electrostrictive Stress: P_eff = P - P_str)
                        float dpi = particles[id].density;
                        float dpj = particles[j].density;
                        
                        float eps_0 = 8.854187817e-12f;
                        float eps_r_diff = permittivity - 1.0f;
                        
                        float eMag_i = length(particles[id].electricField.xyz);
                        float p_str_i = 0.5f * eps_0 * eps_r_diff * particles[id].color * (eMag_i * eMag_i);
                        
                        float eMag_j = length(particles[j].electricField.xyz);
                        float p_str_j = 0.5f * eps_0 * eps_r_diff * particles[j].color * (eMag_j * eMag_j);
                        
                        float4 f_press = -particles[j].mass * (((particles[id].pressure - p_str_i) / (dpi * dpi)) + ((particles[j].pressure - p_str_j) / (dpj * dpj))) * gradW;
                        pressForce += f_press;
                        
                        // B. SPH Monaghan Viscosity Force
                        float4 velDiff = particles[id].velocity - particles[j].velocity;
                        float4 posDiff = myPos - otherPos;
                        if (dot(velDiff.xyz, posDiff.xyz) < 0.0f) {
                            float alpha = grid.sphViscosityAlpha;
                            float beta = grid.sphViscosityBeta;
                            float c = 12.0f;
                            float dot_val = dot(velDiff.xyz, posDiff.xyz);
                            float mu_ij = (h * dot_val) / (d * d + 0.01f * h * h);
                            float PI_ij = (-alpha * c * mu_ij + beta * mu_ij * mu_ij) / (0.5f * (dpi + dpj));
                            viscForce -= particles[j].mass * PI_ij * gradW;
                        }
                    }
                }
            }
        }
    }
    
    particles[id].acceleration += pressForce + viscForce;
    
    // C. Apply True Energy-Conserving Physical Dielectrophoretic (DEP) EHD Body Force
    // f_DEP = -0.5 * eps_0 * |E|^2 * grad(eps_r)
    // where grad(eps_r) is derived from the SPH multi-phase phase color gradient: (permittivity - 1.0) * colorGradient
    float eMag = length(particles[id].electricField.xyz);
    if (eMag > 1.0e3f) {
        float eps_0 = 8.854187817e-12f;
        float eps_r_diff = permittivity - 1.0f;
        float4 grad_eps = particles[id].colorGradient * eps_r_diff;
        float4 depForce = -0.5f * eps_0 * (eMag * eMag) * grad_eps;
        
        // Squeeze the fluid particles under EHD stress
        particles[id].acceleration += depForce / particles[id].mass;
    }
}

// MARK: - 4. 3D Morris CSF Color Gradient Kernel (Pass 1)
kernel void colorGradientKernel(
    device GPUParticle* particles           [[buffer(0)]],
    device int2* hashBuffer                 [[buffer(1)]],
    device int* cellStartBuffer             [[buffer(2)]],
    device int* cellEndBuffer               [[buffer(3)]],
    constant GPUSpatialGridParams& grid     [[buffer(4)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    if (particles[id].phase == 2 || particles[id].phase == 3) return;
    
    float4 myPos = particles[id].position;
    float h = grid.cellSize;
    
    // Calculate 3D Color Gradient
    float4 gradColor = float4(0.0f);
    int3 cellCoord = int3(floor((myPos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
    
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int3 neighborCoord = cellCoord + int3(dx, dy, dz);
                if (neighborCoord.x < 0 || neighborCoord.x >= grid.gridDimensions.x ||
                    neighborCoord.y < 0 || neighborCoord.y >= grid.gridDimensions.y ||
                    neighborCoord.z < 0 || neighborCoord.z >= grid.gridDimensions.z) continue;
                    
                int cellIdx = neighborCoord.x + neighborCoord.y * grid.gridDimensions.x + neighborCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
                int start = cellStartBuffer[cellIdx];
                int end = cellEndBuffer[cellIdx];
                
                if (start == -1) continue;
                
                for (int i = start; i <= end; ++i) {
                    int j = hashBuffer[i].y;
                    if (particles[j].phase == 2 || particles[j].phase == 3) continue;
                    
                    float d = distance(myPos.xyz, particles[j].position.xyz);
                    if (d < h && d > 0.0f) {
                        float4 gradW = SPH_Kernel_GradW(myPos - particles[j].position, h);
                        float dens_j = max(particles[j].density, 0.01f);
                        gradColor += (particles[j].mass / dens_j) * particles[j].color * gradW;
                    }
                }
            }
        }
    }
    
    particles[id].colorGradient = gradColor;
}

// MARK: - 4b. 3D Morris CSF Curvature & Force Kernel (Pass 2)
kernel void curvatureCSFKernel(
    device GPUParticle* particles           [[buffer(0)]],
    device int2* hashBuffer                 [[buffer(1)]],
    device int* cellStartBuffer             [[buffer(2)]],
    device int* cellEndBuffer               [[buffer(3)]],
    constant GPUSpatialGridParams& grid     [[buffer(4)]],
    constant float& sigma                   [[buffer(5)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)grid.numParticles) return;
    if (particles[id].phase == 2 || particles[id].phase == 3) return;
    
    float4 myPos = particles[id].position;
    float h = grid.cellSize;
    float4 gradColor = particles[id].colorGradient;
    float gradNorm = length(gradColor.xyz);
    
    // Curvature divergence calculation in 3D
    float curv = 0.0f;
    int3 cellCoord = int3(floor((myPos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
    
    if (gradNorm > 0.1f) {
        float4 n_i = gradColor / gradNorm;
        
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int3 neighborCoord = cellCoord + int3(dx, dy, dz);
                    if (neighborCoord.x < 0 || neighborCoord.x >= grid.gridDimensions.x ||
                        neighborCoord.y < 0 || neighborCoord.y >= grid.gridDimensions.y ||
                        neighborCoord.z < 0 || neighborCoord.z >= grid.gridDimensions.z) continue;
                        
                    int cellIdx = neighborCoord.x + neighborCoord.y * grid.gridDimensions.x + neighborCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
                    int start = cellStartBuffer[cellIdx];
                    int end = cellEndBuffer[cellIdx];
                    
                    if (start == -1) continue;
                    
                    for (int i = start; i <= end; ++i) {
                        int j = hashBuffer[i].y;
                        if (particles[j].phase == 2 || particles[j].phase == 3) continue;
                        
                        float d = distance(myPos.xyz, particles[j].position.xyz);
                        if (d < h && d > 0.0f) {
                            float4 gradW = SPH_Kernel_GradW(myPos - particles[j].position, h);
                            float dens_j = max(particles[j].density, 0.01f);
                            float gradNorm_j = length(particles[j].colorGradient.xyz);
                            float4 n_j = (gradNorm_j > 0.1f) ? (particles[j].colorGradient / gradNorm_j) : float4(0.0f);
                            
                            curv -= (particles[j].mass / dens_j) * dot((n_i - n_j).xyz, gradW.xyz);
                        }
                    }
                }
            }
        }
    }
    
    particles[id].curvature = curv;
    
    // Apply cohesive Morris CSF surface tension acceleration in 3D
    if (gradNorm > 0.1f) {
        float dens = max(particles[id].density, 0.01f);
        float4 stAccel = (sigma * curv * gradColor) / dens;
        particles[id].acceleration += stAccel;
    }
}

// MARK: - 5. Integration Kernel in 3D
kernel void integrateKernel(
    device GPUParticle* particles           [[buffer(0)]],
    constant float& dt                      [[buffer(1)]],
    constant float4& gravity                [[buffer(2)]],
    constant int& numParticles              [[buffer(3)]],
    device int2* hashBuffer                 [[buffer(4)]],
    device int* cellStartBuffer             [[buffer(5)]],
    device int* cellEndBuffer               [[buffer(6)]],
    constant GPUSpatialGridParams& grid     [[buffer(7)]],
    uint id                                 [[thread_position_in_grid]])
{
    if (id >= (uint)numParticles) return;
    if (particles[id].mass <= 0.0f) return;
    
    float4 oldPos = particles[id].position;
    float4 newPos = oldPos;
    
    if (particles[id].phase == 2 || particles[id].phase == 3) {
        // Standard XPBD Velocity Update: v = (x_new - x_old) / dt
        particles[id].velocity = (particles[id].predictedPosition - oldPos) / dt;
        newPos = particles[id].predictedPosition;
    } else {
        // Add 3D gravity acceleration for fluid
        float4 accel = particles[id].acceleration + gravity;
        
        // Acceleration clamping to prevent numerical explosion
        float accelMag = length(accel.xyz);
        if (accelMag > 3000.0f) {
            accel = float4((accel.xyz / accelMag) * 3000.0f, 0.0f);
        }
        
        // Update velocity
        particles[id].velocity += accel * dt;
        
        // Velocity clamping: Raised to 30.0f to preserve high-frequency dynamic kinetics during rapid zipping
        float4 vel = particles[id].velocity;
        float velMag = length(vel.xyz);
        if (velMag > 30.0f) {
            vel = float4((vel.xyz / velMag) * 30.0f, 0.0f);
            particles[id].velocity = vel;
        }
        
        // Velocity damping
        particles[id].velocity *= 0.992f; 
        
        newPos = oldPos + particles[id].velocity * dt;
        
        // Dynamic Fluid-Structure Boundary Collision Resolution to prevent leaking
        int3 cellCoord = int3(floor((newPos.xyz - grid.gridOrigin.xyz) / grid.cellSize));
        cellCoord = clamp(cellCoord, int3(0), grid.gridDimensions.xyz - 1);
        
        float minSolidDist = 1e9f;
        float4 closestSolidPos = float4(0.0f);
        bool foundSolid = false;
        
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int3 neighborCoord = cellCoord + int3(dx, dy, dz);
                    if (neighborCoord.x < 0 || neighborCoord.x >= grid.gridDimensions.x ||
                        neighborCoord.y < 0 || neighborCoord.y >= grid.gridDimensions.y ||
                        neighborCoord.z < 0 || neighborCoord.z >= grid.gridDimensions.z) continue;
                        
                    int cellIdx = neighborCoord.x + neighborCoord.y * grid.gridDimensions.x + neighborCoord.z * grid.gridDimensions.x * grid.gridDimensions.y;
                    int start = cellStartBuffer[cellIdx];
                    int end = cellEndBuffer[cellIdx];
                    
                    if (start == -1) continue;
                    
                    for (int i = start; i <= end; ++i) {
                        int j = hashBuffer[i].y;
                        if (particles[j].phase == 2 || particles[j].phase == 3) {
                            float d = distance(newPos.xyz, particles[j].position.xyz);
                            if (d < minSolidDist) {
                                minSolidDist = d;
                                closestSolidPos = particles[j].position;
                                foundSolid = true;
                            }
                        }
                    }
                }
            }
        }
        
        if (foundSolid) {
            // Containment threshold (1.2mm prevents micro-leaking under high-voltage compression)
            float threshold = 0.0012f; 
            if (minSolidDist < threshold) {
                float4 dir = newPos - closestSolidPos;
                float d = length(dir.xyz);
                if (d > 1e-6f) {
                    float4 normal = float4(dir.xyz / d, 0.0f);
                    // Push fluid particle back inwards
                    newPos += normal * (threshold - minSolidDist);
                    newPos.w = 1.0f;
                    
                    // Dampen normal velocity component to prevent bouncing penetration
                    float normalVel = dot(particles[id].velocity, normal);
                    if (normalVel < 0.0f) {
                        particles[id].velocity -= normal * normalVel * 1.5f; // rebound damping
                    }
                }
            }
        }
    }
    
    // NaN and Infinite exception protection
    if (isnan(newPos.x) || isnan(newPos.y) || isnan(newPos.z) ||
        isinf(newPos.x) || isinf(newPos.y) || isinf(newPos.z)) {
        newPos = float4(0.0f, 0.0f, 0.0f, 1.0f);
        particles[id].velocity = float4(0.0f);
    }
    
    particles[id].position = newPos;
    particles[id].predictedPosition = newPos;
    
    // 3D Chamber boundary conditions (5cm safety box)
    float border = 0.048f;
    if (particles[id].position.x < -border) {
        particles[id].position.x = -border;
        particles[id].velocity.x *= -0.2f;
    }
    if (particles[id].position.x > border) {
        particles[id].position.x = border;
        particles[id].velocity.x *= -0.2f;
    }
    if (particles[id].position.y < -border) {
        particles[id].position.y = -border;
        particles[id].velocity.y *= -0.2f;
    }
    if (particles[id].position.y > border) {
        particles[id].position.y = border;
        particles[id].velocity.y *= -0.2f;
    }
    if (particles[id].position.z < -border) {
        particles[id].position.z = -border;
        particles[id].velocity.z *= -0.2f;
    }
    if (particles[id].position.z > border) {
        particles[id].position.z = border;
        particles[id].velocity.z *= -0.2f;
    }
}
