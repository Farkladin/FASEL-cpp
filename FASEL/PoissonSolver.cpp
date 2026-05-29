#include "PoissonSolver.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

PoissonSolver::PoissonSolver(int nx, int ny, int nz, float cellSz)
    : gridNx(nx), gridNy(ny), gridNz(nz), origin(simd_make_float4(-0.05f, -0.05f, -0.05f, 0.0f)), cellSize(cellSz) {
    int totalCells = gridNx * gridNy * gridNz;
    permittivityGrid.assign(totalCells, 4.5f * 8.854e-12f);
    potentialGrid.assign(totalCells, 0.0f);
    boundaryGrid.assign(totalCells, 0);
    rScratch.assign(totalCells, 0.0f);
    zScratch.assign(totalCells, 0.0f);
    pScratch.assign(totalCells, 0.0f);
    ApScratch.assign(totalCells, 0.0f);
    invMScratch.assign(totalCells, 1.0f);
    isBoundary.assign(totalCells, false);
}

void PoissonSolver::projectParticlesToGrid(GPUParticle* particles, int numParticles, float appliedVoltage, float liquidPermittivity, bool verbose) {
    /**
     * 3D Particle-to-Grid (P2G) Dielectric Projection
     */
    int totalCells = gridNx * gridNy * gridNz;
    float eps0 = 8.854e-12f;
    
    if (verbose) {
        std::cout << "[Debug 3D] Entered projectParticlesToGrid:\n"
                  << "  - Particle count: " << numParticles << "\n";
    }
    
    // 1. Initialization using std::fill to avoid reallocation overheads and keep it extremely fast
    std::fill(permittivityGrid.begin(), permittivityGrid.end(), liquidPermittivity * eps0);
    std::fill(potentialGrid.begin(), potentialGrid.end(), 0.0f);
    std::fill(boundaryGrid.begin(), boundaryGrid.end(), 0);
    std::fill(rScratch.begin(), rScratch.end(), 0.0f);
    std::fill(zScratch.begin(), zScratch.end(), 0.0f);
    std::fill(pScratch.begin(), pScratch.end(), 0.0f);
    std::fill(ApScratch.begin(), ApScratch.end(), 0.0f);
    std::fill(invMScratch.begin(), invMScratch.end(), 1.0f);
    std::fill(isBoundary.begin(), isBoundary.end(), false);
    
    std::vector<float> weightSum(totalCells, 0.0f);
    std::vector<float> epsSum(totalCells, 0.0f);
    
    for (int i = 0; i < numParticles; ++i) {
        const auto& p = particles[i];
        // Defensive check against NaN and infinity
        if (std::isnan(p.position.x) || std::isinf(p.position.x) ||
            std::isnan(p.position.y) || std::isinf(p.position.y) ||
            std::isnan(p.position.z) || std::isinf(p.position.z)) {
            continue;
        }
        
        // Particle grid coordinates in 3D
        int gx = static_cast<int>(std::floor((p.position.x - origin.x) / cellSize));
        int gy = static_cast<int>(std::floor((p.position.y - origin.y) / cellSize));
        int gz = static_cast<int>(std::floor((p.position.z - origin.z) / cellSize));
        
        if (gx >= 0 && gx < gridNx && gy >= 0 && gy < gridNy && gz >= 0 && gz < gridNz) {
            int cellIdx = gx + gy * gridNx + gz * gridNx * gridNy;
            
            float w = 1.0f;
            weightSum[cellIdx] += w;
            
            float eps = (p.phase == 1) ? (1.0f * eps0) : (liquidPermittivity * eps0);
            epsSum[cellIdx] += eps * w;
            
            if (p.phase == 3) { // solidElectrode
                boundaryGrid[cellIdx] = 1;
                potentialGrid[cellIdx] = (p.gasBubbleID == -10) ? appliedVoltage : 0.0f;
            }
        }
    }
    
    // 2. Determine weighted average permittivity
    for (int i = 0; i < totalCells; ++i) {
        if (weightSum[i] > 0.0f) {
            permittivityGrid[i] = epsSum[i] / weightSum[i];
        }
    }
}

void PoissonSolver::solvePoisson() {
    /**
     * 3D Electrostatic Poisson Equation Solver (Jacobi Preconditioned Conjugate Gradient Method)
     */
    int totalCells = gridNx * gridNy * gridNz;
    float h2 = cellSize * cellSize;
    float invH2 = 1.0f / h2;
    int sliceSize = gridNx * gridNy;
    
    // 1. Precompute isBoundary mask using nested 3D loops to completely eliminate slow division/modulo operations
    #pragma omp parallel for collapse(3)
    for (int gz = 0; gz < gridNz; ++gz) {
        for (int gy = 0; gy < gridNy; ++gy) {
            for (int gx = 0; gx < gridNx; ++gx) {
                int i = gx + gy * gridNx + gz * sliceSize;
                isBoundary[i] = (boundaryGrid[i] == 1 || gx == 0 || gx == gridNx - 1 || 
                                 gy == 0 || gy == gridNy - 1 || gz == 0 || gz == gridNz - 1);
            }
        }
    }
    
    // 2. Precompute the inverse diagonal M^-1 for interior non-Dirichlet cells
    #pragma omp parallel for
    for (int i = 0; i < totalCells; ++i) {
        if (isBoundary[i]) {
            invMScratch[i] = 1.0f;
        } else {
            int im1 = i - 1;
            int ip1 = i + 1;
            int jm1 = i - gridNx;
            int jp1 = i + gridNx;
            int km1 = i - sliceSize;
            int kp1 = i + sliceSize;
            
            float epsR = 0.5f * (permittivityGrid[i] + permittivityGrid[ip1]);
            float epsL = 0.5f * (permittivityGrid[i] + permittivityGrid[im1]);
            float epsU = 0.5f * (permittivityGrid[i] + permittivityGrid[jp1]);
            float epsD = 0.5f * (permittivityGrid[i] + permittivityGrid[jm1]);
            float epsF = 0.5f * (permittivityGrid[i] + permittivityGrid[kp1]);
            float epsB = 0.5f * (permittivityGrid[i] + permittivityGrid[km1]);
            
            float diagEntry = (epsR + epsL + epsU + epsD + epsF + epsB) * invH2;
            invMScratch[i] = 1.0f / std::max(diagEntry, 1e-9f);
        }
    }
    
    // 3. Calculate initial residual r = b - A*Phi (where b = 0)
    computeAp(potentialGrid, rScratch);
    
    double r0Norm = 0.0;
    #pragma omp parallel for reduction(+:r0Norm)
    for (int i = 0; i < totalCells; ++i) {
        if (isBoundary[i]) {
            rScratch[i] = 0.0f; // Dirichlet boundary has no residual
        } else {
            rScratch[i] = -rScratch[i];
        }
        
        // Apply Jacobi Preconditioner
        zScratch[i] = rScratch[i] * invMScratch[i];
        pScratch[i] = zScratch[i];
        r0Norm += static_cast<double>(rScratch[i]) * static_cast<double>(rScratch[i]);
    }
    r0Norm = std::sqrt(r0Norm);
    if (r0Norm < 1e-10) return;
    
    double rzOld = dotProduct(rScratch, zScratch);
    if (std::abs(rzOld) < 1e-30) return;
    
    // 4. Jacobi Preconditioned Conjugate Gradient Iteration (Dynamically scale iterations based on grid size)
    int maxIter = maxIterations;
    bool converged = false;
    for (int iter = 0; iter < maxIter; ++iter) {
        computeAp(pScratch, ApScratch);
        
        // Suppress search directions at Dirichlet boundaries in parallel
        #pragma omp parallel for
        for (int i = 0; i < totalCells; ++i) {
            if (isBoundary[i]) {
                ApScratch[i] = 0.0f;
            }
        }
        
        double pAp = dotProduct(pScratch, ApScratch);
        double alpha = rzOld / std::max(pAp, 1e-9);
        
        #pragma omp parallel for
        for (int i = 0; i < totalCells; ++i) {
            if (!isBoundary[i]) {
                potentialGrid[i] += alpha * pScratch[i];
                rScratch[i] -= alpha * ApScratch[i];
            }
        }
        
        double rNewSq = dotProduct(rScratch, rScratch);
        double currentNorm = std::sqrt(rNewSq);
        if (currentNorm / r0Norm < 1e-6f) { // relative residual tolerance
            converged = true;
            break;
        }
        
        // Apply Preconditioner: z_{k+1} = M^-1 * r_{k+1} in parallel
        #pragma omp parallel for
        for (int i = 0; i < totalCells; ++i) {
            zScratch[i] = rScratch[i] * invMScratch[i];
        }
        
        double rzNew = dotProduct(rScratch, zScratch);
        if (std::abs(rzNew) < 1e-30) break; // Zero-guard before division
        double beta = rzNew / rzOld;
        
        #pragma omp parallel for
        for (int i = 0; i < totalCells; ++i) {
            if (!isBoundary[i]) {
                pScratch[i] = zScratch[i] + beta * pScratch[i];
            } else {
                pScratch[i] = 0.0f;
            }
        }
        rzOld = rzNew;
    }
    
    if (!converged) {
        double rNewSq = dotProduct(rScratch, rScratch);
        if (std::sqrt(rNewSq) > 1e-3f) {
            std::cerr << "[PoissonSolver] Warning: Jacobi CG failed to converge after " 
                      << maxIter << " iterations. Relative residual: " << std::sqrt(rNewSq) / r0Norm << "\n";
        }
    }
}

void PoissonSolver::computeAp(const std::vector<float>& x, std::vector<float>& Ax) {
    /**
     * Discrete Finite Difference Operator for \nabla \cdot ( \epsilon \nabla \phi ) in 3D
     * Enforces explicit Dirichlet ground potential at the outer border boundaries.
     */
    float h2 = cellSize * cellSize;
    float invH2 = 1.0f / h2;
    int sliceSize = gridNx * gridNy;
    int totalCells = gridNx * gridNy * gridNz;
    
    #pragma omp parallel for
    for (int i = 0; i < totalCells; ++i) {
        // Explicit Dirichlet boundaries ( electrodes or outer far-field borders)
        if (isBoundary[i]) {
            Ax[i] = x[i];
            continue;
        }
        
        int im1 = i - 1;
        int ip1 = i + 1;
        int jm1 = i - gridNx;
        int jp1 = i + gridNx;
        int km1 = i - sliceSize;
        int kp1 = i + sliceSize;
        
        float epsR = 0.5f * (permittivityGrid[i] + permittivityGrid[ip1]);
        float epsL = 0.5f * (permittivityGrid[i] + permittivityGrid[im1]);
        float epsU = 0.5f * (permittivityGrid[i] + permittivityGrid[jp1]);
        float epsD = 0.5f * (permittivityGrid[i] + permittivityGrid[jm1]);
        float epsF = 0.5f * (permittivityGrid[i] + permittivityGrid[kp1]); // Front (z+)
        float epsB = 0.5f * (permittivityGrid[i] + permittivityGrid[km1]); // Back (z-)
        
        float termX = epsR * (x[ip1] - x[i]) - epsL * (x[i] - x[im1]);
        float termY = epsU * (x[jp1] - x[i]) - epsD * (x[i] - x[jm1]);
        float termZ = epsF * (x[kp1] - x[i]) - epsB * (x[i] - x[km1]);
        
        Ax[i] = -(termX + termY + termZ) * invH2;
    }
}
 
void PoissonSolver::interpolateElectricFieldToParticles(GPUParticle* particles, int numParticles) {
    /**
     * Eulerian-to-Lagrangian (E2L) Potential and Electric Field Interpolation in 3D
     */
    int sliceSize = gridNx * gridNy;
    float invTwoCellSize = 1.0f / (2.0f * cellSize);
    
    #pragma omp parallel for
    for (int i = 0; i < numParticles; ++i) {
        auto& p = particles[i];
        
        if (std::isnan(p.position.x) || std::isinf(p.position.x) ||
            std::isnan(p.position.y) || std::isinf(p.position.y) ||
            std::isnan(p.position.z) || std::isinf(p.position.z)) {
            p.electricPotential = 0.0f;
            p.electricField = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }
        
        int gx = static_cast<int>(std::floor((p.position.x - origin.x) / cellSize));
        int gy = static_cast<int>(std::floor((p.position.y - origin.y) / cellSize));
        int gz = static_cast<int>(std::floor((p.position.z - origin.z) / cellSize));
        
        if (gx >= 1 && gx < gridNx - 1 && gy >= 1 && gy < gridNy - 1 && gz >= 1 && gz < gridNz - 1) {
            int idx = gx + gy * gridNx + gz * sliceSize;
            
            p.electricPotential = potentialGrid[idx];
            
            int idxR = idx + 1;
            int idxL = idx - 1;
            int idxU = idx + gridNx;
            int idxD = idx - gridNx;
            int idxF = idx + sliceSize;
            int idxB = idx - sliceSize;
            
            float ex = -(potentialGrid[idxR] - potentialGrid[idxL]) * invTwoCellSize;
            float ey = -(potentialGrid[idxU] - potentialGrid[idxD]) * invTwoCellSize;
            float ez = -(potentialGrid[idxF] - potentialGrid[idxB]) * invTwoCellSize;
            
            p.electricField = simd_make_float4(ex, ey, ez, 0.0f);
            
            // Calculate potential Hessian second derivatives for analytical electric field gradient tensor
            float invH2 = 1.0f / (cellSize * cellSize);
            float invFourH2 = 1.0f / (4.0f * cellSize * cellSize);
            
            float phi_xx = (potentialGrid[idxR] - 2.0f * potentialGrid[idx] + potentialGrid[idxL]) * invH2;
            float phi_yy = (potentialGrid[idxU] - 2.0f * potentialGrid[idx] + potentialGrid[idxD]) * invH2;
            float phi_zz = (potentialGrid[idxF] - 2.0f * potentialGrid[idx] + potentialGrid[idxB]) * invH2;
            
            float phi_xy = (potentialGrid[idx + 1 + gridNx] - potentialGrid[idx + 1 - gridNx] - 
                            potentialGrid[idx - 1 + gridNx] + potentialGrid[idx - 1 - gridNx]) * invFourH2;
            float phi_yz = (potentialGrid[idx + gridNx + sliceSize] - potentialGrid[idx + gridNx - sliceSize] - 
                            potentialGrid[idx - gridNx + sliceSize] + potentialGrid[idx - gridNx - sliceSize]) * invFourH2;
            float phi_zx = (potentialGrid[idx + 1 + sliceSize] - potentialGrid[idx + 1 - sliceSize] - 
                            potentialGrid[idx - 1 + sliceSize] + potentialGrid[idx - 1 - sliceSize]) * invFourH2;
            
            // Frobenius norm of electric field gradient tensor: ||grad(E)||_F
            float gradE_norm = std::sqrt(phi_xx * phi_xx + phi_yy * phi_yy + phi_zz * phi_zz + 
                                         2.0f * (phi_xy * phi_xy + phi_yz * phi_yz + phi_zx * phi_zx));
            p.eFieldGradientNorm = gradE_norm;
        } else {
            p.electricPotential = 0.0f;
            p.electricField = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            p.eFieldGradientNorm = 0.0f;
        }
    }
}

float PoissonSolver::dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return static_cast<float>(sum);
}
