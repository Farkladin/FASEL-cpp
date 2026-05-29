#ifndef POISSON_SOLVER_HPP
#define POISSON_SOLVER_HPP

#include <vector>
#include "Particle.hpp"

class PoissonSolver {
public:
    int gridNx;
    int gridNy;
    int gridNz;
    simd_float4 origin;
    float cellSize;
    
    std::vector<float> permittivityGrid;
    std::vector<float> potentialGrid;
    std::vector<int32_t> boundaryGrid;
    int maxIterations = 180;
    
    PoissonSolver(int nx = 32, int ny = 32, int nz = 32, float cellSz = 0.003125f);
    
    void projectParticlesToGrid(GPUParticle* particles, int numParticles, float appliedVoltage, float liquidPermittivity, bool verbose = false);
    
    void solvePoisson();
    
    void interpolateElectricFieldToParticles(GPUParticle* particles, int numParticles);
    
private:
    void computeAp(const std::vector<float>& x, std::vector<float>& Ax);
    float dotProduct(const std::vector<float>& a, const std::vector<float>& b);
    
    // Persistent CG scratch vectors to prevent heap allocation overhead
    std::vector<float> rScratch;
    std::vector<float> zScratch;
    std::vector<float> pScratch;
    std::vector<float> ApScratch;
    std::vector<float> invMScratch;
    std::vector<bool> isBoundary;
};

#endif // POISSON_SOLVER_HPP
