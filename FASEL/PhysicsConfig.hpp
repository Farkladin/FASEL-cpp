#ifndef PHYSICS_CONFIG_HPP
#define PHYSICS_CONFIG_HPP

#include <string>

/**
 * @brief Configuration class to manage, load, and save physical and simulation constants
 * from an external file (e.g., CSV).
 */
struct PhysicsConfig {
    // 1. Paschen's Law Parameters (Breakdown Analyzer)
    float A = 11.25f;            // Pa^-1 * m^-1
    float B = 273.75f;           // V * Pa^-1 * m^-1
    float gammaSE = 0.01f;       // Secondary Electron Emission Coefficient

    // 2. Dielectric Strength/Breakdown Limits
    float liquidBreakdownThreshold = 6.0e7f;   // V/m
    float membraneBreakdownThreshold = 9.0e7f; // V/m (Standard PDMS preset)

    // 3. Fluid Mechanics Properties
    float liquidPermittivity = 2.7f;           // Relative permittivity
    float liquidViscosity = 0.0048f;           // Pa*s
    float surfaceTension = 0.021f;             // N/m (Silicone oil default)
    float gasVolumeFraction = 0.20f;           // phi

    // 4. Actuator Pouch Geometric Specifications
    float pouchWidth = 0.02f;                  // m
    float pouchHeight = 0.04f;                 // m
    float pouchThickness = 0.004f;             // m
    int numLayers = 1;                         // pouch layers

    // 5. XPBD Membrane Mechanics Compliance
    float stretchCompliance = 1.0e-6f;         // Stretch stiffness factor
    float bendingCompliance = 1.0e-5f;         // Bending stiffness factor

    // 6. Simulation Solvers & Limits Control
    float timeStep = 0.000025f;                // dt (seconds)
    float kPouchAnchorYThreshold = 0.43f;      // top/bottom anchor ratio
    float kBoundaryRadiusBox = 0.025f;         // boundary box size
    float kBoundaryRadiusLimit = 0.024f;       // boundary limit size
    float kActiveStrainLeverage = 0.01f;       // force scaling factor
    float appliedVoltage = 7000.0f;            // voltage (V)

    // 7. 3D Spatial Orientation Projection (Degrees)
    float pitch = 90.0f;
    float roll = 0.0f;
    float yaw = 0.0f;

    // 8. Viscoelastic Creep Parameters (polymer membrane relaxation)
    float viscoelasticRelaxationRate = 0.05f;   // gamma (s^-1) (Legacy fallback)
    float viscoelasticMaxStrainLimit = 0.12f;   // max creep limit (e.g. 12%)
    float viscoelasticPronyW1 = 0.30f;          // Prony branch 1 stiffness weight
    float viscoelasticPronyW2 = 0.30f;          // Prony branch 2 stiffness weight
    float viscoelasticPronyW3 = 0.20f;          // Prony branch 3 stiffness weight
    float viscoelasticTau1 = 0.001f;            // Prony branch 1 relaxation time (seconds)
    float viscoelasticTau2 = 0.010f;            // Prony branch 2 relaxation time (seconds)
    float viscoelasticTau3 = 0.100f;            // Prony branch 3 relaxation time (seconds)
    float viscoelasticEyringForce0 = 0.50f;      // Eyring activation reference force (N)

    // 9. Numerical Solver Configurable Tuning Parameters
    float sphViscosityAlpha = 0.15f;            // Monaghan viscosity alpha
    float sphViscosityBeta = 0.30f;             // Monaghan viscosity beta
    int poissonMaxIterations = 180;             // Poisson Conjugate Gradient iteration limit


    /**
     * @brief Load configuration constants from a CSV file.
     * @param filepath Path to the CSV file.
     * @return true if successful, false otherwise.
     */
    bool loadFromCSV(const std::string& filepath);

    /**
     * @brief Save configuration constants to a CSV file.
     * @param filepath Path to the CSV file.
     * @return true if successful, false otherwise.
     */
    bool saveToCSV(const std::string& filepath) const;

    /**
     * @brief Print loaded constants to standard output for verification.
     */
    void print() const;
};

#endif // PHYSICS_CONFIG_HPP
