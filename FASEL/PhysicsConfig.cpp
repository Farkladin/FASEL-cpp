#include "PhysicsConfig.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>

// Helper utility to trim whitespace from standard strings
static std::string trimString(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (std::string::npos == first) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

bool PhysicsConfig::loadFromCSV(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[PhysicsConfig] Warning: Configuration file \"" << filepath 
                  << "\" not found. Writing default configuration and proceeding.\n";
        saveToCSV(filepath);
        return false;
    }

    std::string line;
    int successfullyLoaded = 0;
    while (std::getline(file, line)) {
        line = trimString(line);
        // Ignore empty lines and lines starting with '#'
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::stringstream ss(line);
        std::string key;
        std::string valStr;

        if (std::getline(ss, key, ',') && std::getline(ss, valStr, ',')) {
            key = trimString(key);
            valStr = trimString(valStr);

            if (key.empty() || valStr.empty()) {
                continue;
            }

            try {
                if (key == "A") { A = std::stof(valStr); successfullyLoaded++; }
                else if (key == "B") { B = std::stof(valStr); successfullyLoaded++; }
                else if (key == "gammaSE") { gammaSE = std::stof(valStr); successfullyLoaded++; }
                else if (key == "liquidBreakdownThreshold") { liquidBreakdownThreshold = std::stof(valStr); successfullyLoaded++; }
                else if (key == "membraneBreakdownThreshold") { membraneBreakdownThreshold = std::stof(valStr); successfullyLoaded++; }
                else if (key == "liquidPermittivity") { liquidPermittivity = std::stof(valStr); successfullyLoaded++; }
                else if (key == "liquidViscosity") { liquidViscosity = std::stof(valStr); successfullyLoaded++; }
                else if (key == "surfaceTension") { surfaceTension = std::stof(valStr); successfullyLoaded++; }
                else if (key == "gasVolumeFraction") { gasVolumeFraction = std::stof(valStr); successfullyLoaded++; }
                else if (key == "pouchWidth") { pouchWidth = std::stof(valStr); successfullyLoaded++; }
                else if (key == "pouchHeight") { pouchHeight = std::stof(valStr); successfullyLoaded++; }
                else if (key == "pouchThickness") { pouchThickness = std::stof(valStr); successfullyLoaded++; }
                else if (key == "numLayers") { numLayers = std::stoi(valStr); successfullyLoaded++; }
                else if (key == "stretchCompliance") { stretchCompliance = std::stof(valStr); successfullyLoaded++; }
                else if (key == "bendingCompliance") { bendingCompliance = std::stof(valStr); successfullyLoaded++; }
                else if (key == "timeStep") { timeStep = std::stof(valStr); successfullyLoaded++; }
                else if (key == "kPouchAnchorYThreshold") { kPouchAnchorYThreshold = std::stof(valStr); successfullyLoaded++; }
                else if (key == "kBoundaryRadiusBox") { kBoundaryRadiusBox = std::stof(valStr); successfullyLoaded++; }
                else if (key == "kBoundaryRadiusLimit") { kBoundaryRadiusLimit = std::stof(valStr); successfullyLoaded++; }
                else if (key == "kActiveStrainLeverage") { kActiveStrainLeverage = std::stof(valStr); successfullyLoaded++; }
                else if (key == "appliedVoltage") { appliedVoltage = std::stof(valStr); successfullyLoaded++; }
                else if (key == "pitch") { pitch = std::stof(valStr); successfullyLoaded++; }
                else if (key == "roll") { roll = std::stof(valStr); successfullyLoaded++; }
                else if (key == "yaw") { yaw = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticRelaxationRate") { viscoelasticRelaxationRate = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticMaxStrainLimit") { viscoelasticMaxStrainLimit = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticPronyW1") { viscoelasticPronyW1 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticPronyW2") { viscoelasticPronyW2 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticPronyW3") { viscoelasticPronyW3 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticTau1") { viscoelasticTau1 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticTau2") { viscoelasticTau2 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticTau3") { viscoelasticTau3 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "viscoelasticEyringForce0") { viscoelasticEyringForce0 = std::stof(valStr); successfullyLoaded++; }
                else if (key == "sphViscosityAlpha") { sphViscosityAlpha = std::stof(valStr); successfullyLoaded++; }
                else if (key == "sphViscosityBeta") { sphViscosityBeta = std::stof(valStr); successfullyLoaded++; }
                else if (key == "poissonMaxIterations") { poissonMaxIterations = std::stoi(valStr); successfullyLoaded++; }
                else {
                    std::cerr << "[PhysicsConfig] Warning: Unrecognized parameter key in CSV: \"" << key << "\"\n";
                }
            } catch (const std::exception& e) {
                std::cerr << "[PhysicsConfig] Error parsing value for key \"" << key 
                          << "\" with value string \"" << valStr << "\": " << e.what() << "\n";
            }
        }
    }

    file.close();
    std::cout << "[PhysicsConfig] Loaded " << successfullyLoaded << " constants from \"" << filepath << "\"\n";
    return true;
}

bool PhysicsConfig::saveToCSV(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[PhysicsConfig] Error: Could not write file \"" << filepath << "\"\n";
        return false;
    }

    file << "# =======================================================================\n";
    file << "# FASEL: High-Fidelity HASEL Actuator Simulation Physical Configuration\n";
    file << "# File Format: ConstantName, Value, Optional Description\n";
    file << "# Comments start with '#'. Values are comma-separated.\n";
    file << "# =======================================================================\n\n";

    file << "# --- Section 1: Paschen's Law Dielectric Breakdown Constants ---\n";
    file << "A, " << A << ", Paschen constant A (Pa^-1 * m^-1)\n";
    file << "B, " << B << ", Paschen constant B (V * Pa^-1 * m^-1)\n";
    file << "gammaSE, " << gammaSE << ", Secondary Electron Emission Coefficient (gamma_se)\n\n";

    file << "# --- Section 2: Dielectric Strengths ---\n";
    file << "liquidBreakdownThreshold, " << liquidBreakdownThreshold << ", Liquid dielectric strength (V/m, e.g. 6.0e7 for Silicone Oil)\n";
    file << "membraneBreakdownThreshold, " << membraneBreakdownThreshold << ", Membrane dielectric strength (V/m, e.g. 9.0e7 for PDMS, 3.0e8 for BOPET)\n\n";

    file << "# --- Section 3: Fluid Dynamics & Properties ---\n";
    file << "liquidPermittivity, " << liquidPermittivity << ", Relative permittivity (permittivity of oil, e.g. 2.7 to 4.5)\n";
    file << "liquidViscosity, " << liquidViscosity << ", Dynamic viscosity of liquid medium (Pa*s, e.g. 0.0048 for 5cSt oil)\n";
    file << "surfaceTension, " << surfaceTension << ", Surface tension coefficient between oil/air (N/m, e.g. 0.021)\n";
    file << "gasVolumeFraction, " << gasVolumeFraction << ", Gas volume filling ratio (phi, 0.0 to 1.0)\n\n";

    file << "# --- Section 4: Actuator Shell Geometry ---\n";
    file << "pouchWidth, " << pouchWidth << ", Width of the actuator pouch in meters (e.g. 0.02)\n";
    file << "pouchHeight, " << pouchHeight << ", Height of the actuator pouch in meters (e.g. 0.04)\n";
    file << "pouchThickness, " << pouchThickness << ", 3D thickness of the pouch in meters (e.g. 0.004)\n";
    file << "numLayers, " << numLayers << ", Number of stacked HASEL pouches (e.g. 1)\n\n";

    file << "# --- Section 5: XPBD Structural Mechanics ---\n";
    file << "stretchCompliance, " << stretchCompliance << ", Membrane XPBD stretching compliance (0.0 for inextensible BOPET)\n";
    file << "bendingCompliance, " << bendingCompliance << ", Membrane XPBD bending compliance (e.g. 1.0e-5)\n\n";

    file << "# --- Section 6: SPH & XPBD Numerical Solver Controls ---\n";
    file << "timeStep, " << timeStep << ", Simulation timestep dt (seconds, e.g. 0.000025)\n";
    file << "kPouchAnchorYThreshold, " << kPouchAnchorYThreshold << ", Height ratio to designate membrane anchors/loads (e.g. 0.43)\n";
    file << "kBoundaryRadiusBox, " << kBoundaryRadiusBox << ", Boundary box containment wall radius (m)\n";
    file << "kBoundaryRadiusLimit, " << kBoundaryRadiusLimit << ", Fluid particle outer radius clamp (m)\n";
    file << "kActiveStrainLeverage, " << kActiveStrainLeverage << ", Mechanical leverage force scaling calibration constant\n";
    file << "appliedVoltage, " << appliedVoltage << ", Base applied voltage (V, e.g. 6000)\n\n";

    file << "# --- Section 7: 3D Spatial Orientation & Rotation (Degrees) ---\n";
    file << "pitch, " << pitch << ", Pitch tilt angle\n";
    file << "roll, " << roll << ", Roll tilt angle\n";
    file << "yaw, " << yaw << ", Yaw rotation angle\n\n";

    file << "# --- Section 8: Viscoelastic Creep Parameters ---\n";
    file << "viscoelasticRelaxationRate, " << viscoelasticRelaxationRate << ", Polymer viscoelastic relaxation rate (gamma, s^-1)\n";
    file << "viscoelasticMaxStrainLimit, " << viscoelasticMaxStrainLimit << ", Maximum viscoelastic strain limit factor (e.g. 0.12 for 12% max creep)\n";
    file << "viscoelasticPronyW1, " << viscoelasticPronyW1 << ", Prony branch 1 stiffness weight\n";
    file << "viscoelasticPronyW2, " << viscoelasticPronyW2 << ", Prony branch 2 stiffness weight\n";
    file << "viscoelasticPronyW3, " << viscoelasticPronyW3 << ", Prony branch 3 stiffness weight\n";
    file << "viscoelasticTau1, " << viscoelasticTau1 << ", Prony branch 1 relaxation time (seconds)\n";
    file << "viscoelasticTau2, " << viscoelasticTau2 << ", Prony branch 2 relaxation time (seconds)\n";
    file << "viscoelasticTau3, " << viscoelasticTau3 << ", Prony branch 3 relaxation time (seconds)\n";
    file << "viscoelasticEyringForce0, " << viscoelasticEyringForce0 << ", Eyring activation reference force (N)\n\n";

    file << "# --- Section 9: Numerical Solver Configurable Tuning Parameters ---\n";
    file << "sphViscosityAlpha, " << sphViscosityAlpha << ", SPH Monaghan artificial viscosity alpha constant (e.g. 0.15)\n";
    file << "sphViscosityBeta, " << sphViscosityBeta << ", SPH Monaghan artificial viscosity beta constant (e.g. 0.30)\n";
    file << "poissonMaxIterations, " << poissonMaxIterations << ", Poisson Conjugate Gradient solver maximum iteration limit (e.g. 180)\n";

    file.close();
    std::cout << "[PhysicsConfig] Successfully wrote physical constants configuration to \"" << filepath << "\"\n";
    return true;
}

void PhysicsConfig::print() const {
    std::cout << "▷ [PhysicsConfig] Active Configuration Variables:\n";
    std::cout << "  - [Paschen's Law] A: " << A << " Pa^-1 m^-1, B: " << B << " V/Pa/m, gammaSE: " << gammaSE << "\n";
    std::cout << "  - [Breakdown Thresholds] Liquid: " << (liquidBreakdownThreshold / 1.0e6f) << " MV/m, Membrane: " 
              << (membraneBreakdownThreshold / 1.0e6f) << " MV/m\n";
    std::cout << "  - [Dielectric Medium] Oil Permittivity: " << liquidPermittivity << ", Viscosity: " << liquidViscosity 
              << " Pa*s, Surface Tension: " << surfaceTension << " N/m, Gas Fraction: " << (gasVolumeFraction * 100.0f) << "%\n";
    std::cout << "  - [Geometric Specs] Dimensions: " << (pouchWidth * 100.0f) << " cm (W) x " << (pouchHeight * 100.0f) 
              << " cm (H) x " << (pouchThickness * 100.0f) << " cm (T) [Layers: " << numLayers << "]\n";
    std::cout << "  - [Mechanical Compliance] Stretch: " << stretchCompliance << ", Bending: " << bendingCompliance << "\n";
    std::cout << "  - [Numerical Solver] dt: " << (timeStep * 1.0e6f) << " us, Force Leverage: " << kActiveStrainLeverage << "\n";
    std::cout << "  - [Orientation] Pitch: " << pitch << "°, Roll: " << roll << "°, Yaw: " << yaw << "°\n";
    std::cout << "  - [Viscoelasticity] Legacy Fallback Gamma: " << viscoelasticRelaxationRate << " s^-1, Max Creep Strain Limit: "
              << (viscoelasticMaxStrainLimit * 100.0f) << "%\n";
    std::cout << "  - [Viscoelastic Prony Series] Weights: [" << viscoelasticPronyW1 << ", " << viscoelasticPronyW2 << ", " << viscoelasticPronyW3
              << "], Taus: [" << (viscoelasticTau1 * 1000.0f) << " ms, " << (viscoelasticTau2 * 1000.0f) << " ms, " << (viscoelasticTau3 * 1000.0f) << " ms]\n";
    std::cout << "  - [Viscoelastic Eyring Creep] Activation Ref Force f0: " << viscoelasticEyringForce0 << " N\n";
    std::cout << "  - [SPH Viscosity Tuning] Alpha: " << sphViscosityAlpha << ", Beta: " << sphViscosityBeta << "\n";
    std::cout << "  - [Poisson CG Solver] Max Iterations: " << poissonMaxIterations << "\n";
}
