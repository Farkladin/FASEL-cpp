# FASEL: High-Fidelity HASEL Actuator Simulator

This repository contains the C++20, Swift, and Metal source files for the high-fidelity 3D electrohydraulic simulator of HASEL (Hydraulically Amplified Self-healing Electrostatic) actuators. The simulator integrates smoothed particle hydrodynamics (SPH), extended position-based dynamics (XPBD), Poisson electrostatic field equations, and micro-bubble dielectric breakdown diagnostics (Paschen's Law).

---

## 1. Requirements

To compile and run this simulator, the following environment is required:
*   **Operating System**: macOS (designed specifically for Apple Silicon Unified Memory Architecture).
*   **Compiler**: Apple Clang supporting C++20 standards.
*   **Graphics API**: Apple Metal Framework.
*   **External Library**: Apple `metal-cpp` wrapper headers.
*   **Language Bridge**: Swift compiler (`swiftc`) and Objective-C runtime libraries (for the cryptographically secure deterministic PRNG bridge).

---

## 2. Compilation and Build Instructions

The simulator requires compiling C++ files, compiling the Swift random number generator bridge, and linking against macOS frameworks.

### Method A: Command Line Compilation
Ensure that Apple `metal-cpp` headers are installed and accessible. You can compile the project using a single command sequence from your terminal:

```bash
# 1. Compile the Swift bridge source to an object file
swiftc -c SwiftCrypto.swift -o SwiftCrypto.o -parse-as-library

# 2. Compile and link all C++ source files alongside the Swift object
clang++ -std=c++20 -Os -Wall -Wextra \
  -I/path/to/metal-cpp \
  main.cpp SPHXSolver.cpp PoissonSolver.cpp BreakdownAnalyzer.cpp PhysicsConfig.cpp MetalMacro.cpp SwiftCrypto.o \
  -framework Metal -framework QuartzCore -framework Foundation \
  -L/usr/lib/swift -fobjc-link-runtime \
  -o FASEL-cpp
```

*Note: Replace `/path/to/metal-cpp` with the actual path to your Apple `metal-cpp` directory.*

### Method B: Xcode Project Setup (Recommended)
If configuring through the Xcode IDE:
1. Create a new **Command Line Tool** project in Xcode.
2. Select **C++** as the primary language.
3. Import all `.cpp`, `.hpp`, `.swift`, and `.metal` files from this directory into the project target.
4. Under project **Build Settings**:
    *   Set **C++ Language Dialect** to **C++20** (`-std=c++20`).
    *   Add the path containing the `metal-cpp` folder to **Header Search Paths**.
    *   Add `-L/usr/lib/swift` and `-fobjc-link-runtime` to **Other Linker Flags**.
5. Compile and run the target. The Xcode build system natively compiles `Shaders.metal` into `default.metallib` during build phases.

---

## 3. Command Line Interface (CLI) Flags and Options

The executable supports comprehensive runtime configurations. Specify arguments using the `--flag value` format.

### General Simulation Controls
*   `--mode [fast | medium | high]`  
    Set the physical mesh and solver resolution.
    *   `fast`: Analytical effective medium quasi-static solver. Runs instantly.
    *   `medium` (Default): 1,010 fluid/boundary particles ($0.80\text{ mm}$ SPH spacing). High convergence rate.
    *   `high`: 2,472 fluid/boundary particles ($0.50\text{ mm}$ SPH spacing). Maximum spatial fidelity.
*   `--steps [integer]`  
    Maximum numerical solver steps to execute (Default: `1500` steps).
*   `--config [path]`  
    Path to the physical constants CSV configuration file (Default: `physics_constants.csv`).
*   `--output [path]`  
    Path to export the final particle-state breakdown diagnostics CSV (Default: `report.csv`).
*   `--verbose`  
    Enable detailed step-by-step solver projection logs on stdout.

### Geometric and Structural Parameters
*   `--phi [float]`  
    Gas volume fraction inside the pouch. Value range is `0.0` (pure liquid) to `1.0` (pure gas) (Default: `0.20` / 20%).
*   `--layers [integer]` or `-l [integer]`  
    Number of stacked electrohydraulic pouch layers to simulate (Default: `1`).
*   `--pouch-width [float]`  
    Total physical width of the pouch in meters (Default: `0.02` / 2 cm).
*   `--pouch-height [float]`  
    Total physical height of the pouch in meters (Default: `0.04` / 4 cm).
*   `--pouch-thickness [float]`  
    Out-of-plane physical thickness in meters (Default: `0.004` / 4 mm).
*   `--pouch-material [bopet | bopp | tpu | pdms | ecoflex | custom]`  
    Outer membrane preset which maps mechanical compliance and dielectric breakdown strengths.
    *   `bopet`: Mylar shell. Strict inextensibility (`stretch = 0.0`), high bending stiffness (`5e-7`), breakdown strength `3.0e8 V/m`.
    *   `bopp`: Polypropylene shell. Strict inextensibility (`stretch = 0.0`), bending `1.0e-6`, breakdown strength `2.0e8 V/m`.
    *   `tpu`: Elastomeric Polyurethane. Elastic stretch (`1e-7`), bending `1e-6`, breakdown `1.2e8 V/m`.
    *   `pdms`: High elasticity (`1e-6`), bending `1e-5`, breakdown `9.0e7 V/m`.
    *   `ecoflex`: Extreme compliance (`5e-6`), bending `5e-5`, breakdown `3.0e7 V/m`.
    *   `custom`: Relies strictly on manually configured compliance parameters.
*   `--stretch-compliance [float]`  
    XPBD tensile distance constraint compliance (Default: `1.0e-6`). Set to `0.0` for inextensible BOPET/BOPP.
*   `--bending-compliance [float]`  
    XPBD out-of-plane dihedral angle bending constraint compliance (Default: `1.0e-5`).

### Electromechanical and Fluid Options
*   `--voltage [float]`  
    Constant amplitude of the applied step voltage waveform in Volts (Default: `7000.0` V).
*   `--sweep`  
    Triggers a sequential electromechanical voltage sweep from $1,000\text{ V}$ to $12,000\text{ V}$ in $10$ discrete steps.
*   `--liquid-permittivity [float]`  
    Relative permittivity ($\epsilon_r$) of the dielectric fluid (Default: `2.7` for silicone oil).
*   `--liquid-viscosity [float]`  
    Viscous shear damping coefficient ($\mu$) of the fluid in $\text{Pa}\cdot\text{s}$ (Default: `0.0048` for silicone oil, `0.033` for FR3 soybean oil).
*   `--surface-tension [float]`  
     Morris Color-Gradient CSF SPH surface tension coefficient in $\text{N/m}$ (Default: `0.021`).

### Kinematic Motion and Boundary Loads
*   `--motion-freq-trans [float]`  
    Real-time translational harmonic vibration frequency (Hz) applied to pouch frame anchors.
*   `--motion-amp-trans [float]`  
    Translation vibration amplitude in meters.
*   `--motion-freq-rot [float]`  
    Real-time rotational harmonic yaw/sway frequency (Hz) applied to pouch frame anchors.
*   `--motion-amp-rot [float]`  
    Rotational vibration amplitude in degrees.
*   `--ext-force-x [float]`  
    External deadweight force component acting along the x-axis (Newtons) applied to designated load particles.
*   `--ext-force-y [float]`  
    External deadweight force component acting along the y-axis (Newtons) applied to designated load particles (Default: `0.0`).
*   `--pitch [float]`  
    3D spatial pitch rotation angle (degrees) to project the gravitational vector onto the simulation plane.
*   `--roll [float]`  
    3D spatial roll rotation angle (degrees).
*   `--yaw [float]`  
    3D spatial yaw rotation angle (degrees).

### Transient Dynamic Test Mode
*   `--transient-test`  
    Overrides standard step settings and runs a dedicated 1200-step transient response sweep ($30.0\text{ ms}$ physical time):
    1.  **Steps 1 to 200 (0.0 ms to 5.0 ms)**: Voltage is OFF ($0\text{ V}$) to establish quiescent sag stabilization.
    2.  **Steps 201 to 700 (5.0 ms to 17.5 ms)**: Voltage is ON (amplitude set by `--voltage`) to characterize contraction dynamics.
    3.  **Steps 701 to 1200 (17.5 ms to 30.0 ms)**: Voltage is OFF ($0\text{ V}$) to analyze passive viscoelastic relaxation.
    *   Prints a post-simulation dynamic characterization report, containing rise time ($t_{10} - t_{90}$), passive decay time ($t_{90} - t_{10}$), average contraction strain rates ($\dot{\epsilon}$ in $\%/\text{s}$), and outputs an **ASCII Timeline Strain Plot** to standard output. Saves the evaluation report to `transient_response_evaluation.md`.

---

## 4. Physical Constants CSV Configuration File

If the path targeted by `--config` is missing at execution startup, the simulator automatically bootstraps and writes a template file containing standard default variables. You can edit this file (`physics_constants.csv`) to alter physical properties without recompiling the source code.

### Standard CSV Entry Layout:
```csv
A, 11.25, Townsend ionization parameter A (Pa^-1 m^-1)
B, 273.75, Townsend ionization parameter B (V/Pa/m)
gammaSE, 0.01, Secondary electron emission coefficient
liquidBreakdownThreshold, 60000000.0, Dielectric liquid breakdown strength (V/m)
liquidPermittivity, 2.7, Fluid relative permittivity
liquidViscosity, 0.0048, Fluid dynamic viscosity (Pa*s)
surfaceTension, 0.021, SPH interface surface tension (N/m)
gasVolumeFraction, 0.2, Initial gas volume fraction (0.0 - 1.0)
appliedVoltage, 7000.0, Base applied voltage (V)
pouchHeight, 0.04, Total pouch height (m)
pouchWidth, 0.02, Total pouch width (m)
pouchThickness, 0.004, Out-of-plane pouch thickness (m)
pitch, 90.0, Pitch angle for gravity projection (degrees)
roll, 0.0, Roll angle for gravity projection (degrees)
yaw, 0.0, Yaw angle for gravity projection (degrees)
stretchCompliance, 0.0, XPBD membrane stretch compliance
bendingCompliance, 0.00001, XPBD membrane bending compliance
membraneBreakdownThreshold, 90000000.0, Solid membrane breakdown strength (V/m)
numLayers, 1, Stacking layers count
timeStep, 0.000025, Simulation timestep dt (seconds)
kPouchAnchorYThreshold, 0.43, Height ratio for boundary anchors
kBoundaryRadiusBox, 0.015, Out-of-plane SPH boundary buffer
kBoundaryRadiusLimit, 0.012, Safe out-of-plane particle boundary limit
kActiveStrainLeverage, 0.01, Dynamic strain evaluation scaling
viscoelasticRelaxationRate, 0.05, Membrane relaxation rate (s^-1)
viscoelasticMaxStrainLimit, 0.12, Membrane max viscoelastic creep strain
viscoelasticPronyW1, 0.3, Prony series weight 1
viscoelasticPronyW2, 0.3, Prony series weight 2
viscoelasticPronyW3, 0.2, Prony series weight 3
viscoelasticTau1, 0.001, Prony relaxation time 1 (seconds)
viscoelasticTau2, 0.01, Prony relaxation time 2 (seconds)
viscoelasticTau3, 0.1, Prony relaxation time 3 (seconds)
viscoelasticEyringForce0, 0.5, Eyring activation force scale (N)
sphViscosityAlpha, 0.15, SPH artificial viscosity parameter alpha
sphViscosityBeta, 0.3, SPH artificial viscosity parameter beta
poissonMaxIterations, 180, Electrostatic JPCG solver maximum iterations
```

---

## 5. Standard Output Diagnostics

During numerical solving, the tool prints periodic system metrics:
*   **Step**: Current computational step.
*   **t**: Elapsed simulator physical time (seconds).
*   **Strain**: 3D Frame-Invariant Centroid-to-Centroid Actuation Strain.
*   **Max Force**: Sum of out-of-plane tensile blocking forces (Newtons) accumulated across physical boundary constraints at anchors.
*   **Peak DBI**: Max local Electric Field to Paschen Threshold ratio. Values $\ge 1.0$ indicate gaseous dielectric breakdown (sparks present).
