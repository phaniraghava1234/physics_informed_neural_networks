# Physics-Informed Neural Networks (PINNs) Portfolio

Welcome to my portfolio of Physics-Informed Neural Networks (PINNs). This repository contains a collection of 5 distinct Partial Differential Equations (PDEs) solved using deep learning in PyTorch. 

The primary objective of this project is to demonstrate how neural networks can be trained **without any internal domain data**, relying strictly on the governing physical laws (PDEs) and boundary/initial conditions to predict complex fluid and system dynamics.

All code is developed and tested using Google Colab.

## 🗂️ Portfolio Structure

### Problem 1: 1D Burgers' Equation
**Focus:** Non-linear convection-diffusion and shockwave formation.
* 👉 **[Click here for the detailed Problem 1 explanation](./PROBLEM1_README.md)**
* 👉 **[Click here for the Python/Colab Code](./1D_burgers_PINN.ipynb)**

### Problem 2: 1D Euler Equations (Sod Shock Tube)
**Focus:** Compressible Aerodynamics, Shockwaves, and Artificial Viscosity.
The 1D Euler equations govern compressible, inviscid fluid flow. This problem demonstrates the challenge of training deep learning models on sharp discontinuities (shocks, contact surfaces) and uses Artificial Viscosity to stabilize the network.
* 👉 **[Click here for the detailed Problem 2 explanation](./PROBLEM2_README.md)**
* 👉 **[Click here for the Python/Colab Code](./1d_Euler_Shock_Tube_PINN.ipynb)**

### Problem 3: 2D Incompressible Navier-Stokes Equations
**Focus:** 2D Aerodynamics, Kovasznay Flow, and Steady Incompressible Fluids.
The Navier-Stokes equations are the ultimate governing equations of fluid dynamics. In this problem, a PINN learns the 2D velocity and pressure fields of the classic Kovasznay flow (modeling the wake behind a grid) relying purely on boundary conditions and physics losses.
* 👉 **[Click here for the detailed Problem 3 explanation](./PROBLEM3_README.md)**
* 👉 **[Click here for the Python/Colab Code](./2D_Navier_Stokes_PINN.ipynb)**

### Problem 4: Pitch-Plunge Aeroelasticity System
**Focus:** Flight Dynamics, Coupled ODEs, Spectral Bias, and Fourier Features.
Solving time-dependent, 2nd-order coupled structural dynamics. In this problem, a PINN is trained to predict the damped aeroelastic response (pitching and plunging) of an aircraft wing section, overcoming Spectral Bias using Fourier Feature Embeddings.
* 👉 **[Click here for the detailed Problem 4 explanation](./PROBLEM4_README.md)**
* 👉 **[Click here for the Python/Colab Code](./Pitch_Plunge_Aeroelasticity_PINN.ipynb)**

### Problem 5: 2D Transient Heat Equation
**Focus:** Thermodynamics, 3D Spatiotemporal Inputs, and Mixed Partial Derivatives.
Solving parabolic PDEs over a 2D spatial domain and time. The PINN simultaneously maps $X$, $Y$, and $T$ inputs to predict the diffusion of a heat spike across a metal plate, demonstrating the network's ability to learn 3D volumetric physics data.
* 👉 **[Click here for the detailed Problem 5 explanation](./PROBLEM5_README.md)**
* 👉 **[Click here for the Python/Colab Code](./2D_Heat_Equation_PINN.ipynb)**

### Problem 6: Aeroacoustics (Helmholtz Sound Scattering)
**Focus:** Time-Harmonic Acoustics, Complex Fields, SIREN Networks, and Scattered Field Formulation.
For the Grand Finale, a PINN is engineered to predict the scattering of high-frequency sound waves by an aerospace blunt body. This project documents the architectural evolution required to overcome severe phase-shifting, Fourier streaking, and wave propagation failures using Sine Representation Networks (SIREN) and Scattered Field physics mapping.
* 👉 **[Click here for the detailed Problem 6 explanation](./PROBLEM6_README.md)**

## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch
* **Environment:** Google Colab / Local Windows PC (RTX 2060)
* **Approach:** Continuous time, data-free interior domain training using automatic differentiation (`torch.autograd`).

## 👨‍💻 About
This repository was built by [phaniraghava1234](https://github.com/phaniraghava1234/physics_informed_neural_networks) to showcase advanced scientific machine learning techniques applied to aerospace and physics domains.
"""
