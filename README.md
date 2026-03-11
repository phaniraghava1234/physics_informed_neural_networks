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
* 👉 **[Click here for the Python/Colab Code](./Pitch-Plunge Aeroelasticity PINN.ipynb)**

### ⏳ Future Work (Coming Soon)
* **Problem 5:** 2D Transient Heat Equation (Thermodynamics)
* **Problem 6:** 2D Wave Equation (Acoustics & Vibrations)

## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch
* **Environment:** Google Colab
* **Approach:** Continuous time, data-free interior domain training using automatic differentiation (`torch.autograd`).

## 👨‍💻 About
This repository was built by [phaniraghava1234](https://github.com/phaniraghava1234/physics_informed_neural_networks) to showcase advanced scientific machine learning techniques applied to aerospace and physics domains.
