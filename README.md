# Physics-Informed Neural Networks (PINNs) Portfolio

Welcome to my portfolio of Physics-Informed Neural Networks (PINNs). This repository contains a collection of 5 distinct Partial Differential Equations (PDEs) solved using deep learning in PyTorch. 

The primary objective of this project is to demonstrate how neural networks can be trained **without any internal domain data**, relying strictly on the governing physical laws (PDEs) and boundary/initial conditions to predict complex fluid and system dynamics.

All code is developed and tested using Google Colab.

## 🗂️ Portfolio Structure

### Problem 1: 1D Burgers' Equation
**Focus:** Non-linear convection-diffusion and shockwave formation.
* 👉 **[Click here for the detailed Problem 1 explanation](./PROBLEM1_README.md)**
* 👉 **[Click here for the Python/Colab Code](./burgers_pinn.ipynb)**

### Problem 2: 1D Euler Equations (Sod Shock Tube)
**Focus:** Compressible Aerodynamics, Shockwaves, and Artificial Viscosity.
The 1D Euler equations govern compressible, inviscid fluid flow. This problem demonstrates the challenge of training deep learning models on sharp discontinuities (shocks, contact surfaces) and uses Artificial Viscosity to stabilize the network.
* 👉 **[Click here for the detailed Problem 2 explanation](./PROBLEM2_README.md)**
* 👉 **[Click here for the Python/Colab Code](./euler_pinn.ipynb)**

### ⏳ Future Work (Coming Soon)
* **Problem 3:** 2D Incompressible Navier-Stokes Equations (Aerodynamics - Flow over a cylinder)
* **Problem 4:** Pitch-Plunge Aeroelasticity System (Flight Dynamics / Flutter Analysis)
* **Problem 5:** 2D Heat/Wave Equation (Thermodynamics/Acoustics)

## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch
* **Environment:** Google Colab
* **Approach:** Continuous time, data-free interior domain training using automatic differentiation (`torch.autograd`).

## 👨‍💻 About
This repository was built by [phaniraghava1234](https://github.com/phaniraghava1234/physics_informed_neural_networks) to showcase advanced scientific machine learning techniques applied to aerospace and physics domains.
