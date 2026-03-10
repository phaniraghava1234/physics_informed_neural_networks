# Physics-Informed Neural Networks (PINNs) Portfolio

Welcome to my portfolio of Physics-Informed Neural Networks (PINNs). This repository contains a collection of 5 distinct Partial Differential Equations (PDEs) solved using deep learning in PyTorch. 

The primary objective of this project is to demonstrate how neural networks can be trained **without any internal domain data**, relying strictly on the governing physical laws (PDEs) and boundary/initial conditions to predict complex fluid and system dynamics.

All code is developed and tested using Google Colab.

## 🗂️ Portfolio Structure

### [Problem 1: 1D Burgers' Equation](./Problem1_Burgers_Equation/)
**Focus:** Non-linear convection-diffusion and shockwave formation.
The Burgers' equation is a fundamental PDE in fluid mechanics. In this problem, a PINN is trained to capture the formation of a sharp shockwave over time. The results are rigorously validated against the exact analytical solution derived via the Cole-Hopf transformation. 
* 👉 **[Click here for the detailed Problem 1 explanation and code.](./Problem1_README.md/)**

### ⏳ Future Work (Coming Soon)
Over the course of this project, four additional PDEs will be added, with a strong emphasis on aerodynamics and flight dynamics:
* **Problem 2:** 1D Euler Equations (Aerodynamics - Sod Shock Tube Problem)
* **Problem 3:** 2D Incompressible Navier-Stokes Equations (Aerodynamics - Flow over a cylinder)
* **Problem 4:** Pitch-Plunge Aeroelasticity System (Flight Dynamics / Flutter Analysis)
* **Problem 5:** 2D Heat/Wave Equation (Thermodynamics/Acoustics)

## 🛠️ Tech Stack & Methodology
* **Framework:** PyTorch
* **Environment:** Google Colab
* **Approach:** Continuous time, data-free interior domain training using automatic differentiation (`torch.autograd`).

## 👨‍💻 About
This repository was built by [phaniraghava1234](https://github.com/phaniraghava1234/physics_informed_neural_networks) to showcase advanced scientific machine learning techniques applied to aerospace and physics domains.
