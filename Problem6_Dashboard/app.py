import os
# CRITICAL FIX 1: Completely hide the GPU from PyTorch so it loads instantly
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import matplotlib
# CRITICAL FIX 2: Force matplotlib to run in "headless" mode so it doesn't freeze Windows
matplotlib.use('Agg') 

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.special as sp

# --- 1. Page Configuration ---
st.set_page_config(page_title="Aeroacoustics PINN", layout="wide")
st.title("🛩️ Aeroacoustics Physics-Informed Neural Network")
st.markdown("""
Welcome to the interactive dashboard for **Problem 6: Acoustic Scattering by a Rigid Strut**. 
This neural network was trained **without any simulation data**, using only the physical laws of the Helmholtz equation and a SIREN (Sine Representation) architecture via a Scattered Field Formulation.
""")

# --- 2. Load Model Definition ---
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=10.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(nn.Module):
    def __init__(self, layers, omega_0=10.0):
        super().__init__()
        self.net = nn.ModuleList()
        for i in range(len(layers)-2):
            self.net.append(SineLayer(layers[i], layers[i+1], is_first=(i==0), omega_0=omega_0))
        self.net.append(nn.Linear(layers[-2], layers[-1]))
    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)
        for layer in self.net: out = layer(out)
        return out[:, 0:1], out[:, 1:2]

@st.cache_resource
def load_model():
    device = torch.device('cpu') # Force CPU for web deployment
    layers = [2, 128, 128, 128, 128, 2]
    model = SIREN(layers, omega_0=10.0).to(device)
    try:
        model.load_state_dict(torch.load('siren_aeroacoustics.pth', map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'siren_aeroacoustics.pth' not found! Please upload it.")
        return None

model = load_model()

# --- 3. Exact Math Function ---
@st.cache_data
def exact_acoustic_scattering(x, y, k=4.0, a=0.5):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    p_inc = np.exp(1j * k * x)
    p_s = np.zeros_like(p_inc, dtype=np.complex128)
    for m in range(25): 
        eps = 1 if m == 0 else 2
        J_prime = sp.jvp(m, k * a)
        H_prime = sp.h1vp(m, k * a)
        term = -eps * (1j**m) * (J_prime / H_prime) * sp.hankel1(m, k * r) * np.cos(m * theta)
        p_s += term
    p_tot = p_inc + p_s
    if isinstance(r, np.ndarray):
        p_tot[r < a] = np.nan; p_s[r < a] = np.nan
    return np.real(p_s), np.imag(p_s), np.real(p_tot), np.imag(p_tot)

# --- 4. Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
field_type = st.sidebar.radio("Acoustic Field to View:", ["Total Field (Actual Sound)", "Scattered Field Only (Echo)"])
view_mode = st.sidebar.selectbox("Measurement Metric:", ["Real Pressure (Pa)", "Sound Pressure Level (SPL dB)"])

st.sidebar.markdown("---")
st.sidebar.header("Interactive 1D Cross-Section")
y_slice = st.sidebar.slider("Y-Axis Cross-Section (Slice through the wave)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

# --- 5. Data Generation ---
k_wave = 4.0; a_cyl = 0.5; R_out = 2.0
grid_points = 200 
x_grid = np.linspace(-R_out, R_out, grid_points)
y_grid = np.linspace(-R_out, R_out, grid_points)
X, Y = np.meshgrid(x_grid, y_grid)

X_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
Y_flat = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)

if model:
    with torch.no_grad():
        prs_pred, pis_pred = model(X_flat, Y_flat)
        prs_pred = prs_pred.numpy().reshape(grid_points, grid_points)
        pis_pred = pis_pred.numpy().reshape(grid_points, grid_points)

    prs_exact, pis_exact, prt_exact, pit_exact = exact_acoustic_scattering(X, Y)

    if field_type == "Scattered Field Only (Echo)":
        pr_target_pred = prs_pred
        pi_target_pred = pis_pred
        pr_target_exact = prs_exact
        pi_target_exact = pis_exact
    else:
        # Reconstruct Total Field
        p_inc_r = np.cos(k_wave * X)
        p_inc_i = np.sin(k_wave * X)
        pr_target_pred = prs_pred + p_inc_r
        pi_target_pred = pis_pred + p_inc_i
        pr_target_exact = prt_exact
        pi_target_exact = pit_exact

    # Mask cylinder
    r_mesh = np.sqrt(X**2 + Y**2)
    pr_target_pred[r_mesh < a_cyl] = np.nan
    pi_target_pred[r_mesh < a_cyl] = np.nan

    if view_mode == "Real Pressure (Pa)":
        plot_exact = pr_target_exact
        plot_pred = pr_target_pred
        cmap = 'RdBu'
        vmin, vmax = -2.0, 2.0
        label = "Pressure (Pa)"
    else: # SPL
        p_ref = 2e-5
        amp_exact = np.sqrt(pr_target_exact**2 + pi_target_exact**2)
        amp_pred = np.sqrt(pr_target_pred**2 + pi_target_pred**2)
        plot_exact = 20 * np.log10(amp_exact / (np.sqrt(2) * p_ref) + 1e-5)
        plot_pred = 20 * np.log10(amp_pred / (np.sqrt(2) * p_ref) + 1e-5)
        cmap = 'magma'
        vmin, vmax = 60.0, 98.0
        label = "SPL (dB)"

    # --- 6. 2D Contour Plotting ---
    col1, col2 = st.columns(2)
    
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    c1 = ax1.contourf(X, Y, plot_exact, levels=np.linspace(vmin, vmax, 100), cmap=cmap, extend='both')
    ax1.add_patch(patches.Circle((0, 0), a_cyl, color='black'))
    ax1.axhline(y_slice, color='green', linestyle='--', alpha=0.7) # Show slice line
    fig1.colorbar(c1, ax=ax1, label=label)
    ax1.set_title("Exact Mathematical Solution")
    col1.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    c2 = ax2.contourf(X, Y, plot_pred, levels=np.linspace(vmin, vmax, 100), cmap=cmap, extend='both')
    ax2.add_patch(patches.Circle((0, 0), a_cyl, color='black'))
    ax2.axhline(y_slice, color='green', linestyle='--', alpha=0.7) # Show slice line
    fig2.colorbar(c2, ax=ax2, label=label)
    ax2.set_title("PINN Prediction")
    col2.pyplot(fig2)

    # --- 7. 1D Slice Plotting ---
    st.markdown(f"### 1D Cross-Section at $Y = {y_slice:.1f}$")
    
    # Get closest index for the slice
    y_idx = (np.abs(y_grid - y_slice)).argmin()
    slice_exact = plot_exact[y_idx, :]
    slice_pred = plot_pred[y_idx, :]
    
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(x_grid, slice_exact, 'b-', lw=3, label="Exact Analytical")
    ax3.plot(x_grid, slice_pred, 'r--', lw=2, label="PINN Prediction")
    
    # Shade the cylinder if the slice goes through it
    if abs(y_slice) < a_cyl:
        x_intersect = np.sqrt(a_cyl**2 - y_slice**2)
        ax3.axvspan(-x_intersect, x_intersect, color='gray', alpha=0.3, label='Rigid Strut')
        
    ax3.set_xlabel("X coordinate")
    ax3.set_ylabel(label)
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)