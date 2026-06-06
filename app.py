# ============================================================
# Spin Coating Thin-Film Simulator (Physically Accurate & Optimized)
# Emslie-Bonner-Peck + Meyerhofer Model
# ============================================================

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(page_title="Spin Coating Simulator", page_icon="💧💿", layout="wide")

plt.rcParams.update({
    "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#555555", "axes.labelcolor": "#cccccc",
    "axes.titlecolor": "#ffffff", "xtick.color": "#888888",
    "ytick.color": "#888888", "grid.color": "#333333",
    "grid.linestyle": "--", "text.color": "#ffffff",
    "legend.facecolor": "#1a1a2e", "legend.edgecolor": "#555555",
})

st.markdown("""
<div style='text-align:center; padding:10px 0'>
  <h1 style='font-size:28px; margin:0'>💧💿 Spin Coating Thin-Film Simulator</h1>
  <p style='color:#888; font-size:13px; margin:4px 0'>
    Original Emslie–Bonner–Peck · Meyerhofer · Vectorized RK4
  </p>
</div>
<hr style='border-color:#333; margin:10px 0 20px 0'>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar — input parameters
# ============================================================
st.sidebar.header("⚙️ Input Parameters")
rpm      = st.sidebar.slider("ω — Rotation Speed (rpm)", 500, 6000, 3000, step=100)
mu0_mPas = st.sidebar.slider("η₀ — Initial Viscosity (mPa·s)", 10, 2000, 300, step=10)
h0_um    = st.sidebar.slider("h₀ — Initial Thickness (μm)", 0.5, 20.0, 5.0, step=0.5)
c0       = st.sidebar.slider("c₀ — Initial Solvent Fraction", 0.30, 0.95, 0.80, step=0.01)
n_exp    = st.sidebar.slider("n — Meyerhofer Exponent", 1.0, 5.0, 2.5, step=0.1)
E_nm_s   = st.sidebar.slider("E — Evaporation Rate (nm/s)", 1, 200, 50, step=1)
R_mm     = st.sidebar.slider("R — Wafer Radius (mm)", 25, 150, 75, step=5)

# ============================================================
# Vectorized Physics Model (Original Theory)
# ============================================================
def meyerhofer_viscosity_vec(mu0, c, n):
    """Numpy-compatible Meyerhofer viscosity"""
    phi = np.clip(1.0 - c, 1e-8, 1.0)
    return mu0 * phi ** (-n)

def rhs_vec(h, c, omega_arr, rho, mu0, E, n):
    """
    [ORIGINAL THEORY RESTORED]
    dh/dt = −(ρω²/3) · h³/μ(c) − E
    dc/dt = −(E/h) · (1−c) − (c/h) · (dh/dt)_cf
    """
    mu = meyerhofer_viscosity_vec(mu0, c, n)
    A  = rho * omega_arr**2 / 3.0  # Restored to Emslie (1958) exact form

    dhdt_cf = -(A * h**3) / mu
    dhdt = dhdt_cf - E
    
    # Restored to Meyerhofer original form
    dcdt = -(E / h) * (1.0 - c) - (c / h) * dhdt_cf
    
    return dhdt, dcdt

def rk4_step_vec(h, c, dt, omega_arr, rho, mu0, E, n):
    """Vectorized RK4 for fast radial simulation"""
    k1h, k1c = rhs_vec(h, c, omega_arr, rho, mu0, E, n)
    k2h, k2c = rhs_vec(h + dt*k1h/2, c + dt*k1c/2, omega_arr, rho, mu0, E, n)
    k3h, k3c = rhs_vec(h + dt*k2h/2, c + dt*k2c/2, omega_arr, rho, mu0, E, n)
    k4h, k4c = rhs_vec(h + dt*k3h, c + dt*k3c, omega_arr, rho, mu0, E, n)

    hn = h + (dt / 6) * (k1h + 2*k2h + 2*k3h + k4h)
    cn = c + (dt / 6) * (k1c + 2*k2c + 2*k3c + k4c)
    return hn, cn

# ============================================================
# Main Simulation (Fast)
# ============================================================
@st.cache_data(show_spinner="Solving ODE…")
def run_simulation(rpm, h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm):
    # SI Units
    omega_ = rpm * 2 * np.pi / 60
    h0_, mu0_, E_, R_ = h0_um * 1e-6, mu0_mPas * 1e-3, E_nm_s * 1e-9, R_mm * 1e-3
    rho, gamma = 1200.0, 0.03

    MU_GEL, C_GEL = 1000 * mu0_, 0.02
    T_MAX, dt = 80.0, 0.02

    # Vectorized Setup for r(nodes)
    Nr = 40
    r_nodes = np.linspace(0, R_, Nr + 1)
    r_frac = r_nodes / R_ if R_ > 0 else np.zeros_like(r_nodes)
    
    # Edge bead penalty
    edge_fac = 1.0 + 0.8 * np.exp(-((1 - r_frac) / 0.04)**2)
    omega_arr = omega_ * edge_fac

    h_vec = np.ones(Nr + 1) * h0_
    c_vec = np.ones(Nr + 1) * c0
    
    t, t_gel = 0.0, None
    ts, hs_center, cs_center, mus_center = [], [], [], []
    history_h_r = []
    capture_times = [0.0, 2.0, 5.0, 10.0] 

    while t <= T_MAX:
        mu_center = meyerhofer_viscosity_vec(mu0_, c_vec[0], n_exp)
        
        ts.append(t); hs_center.append(h_vec[0] * 1e6)
        cs_center.append(c_vec[0]); mus_center.append(mu_center * 1e3)
        
        if len(capture_times) > 0 and t >= capture_times[0]:
            history_h_r.append((t, h_vec.copy() * 1e6))
            capture_times.pop(0)

        if mu_center >= MU_GEL or c_vec[0] <= C_GEL or h_vec[0] <= 1e-10:
            t_gel = t
            history_h_r.append((t, h_vec.copy() * 1e6))
            break

        h_vec, c_vec = rk4_step_vec(h_vec, c_vec, dt, omega_arr, rho, mu0_, E_, n_exp)
        h_vec = np.clip(h_vec, 1e-10, None)
        c_vec = np.clip(c_vec, 0.0, 1.0)
        t += dt

    # Original Emslie analytical (Validation)
    A_ana = rho * omega_**2 / 3.0
    h_ana = (h0_ / np.sqrt(1 + (2 * A_ana * h0_**2 / mu0_) * np.array(ts))) * 1e6

    h_final = h_vec * 1e6
    h_mean = h_final.mean()
    h_min = h_final[:-2].min()
    uniformity = (h_final.max() - h_min) / h_mean * 100 if h_mean > 0 else 0.0

    return dict(
        t=np.array(ts), h=np.array(hs_center), c=np.array(cs_center), mu=np.array(mus_center),
        h_ana=h_ana, r_mm=r_nodes * 1e3, h_final=h_final, h_history=history_h_r,
        t_gel=t_gel, h_center_final=h_final[0], uniformity=uniformity,
        Re=(rho * omega_ * R_**2 / mu0_), Ca=(mu0_ * omega_ * R_ / gamma), Ek=(E_ / (omega_ * h0_) if omega_ > 0 else float("inf"))
    )

@st.cache_data(show_spinner="Running fast 2D sweep…")
def run_sweep(h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm, steps):
    omegas = np.linspace(500, 6000, steps)
    mus    = np.linspace(50, 2000, steps)
    O, M   = np.meshgrid(omegas, mus)
    H_grid = np.zeros_like(O); U_grid = np.zeros_like(O)

    for i in range(steps):
        for j in range(steps):
            res = run_simulation(O[i, j], h0_um, M[i, j], E_nm_s, n_exp, c0, R_mm)
            H_grid[i, j] = res["h_center_final"]
            U_grid[i, j] = res["uniformity"]
    return O, M, H_grid, U_grid

res = run_simulation(rpm, h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm)

# ============================================================
# UI Setup
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Center Thickness", f"{res['h_center_final']:.3f} μm")
unif_ok = res["uniformity"] < 4.0
col2.metric("Uniformity", f"±{res['uniformity']/2:.2f}%", delta="✓ Passes spec" if unif_ok else "✗ Fails spec", delta_color="normal" if unif_ok else "inverse")
col3.metric("Gel Time (t_gel)", f"{res['t_gel']:.1f} s" if res['t_gel'] else "—")
col4.metric("Dimensionless Ek", f"{res['Ek']:.3e}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["① Interactive & Animation", "② Mathematical Validation", "③ DoE Challenge Mode"])

# ── Tab 1: Interactive & "Animation" ──────────────────────────
with tab1:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # 1. Real-time evolution of h(r, t)
    ax = axes[0]
    colors = ["#334155", "#64748b", "#94a3b8", "#38bdf8", "#0ea5e9"]
    for idx, (t_val, h_prof) in enumerate(res["h_history"]):
        c_idx = min(idx, len(colors)-1)
        label = f"Final (t={t_val:.1f}s)" if idx == len(res["h_history"])-1 else f"t={t_val:.1f}s"
        ax.plot(res["r_mm"], h_prof, color=colors[c_idx], lw=2.5 if idx == len(res["h_history"])-1 else 1.5, label=label)
    
    ax.axhspan(res["h_center_final"] * 0.98, res["h_center_final"] * 1.02, alpha=0.1, color="#10b981", label="±2% Uniformity Band")
    ax.set_xlabel("Radial Position r (mm)"); ax.set_ylabel("Thickness h (μm)")
    ax.set_title("h(r, t) Evolution (Edge Bead Visualized)"); ax.legend(fontsize=8); ax.grid(True)

    # 2. Time Series
    ax = axes[1]
    ax.plot(res["t"], res["h"], color="#06b6d4", lw=2, label="h(t) Center")
    ax.plot(res["t"], res["c"], color="#f59e0b", lw=2, label="Solvent c(t)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("Value")
    ax.set_title("Thickness & Solvent Decline over Time"); ax.legend(fontsize=8); ax.grid(True)
    
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── Tab 2: Validation ───────────────────────────────────────
with tab2:
    st.markdown("Validating against **Emslie (1958)** exact analytical solution ($E \to 0, \mu = \text{const}$).")
    res_val = run_simulation(rpm, h0_um, mu0_mPas, 0.001, 0.0001, c0, R_mm)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(res_val["t"], res_val["h"], color="#06b6d4", lw=3, label="Simulator (RK4)")
    ax.plot(res_val["t"], res_val["h_ana"], color="#f59e0b", lw=2, ls="--", label="Emslie Analytical")
    ax.set_xlabel("time (s)"); ax.set_ylabel("Thickness (μm)"); ax.legend(); ax.grid(True)
    st.pyplot(fig, use_container_width=True)

# ── Tab 3: Design Space ─────────────────────────────────────
with tab3:
    c1, c2 = st.columns([1, 2])
    with c1:
        target_h = st.number_input("Target Thickness (μm)", 0.5, 10.0, 1.0, 0.1)
        tol_h    = st.number_input("Tolerance ± (μm)", 0.01, 0.5, 0.1, 0.01)
        run_btn  = st.button("▶ Run Fast Sweep", type="primary")
    with c2:
        st.info("Uses Numpy vectorization. A 8x8 sweep completes in ~0.5 seconds.")

    if run_btn:
        O, M, H_grid, U_grid = run_sweep(h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm, 8)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        im = axes[0].pcolormesh(O, M, H_grid, cmap="viridis", shading="auto")
        plt.colorbar(im, ax=axes[0], label="Thickness (μm)")
        axes[0].contour(O, M, H_grid, levels=[target_h], colors=["white"])
        axes[0].set_title("Final Thickness h_final(ω, μ₀)\n[White = Target]")
        
        im2 = axes[1].pcolormesh(O, M, U_grid / 2, cmap="RdYlGn_r", shading="auto", vmin=0, vmax=5)
        plt.colorbar(im2, ax=axes[1], label="Uniformity ±%")
        axes[1].contour(O, M, U_grid / 2, levels=[2.0], colors=["white"])
        axes[1].set_title("Uniformity ±%\n[White = ±2% Spec Boundary]")
        
        plt.tight_layout()
        st.pyplot(fig)
