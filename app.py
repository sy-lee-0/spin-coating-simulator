# ============================================================
# Spin Coating Thin-Film Simulator
# Emslie-Bonner-Peck + Meyerhofer Model
#
# How to run:
#   pip install streamlit numpy matplotlib pandas
#   streamlit run app.py
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

st.set_page_config(
    page_title="Spin Coating Simulator",
    page_icon="🌀",
    layout="wide"
)

# ============================================================
# Matplotlib dark theme
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "axes.edgecolor":   "#555555",
    "axes.labelcolor":  "#cccccc",
    "axes.titlecolor":  "#ffffff",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "grid.color":       "#333333",
    "grid.linestyle":   "--",
    "text.color":       "#ffffff",
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#555555",
    "lines.linewidth":  2.0,
})

# ============================================================
# Page title
# ============================================================

st.markdown("""
<div style='text-align:center; padding:10px 0'>
  <h1 style='font-size:28px; margin:0'>🌀 Spin Coating Thin-Film Simulator</h1>
  <p style='color:#888; font-size:13px; margin:4px 0'>
    Emslie–Bonner–Peck (1958)  ·  Meyerhofer (1978)  ·  RK4 Numerical Integration
  </p>
</div>
<hr style='border-color:#333; margin:10px 0 20px 0'>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar — input parameters
# ============================================================

st.sidebar.header("⚙️ Input Parameters")

st.sidebar.markdown("**Rotation**")
rpm = st.sidebar.slider("ω — Rotation Speed (rpm)", 500, 6000, 3000, step=100)

st.sidebar.markdown("**Fluid Properties**")
mu0_mPas = st.sidebar.slider("η₀ — Initial Viscosity (mPa·s)", 10, 2000, 300, step=10)
h0_um    = st.sidebar.slider("h₀ — Initial Thickness (μm)", 0.5, 20.0, 5.0, step=0.5)
c0       = st.sidebar.slider("c₀ — Initial Solvent Fraction", 0.30, 0.95, 0.80, step=0.01)
n_exp    = st.sidebar.slider("n — Meyerhofer Exponent", 1.0, 5.0, 2.5, step=0.1)

st.sidebar.markdown("**Process Conditions**")
E_nm_s = st.sidebar.slider("E — Evaporation Rate (nm/s)", 1, 200, 50, step=1)
R_mm   = st.sidebar.slider("R — Wafer Radius (mm)", 25, 150, 75, step=5)

st.sidebar.markdown("---")
st.sidebar.caption("Ref: Emslie, Bonner & Peck (1958) · Meyerhofer (1978)")
st.sidebar.caption("SKKU Chem. Eng. · Fluid Mechanics 2026")

# ============================================================
# Unit conversion (SI)
# ============================================================

omega  = rpm    * 2 * np.pi / 60   # rad/s
h0     = h0_um  * 1e-6              # m
mu0    = mu0_mPas * 1e-3            # Pa·s
E      = E_nm_s * 1e-9              # m/s
R      = R_mm   * 1e-3              # m
rho    = 1200.0                     # kg/m³  (typical PR solution)
gamma  = 0.03                       # N/m    (surface tension)

# ============================================================
# Physics model functions
# ============================================================

def meyerhofer_viscosity(mu0, c, n):
    """
    Meyerhofer (1978) viscosity model.
    μ(c) = μ₀ · (1 − c)^(−n)
    c : solvent volume fraction (0~1)
    As c→0 (solvent evaporates), μ→∞  →  gelation
    """
    phi = 1.0 - c
    if phi <= 1e-8:
        return 1e12
    return mu0 * phi ** (-n)


def rhs(h, c, omega, rho, mu0, E, n):
    """
    Compute the right-hand side of the EBP + Meyerhofer ODE system.

    dh/dt = −(ρω²/3) · h³/μ(c)  −  E      [film thickness ODE]
    dc/dt = −(E/h)·(1−c) − (c/h)·(dh/dt)cf [solvent concentration ODE]
    """
    mu  = meyerhofer_viscosity(mu0, c, n)
    A   = rho * omega**2 / 3.0

    dhdt    = -(A * h**3) / mu - E            # total dh/dt
    dhdt_cf = -(A * h**3) / mu                # centrifugal term only
    dcdt    = -(E / h) * (1.0 - c) - (c / h) * dhdt_cf

    return dhdt, dcdt


def rk4_step(h, c, dt, omega, rho, mu0, E, n):
    """
    Single RK4 step.
    Advances state vector y = [h, c] by one time step dt.
    """
    k1h, k1c = rhs(h,              c,              omega, rho, mu0, E, n)
    k2h, k2c = rhs(h + dt*k1h/2,  c + dt*k1c/2,  omega, rho, mu0, E, n)
    k3h, k3c = rhs(h + dt*k2h/2,  c + dt*k2c/2,  omega, rho, mu0, E, n)
    k4h, k4c = rhs(h + dt*k3h,    c + dt*k3c,    omega, rho, mu0, E, n)

    hn = h + (dt / 6) * (k1h + 2*k2h + 2*k3h + k4h)
    cn = c + (dt / 6) * (k1c + 2*k2c + 2*k3c + k4c)
    return hn, cn


@st.cache_data(show_spinner="Solving ODE (RK4)…")
def run_simulation(rpm, h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm):
    """
    Main simulation function.
    Accepts parameters and returns time series + radial profile.
    st.cache_data recomputes automatically when sliders change.
    """
    # ── Unit conversion ────────────────────────────────────────
    omega_ = rpm    * 2 * np.pi / 60
    h0_    = h0_um  * 1e-6
    mu0_   = mu0_mPas * 1e-3
    E_     = E_nm_s * 1e-9
    R_     = R_mm   * 1e-3

    MU_GEL = 1000 * mu0_     # gelation threshold viscosity
    C_GEL  = 0.02             # gelation threshold solvent fraction
    T_MAX  = 80.0             # maximum integration time (s)
    DT_MIN = 1e-5
    DT_MAX = 0.05

    # ── Center-film (r=0) time series integration ─────────────
    h, c, t = h0_, c0, 0.0
    ts, hs, cs, mus, regimes = [], [], [], [], []
    t_gel = None

    while t <= T_MAX:
        mu_ = meyerhofer_viscosity(mu0_, c, n_exp)
        A_  = rho * omega_**2 / 3.0
        cf_ = abs(A_ * h**3 / mu_)
        reg = "rotation" if cf_ > E_ else "evaporation"

        ts.append(t); hs.append(h * 1e6)
        cs.append(c); mus.append(mu_ * 1e3); regimes.append(reg)

        if mu_ >= MU_GEL or c <= C_GEL or h <= 1e-10:
            t_gel = t
            break

        # Adaptive time step
        dhdt_mag = abs(-(A_ * h**3) / mu_ - E_)
        dt_adap  = min(DT_MAX, 0.005 * h / dhdt_mag) if dhdt_mag > 0 else DT_MAX
        dt       = max(DT_MIN, min(dt_adap, T_MAX - t))

        h, c = rk4_step(h, c, dt, omega_, rho, mu0_, E_, n_exp)
        h = max(h, 1e-10)
        c = float(np.clip(c, 0.0, 1.0))
        t += dt

    t_arr  = np.array(ts)
    h_arr  = np.array(hs)
    c_arr  = np.array(cs)
    mu_arr = np.array(mus)

    # ── Emslie (1958) analytical solution (validation, E=0, μ=const) ──
    A_ana  = rho * omega_**2 / 3.0
    denom  = np.sqrt(1 + (2 * A_ana * h0_**2 / mu0_) * t_arr)
    h_ana  = h0_ / denom * 1e6

    # ── Radial profile h(r) ────────────────────────────────────
    Nr      = 40
    r_nodes = np.linspace(0, R_, Nr + 1)
    h_r     = np.zeros(Nr + 1)
    t_final = t_gel if t_gel else T_MAX

    for i, r_i in enumerate(r_nodes):
        r_frac   = r_i / R_ if R_ > 0 else 0
        # Edge bead: capillary retardation at contact line model
        edge_fac = 1.0 + 0.8 * np.exp(-((1 - r_frac) / 0.04)**2)
        omega_r  = omega_ * edge_fac

        hr, cr = h0_, c0
        tr     = 0.0
        DT_R   = 0.02

        while tr < t_final:
            mu_r = meyerhofer_viscosity(mu0_, cr, n_exp)
            if mu_r >= MU_GEL or cr <= C_GEL:
                break
            dt_r   = min(DT_R, t_final - tr)
            hr, cr = rk4_step(hr, cr, dt_r, omega_r, rho, mu0_, E_, n_exp)
            hr = max(hr, 1e-10)
            cr = float(np.clip(cr, 0.0, 1.0))
            tr += dt_r

        h_r[i] = hr * 1e6

    # ── Uniformity ────────────────────────────────────────
    h_mean      = h_r.mean()
    h_max       = h_r.max()
    h_min       = h_r[:-2].min()   # exclude edge bead node
    uniformity  = (h_max - h_min) / h_mean * 100 if h_mean > 0 else 0.0

    # ── Dimensionless numbers ──────────────────────────────────
    Re = rho * omega_ * R_**2 / mu0_
    Ca = mu0_ * omega_ * R_ / gamma
    Ek = E_ / (omega_ * h0_) if omega_ > 0 else float("inf")

    return dict(
        t=t_arr, h=h_arr, c=c_arr, mu=mu_arr, regime=regimes,
        h_ana=h_ana,
        r_mm=r_nodes * 1e3, h_r=h_r,
        t_gel=t_gel, h_final=h_arr[-1], uniformity=uniformity,
        Re=Re, Ca=Ca, Ek=Ek,
    )


@st.cache_data(show_spinner="Running design sweep…")
def run_sweep(h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm, steps):
    """
    2D parameter sweep over (omega_rpm, mu0_mPas).
    Computes final thickness and uniformity on a steps×steps grid.
    """
    omegas = np.linspace(500,  6000, steps)
    mus    = np.linspace(50,   2000, steps)
    O, M   = np.meshgrid(omegas, mus)
    H_grid = np.zeros_like(O)
    U_grid = np.zeros_like(O)

    for i in range(steps):
        for j in range(steps):
            res = run_simulation(
                rpm=float(O[i, j]), h0_um=h0_um,
                mu0_mPas=float(M[i, j]), E_nm_s=E_nm_s,
                n_exp=n_exp, c0=c0, R_mm=R_mm,
            )
            H_grid[i, j] = res["h_final"]
            U_grid[i, j] = res["uniformity"]

    return O, M, H_grid, U_grid


# ============================================================
# Run simulation
# ============================================================

res = run_simulation(rpm, h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm)

# ============================================================
# Header summary metrics
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Final Thickness", f"{res['h_final']:.3f} μm")

unif_ok = res["uniformity"] < 4.0
col2.metric(
    "Uniformity",
    f"±{res['uniformity']/2:.2f}%",
    delta="✓ Passes ±2% spec" if unif_ok else "✗ Fails ±2% spec",
    delta_color="normal" if unif_ok else "inverse",
)

col3.metric(
    "Gel Time",
    f"{res['t_gel']:.1f} s" if res["t_gel"] else "— (not reached)",
)

pct_rot = int(np.mean([x == "rotation" for x in res["regime"]]) * 100)
col4.metric("Rotation-dominated", f"{pct_rot}%")

st.markdown("---")

# ============================================================
# Tab layout
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "① Interactive",
    "② Validation  (Analytical Limits)",
    "③ Design Space  (ω–μ₀ Sweep)",
])

# ============================================================
# ① Interactive
# ============================================================

with tab1:

    # ── Dimensionless numbers ──────────────────────────────────
    st.subheader("📐 Dimensionless Numbers")
    d1, d2, d3 = st.columns(3)

    d1.info(
        f"**Re = {res['Re']:.2e}**\n\n"
        f"ρωR²/μ₀ — Inertia / Viscous\n\n"
        f"{'✓ Re·ε² ≪ 1 → thin-film valid' if res['Re']*(h0_um*1e-6/R)**2 < 0.01 else '⚠ Check thin-film assumption'}"
    )
    d2.info(
        f"**Ca = {res['Ca']:.2e}**\n\n"
        f"μ₀ωR/γ — Viscous / Surface tension\n\n"
        f"{'✓ Ca ≫ 1 → surface tension negligible' if res['Ca'] > 10 else '⚠ Surface tension may matter'}"
    )
    d3.info(
        f"**Ek = {res['Ek']:.2e}**\n\n"
        f"E/(ω·h₀) — Evaporation / Centrifugal\n\n"
        f"{'Evap-dominated' if res['Ek'] > 0.1 else 'Rotation → Evap transition'}"
    )

    st.markdown("---")

    # ── Time series charts (h, c, μ) ──────────────────────────
    st.subheader("📈 Time Series: h(t) · c(t) · μ(t)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    # h(t)
    ax = axes[0]
    ax.plot(res["t"], res["h"], color="#06b6d4", lw=2.5, label="h(t)")
    if res["t_gel"]:
        ax.axvline(res["t_gel"], color="#ef4444", ls="--", lw=1.2,
                   label=f"t_gel = {res['t_gel']:.1f} s")
    # Background shading by flow regime
    t_arr, regime = res["t"], res["regime"]
    start_idx = 0
    for k in range(1, len(t_arr)):
        if regime[k] != regime[k-1] or k == len(t_arr) - 1:
            color = "#06b6d4" if regime[start_idx] == "rotation" else "#f59e0b"
            ax.axvspan(t_arr[start_idx], t_arr[k], alpha=0.08, color=color, lw=0)
            start_idx = k
    ax.set_xlabel("time (s)"); ax.set_ylabel("h (μm)")
    ax.set_title("Film Thickness h(t)"); ax.legend(fontsize=8); ax.grid(True)

    # c(t)
    ax = axes[1]
    ax.plot(res["t"], res["c"], color="#f59e0b", lw=2)
    ax.axhline(0.02, color="#ef4444", ls=":", lw=1, label="c_gel = 0.02")
    if res["t_gel"]:
        ax.axvline(res["t_gel"], color="#ef4444", ls="--", lw=1.2)
    ax.set_xlabel("time (s)"); ax.set_ylabel("c (solvent fraction)")
    ax.set_title("Solvent Fraction c(t)"); ax.legend(fontsize=8); ax.grid(True)

    # μ(t) — log scale
    ax = axes[2]
    ax.semilogy(res["t"], res["mu"], color="#a78bfa", lw=2)
    ax.axhline(mu0_mPas * 1000, color="#ef4444", ls=":", lw=1, label="μ_gel")
    if res["t_gel"]:
        ax.axvline(res["t_gel"], color="#ef4444", ls="--", lw=1.2, label="t_gel")
    ax.set_xlabel("time (s)"); ax.set_ylabel("μ (mPa·s)  [log]")
    ax.set_title("Viscosity μ(t)  [Meyerhofer]"); ax.legend(fontsize=8)
    ax.grid(True, which="both")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ── Radial thickness profile ───────────────────────────────
    st.subheader("🔄 Radial Thickness Profile h(r) — Final Film + Edge Bead")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(res["r_mm"], res["h_r"], color="#06b6d4", lw=2.5)
    ax.axhline(res["h_final"], color="#3b82f6", ls="--", lw=1.2,
               label=f"center = {res['h_final']:.3f} μm")
    h_mean = res["h_r"].mean()
    ax.axhspan(h_mean * 0.98, h_mean * 1.02, alpha=0.12, color="#10b981",
               label="±2% band")
    ax.set_xlabel("Radial position r (mm)")
    ax.set_ylabel("Final thickness h (μm)")
    ax.set_title("Radial Thickness Profile — Edge Bead at r = R")
    ax.legend(fontsize=9); ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        "Edge bead: capillary retardation at the contact line (r = R). "
        "Green band = ±2% uniformity window."
    )


# ============================================================
# ② Validation
# ============================================================

with tab2:

    st.markdown("""
    Validates the RK4 numerical solution against three analytical limit cases.
    """)

    # ── Test 1: E→0, μ=const → Emslie (1958) ─────────────
    with st.expander("✅ Test 1 — E → 0, μ = const  →  Emslie (1958) exact solution", expanded=True):

        res1 = run_simulation(rpm, h0_um, mu0_mPas,
                              E_nm_s=0.001,   # E → 0
                              n_exp=0.0001,   # n → 0 (const μ)
                              c0=c0, R_mm=R_mm)

        # Emslie analytical solution
        A_   = rho * (rpm * 2 * np.pi / 60)**2 / 3
        den  = np.sqrt(1 + (2 * A_ * (h0_um*1e-6)**2 / (mu0_mPas*1e-3)) * res1["t"])
        h_an = (h0_um * 1e-6) / den * 1e6

        err1    = np.abs(res1["h"] - h_an) / h_an * 100
        max_err = err1.max()

        c_left, c_right = st.columns([3, 1])
        with c_left:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(res1["t"], res1["h"], color="#06b6d4", lw=2.5, label="Simulator (RK4)")
            ax1.plot(res1["t"], h_an,      color="#f59e0b", lw=2, ls="--",
                     label="Emslie (1958) analytical")
            ax1.set_ylabel("h (μm)"); ax1.legend(fontsize=9); ax1.grid(True)
            ax1.set_title("h(t): RK4 vs Emslie (1958)  [E→0, μ=const]")

            ax2.plot(res1["t"], err1, color="#ef4444", lw=1.5)
            ax2.set_ylabel("Error (%)"); ax2.set_xlabel("time (s)"); ax2.grid(True)
            ax2.set_title("Residual: |numerical − analytical| / analytical × 100%")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            st.metric("Max Error", f"{max_err:.4f}%")
            st.metric("Result", "✓ PASS" if max_err < 0.5 else "✗ FAIL")
            st.markdown(r"""
**ODE (E=0):**
$$\frac{dh}{dt} = -\frac{\rho\omega^2}{3\mu_0}h^3$$

**Exact solution:**
$$h(t) = \frac{h_0}{\sqrt{1 + \frac{2\rho\omega^2 h_0^2}{3\mu_0}t}}$$
""")

    # ── Test 2: ω→0 → linear evaporation ─────────────────
    with st.expander("✅ Test 2 — ω → 0  →  Linear evaporation: h(t) = h₀ − E·t"):

        res2  = run_simulation(10, h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm)
        E_um_ = E_nm_s * 1e-3   # nm/s → μm/s
        h_lin = np.maximum(0, h0_um - E_um_ * res2["t"])
        err2  = np.where(h_lin > 0, np.abs(res2["h"] - h_lin) / h_lin * 100, 0)
        max_err2 = err2.max()

        c_left, c_right = st.columns([3, 1])
        with c_left:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]})
            ax1.plot(res2["t"], res2["h"], color="#06b6d4", lw=2.5, label="Simulator (ω=10 rpm)")
            ax1.plot(res2["t"], h_lin,     color="#f59e0b", lw=2, ls="--",
                     label="h₀ − E·t  (analytical)")
            ax1.set_ylabel("h (μm)"); ax1.legend(fontsize=9); ax1.grid(True)
            ax1.set_title("h(t): ω→0 limit  (pure evaporation)")

            ax2.plot(res2["t"], err2, color="#ef4444", lw=1.5)
            ax2.set_ylabel("Error (%)"); ax2.set_xlabel("time (s)"); ax2.grid(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            st.metric("Max Error", f"{max_err2:.4f}%")
            st.metric("Result", "✓ PASS" if max_err2 < 3.0 else "✗ FAIL")
            st.markdown(r"""
**When ω → 0:**
$$\frac{dh}{dt} = -E$$

**Exact solution:**
$$h(t) = h_0 - E \cdot t$$
""")

    # ── Test 3: n→large (μ→∞) → evap-only ───────────────
    with st.expander("✅ Test 3 — n → large  (μ → ∞, early gelation)  →  dh/dt ≈ −E"):

        res3  = run_simulation(rpm, h0_um, mu0_mPas, E_nm_s,
                               n_exp=8.0, c0=0.5, R_mm=R_mm)
        E_um3 = E_nm_s * 1e-3
        h_ref = np.maximum(0, h0_um - E_um3 * res3["t"])

        c_left, c_right = st.columns([3, 1])
        with c_left:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(res3["t"], res3["h"], color="#06b6d4", lw=2.5, label="Simulator (n=8)")
            ax.plot(res3["t"], h_ref,     color="#f59e0b", lw=2, ls="--",
                    label="Evap-only reference")
            ax.set_xlabel("time (s)"); ax.set_ylabel("h (μm)")
            ax.set_title("High-n limit: centrifugal term suppressed by large μ")
            ax.legend(fontsize=9); ax.grid(True)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with c_right:
            last_ref = h_ref[-1] if len(h_ref) > 0 and h_ref[-1] > 0 else 1
            err3 = abs(res3["h"][-1] - last_ref) / last_ref * 100
            st.metric("Final-point Error", f"{err3:.2f}%")
            st.metric("Result", "✓ PASS" if err3 < 15 else "✗ FAIL")
            st.markdown(r"""
**When μ → ∞ (large n):**
$$-\frac{\rho\omega^2 h^3}{3\mu} \to 0$$

Film thins by evaporation only.
""")

    # ── Validation summary table ──────────────────────────────
    st.markdown("---")
    st.subheader("📋 Validation Summary")
    df_val = pd.DataFrame([
        {"Test": "T1", "Condition": "E→0, μ=const",
         "Reference": "Emslie (1958)",
         "Max Error": f"{max_err:.4f}%",
         "Result": "✓ PASS" if max_err < 0.5 else "✗ FAIL"},
        {"Test": "T2", "Condition": "ω→0",
         "Reference": "h(t) = h₀ − E·t",
         "Max Error": f"{max_err2:.4f}%",
         "Result": "✓ PASS" if max_err2 < 3.0 else "✗ FAIL"},
        {"Test": "T3", "Condition": "n=8 (μ→∞)",
         "Reference": "dh/dt ≈ −E",
         "Max Error": f"{err3:.2f}%",
         "Result": "✓ PASS" if err3 < 15 else "✗ FAIL"},
    ])
    st.dataframe(df_val, use_container_width=True, hide_index=True)


# ============================================================
# ③ Design Space
# ============================================================

with tab3:

    st.subheader("⚙️ Sweep Settings")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        steps     = st.select_slider("Grid resolution (N×N)", [6, 8, 10, 12], value=8)
        target_h  = st.number_input("Target thickness (μm)", 0.1, 20.0, 1.0, step=0.1)
        tol_h     = st.number_input("Tolerance ± (μm)",     0.01, 1.0,  0.1, step=0.01)
        run_btn   = st.button("▶ Run Sweep", type="primary", use_container_width=True)

    with col_s2:
        st.info(
            f"**{steps}×{steps} = {steps**2} simulations**\n\n"
            f"- ω ∈ [500, 6000] rpm\n"
            f"- μ₀ ∈ [50, 2000] mPa·s\n\n"
            f"Uses the current E, n, h₀, c₀ values from the sidebar.\n\n"
            f"Searches for conditions that achieve {target_h:.2f} ± {tol_h:.2f} μm."
        )

    if not run_btn:
        st.caption("👆 Press **Run Sweep** to generate the design map.")
        st.stop()

    # ── Run sweep ──────────────────────────────────────────────
    O, M, H_grid, U_grid = run_sweep(
        h0_um, mu0_mPas, E_nm_s, n_exp, c0, R_mm, steps
    )

    st.markdown("---")
    st.subheader("🗺️ Design Maps")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # h_final heatmap
    ax = axes[0]
    im = ax.pcolormesh(O, M, H_grid, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="h_final (μm)")
    # Target thickness contour lines
    try:
        cs = ax.contour(O, M, H_grid,
                        levels=[target_h - tol_h, target_h, target_h + tol_h],
                        colors=["#f59e0b", "white", "#f59e0b"],
                        linewidths=[1.2, 2.0, 1.2],
                        linestyles=["--", "-", "--"])
        ax.clabel(cs, fmt="%.2f μm", fontsize=8)
    except Exception:
        pass
    # Hatch region that fails the ±2% spec
    ax.contourf(O, M, U_grid, levels=[4.0, U_grid.max() + 1],
                colors=["#ef4444"], alpha=0.15)
    ax.set_xlabel("ω (rpm)"); ax.set_ylabel("μ₀ (mPa·s)")
    ax.set_title("Final Thickness h_final(ω, μ₀)\n[white contour = target, red = fails ±2% spec]")

    # Uniformity heatmap
    ax = axes[1]
    im2 = ax.pcolormesh(O, M, U_grid / 2, cmap="RdYlGn_r",
                         shading="auto", vmin=0, vmax=5)
    plt.colorbar(im2, ax=ax, label="Uniformity ±%")
    try:
        ax.contour(O, M, U_grid / 2, levels=[2.0],
                   colors=["white"], linewidths=[2.0])
    except Exception:
        pass
    ax.set_xlabel("ω (rpm)"); ax.set_ylabel("μ₀ (mPa·s)")
    ax.set_title("Uniformity ±%\n[white line = ±2% spec boundary]")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Target thickness search ────────────────────────────────
    st.markdown("---")
    st.subheader(f"🎯 Feasible Conditions  ({target_h:.2f} ± {tol_h:.2f} μm)")

    mask = (np.abs(H_grid - target_h) <= tol_h) & (U_grid < 4.0)
    hits = []
    for i in range(steps):
        for j in range(steps):
            if mask[i, j]:
                hits.append({
                    "ω (rpm)"      : int(O[i, j]),
                    "μ₀ (mPa·s)"  : int(M[i, j]),
                    "h_final (μm)": round(float(H_grid[i, j]), 3),
                    "Uniformity ±%": round(float(U_grid[i, j] / 2), 2),
                })

    if hits:
        st.dataframe(pd.DataFrame(hits), use_container_width=True, hide_index=True)
        st.success(f"✓ {len(hits)} condition(s) satisfy both the target thickness and the ±2% uniformity spec.")
    else:
        st.warning("No feasible combinations found — try widening the target thickness or tolerance.")

    # ── EBP scaling law ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📐 EBP Scaling Law:  h_final ∝ ω^(−2/3)")

    mid      = steps // 2
    o_1d     = O[mid, :]
    h_1d     = H_grid[mid, :]
    mu_label = M[mid, 0]

    coeffs = np.polyfit(np.log(o_1d), np.log(h_1d), 1)
    h_fit  = np.exp(np.polyval(coeffs, np.log(o_1d)))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.loglog(o_1d, h_1d, "o-", color="#06b6d4", lw=2, ms=6, label="Simulator")
    ax.loglog(o_1d, h_fit, "--", color="#f59e0b", lw=2,
              label=f"Fit: h ∝ ω^({coeffs[0]:.2f})  (theory: −0.667)")
    ax.set_xlabel("ω (rpm)"); ax.set_ylabel("h_final (μm)")
    ax.set_title(f"h_final vs ω  at μ₀ = {mu_label:.0f} mPa·s")
    ax.legend(fontsize=9); ax.grid(True, which="both")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.caption(
        f"Fitted exponent = **{coeffs[0]:.3f}**  (EBP theory: **−0.667**). "
        "Deviation from −2/3 is due to Meyerhofer evaporation-viscosity coupling."
    )

# ============================================================
# End
# ============================================================

st.success("✅ Simulation Complete")
