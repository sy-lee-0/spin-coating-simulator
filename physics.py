"""
physics.py — Spin Coating Physics Engine
=========================================
Emslie–Bonner–Peck (1958) thin-film theory
+ Meyerhofer (1978) solvent-evaporation viscosity model

State vector:  y = [h, c]
  h  :  film thickness          (m)
  c  :  solvent volume fraction (dimensionless, 0-1)

All inputs/outputs in SI unless a helper converts units.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpinParams:
    """Physical parameters for one spin-coating run (SI units)."""
    omega_rpm : float = 3000.0   # angular velocity (rpm)
    h0_um     : float = 5.0      # initial film thickness (μm)
    mu0_mPas  : float = 300.0    # initial dynamic viscosity (mPa·s)
    E_nm_s    : float = 50.0     # solvent evaporation rate (nm/s)
    n_exp     : float = 2.5      # Meyerhofer viscosity exponent
    rho       : float = 1200.0   # fluid density (kg/m³)
    c0        : float = 0.80     # initial solvent volume fraction
    t_max_s   : float = 60.0     # max integration time (s)
    R_mm      : float = 75.0     # wafer radius (mm)
    Nr        : int   = 40       # radial nodes for h(r) profile

    # Derived SI values (set in __post_init__)
    omega   : float = field(init=False)   # rad/s
    h0      : float = field(init=False)   # m
    mu0     : float = field(init=False)   # Pa·s
    E       : float = field(init=False)   # m/s
    R       : float = field(init=False)   # m

    def __post_init__(self):
        self.omega = self.omega_rpm * 2 * np.pi / 60
        self.h0    = self.h0_um  * 1e-6
        self.mu0   = self.mu0_mPas * 1e-3
        self.E     = self.E_nm_s   * 1e-9
        self.R     = self.R_mm     * 1e-3


@dataclass
class SimResult:
    """Output of one runSimulation() call."""
    t       : np.ndarray   # time array (s)
    h_um    : np.ndarray   # center film thickness (μm)
    c       : np.ndarray   # solvent fraction
    mu_mPas : np.ndarray   # viscosity (mPa·s)
    regime  : list         # 'rotation' or 'evaporation' per step

    r_mm    : np.ndarray   # radial positions (mm)
    h_r_um  : np.ndarray   # final radial thickness profile (μm)

    h_ana_um: np.ndarray   # Emslie (1958) analytical h(t) for validation

    t_gel         : Optional[float]  # gelation time (s), None if not reached
    h_final_um    : float            # final center thickness (μm)
    uniformity_pct: float            # (h_max−h_min)/h_mean × 100 %

    Re : float   # Reynolds number
    Ca : float   # Capillary number
    Ek : float   # Evaporation number


# ──────────────────────────────────────────────────────────────────────────────
# Core physics functions
# ──────────────────────────────────────────────────────────────────────────────

def meyerhofer_viscosity(mu0: float, c: float, n: float) -> float:
    """
    Meyerhofer (1978) viscosity model.
    μ(c) = μ₀ · (1 − c)^(−n)

    As c → 0 (all solvent evaporated), μ → ∞  (gelation).
    """
    phi = 1.0 - c          # polymer volume fraction
    if phi <= 1e-8:
        return 1e12        # fully gelled — return large sentinel
    return mu0 * phi ** (-n)


def _rhs(h: float, c: float, p: SpinParams) -> tuple[float, float]:
    """
    Right-hand side of the EBP + Meyerhofer ODE system.

    dh/dt = −(ρω²/3) · h³/μ(c)  −  E          [centrifugal + evaporation]
    dc/dt = −(E/h)·(1−c) − (c/h)·(dh/dt)|cf    [solvent mass balance]

    Returns (dh/dt, dc/dt).
    """
    mu  = meyerhofer_viscosity(p.mu0, c, p.n_exp)
    A   = p.rho * p.omega ** 2 / 3.0             # centrifugal prefactor

    # Film-thickness ODE  (Emslie 1958 + evaporation term)
    dhdt = -(A * h**3) / mu - p.E

    # Solvent-concentration ODE  (Meyerhofer 1978)
    dhdt_cf = -(A * h**3) / mu                   # centrifugal part only
    dcdt    = -(p.E / h) * (1.0 - c) - (c / h) * dhdt_cf

    return dhdt, dcdt


def _rk4_step(h: float, c: float, dt: float, p: SpinParams) -> tuple[float, float]:
    """
    Single RK4 step for the [h, c] state vector.
    """
    k1h, k1c = _rhs(h,              c,              p)
    k2h, k2c = _rhs(h + dt*k1h/2,  c + dt*k1c/2,  p)
    k3h, k3c = _rhs(h + dt*k2h/2,  c + dt*k2c/2,  p)
    k4h, k4c = _rhs(h + dt*k3h,    c + dt*k3c,    p)

    hn = h + (dt / 6) * (k1h + 2*k2h + 2*k3h + k4h)
    cn = c + (dt / 6) * (k1c + 2*k2c + 2*k3c + k4c)
    return hn, cn


# ──────────────────────────────────────────────────────────────────────────────
# Main integrator
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(p: SpinParams) -> SimResult:
    """
    Integrate the EBP+Meyerhofer ODE from t=0 to t_max or gelation.
    Returns a SimResult with full time series, radial profile, and validation data.
    """
    MU_GEL  = 1000 * p.mu0   # gelation threshold: μ > 1000 μ₀
    C_GEL   = 0.02            # gelation threshold: c < 2 % solvent
    DT_MIN  = 1e-5            # minimum allowed time step (s)
    DT_MAX  = 0.05            # maximum allowed time step (s)

    # ── Center-film integration ────────────────────────────────────────────
    h, c, t = p.h0, p.c0, 0.0
    ts, hs, cs, mus, regimes = [], [], [], [], []
    t_gel = None

    while t <= p.t_max_s:
        mu  = meyerhofer_viscosity(p.mu0, c, p.n_exp)
        A   = p.rho * p.omega**2 / 3.0
        cf  = abs(A * h**3 / mu)          # |centrifugal thinning rate|
        ev  = p.E                          # evaporation thinning rate
        reg = 'rotation' if cf > ev else 'evaporation'

        ts.append(t); hs.append(h * 1e6); cs.append(c)
        mus.append(mu * 1e3); regimes.append(reg)

        # Gelation check
        if mu >= MU_GEL or c <= C_GEL or h <= 1e-10:
            t_gel = t
            break

        # Adaptive time step
        dhdt_mag = abs(-(A * h**3) / mu - p.E)
        if dhdt_mag > 0:
            dt_adap = min(DT_MAX, 0.005 * h / dhdt_mag)
        else:
            dt_adap = DT_MAX
        dt = max(DT_MIN, min(dt_adap, p.t_max_s - t))

        h, c = _rk4_step(h, c, dt, p)
        h = max(h, 1e-10)
        c = np.clip(c, 0.0, 1.0)
        t += dt

    t_arr  = np.array(ts)
    h_arr  = np.array(hs)
    c_arr  = np.array(cs)
    mu_arr = np.array(mus)

    h_final = h_arr[-1]

    # ── Emslie (1958) analytical solution for validation (E=0, μ=const) ───
    A_ana = p.rho * p.omega**2 / 3.0
    denom = np.sqrt(1 + (2 * A_ana * p.h0**2 / p.mu0) * t_arr)
    h_ana_um = (p.h0 / denom) * 1e6

    # ── Radial profile h(r) at final time ─────────────────────────────────
    t_final = t_gel if t_gel else p.t_max_s
    r_nodes = np.linspace(0, p.R, p.Nr + 1)
    h_r = np.zeros(p.Nr + 1)

    for i, r in enumerate(r_nodes):
        r_frac    = r / p.R
        # Edge-bead capillary retardation model (contact-line pinning)
        edge_fac  = 1.0 + 0.8 * np.exp(-((1 - r_frac) / 0.04)**2)
        omega_eff = p.omega * edge_fac

        # Build a local SpinParams with modified omega for this radial node
        p_r           = SpinParams.__new__(SpinParams)
        p_r.__dict__  = dict(p.__dict__)     # shallow copy
        p_r.omega     = omega_eff
        p_r.t_max_s   = t_final

        hr, cr = p.h0, p.c0
        tr     = 0.0
        DT_R   = 0.02

        while tr < t_final:
            mu_r = meyerhofer_viscosity(p.mu0, cr, p.n_exp)
            if mu_r >= MU_GEL or cr <= C_GEL:
                break
            dt_step = min(DT_R, t_final - tr)
            hr, cr  = _rk4_step(hr, cr, dt_step, p_r)
            hr = max(hr, 1e-10)
            cr = np.clip(cr, 0.0, 1.0)
            tr += dt_step

        h_r[i] = hr * 1e6

    # ── Uniformity metric ──────────────────────────────────────────────────
    h_mean = h_r.mean()
    h_max  = h_r.max()
    h_min  = h_r[:-2].min()   # exclude edge-bead node
    uniformity = (h_max - h_min) / h_mean * 100 if h_mean > 0 else 0.0

    # ── Dimensionless numbers ──────────────────────────────────────────────
    gamma = 0.03                                       # surface tension ~30 mN/m
    Re = p.rho * p.omega * p.R**2 / p.mu0
    Ca = p.mu0 * p.omega * p.R / gamma
    Ek = p.E / (p.omega * p.h0) if p.omega > 0 else np.inf

    return SimResult(
        t=t_arr, h_um=h_arr, c=c_arr, mu_mPas=mu_arr, regime=regimes,
        r_mm=r_nodes * 1e3, h_r_um=h_r,
        h_ana_um=h_ana_um,
        t_gel=t_gel, h_final_um=h_final, uniformity_pct=uniformity,
        Re=Re, Ca=Ca, Ek=Ek,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Design-space sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep_design_space(
    base: SpinParams,
    omega_range: tuple[float, float] = (500, 6000),
    mu_range:    tuple[float, float] = (50, 2000),
    steps: int = 10,
) -> dict:
    """
    Sweep (omega_rpm, mu0_mPas) on a steps×steps grid.
    Returns dict of 2-D arrays: omega_grid, mu_grid, h_grid, unif_grid, pass_grid, tgel_grid.
    """
    omegas = np.linspace(omega_range[0], omega_range[1], steps)
    mus    = np.linspace(mu_range[0],    mu_range[1],    steps)

    O, M = np.meshgrid(omegas, mus)          # shape (steps, steps)
    H    = np.zeros_like(O)
    U    = np.zeros_like(O)
    T    = np.full_like(O, np.nan)

    for i in range(steps):
        for j in range(steps):
            p = SpinParams(
                omega_rpm = float(O[i, j]),
                mu0_mPas  = float(M[i, j]),
                h0_um     = base.h0_um,
                E_nm_s    = base.E_nm_s,
                n_exp     = base.n_exp,
                rho       = base.rho,
                c0        = base.c0,
                t_max_s   = base.t_max_s,
                R_mm      = base.R_mm,
                Nr        = 10,             # coarser for speed
            )
            r = run_simulation(p)
            H[i, j] = r.h_final_um
            U[i, j] = r.uniformity_pct
            T[i, j] = r.t_gel if r.t_gel else np.nan

    return {
        'omega': O, 'mu': M,
        'h':     H, 'unif': U, 'tgel': T,
        'pass':  (U < 4.0),     # ±2 % spec → full range < 4 %
    }
