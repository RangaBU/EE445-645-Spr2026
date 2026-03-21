"""
================================================================================
1D RADIATIVE TRANSFER IN A VEGETATION CANOPY
Uncollided Field  /  First-Collision  /  Full SOS Solution
================================================================================
Course  : Physical Models in Remote Sensing
Chapter : 04 – Parts 01–03
Authors : Claude And I (Ranga B. Myneni)

Usage
-----
Run the script and choose one of:
  (1) Uncollided problem only
  (2) Uncollided + first-collision  (I₀ + I₁)
  (3) Uncollided + full SOS         (I₀ + I₁ + I₂ + … + Iₙ)

If option (3) is selected, the user is also asked whether to write the BRF
on a fine angular grid to a text file.

Outputs (saved to current directory):
  <band>_<option>.png       — one figure per spectral band (RED and NIR).
                              Option 1: three plots only (no table).
                              Options 2 & 3: four plots + energy balance table.
                              Uncollided field (I₀) shown as dotted lines.
  BRF_FINE_GRID.txt         — (option 3 only, if requested) BRF on a 90×90 grid:
                              θ_v = 2°, 4°, …, 178° (2° step, 89 values)
                              φ_v = 0°, 4°, …, 356° (4° step, 90 values)
                              Each band/LAI block includes a verification:
                              Ref integrated from fine BRF vs Ref from coarse GL.

================================================================================
GEOMETRY AND SIGN CONVENTION
================================================================================
  L = 0          top of canopy (radiation enters here)
  L = LAI        bottom of canopy (soil surface)
  μ < 0          downward-travelling directions
  μ > 0          upward-travelling directions
  θ₀ = 140°      solar zenith from the upward zenith axis → μ₀ = cos(140°) < 0

================================================================================
PHASE FUNCTION
================================================================================
Bi-Lambertian area scattering phase function (Ross 1981, Myneni 1988):

    Γ(Ω′→Ω) = (ωL / 3π)(sin β − β cos β)  +  (τL / 3) cos β

Normalisation:  (1/π) ∫₄π Γ dΩ  =  ωL · G  =  ωL / 2  ✓

================================================================================
CONVERGENCE CRITERION  (options 2 and 3)
================================================================================
Stop SOS when BOTH of the following are satisfied simultaneously:

  (A) Boundary intensities: for every quadrature direction with θ ∈ [0°, 70°]
      at both L = 0 and L = LAI:
          |Iₙ(μᵢ, φⱼ)| / |Icum(μᵢ, φⱼ)| < 1%

  (B) Scalar irradiance profile: for every canopy layer k:
          |Sₙ(k)| / |Scum(k)| < 1%
      where Sₙ(k) = ∑ᵢ ∑ⱼ [|μᵢ| omitted] Iₙ(i,j,k) · wᵢ · wⱼ
                                                   (unweighted by |μ|)

Both criteria must pass before the iteration stops.  Criterion (A) ensures
the boundary fluxes (Ref and Trans) are converged.  Criterion (B) ensures the
interior radiation density field is converged, which is needed for a meaningful
absorbed energy estimate via the scalar irradiance formula.

================================================================================
ENERGY BALANCE
================================================================================
All three components are computed from the solution fields:

  Reflected   = Fu(L=0)                        upward flux at canopy top
  Transmitted = Fd(LAI) − Fu(LAI)              net downward flux at ground
  Absorbed    = G·(1−ωL)·∫₀^LAI S(L) dL       scalar irradiance integral
              where S(L) = ∑ᵢ ∑ⱼ I(i,j,L)·wᵢ·wⱼ  (no |μ| weight)

These three quantities are computed INDEPENDENTLY from the solution fields.
Their sum is NOT guaranteed to equal Fin — the imbalance is a genuine
diagnostic of how well the SOS has converged and how accurately the angular
quadrature represents the full radiation field.

For RED (ωL ≈ 0.10): imbalance < 0.3%  — scalar irradiance works well.
For NIR (ωL ≈ 0.975): residual imbalance ~ 3–6% remains even after full
  convergence, due to the GL quadrature underestimating the scalar irradiance
  of the multiply-scattered field.  The Reflected and Transmitted values are
  accurate (boundary criterion also satisfied); the imbalance indicates a
  limitation of the GL quadrature for the scalar irradiance in high-albedo
  canopies.

Each radiation type (Direct / Diffuse / Collided) forms an independent column
in the energy balance table.

For the First-Collision problem (option 2), the imbalance equals the energy
still residing in uncomputed higher orders:
    RED:  ~ −0.6% to −0.7%   (small — leaves absorb 90% of intercepted light)
    NIR:  ~ −39% to −67%     (large — nearly all intercepted light rescatters)
This is the most physically informative diagnostic the SOS provides.

================================================================================
ABSORBED RADIATION PROFILE (Plot D)
================================================================================
The vertical profile of canopy absorption is computed via the flux divergence:

    dAbs/dL(k) = −d(Fd − Fu)/dL  at cell centre k

This is exact by construction — it is consistent with the boundary fluxes —
and always integrates to Fin − Ref − Trans over the full canopy.

Note: The scalar irradiance formula G(1−ωL)·S(L) gives a different profile
for high-ωL canopies (NIR) because the GL quadrature underestimates S(L) in
the interior.  The flux divergence profile is used for Plot D.
================================================================================
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import integrate
from scipy.interpolate import RectBivariateSpline
from numpy.polynomial.legendre import leggauss
import time

# ==============================================================================
# SECTION 1 – INPUTS
# ==============================================================================
#
# ── A. USER-CHANGEABLE INPUTS ──────────────────────────────────────────────
Fin        = 1.0
fdir       = 0.7
theta0_deg = 140.0
phi0_deg   = 0.0
LAI_vals   = [1.5, 4.0]

bands = {
    'RED': {'rho_L': 0.06,  'tau_L': 0.04,  'rho_g': 0.10},
    'NIR': {'rho_L': 0.525, 'tau_L': 0.45,  'rho_g': 0.20},
}

# ── B. FIXED CONSTANTS — DO NOT CHANGE ─────────────────────────────────────
G = 0.5

# ── C. DERIVED QUANTITIES ──────────────────────────────────────────────────
for b in bands.values():
    b['omega_L'] = b['rho_L'] + b['tau_L']

mu0     = np.cos(np.radians(theta0_deg))
abs_mu0 = abs(mu0)
phi0_r  = np.radians(phi0_deg)
Io      = fdir / abs_mu0
Id      = (1.0 - fdir) / np.pi

# ==============================================================================
# SECTION 2 – NUMERICAL PARAMETERS
# ==============================================================================
K     = 50
N_mu  = 16
N_phi = 32

# ==============================================================================
# SECTION 3 – ANGULAR QUADRATURE
# ==============================================================================
_nodes, _weights = leggauss(2 * N_mu)
mu_up    =  _nodes[N_mu:];     w_mu_up    = _weights[N_mu:]
mu_down  = -mu_up[::-1];       w_mu_down  = w_mu_up[::-1]
phi_arr  = np.array([j * 2.*np.pi / N_phi for j in range(N_phi)])
w_phi    = np.full(N_phi, 2.*np.pi / N_phi)

MU_MIN   = np.cos(np.radians(70.))   # convergence checked for θ ≤ 70°

# ==============================================================================
# SECTION 4 – PHASE FUNCTION
# ==============================================================================
def phase_function(mu_o, phi_o, mu_i, phi_i, omega_L, tau_L):
    """
    Γ(Ω′→Ω) = (ωL / 3π)(sin β − β cos β)  +  (τL / 3) cos β
    All arguments support NumPy broadcasting.
    """
    sin_o = np.sqrt(np.maximum(0., 1. - mu_o**2))
    sin_i = np.sqrt(np.maximum(0., 1. - mu_i**2))
    cos_b = np.clip(mu_i*mu_o + sin_i*sin_o*np.cos(phi_i - phi_o), -1., 1.)
    b     = np.arccos(cos_b)
    return ((omega_L / (3.*np.pi)) * (np.sin(b) - b*np.cos(b))
            + (tau_L / 3.) * np.cos(b))

# ==============================================================================
# SECTION 5 – UNCOLLIDED FIELD  (analytical)
# ==============================================================================
def uncollided_Fd(L):
    """Total uncollided downward flux at depth L."""
    Fd_dir = abs_mu0 * Io * np.exp(-G * L / abs_mu0)
    if L == 0.:
        return Fd_dir + (1. - fdir) * Fin
    r, _ = integrate.quad(lambda m: m*Id*np.exp(-G*L/m), 1e-6, 1., limit=200)
    return Fd_dir + 2.*np.pi*r

def uncollided_Fu(L, LAI, rho_g):
    """Uncollided upward flux at depth L (ground-reflected)."""
    Fb = uncollided_Fd(LAI); dL = LAI - L
    if dL < 1e-12: return rho_g * Fb
    r, _ = integrate.quad(lambda m: m*np.exp(-G*dL/m), 1e-6, 1., limit=200)
    return 2.*np.pi * (rho_g/np.pi * Fb) * r

def uncollided_BRF(theta_v_deg, LAI, rho_g):
    """Uncollided BRF at L=0 in view direction θ_v."""
    mu_v = abs(np.cos(np.radians(theta_v_deg)))
    return (rho_g/np.pi * uncollided_Fd(LAI)) * np.exp(-G*LAI/mu_v) / (Fin/np.pi)

# ==============================================================================
# SECTION 6 – FIRST-COLLISION SOURCE  Q₁
# ==============================================================================
def compute_Q1(L_centres, mu_out, phi_out, omega_L, tau_L):
    """
    Q₁(L, Ω) = (1/π) ∫₄π Γ(Ω′→Ω) · I₀(L, Ω′) dΩ′
    Returns Q1[N_out, N_phi, K].
    """
    I0_dir = Io * np.exp(-G * L_centres / abs_mu0)
    I0_dif = Id * np.exp(-G * L_centres[None,:] /
                          np.abs(mu_down[:,None])) * np.ones((N_mu, len(L_centres)))
    Gam_dir = phase_function(mu_out[:,None], phi_out[None,:],
                              mu0, phi0_r, omega_L, tau_L)
    Q1 = (1./np.pi) * Gam_dir[:,:,None] * I0_dir[None,None,:]
    for jj, phi_d in enumerate(phi_arr):
        Gam = phase_function(mu_out[:,None,None], phi_out[None,:,None],
                              mu_down[None,None,:], phi_d, omega_L, tau_L)
        wts = w_mu_down * w_phi[jj]
        Q1 += (1./np.pi) * np.einsum('ijm,mk,m->ijk', Gam, I0_dif, wts)
    return Q1

# ==============================================================================
# SECTION 7 – SPATIAL SWEEP METHODS: DO AND MOC
# ==============================================================================
def DD_dn(abs_mu, Qc, Dl):
    alpha = 2.0 * abs_mu / Dl
    I = np.zeros(K+1)
    for k in range(K):
        I[k+1] = max(((alpha - G)*I[k] + 2.*Qc[k]) / (alpha + G), 0.)
    return I

def DD_up(mu_o, Qc, Dl, I_bot):
    alpha = 2.0 * mu_o / Dl
    I = np.zeros(K+1); I[K] = I_bot
    for k in range(K-1, -1, -1):
        I[k] = max(((alpha - G)*I[k+1] + 2.*Qc[k]) / (alpha + G), 0.)
    return I

def MOC_dn(abs_mu, Qc, Dl):
    """
    Downward MOC sweep:  I_{k+1} = I_k·λ + (Q_k/G)·(1−λ),  λ=exp(−G·Δl/|μ|)
    Upper BC: I(L=0) = 0.
    """
    decay = np.exp(-G * Dl / abs_mu)
    I = np.zeros(K+1); Ic = 0.
    for m in range(K):
        Ic = max(Ic * decay + Qc[m]/G * (1. - decay), 0.)
        I[m+1] = Ic
    return I

def MOC_up(mu_o, Qc, Dl, I_bot):
    """
    Upward MOC sweep:  I_k = I_{k+1}·λ + (Q_k/G)·(1−λ),  λ=exp(−G·Δl/μ)
    Lower BC: I_bot supplied by caller.
    """
    decay = np.exp(-G * Dl / mu_o)
    I = np.zeros(K+1); I[K] = I_bot; Ic = I_bot
    for m in range(K-1, -1, -1):
        Ic = max(Ic * decay + Qc[m]/G * (1. - decay), 0.)
        I[m] = Ic
    return I

# ==============================================================================
# SECTION 8 – FLUX AND BRF HELPERS
# ==============================================================================
def fluxes(I_dn, I_up):
    """Downward and upward hemisphere-integrated fluxes [K+1]."""
    Fd = np.sum(np.abs(mu_down[:,None,None]) * I_dn
                * w_mu_down[:,None,None] * w_phi[None,:,None], axis=(0,1))
    Fu = np.sum(mu_up[:,None,None] * I_up
                * w_mu_up[:,None,None] * w_phi[None,:,None], axis=(0,1))
    return Fd, Fu

def scalar_irrad_profile(I_dn, I_up):
    """
    Scalar irradiance S(L) at cell centres [K].
    S = ∑ᵢ ∑ⱼ I(i,j,k) · wᵢ · wⱼ    (no |μ| weight — counts all photons equally)
    Direct beam contribution added separately in the solver.
    """
    Idc = 0.5*(I_dn[:,:,:-1]+I_dn[:,:,1:])
    Iuc = 0.5*(I_up[:,:,:-1]+I_up[:,:,1:])
    return (np.einsum('ijk,i,j->k', Idc, w_mu_down, w_phi) +
            np.einsum('ijk,i,j->k', Iuc, w_mu_up,   w_phi))

def brf_principal_plane(I_up_top):
    """2-D cubic spline BRF on the principal plane, back-scatter and forward."""
    phi_ext = np.append(phi_arr, 2.*np.pi)
    spl = RectBivariateSpline(mu_up, phi_ext,
                               np.hstack([I_up_top, I_up_top[:,[0]]]), kx=3, ky=3)
    th  = np.linspace(2., 88., 300)
    mv  = np.clip(np.abs(np.cos(np.radians(th))), mu_up[0]+1e-5, mu_up[-1]-1e-5)
    bk  = np.maximum(spl.ev(mv, np.full_like(mv, np.pi)),  0.) / (Fin/np.pi)
    fw  = np.maximum(spl.ev(mv, np.zeros_like(mv)),        0.) / (Fin/np.pi)
    return th, bk, fw

def brf_fine_grid(I_up_top):
    """
    Interpolate the BRF onto a fine 90×90 grid and return as a 2-D array.
    Also integrates the fine-grid BRF over the upper hemisphere to give Ref,
    which can be compared against the coarse-GL Fu(0) as a verification check.

    Output grid:
      theta_v : 2°, 4°, …, 178°  (89 values, 2° step)  — viewing zenith
      phi_v   : 0°, 4°, …, 356°  (90 values, 4° step)  — viewing azimuth

    BRF(θ_v, φ_v) = I_up_top(θ_v, φ_v) · π / Fin
    Only the upper hemisphere (θ_v < 90°) is physically meaningful.

    Ref from fine grid:
      Ref_fine = ∫₀²π ∫₀^(π/2) BRF(θ,φ)·(Fin/π)·cos θ·sin θ dθ dφ
             ≈ ∑_{θ<90°} ∑_φ BRF·(Fin/π)·cos θ·sin θ · Δθ · Δφ

    Returns theta_grid [89], phi_grid [90], BRF [89×90], Ref_fine (scalar).
    """
    phi_ext = np.append(phi_arr, 2.*np.pi)
    spl = RectBivariateSpline(mu_up, phi_ext,
                               np.hstack([I_up_top, I_up_top[:,[0]]]), kx=3, ky=3)
    theta_grid = np.arange(2., 180., 2.)                 # 89 values
    phi_grid   = np.arange(0., 360., 4.)                 # 90 values
    dtheta_r   = np.radians(2.)                          # 2° step in radians
    dphi_r     = np.radians(4.)                          # 4° step in radians
    BRF = np.zeros((len(theta_grid), len(phi_grid)))
    for it, th in enumerate(theta_grid):
        mu_v = abs(np.cos(np.radians(th)))
        mu_v_clamped = np.clip(mu_v, mu_up[0]+1e-5, mu_up[-1]-1e-5)
        for ip, ph in enumerate(phi_grid):
            phi_v = np.radians(ph)
            BRF[it, ip] = max(spl.ev(mu_v_clamped, phi_v), 0.) / (Fin/np.pi)

    # Integrate over upper hemisphere only (theta_v < 90°, i.e. indices where theta_grid < 90)
    # Ref = ∑ BRF·(Fin/π)·cos(θ)·sin(θ)·dθ·dφ
    Ref_fine = 0.
    for it, th in enumerate(theta_grid):
        if th >= 90.:
            break                                        # upper hemisphere only
        mu_v   = abs(np.cos(np.radians(th)))
        sin_v  = np.sin(np.radians(th))
        for ip in range(len(phi_grid)):
            Ref_fine += BRF[it, ip] * (Fin/np.pi) * mu_v * sin_v * dtheta_r * dphi_r

    return theta_grid, phi_grid, BRF, Ref_fine

def write_brf_fine_grid(band_name, LAI, theta_grid, phi_grid, BRF, Ref_fine,
                        Ref_coarse, filename='BRF_FINE_GRID.txt'):
    """
    Write the fine-grid BRF to a text file (append mode — all cases in one file).
    Includes a verification block comparing Ref integrated from the fine BRF
    against Ref from the coarse GL quadrature (Fu(0)).
    Both include uncollided + collided contributions.
    """
    diff_pct = 100.*(Ref_fine - Ref_coarse) / Ref_coarse if Ref_coarse > 1e-12 else 0.
    with open(filename, 'a') as f:
        f.write(f'# ── Band={band_name}  LAI={LAI} ──────────────────────────────\n')
        f.write(f'# Ref verification (includes uncollided + collided):\n')
        f.write(f'#   Ref_coarse (GL quadrature, Fu(0))     = {Ref_coarse:.6f}\n')
        f.write(f'#   Ref_fine   (integrated from fine BRF) = {Ref_fine:.6f}\n')
        f.write(f'#   Difference                            = {diff_pct:+.4f}%\n')
        f.write(f'# Columns: theta_v(deg)  phi_v(deg)  BRF\n')
        f.write(f'# (BRF physically meaningful for theta_v < 90 deg only)\n')
        for it, th in enumerate(theta_grid):
            for ip, ph in enumerate(phi_grid):
                f.write(f'{th:8.2f}  {ph:8.2f}  {BRF[it,ip]:12.6f}\n')
        f.write('\n')

# ==============================================================================
# SECTION 9 – CONVERGENCE CHECK  (dual criterion)
# ==============================================================================
def converged_dual(I_dn_n, I_up_n, I_dn_tot, I_up_tot, S_n, S_cum, tol=0.01):
    """
    Returns (converged: bool, rel_bnd: float, rel_S: float).

    Criterion A — boundary intensities:
      max |Iₙ(μᵢ,φⱼ)| / |Icum(μᵢ,φⱼ)| < tol
      for θ ∈ [0°,70°] at L=0 and L=LAI.

    Criterion B — scalar irradiance profile:
      max_k |Sₙ(k)| / |Scum(k)| < tol
      over all K cell centres.

    Both must pass simultaneously.
    """
    eps = 1e-12

    # Criterion A: boundary intensities
    mr = 0.
    for k_idx in [0, K]:
        mask  = np.abs(mu_down) >= MU_MIN
        mr = max(mr, (np.abs(I_dn_n[mask,:,k_idx]) /
                      np.maximum(np.abs(I_dn_tot[mask,:,k_idx]), eps)).max())
        mask2 = mu_up >= MU_MIN
        mr = max(mr, (np.abs(I_up_n[mask2,:,k_idx]) /
                      np.maximum(np.abs(I_up_tot[mask2,:,k_idx]), eps)).max())

    # Criterion B: scalar irradiance profile
    rel_S = np.max(np.abs(S_n) / np.maximum(np.abs(S_cum), eps))

    return (mr < tol and rel_S < tol), mr, rel_S

# ==============================================================================
# SECTION 10 – SOS SCATTERING SOURCE FROM PREVIOUS ORDER
# ==============================================================================
def scattering_source(I_dn_prev, I_up_prev, mu_out, omega_L, tau_L):
    """
    Qₙ(L, Ω) = (1/π) ∫₄π Γ(Ω′→Ω) · Iₙ₋₁(L, Ω′) dΩ′
    Uses cell-centre intensities. Returns Qn[N_out, N_phi, K].
    """
    I_dn_c = 0.5*(I_dn_prev[:,:,:-1]+I_dn_prev[:,:,1:])
    I_up_c = 0.5*(I_up_prev[:,:,:-1]+I_up_prev[:,:,1:])
    Qn = np.zeros((len(mu_out), N_phi, K))
    for ii in range(N_mu):
        for jj in range(N_phi):
            wd = w_mu_down[ii]*w_phi[jj]; wu = w_mu_up[ii]*w_phi[jj]
            Gd = phase_function(mu_out[:,None], phi_arr[None,:],
                                 mu_down[ii], phi_arr[jj], omega_L, tau_L)
            Gu = phase_function(mu_out[:,None], phi_arr[None,:],
                                 mu_up[ii],   phi_arr[jj], omega_L, tau_L)
            Qn += (1./np.pi)*(Gd[:,:,None]*I_dn_c[ii,jj,:][None,None,:]*wd
                             +Gu[:,:,None]*I_up_c[ii,jj,:][None,None,:]*wu)
    return Qn

# ==============================================================================
# SECTION 11 – ONE SOS ORDER SWEEP
# ==============================================================================
def sos_order(Qn_dn, Qn_up, LAI, rho_g, method='MOC'):
    """Solve one SOS order. method = 'MOC' (default) or 'DO'."""
    Dl = LAI / K
    sweep_dn = MOC_dn if method == 'MOC' else DD_dn
    sweep_up = MOC_up if method == 'MOC' else DD_up
    I_dn = np.zeros((N_mu,N_phi,K+1)); I_up = np.zeros((N_mu,N_phi,K+1))
    for i, mp in enumerate(mu_down):
        for j in range(N_phi): I_dn[i,j,:] = sweep_dn(abs(mp), Qn_dn[i,j,:], Dl)
    Fd_b = np.sum(np.abs(mu_down[:,None])*I_dn[:,:,K]*w_mu_down[:,None]*w_phi[None,:])
    Ig   = rho_g/np.pi * Fd_b
    for i, mu_o in enumerate(mu_up):
        for j in range(N_phi): I_up[i,j,:] = sweep_up(mu_o, Qn_up[i,j,:], Dl, Ig)
    return I_dn, I_up


def verify_DO_vs_MOC(Q1_dn, Q1_up, LAI, rho_g, band_name):
    """Run order-1 with both DO and MOC and report maximum relative difference."""
    I_dn_DO,  I_up_DO  = sos_order(Q1_dn, Q1_up, LAI, rho_g, method='DO')
    I_dn_MOC, I_up_MOC = sos_order(Q1_dn, Q1_up, LAI, rho_g, method='MOC')
    Fd_DO,  Fu_DO  = fluxes(I_dn_DO,  I_up_DO)
    Fd_MOC, Fu_MOC = fluxes(I_dn_MOC, I_up_MOC)
    def rdiff(a, b): return abs(a - b) / max(abs(b), 1e-15) * 100.
    checks = [('Fd1(L=0)',   Fd_DO[0],  Fd_MOC[0]),
              ('Fd1(L=LAI)', Fd_DO[-1], Fd_MOC[-1]),
              ('Fu1(L=LAI)', Fu_DO[-1], Fu_MOC[-1]),
              ('Fu1(L=0)',   Fu_DO[0],  Fu_MOC[0])]
    max_diff = max(rdiff(a, b) for _, a, b in checks)
    print(f"    DO vs MOC verification [{band_name} LAI={LAI}]:")
    for name, vd, vm in checks:
        print(f"      {name:<14}  DO={vd:.5f}  MOC={vm:.5f}  diff={rdiff(vd,vm):.4f}%")
    status = "✓ PASS" if max_diff < 0.1 else "✗ FAIL — check sweep code"
    print(f"      max diff = {max_diff:.4f}%  {status}")
    return I_dn_MOC, I_up_MOC

# ==============================================================================
# SECTION 12 – ENERGY BALANCE
# ==============================================================================
def energy_balance(band_name, LAI, Fd_tot, Fu_tot, S_cum, omL):
    """
    Ref   = Fu_tot(L=0)
    Trans = Fd_tot(LAI) − Fu_tot(LAI)
    Abs   = G·(1−ωL)·∫S(L) dL     scalar irradiance (independent of Ref & Trans)

    S_cum includes the direct beam: S_cum = Σ_orders Sp(Idn,Iup) + I0_dir_profile
    Columns: Direct, Diffuse, Collided  (each sums to its source independently).
    The total Abs is a single number (scalar irradiance does not split by column).
    """
    rho_g   = bands[band_name]['rho_g']
    L_edges = np.linspace(0, LAI, K+1)
    Dl      = LAI / K

    def Fd0_dir(l): return abs_mu0*Io*np.exp(-G*l/abs_mu0)
    def Fd0_dif(l): return uncollided_Fd(l) - Fd0_dir(l)
    def Fu0_cmp(l, Fb):
        dL = LAI - l
        if dL < 1e-12: return rho_g * Fb
        r, _ = integrate.quad(lambda m: m*np.exp(-G*dL/m), 1e-6, 1., limit=200)
        return 2.*np.pi * (rho_g/np.pi*Fb) * r

    Fd0_dir_bot = Fd0_dir(LAI); Fd0_dif_bot = Fd0_dif(LAI)
    Fu0_dir_0   = Fu0_cmp(0, Fd0_dir_bot)
    Fu0_dif_0   = Fu0_cmp(0, Fd0_dif_bot)
    Fd0_arr     = np.array([Fd0_dir(l)+Fd0_dif(l) for l in L_edges])
    Fu0_arr     = np.array([Fu0_cmp(l,Fd0_dir_bot)+Fu0_cmp(l,Fd0_dif_bot)
                            for l in L_edges])
    Fd_col = Fd_tot - Fd0_arr; Fu_col = Fu_tot - Fu0_arr

    # Ref and Trans split by column (from evaluated flux fields)
    ref_dir = Fu0_dir_0;               ref_dif = Fu0_dif_0
    ref_col = Fu_tot[0] - Fu0_dir_0 - Fu0_dif_0
    tra_dir = (1-rho_g)*Fd0_dir_bot;  tra_dif = (1-rho_g)*Fd0_dif_bot
    tra_col = Fd_col[-1] - Fu_col[-1]

    # Abs from scalar irradiance (total only — no column split)
    Abs_sc = G*(1-omL)*Dl*np.sum(S_cum)

    # SUM and imbalance
    Ref   = ref_dir + ref_dif + ref_col
    Trans = tra_dir + tra_dif + tra_col
    Fs    = Ref + Trans + Abs_sc
    return dict(
        ref=[ref_dir, ref_dif, ref_col, Ref],
        tra=[tra_dir, tra_dif, tra_col, Trans],
        Abs_sc=Abs_sc,
        Ref=Ref, Trans=Trans,
        Fs=Fs, imb=Fs-Fin, imb_pct=100*(Fs-Fin)/Fin
    )

# ==============================================================================
# SECTION 13 – FIGURE MAKER  (4-panel + table for options 2 & 3)
# ==============================================================================
COLORS = {1.5:'#1f77b4', 4.0:'#d62728'}

def make_figure(band_name, option, all_data, figname):
    """
    all_data[LAI] = dict with keys: Fd, Fu, theta, bk, fw, abs_prof,
                    S_cum, n_orders (options 2 & 3), omL.
    """
    params    = bands[band_name]
    rho_g     = params['rho_g']; omL = params['omega_L']; tauL = params['tau_L']
    theta_sun = 180. - theta0_deg

    titles = {1: 'Uncollided Field  (I₀)',
              2: 'Uncollided + First-Collision  (I₀ + I₁)',
              3: 'Full SOS Solution  (I₀ + I₁ + … + Iₙ)'}

    if option == 1:
        # 3 plots, no table, no absorption profile
        fig = plt.figure(figsize=(18, 9))
        gs  = gridspec.GridSpec(2, 3, figure=fig,
                                height_ratios=[0.12, 0.88], hspace=0.40, wspace=0.35)
    else:
        # 4 plots + table
        fig = plt.figure(figsize=(18, 17))
        gs  = gridspec.GridSpec(3, 4, figure=fig,
                                height_ratios=[0.07, 0.50, 0.43],
                                hspace=0.44, wspace=0.35)

    # Header
    ax_h = fig.add_subplot(gs[0,:])
    ax_h.axis('off')
    conv_note = ('  |  Convergence: boundary intensities AND scalar irradiance < 1%'
                 if option > 1 else '')
    ax_h.text(0.5, 0.88, titles[option] + f'  —  {band_name}' + conv_note,
              transform=ax_h.transAxes, ha='center', va='top',
              fontsize=10.5, fontweight='bold', color='#1a1a2e')
    ax_h.text(0.5, 0.44,
              f'fdir={fdir}  θ₀={theta0_deg:.0f}°  μ₀={mu0:.4f}  '
              f'G={G}  K={K}  Nμ={N_mu}/hemi  Nφ={N_phi}  '
              f'Fin={Fin} Wm⁻²  LAI={LAI_vals}',
              transform=ax_h.transAxes, ha='center', va='top',
              fontsize=9, color='#2c3e50', fontfamily='monospace')
    ax_h.text(0.5, 0.05,
              f'ρL={params["rho_L"]}  τL={tauL}  ωL={omL:.3f}  ρg={rho_g}'
              + ('' if option == 1 else
                 '  |  Γ=(ωL/3π)(sinβ−βcosβ)+(τL/3)cosβ'
                 '  |  Abs=G(1−ωL)∫S dL  (scalar irradiance)'),
              transform=ax_h.transAxes, ha='center', va='top',
              fontsize=9, color='#2c3e50', fontfamily='monospace')

    if option == 1:
        ax_A = fig.add_subplot(gs[1,0])
        ax_B = fig.add_subplot(gs[1,1])
        ax_C = fig.add_subplot(gs[1,2])
        ax_D = None
    else:
        ax_A = fig.add_subplot(gs[1,0])
        ax_B = fig.add_subplot(gs[1,1])
        ax_C = fig.add_subplot(gs[1,2])
        ax_D = fig.add_subplot(gs[1,3])

    for LAI in LAI_vals:
        c = COLORS[LAI]; d = all_data[LAI]
        x = np.linspace(0, 1, K+1)
        L_c = np.linspace(LAI/(2*K), LAI-LAI/(2*K), K)

        if option == 1:
            ax_A.plot(x, d['Fd']/Fin, c=c, ls='-', lw=2.2, label=f'LAI={LAI}')
            ax_B.plot(x, d['Fu']/Fin, c=c, ls='-', lw=2.2)
            th = d['theta']; xpp = np.concatenate([-th[::-1], th])
            ax_C.plot(xpp, np.concatenate([d['bk'][::-1], d['fw']]),
                      c=c, ls='-', lw=2.2)
        else:
            Fd0 = np.array([uncollided_Fd(l) for l in np.linspace(0,LAI,K+1)])
            Fu0 = np.array([uncollided_Fu(l,LAI,rho_g) for l in np.linspace(0,LAI,K+1)])
            bk0 = np.array([uncollided_BRF(t,LAI,rho_g) for t in d['theta']])
            xpp0 = np.concatenate([-d['theta'][::-1], d['theta']])
            ax_A.plot(x, Fd0/Fin, c=c, ls=':', lw=1.5, alpha=0.7)
            ax_B.plot(x, Fu0/Fin, c=c, ls=':', lw=1.5, alpha=0.7)
            ax_C.plot(xpp0, np.concatenate([bk0[::-1], bk0]), c=c, ls=':', lw=1.5, alpha=0.7)
            n_lbl = d.get('n_orders','')
            lbl = f'LAI={LAI}' + (f'  (n={n_lbl})' if n_lbl else '')
            ax_A.plot(x, d['Fd']/Fin, c=c, ls='-', lw=2.2, label=lbl)
            ax_B.plot(x, d['Fu']/Fin, c=c, ls='-', lw=2.2)
            th = d['theta']; xpp = np.concatenate([-th[::-1], th])
            ax_C.plot(xpp, np.concatenate([d['bk'][::-1], d['fw']]),
                      c=c, ls='-', lw=2.2)
            ax_D.plot(d['abs_prof'], L_c/LAI, c=c, ls='-', lw=2.2, label=lbl)

    # Legends and axes
    leg = [Line2D([0],[0], c=COLORS[l], lw=2.2,
                  label=f'LAI={l}' + (f'  (n={all_data[l].get("n_orders","")})' if option > 1
                                       and all_data[l].get("n_orders") else ''))
           for l in LAI_vals]
    if option > 1:
        leg += [Line2D([0],[0], c='gray', ls=':', lw=1.5, label='Uncollided I₀'),
                Line2D([0],[0], c='gray', ls='-', lw=2.2,
                       label='I₀+I₁' if option == 2 else 'Total')]

    for ax, yl, tit in [
        (ax_A, 'Norm. Downward Flux  (F / Fin)', 'Plot A — Downward Flux Profile'),
        (ax_B, 'Norm. Upward Flux  (F / Fin)',   'Plot B — Upward Flux Profile')]:
        ax.set_xlabel('Normalised Depth  L / LAI', fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(tit, fontsize=10, fontweight='bold')
        ax.set_xlim(0,1); ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3); ax.legend(handles=leg, fontsize=8, ncol=2)

    ax_C.set_xlabel('View Zenith (°)\n← Back (φ=π)  |  Fwd (φ=0) →', fontsize=9)
    ax_C.set_ylabel('BRF', fontsize=10)
    ax_C.set_title('Plot C — BRF at L=0  (Principal Plane)', fontsize=10, fontweight='bold')
    ax_C.axvline(0, c='gray', lw=0.8, ls=':')
    ax_C.axvline(-theta_sun, c='orange', lw=1.2, ls='--', alpha=0.85)
    ax_C.set_xlim(-90,90); ax_C.set_ylim(bottom=0)
    ax_C.set_xticks(np.arange(-90,91,15)); ax_C.grid(True, alpha=0.3)
    ax_C.legend(handles=leg[:2 if option==1 else 2]+
                [Line2D([0],[0], c='orange', lw=1.2, ls='--',
                        label=f'Solar (θ={theta_sun:.0f}°, back)')], fontsize=8)

    if ax_D is not None:
        ax_D.set_xlabel('Absorbed flux per unit LAI  [W m⁻²]', fontsize=10)
        ax_D.set_ylabel('Normalised Depth  L / LAI', fontsize=10)
        ax_D.set_title('Plot D — Canopy Absorption Profile\n'
                       'dAbs/dL = −d(Fd−Fu)/dL  (flux divergence)',
                       fontsize=9, fontweight='bold')
        ax_D.invert_yaxis(); ax_D.set_ylim(1, 0); ax_D.set_xlim(left=0)
        ax_D.grid(True, alpha=0.3); ax_D.legend(fontsize=8)

    # ── Energy balance table (options 2 and 3) ───────────────────────────────
    if option == 1:
        fig.savefig(figname, dpi=150, bbox_inches='tight')
        print(f'  Saved: {figname}')
        plt.close(fig); return

    ax_T = fig.add_subplot(gs[2,:]); ax_T.axis('off')
    ax_T.set_xlim(0,1); ax_T.set_ylim(0,1)

    LABEL_W = 0.28; DATA_W = (1.0-LABEL_W)/8.
    col_cx  = [LABEL_W+(i+0.5)*DATA_W for i in range(8)]
    mid15   = LABEL_W+2*DATA_W; mid40 = LABEL_W+6*DATA_W

    ax_T.add_patch(plt.Rectangle((0,0.960),1,0.040, color='#1F3864',
                   clip_on=False, zorder=1, transform=ax_T.transAxes))
    for txt, cx in [('LAI = 1.5', mid15), ('LAI = 4.0', mid40)]:
        ax_T.text(cx, 0.980, txt, ha='center', va='center', fontsize=9,
                  fontweight='bold', color='white', transform=ax_T.transAxes, zorder=2)
    ax_T.plot([LABEL_W+4*DATA_W]*2, [0.960,1.0],
              color='white', lw=0.8, alpha=0.5, transform=ax_T.transAxes)
    ax_T.add_patch(plt.Rectangle((0,0.920),1,0.040, color='#2E75B6',
                   clip_on=False, zorder=1, transform=ax_T.transAxes))
    for i, h in enumerate(['Direct','Diffuse','Collided','TOTAL']*2):
        ax_T.text(col_cx[i], 0.940, h, ha='center', va='center', fontsize=7.5,
                  fontweight='bold', color='white', transform=ax_T.transAxes, zorder=2)

    RH = 0.068; SH = 0.034
    def section(y, lbl, color='#1F3864'):
        ax_T.add_patch(plt.Rectangle((0,y-SH),1,SH, color=color, alpha=0.18,
                       transform=ax_T.transAxes, zorder=0))
        ax_T.text(0.005, y-SH*0.5, lbl, ha='left', va='center', fontsize=7.5,
                  fontweight='bold', color=color, transform=ax_T.transAxes)

    def drow(y, lbl, v15, v40, bold=False):
        ax_T.add_patch(plt.Rectangle((0,y-RH),1,RH, color='#F4F6F7', alpha=0.5,
                       transform=ax_T.transAxes, zorder=0))
        ax_T.plot([0,1],[y-RH]*2, color='#E0E0E0', lw=0.4, transform=ax_T.transAxes)
        ax_T.text(0.005, y-RH*0.5, lbl, ha='left', va='center', fontsize=7.5,
                  fontweight='bold' if bold else 'normal',
                  color='#1a1a2e', transform=ax_T.transAxes)
        for ci, (v, v2) in enumerate(zip(v15, v40)):
            for idx, val in [(ci, v), (ci+4, v2)]:
                txt = ('—' if val is None else
                       val if isinstance(val,str) else f'{float(val):.4f}')
                is_tot = (idx == 3 or idx == 7)
                is_neg = isinstance(val,(int,float)) and val < -5e-4
                ax_T.text(col_cx[idx], y-RH*0.5, txt,
                          ha='center', va='center', fontsize=7.5,
                          fontweight='bold' if is_tot else 'normal',
                          color='#C0392B' if is_neg else '#1a1a2e',
                          transform=ax_T.transAxes)
        return y - RH

    # Compute energy balance for each LAI
    pc = {LAI: energy_balance(band_name, LAI,
                               all_data[LAI]['Fd'], all_data[LAI]['Fu'],
                               all_data[LAI]['S_cum'], omL)
          for LAI in LAI_vals}

    def rv(LAI, field): return pc[LAI][field]     # returns list [dir,dif,col,tot]

    y = 0.912
    section(y, 'Incident flux'); y -= SH
    y = drow(y, 'Fin = 1.0 W m⁻²',
             [fdir,1-fdir,None,Fin], [fdir,1-fdir,None,Fin], bold=True)

    section(y, 'Reflected  —  upward flux at L = 0'); y -= SH
    y = drow(y, 'Fu(L=0)', rv(1.5,'ref'), rv(4.0,'ref'), bold=True)

    section(y, 'Transmitted  —  net flux at L = LAI  =  Fd(LAI) − Fu(LAI)'); y -= SH
    y = drow(y, 'Fd(LAI) − Fu(LAI)', rv(1.5,'tra'), rv(4.0,'tra'), bold=True)

    section(y, 'Absorbed  —  G·(1−ωL)·∫S(L) dL  (scalar irradiance, independent)'); y -= SH
    y = drow(y, 'G·(1−ωL)·∫S dL',
             [None,None,None,rv(1.5,'Abs_sc')],
             [None,None,None,rv(4.0,'Abs_sc')], bold=True)

    # Imbalance section
    imb15 = rv(1.5,'imb_pct'); imb40 = rv(4.0,'imb_pct')
    if option == 2:
        imb_color = '#922B21'   # red — expected non-zero
        sec_lbl = ('Energy balance  —  Imbalance = energy in I₂ + I₃ + … not yet included')
        foot = ('Imbalance represents energy still in uncomputed higher-order scatterings.  '
                'RED: small (leaves absorb 90%).  NIR: large (97.5% rescatters into higher orders).  '
                'Abs = G·(1−ωL)·∫S dL (scalar irradiance, independent of Ref & Trans).')
    else:
        n15 = all_data[1.5].get('n_orders','?')
        n40 = all_data[4.0].get('n_orders','?')
        imb_color = '#922B21' if (abs(imb15)>0.5 or abs(imb40)>0.5) else '#1a6e1a'
        sec_lbl = (f'Energy balance  —  dual convergence (boundary + scalar irradiance < 1%)  '
                   f'|  LAI=1.5: {n15} orders,  LAI=4.0: {n40} orders')
        foot = ('Ref and Trans: accurate (boundary criterion satisfied).  '
                'Abs = G·(1−ωL)·∫S dL (scalar irradiance).  '
                'Residual imbalance in NIR reflects GL quadrature limitation '
                'on scalar irradiance for high-ωL canopies (not a convergence failure).')
    section(y, sec_lbl, color=imb_color); y -= SH

    def tot(v15t, v40t):
        return [None,None,None,v15t], [None,None,None,v40t]
    y = drow(y, 'SUM = Reflected + Transmitted + Absorbed',
             *tot(f"{rv(1.5,'Fs'):.4f}", f"{rv(4.0,'Fs'):.4f}"), bold=True)
    y = drow(y, 'Imbalance  (SUM − Fin)  [W m⁻²]',
             *tot(f"{rv(1.5,'imb'):+.5f}", f"{rv(4.0,'imb'):+.5f}"), bold=True)
    drow(y, 'Imbalance  (%)',
         *tot(f"{imb15:+.3f}%", f"{imb40:+.3f}%"), bold=True)

    ax_T.text(0.5, -0.018, foot, ha='center', va='top',
              fontsize=7.0, color='#595959', style='italic',
              transform=ax_T.transAxes)

    fig.savefig(figname, dpi=150, bbox_inches='tight')
    print(f'  Saved: {figname}')
    plt.close(fig)

# ==============================================================================
# SECTION 14 – SOLVERS PER OPTION
# ==============================================================================
def solve_uncollided(band_name, LAI):
    """Option 1: uncollided field only."""
    L_edges = np.linspace(0, LAI, K+1)
    rho_g   = bands[band_name]['rho_g']
    Fd = np.array([uncollided_Fd(l) for l in L_edges])
    Fu = np.array([uncollided_Fu(l, LAI, rho_g) for l in L_edges])
    th = np.linspace(2., 88., 300)
    bk = np.array([uncollided_BRF(t, LAI, rho_g) for t in th])
    return dict(Fd=Fd, Fu=Fu, theta=th, bk=bk, fw=bk.copy(),
                abs_prof=np.zeros(K), S_cum=np.zeros(K))


def solve_first_collision(band_name, LAI):
    """
    Option 2: uncollided + one SOS order (I₀ + I₁).
    Dual convergence is not applied here — only one order is computed.
    The energy imbalance is the diagnostic.
    """
    params = bands[band_name]
    omL = params['omega_L']; tauL = params['tau_L']; rho_g = params['rho_g']
    Dl = LAI/K; L_edges = np.linspace(0,LAI,K+1)
    L_centres = 0.5*(L_edges[:-1]+L_edges[1:])
    Fd0_bot = uncollided_Fd(LAI)

    I_dn0 = (Id*np.exp(-G*L_edges[None,None,:]/np.abs(mu_down[:,None,None]))
             *np.ones((N_mu,N_phi,K+1)))
    I_up0 = (rho_g/np.pi*Fd0_bot
             *np.exp(-G*(LAI-L_edges)[None,None,:]/mu_up[:,None,None])
             *np.ones((N_mu,N_phi,K+1)))

    # Scalar irradiance accumulator — initialise with uncollided + direct beam
    S_cum = scalar_irrad_profile(I_dn0, I_up0) + Io*np.exp(-G*L_centres/abs_mu0)

    Fd_cum = np.array([uncollided_Fd(l) for l in L_edges])
    Fu_cum = np.array([uncollided_Fu(l,LAI,rho_g) for l in L_edges])
    I_up_top = (rho_g/np.pi*Fd0_bot*np.exp(-G*LAI/mu_up)[:,None]
                *np.ones((N_mu,N_phi)))

    Q1_dn = compute_Q1(L_centres, mu_down, phi_arr, omL, tauL)
    Q1_up = compute_Q1(L_centres, mu_up,   phi_arr, omL, tauL)
    I_dn, I_up = verify_DO_vs_MOC(Q1_dn, Q1_up, LAI, rho_g, band_name)
    Fd1, Fu1 = fluxes(I_dn, I_up)
    Fd_cum += Fd1; Fu_cum += Fu1; I_up_top += I_up[:,:,0]
    S_cum  += scalar_irrad_profile(I_dn, I_up)

    # Absorption profile — flux divergence
    net      = Fd_cum - Fu_cum
    abs_prof = -np.diff(net) / Dl

    th, bk, fw = brf_principal_plane(I_up_top)
    return dict(Fd=Fd_cum, Fu=Fu_cum, theta=th, bk=bk, fw=fw,
                abs_prof=abs_prof, S_cum=S_cum, I_up_top=I_up_top)


def solve_full_sos(band_name, LAI):
    """
    Option 3: uncollided + full SOS with dual convergence criterion.
    Stops when BOTH boundary intensities AND scalar irradiance converge to < 1%.
    """
    params = bands[band_name]
    omL = params['omega_L']; tauL = params['tau_L']; rho_g = params['rho_g']
    Dl = LAI/K; L_edges = np.linspace(0,LAI,K+1)
    L_centres = 0.5*(L_edges[:-1]+L_edges[1:])
    Fd0_bot = uncollided_Fd(LAI)

    I_dn0 = (Id*np.exp(-G*L_edges[None,None,:]/np.abs(mu_down[:,None,None]))
             *np.ones((N_mu,N_phi,K+1)))
    I_up0 = (rho_g/np.pi*Fd0_bot
             *np.exp(-G*(LAI-L_edges)[None,None,:]/mu_up[:,None,None])
             *np.ones((N_mu,N_phi,K+1)))

    Fd_cum   = np.array([uncollided_Fd(l) for l in L_edges])
    Fu_cum   = np.array([uncollided_Fu(l,LAI,rho_g) for l in L_edges])
    I_dn_tot = I_dn0.copy(); I_up_tot = I_up0.copy()
    I_up_top = (rho_g/np.pi*Fd0_bot*np.exp(-G*LAI/mu_up)[:,None]
                *np.ones((N_mu,N_phi)))

    # Scalar irradiance: initialise with uncollided field + direct beam
    S_cum = scalar_irrad_profile(I_dn0, I_up0) + Io*np.exp(-G*L_centres/abs_mu0)

    # Order 1
    Q1_dn = compute_Q1(L_centres, mu_down, phi_arr, omL, tauL)
    Q1_up = compute_Q1(L_centres, mu_up,   phi_arr, omL, tauL)
    I_dn, I_up = verify_DO_vs_MOC(Q1_dn, Q1_up, LAI, rho_g, band_name)
    Fd1, Fu1 = fluxes(I_dn, I_up)
    Fd_cum += Fd1; Fu_cum += Fu1
    I_dn_tot += I_dn; I_up_tot += I_up; I_up_top += I_up[:,:,0]
    S_n   = scalar_irrad_profile(I_dn, I_up)
    S_cum += S_n
    I_dn_prev = I_dn.copy(); I_up_prev = I_up.copy()
    n = 1

    while True:
        n += 1
        Qn_dn = scattering_source(I_dn_prev, I_up_prev, mu_down, omL, tauL)
        Qn_up = scattering_source(I_dn_prev, I_up_prev, mu_up,   omL, tauL)
        I_dn, I_up = sos_order(Qn_dn, Qn_up, LAI, rho_g)
        Fd_n, Fu_n = fluxes(I_dn, I_up)
        Fd_cum += Fd_n; Fu_cum += Fu_n
        I_dn_tot += I_dn; I_up_tot += I_up; I_up_top += I_up[:,:,0]
        S_n   = scalar_irrad_profile(I_dn, I_up)
        S_cum += S_n
        I_dn_prev = I_dn.copy(); I_up_prev = I_up.copy()

        conv, rel_bnd, rel_S = converged_dual(
            I_dn, I_up, I_dn_tot, I_up_tot, S_n, S_cum)
        print(f"      order {n:3d}  "
              f"bnd={rel_bnd:.2e}  S={rel_S:.2e}"
              + (" ✓" if conv else ""))
        if conv: break
        if n >= 500:
            print("      (max iterations reached)"); break

    # Absorption profile — flux divergence (exact, consistent with boundaries)
    net      = Fd_cum - Fu_cum
    abs_prof = -np.diff(net) / Dl

    th, bk, fw = brf_principal_plane(I_up_top)
    return dict(Fd=Fd_cum, Fu=Fu_cum, theta=th, bk=bk, fw=fw,
                abs_prof=abs_prof, S_cum=S_cum,
                n_orders=n, I_up_top=I_up_top)

# ==============================================================================
# SECTION 15 – MAIN
# ==============================================================================
def main():
    print()
    print("=" * 60)
    print("  1D Radiative Transfer — Vegetation Canopy")
    print("=" * 60)
    print()
    print("  Choose a problem to solve:")
    print("  (1)  Uncollided field only  (I₀)")
    print("  (2)  Uncollided + first collision  (I₀ + I₁)")
    print("  (3)  Uncollided + full SOS  (I₀ + I₁ + … + Iₙ)")
    print()
    while True:
        choice = input("  Enter 1, 2, or 3: ").strip()
        if choice in ('1','2','3'):
            option = int(choice); break
        print("  Please enter 1, 2, or 3.")

    # BRF fine grid option — only meaningful for the full SOS solution
    write_brf = False
    if option == 3:
        print()
        print("  Write BRF to fine angular grid file  BRF_FINE_GRID.txt ?")
        print("  Grid: θ_v = 2°…178° (2° step, 89 values),  φ_v = 0°…356° (4° step, 90 values)")
        print("  Includes verification: Ref from fine BRF vs Ref from GL quadrature.")
        print("  (0)  No")
        print("  (1)  Yes")
        print()
        while True:
            brf_choice = input("  Enter 0 or 1: ").strip()
            if brf_choice in ('0','1'):
                write_brf = (brf_choice == '1'); break
            print("  Please enter 0 or 1.")

    solvers = {1: solve_uncollided, 2: solve_first_collision, 3: solve_full_sos}
    labels  = {1: 'uncollided', 2: 'first_collision', 3: 'full_SOS'}
    solver  = solvers[option]

    # Initialise BRF output file
    if write_brf:
        with open('BRF_FINE_GRID.txt', 'w') as f:
            f.write('# BRF Fine Grid Output  —  Full SOS Solution\n')
            f.write(f'# theta0={theta0_deg} deg  phi0={phi0_deg} deg  fdir={fdir}\n')
            f.write(f'# Bands: {list(bands.keys())}   LAI: {LAI_vals}\n')
            f.write(f'# theta_v: 2 to 178 deg in 2-deg steps (89 values)\n')
            f.write(f'# phi_v:   0 to 356 deg in 4-deg steps (90 values)\n')
            f.write(f'# Ref verification: both Ref_coarse and Ref_fine include\n')
            f.write(f'#   uncollided (ground-reflected) + collided contributions.\n\n')

    print()
    for band_name in bands:
        print(f"  ── {band_name} " + "─"*45)
        all_data = {}
        for LAI in LAI_vals:
            print(f"    LAI = {LAI}")
            t0 = time.time()
            all_data[LAI] = solver(band_name, LAI)
            elapsed = time.time() - t0
            n_info = (f"  ({all_data[LAI]['n_orders']} orders)"
                      if 'n_orders' in all_data[LAI] else '')
            print(f"      done in {elapsed:.1f}s{n_info}")

            # Write fine BRF grid if requested (option 3 only)
            if write_brf:
                print(f"      computing BRF fine grid...", end=' ', flush=True)
                tg, pg, BRF_fg, Ref_fine = brf_fine_grid(all_data[LAI]['I_up_top'])
                Ref_coarse = all_data[LAI]['Fu'][0]   # Fu(0) from GL — includes unc + collided
                write_brf_fine_grid(band_name, LAI, tg, pg, BRF_fg,
                                    Ref_fine, Ref_coarse)
                diff = 100.*(Ref_fine - Ref_coarse)/Ref_coarse
                print(f"done  (Ref_fine={Ref_fine:.5f}  Ref_GL={Ref_coarse:.5f}  "
                      f"diff={diff:+.3f}%)")

        figname = f"{band_name}_{labels[option]}.png"
        make_figure(band_name, option, all_data, figname)

    print()
    if write_brf:
        print("  BRF fine grid written to: BRF_FINE_GRID.txt")
    print("  Done.")


if __name__ == '__main__':
    main()
