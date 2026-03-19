"""
================================================================================
UNCOLLIDED 1D RADIATIVE TRANSFER IN A VEGETATION CANOPY
================================================================================

Course  : Physical Models in Remote Sensing
Chapter : 04 – Part 01: 1D RT Problem Setup
Topic   : Solution of the Uncollided Radiation Field
Authors : Claude And I (Ranga B. Myneni)

--------------------------------------------------------------------------------
WHAT THIS CODE DOES
--------------------------------------------------------------------------------

This program solves the *uncollided* part of the 1D Radiative Transfer (RT)
problem for a horizontally homogeneous vegetation canopy. "Uncollided" means
we track only photons that have NOT yet hit a leaf since entering the canopy.

We compute and plot:
  Plot A – Downward uncollided flux  (direct solar + diffuse sky) vs. depth
  Plot B – Upward   uncollided flux  (ground-reflected beam)      vs. depth
  Plot C – Upward   uncollided BRF   at the canopy top, in the principal plane

--------------------------------------------------------------------------------
THE PHYSICS IN PLAIN ENGLISH
--------------------------------------------------------------------------------

Imagine you are standing inside a forest looking upward.  Some sunlight reaches
you directly without hitting any leaf (that is the *uncollided* direct beam).
Some scattered sky light also reaches you without hitting a leaf (uncollided
diffuse).  When both components finally hit the ground, the ground reflects them
back upward.  Those ground-reflected photons travel back up through the canopy;
the ones that escape without hitting a leaf on the way up are the *uncollided
upward* field.

Because uncollided photons never scatter off leaves, the mathematics reduces to
simple exponential attenuation (the Beer–Lambert law), making this part of the
problem analytically tractable — no iteration required.

--------------------------------------------------------------------------------
COORDINATE SYSTEM
--------------------------------------------------------------------------------

  • The z-axis points DOWNWARD into the canopy.
  • Instead of geometric depth z, we use the cumulative Leaf Area Index L:
        L = 0        at the TOP of the canopy
        L = LAI      at the BOTTOM (ground surface)
    This is natural because what matters for photon interception is how much
    leaf area a photon has passed through, not how many metres it has travelled.

  • Angles θ are measured from the NEGATIVE z-axis (i.e., from the zenith).
        μ = cos(θ)
        μ < 0  →  downward direction  (θ > 90° from zenith)
        μ > 0  →  upward direction    (θ < 90° from zenith)

  • The solar zenith angle θ₀ = 140° means the sun is 40° from the horizon,
    giving μ₀ = cos(140°) ≈ −0.766  (a downward direction, as expected).

--------------------------------------------------------------------------------
THE GOVERNING EQUATION FOR UNCOLLIDED PHOTONS
--------------------------------------------------------------------------------

The full 1D Radiative Transfer Equation is:

    −μ ∂I/∂L + G(Ω) I(L,Ω) = scattering source

For UNCOLLIDED photons, the scattering source is zero (by definition — these
photons have never scattered). So the equation simplifies to:

    −μ ∂I₀/∂L + G(Ω) I₀(L,Ω) = 0

This is a first-order ODE in L with the solution:

    I₀(L, Ω) = I₀(L_boundary, Ω) × exp( −G × |L − L_boundary| / |μ| )

This is exactly the Beer–Lambert law.  The term G/|μ| is the effective
extinction coefficient: it accounts for the fact that a photon travelling at
angle θ (with cosine μ) traverses more leaf area per unit depth than a
photon going straight down.

--------------------------------------------------------------------------------
GEOMETRY FACTOR G
--------------------------------------------------------------------------------

G(Ω) is the mean projected area of leaves per unit leaf area in direction Ω.
For a SPHERICAL leaf normal distribution (leaves point equally in all
directions), it can be shown analytically that:

    G = 0.5   for ALL directions Ω

This is a great simplification: the extinction coefficient G/|μ| depends only
on the view/illumination angle, not on the leaf orientation details.

--------------------------------------------------------------------------------
FLUX vs. INTENSITY
--------------------------------------------------------------------------------

  • Intensity  I(L, Ω)  [W m⁻² sr⁻¹]:  power per unit area per unit solid angle.
  • Flux       F(L)      [W m⁻²]:        power per unit area, integrated over
                                          a hemisphere.

The flux is obtained by integrating intensity weighted by |μ| = |cos θ|:

    F↓(L) = ∫₀²π dφ ∫₋₁⁰ |μ| I(L,μ,φ) dμ
           = 2π ∫₀¹ μ I(L,−μ) dμ      (downward hemisphere, μ > 0 here)

The factor |μ| appears because flux measures the component of energy flow
normal to a horizontal surface (think of sunlight hitting a tilted table — the
effective illumination depends on the tilt angle).

--------------------------------------------------------------------------------
NORMALISED QUANTITIES
--------------------------------------------------------------------------------

All fluxes are divided by the total incident flux Fin = 1 W m⁻² to give
dimensionless numbers between 0 and 1.  This makes it easy to see what
fraction of incoming radiation survives to each depth.

BRF (Bidirectional Reflectance Factor) is the upward intensity normalised by
the flux that would come from a perfect Lambertian reflector:

    BRF = I_up(L=0, Ω) / (Fin / π)

A perfectly white Lambertian surface has BRF = 1 in all directions.
Vegetation canopies have BRF ≪ 1 in the RED and < 1 in the NIR.

--------------------------------------------------------------------------------
DEPENDENCIES (standard Python scientific libraries)
--------------------------------------------------------------------------------

  numpy    – array maths
  scipy    – numerical integration  (scipy.integrate.quad)
  matplotlib – plotting

Install with:  pip install numpy scipy matplotlib
================================================================================
"""

# ── 0. IMPORTS ────────────────────────────────────────────────────────────────
# We import only what we need, and give each import a short comment so students
# know why it is here.

import numpy as np                     # array maths, trig functions, exp, etc.
import matplotlib.pyplot as plt        # plotting
import matplotlib.gridspec as gridspec # flexible subplot layout
from matplotlib.lines import Line2D    # for building custom legend entries
from scipy import integrate            # numerical integration (quad function)


# ==============================================================================
# SECTION 1 – MODEL INPUTS
# ==============================================================================
# All physical parameters are defined here in one place.  If you want to change
# any of them (e.g., try a different solar angle), you only need to edit this
# section — nothing else in the code needs to change.

# ── 1a. Radiation field inputs ─────────────────────────────────────────────────

Fin  = 1.0    # Total incident flux at the top of the canopy [W m⁻²]
              # We normalise everything by Fin, so setting it to 1 is convenient.

fdir = 0.7    # Fraction of Fin that arrives as DIRECT solar radiation.
              # The remaining (1 - fdir) = 0.3 is DIFFUSE sky radiation.
              # fdir = 1 → perfectly clear sky  (only direct sun)
              # fdir = 0 → completely overcast   (only diffuse sky)

# ── 1b. Solar geometry ─────────────────────────────────────────────────────────

theta0_deg = 140.0   # Solar zenith angle [degrees], measured from the UPWARD
                     # zenith (negative z-axis in our convention).
                     # θ₀ = 140° means the sun is 40° above the horizon.
                     # A downward solar direction requires θ₀ > 90°.

phi0_deg   = 0.0     # Solar azimuth angle [degrees].  We set the reference
                     # direction (φ = 0°) toward the sun, so the principal
                     # plane has φ = 0° (forward scatter) and φ = 180°
                     # (backscatter).

# Convert angles to radians for use with numpy trig functions
theta0 = np.radians(theta0_deg)
phi0   = np.radians(phi0_deg)

# Compute μ₀ = cos(θ₀).  This will be negative because θ₀ > 90°.
mu0     = np.cos(theta0)      # ≈ −0.766  (downward direction)
abs_mu0 = abs(mu0)            # |μ₀| ≈  0.766  (used in many formulas below)

# ── 1c. Canopy structure (wavelength-independent) ──────────────────────────────

G        = 0.5     # Geometry factor for a SPHERICAL leaf normal distribution.
                   # G = 0.5 is the exact analytical result for this case.
                   # It is the mean projected leaf area in any direction.

LAI_vals = [1.5, 4.0]   # We solve for TWO canopy depths:
                         #   LAI = 1.5  → sparse/moderate canopy (e.g., grassland)
                         #   LAI = 4.0  → dense canopy (e.g., closed forest)
                         # Plotting both on the same axes allows direct comparison.

# ── 1d. Wavelength-dependent optical properties ────────────────────────────────
# We solve for two spectral bands: RED and NIR.
# Each band has its own leaf reflectance, leaf transmittance, and ground reflectance.
# Leaf albedo ω_L = ρ_L + τ_L  (total fraction of intercepted light that is
# scattered; the rest, 1 − ω_L, is absorbed by the leaf).

bands = {
    'RED': {
        'rho_L': 0.06,    # Leaf reflectance (fraction reflected back)
        'tau_L': 0.04,    # Leaf transmittance (fraction transmitted through)
        'rho_g': 0.10,    # Ground (soil) reflectance
        # ω_L will be added below
    },
    'NIR': {
        'rho_L': 0.525,   # Leaves are much more reflective in NIR
        'tau_L': 0.45,    # Leaves transmit a lot of NIR (they are nearly transparent)
        'rho_g': 0.20,    # Soil is more reflective in NIR than RED
    },
}

# Compute derived quantity ω_L = ρ_L + τ_L for each band
for band in bands.values():
    band['omega_L'] = band['rho_L'] + band['tau_L']

# ── 1e. Intensities at the upper boundary ──────────────────────────────────────
# These are set by the boundary condition at L = 0 (top of canopy).

# Direct solar intensity [W m⁻² sr⁻¹]:
#   The solar beam carries flux fdir × Fin in direction Ω₀.
#   Flux = |μ₀| × Intensity  →  Intensity = Flux / |μ₀|
Io = fdir / abs_mu0

# Isotropic diffuse sky intensity [W m⁻² sr⁻¹]:
#   The diffuse sky radiates equally in all downward directions.
#   Integrating an isotropic intensity Id over the downward hemisphere gives:
#       F_diffuse = ∫₀²π dφ ∫₀¹ μ Id dμ = π × Id
#   So Id = F_diffuse / π = (1 − fdir) × Fin / π
Id = (1.0 - fdir) / np.pi


# ==============================================================================
# SECTION 2 – CORE PHYSICS FUNCTIONS
# ==============================================================================
# Each function below corresponds to a specific physical quantity.
# Read the docstring to understand what it computes and why.

# ── 2a. Beer–Lambert transmission probability ──────────────────────────────────

def transmission(mu, L1, L2):
    """
    Compute the probability that a photon travelling in direction μ passes
    from depth L1 to depth L2 WITHOUT hitting any leaf.

    This is the Beer–Lambert law for an anisotropic medium:

        T(L1 → L2, μ) = exp( −G × |L2 − L1| / |μ| )

    Physical interpretation:
      • G × |L2 − L1| / |μ|  is the optical path length — the total leaf area
        that the photon's path passes through between L1 and L2.
      • A photon travelling at angle θ (cosine μ) crosses more leaf area per
        unit depth than one going straight down, by a factor of 1/|μ|.
      • The exponential gives the survival probability.

    Parameters
    ----------
    mu : float
        cos(θ) for the photon's direction.  Sign indicates up/down but
        only |μ| enters the formula (distance is always positive).
    L1, L2 : float
        Start and end cumulative LAI values.  Order does not matter
        because we take |L2 − L1|.

    Returns
    -------
    float
        Probability in [0, 1].  Returns 1 if L1 == L2 (no path, no attenuation).
    """
    optical_path = G * abs(L2 - L1) / abs(mu)
    return np.exp(-optical_path)


# ── 2b. Downward direct solar flux ─────────────────────────────────────────────

def downward_direct_flux(L):
    """
    Uncollided downward DIRECT solar flux at depth L.

    The direct solar beam enters the canopy top (L = 0) as a delta function
    in direction Ω₀.  Travelling to depth L, it is attenuated by the
    Beer–Lambert transmission factor.

    Physics:
        F↓_direct(L) = |μ₀| × I₀ × T(0 → L, μ₀)
                     = fdir × Fin × exp( −G × L / |μ₀| )

    The factor |μ₀| converts intensity to flux (projects onto horizontal surface).
    Note that |μ₀| × I₀ = fdir × Fin — the initial direct flux — which is a
    useful sanity check: at L = 0, F↓_direct = 0.7 × 1 = 0.7 W m⁻².

    Parameters
    ----------
    L : float or numpy array
        Cumulative LAI [m² m⁻²] at which to evaluate the flux.

    Returns
    -------
    float or numpy array
        Direct downward flux [W m⁻²].
    """
    return abs_mu0 * Io * transmission(mu0, 0.0, L)


# ── 2c. Downward diffuse sky flux ──────────────────────────────────────────────

def _diffuse_integrand(mu, L):
    """
    Integrand for computing the downward diffuse flux at depth L.

    For an isotropic sky (intensity Id in all downward directions), the
    contribution to the downward flux from photons at cosine angle μ is:

        dF = 2π × μ × Id × exp(−G × L / μ) × dμ

    where:
      • 2π comes from integrating over all azimuth angles φ (no azimuth
        dependence for an isotropic sky)
      • μ is the flux projection factor (Lambert cosine law)
      • Id is the sky intensity (constant for isotropic sky)
      • exp(−G L / μ) is the Beer–Lambert survival probability for a photon
        at angle arccos(μ) travelling from L=0 to L

    Parameters
    ----------
    mu : float
        |cos θ| for the downward direction (positive, in [0, 1]).
    L  : float
        Depth (cumulative LAI) at which we want the flux.

    Returns
    -------
    float
        Value of the integrand at this (mu, L) pair.
    """
    return mu * Id * np.exp(-G * L / mu)


def downward_diffuse_flux(L):
    """
    Uncollided downward DIFFUSE sky flux at depth L.

    We integrate the diffuse sky intensity over the entire downward hemisphere
    (all directions with μ ∈ [0,1] and φ ∈ [0, 2π]):

        F↓_diff(L) = 2π ∫₀¹ μ × Id × exp(−G L / μ) dμ

    At L = 0 (canopy top), this must equal (1 − fdir) × Fin = 0.3 W m⁻²
    (the total diffuse sky flux before any attenuation).  This is a useful
    check: 2π ∫₀¹ μ Id dμ = 2π × Id × 1/2 = π × Id = (1−fdir) × Fin ✓

    The integral has no closed form for general L, so we evaluate it
    numerically using scipy.integrate.quad (Gaussian quadrature).

    Parameters
    ----------
    L : float
        Depth (cumulative LAI).  Must be a scalar (not an array) because
        scipy.integrate.quad works on scalars.

    Returns
    -------
    float
        Diffuse downward flux [W m⁻²].
    """
    if L == 0.0:
        # At the top, no attenuation has occurred yet.
        # Return the exact value directly (avoids a trivial integral).
        return (1.0 - fdir) * Fin

    # scipy.integrate.quad(function, lower_limit, upper_limit, args=(extra_args,))
    # We integrate from a tiny value near 0 (avoid division by zero at μ=0)
    # to 1 (nadir, μ=1).
    # 'limit=200' allows more subdivision intervals for accuracy.
    # 'epsabs' and 'epsrel' set absolute and relative error tolerances.
    result, error = integrate.quad(
        _diffuse_integrand,    # function to integrate
        1e-6,                  # lower limit: μ ≈ 0  (near-horizontal direction)
        1.0,                   # upper limit: μ = 1  (nadir direction)
        args=(L,),             # extra argument passed to the integrand
        limit=200,
        epsabs=1e-10,
        epsrel=1e-10
    )

    # Multiply by 2π (azimuthal integration — sky is uniform in φ)
    return 2.0 * np.pi * result


# ── 2d. Upward flux from ground reflection ─────────────────────────────────────

def _upward_integrand(mu, delta_L):
    """
    Integrand for computing the upward flux at a depth L above the ground.

    After the downward uncollided flux hits the Lambertian ground, it is
    reflected isotropically (equally in all upward directions) with
    intensity I_ground = ρ_g / π × F↓(L=LAI).

    The contribution to the upward flux at depth L from photons at upward
    angle arccos(μ) is:

        dF = 2π × μ × I_ground × exp(−G × delta_L / μ) × dμ

    where delta_L = LAI − L is the distance from the observation point
    to the ground.

    Parameters
    ----------
    mu      : float
        cos θ for the upward direction (positive, in [0, 1]).
    delta_L : float
        LAI − L, the remaining LAI between the observation point and the ground.

    Returns
    -------
    float
        Value of the integrand.
    """
    return mu * np.exp(-G * delta_L / mu)


def upward_flux_from_ground(L, LAI, rho_g, Fd_bottom):
    """
    Uncollided upward flux at depth L due to Lambertian ground reflection.

    Physics step by step:
    1. The downward uncollided flux Fd_bottom hits the ground at L = LAI.
    2. The Lambertian ground reflects it isotropically with reflectance ρ_g:
           I_ground(Ω_up) = ρ_g / π × Fd_bottom   [W m⁻² sr⁻¹]
       (same intensity in all upward directions — this is the definition of
       Lambertian / isotropic reflectance)
    3. Photons from the ground travel upward and are attenuated by the canopy
       over the distance delta_L = LAI − L from ground to observation point.
    4. The upward flux at depth L is found by integrating over all upward
       directions:
           F↑(L) = 2π ∫₀¹ μ × I_ground × exp(−G × delta_L / μ) dμ
                 = 2π × (ρ_g / π) × Fd_bottom × ∫₀¹ μ exp(−G δL / μ) dμ

    We call this function SEPARATELY for the direct-beam component of Fd_bottom
    and the diffuse component.  Because the equation is linear, the results add.

    Parameters
    ----------
    L          : float
        Depth (cumulative LAI) where we want the upward flux.
    LAI        : float
        Total leaf area index of the canopy (bottom boundary).
    rho_g      : float
        Ground reflectance (dimensionless, 0 to 1).
    Fd_bottom  : float
        The downward uncollided flux at L = LAI [W m⁻²].
        Call this function twice — once with Fd_direct(LAI), once with
        Fd_diffuse(LAI) — then add the results.

    Returns
    -------
    float
        Upward flux component [W m⁻²].
    """
    delta_L = LAI - L    # distance from observation point L to the ground

    if delta_L < 1e-12:
        # We are AT the ground boundary.  The reflected intensity is ρ_g × Fd_bottom,
        # integrated over the upward hemisphere:
        #   F↑ = ∫ μ (ρ_g/π × Fd_bottom) dΩ = ρ_g × Fd_bottom
        return rho_g * Fd_bottom

    # Ground-reflected intensity (isotropic upward) [W m⁻² sr⁻¹]
    I_ground = rho_g / np.pi * Fd_bottom

    # Numerically integrate over all upward directions (μ from 0 to 1)
    result, error = integrate.quad(
        _upward_integrand,
        1e-6,          # avoid μ = 0 (horizontal direction, infinite path)
        1.0,           # nadir view (μ = 1)
        args=(delta_L,),
        limit=200,
        epsabs=1e-10,
        epsrel=1e-10
    )

    return 2.0 * np.pi * I_ground * result


# ── 2e. Upward BRF at canopy top ───────────────────────────────────────────────

def compute_BRF(theta_view_deg, LAI, rho_g, Fd_total_at_ground):
    """
    Bidirectional Reflectance Factor (BRF) of the upward uncollided intensity
    at the TOP of the canopy (L = 0) for a given view direction.

    Physics:
    1. The Lambertian ground reflects the total downward uncollided flux as
       an isotropic upward intensity:
           I_ground = ρ_g / π × Fd_total(L=LAI)
    2. This intensity is attenuated by Beer–Lambert as it travels from the
       ground (L = LAI) back up to the canopy top (L = 0):
           I↑(L=0, μ_view) = I_ground × exp(−G × LAI / μ_view)
    3. BRF is defined relative to a perfect Lambertian surface:
           BRF = I↑(L=0) / (Fin / π)

    Note: Because I_ground is ISOTROPIC (same in all upward directions) and
    the canopy transmittance exp(−G LAI / μ) depends only on μ (not on φ),
    the BRF does not depend on the azimuth φ of the view direction.  This
    means the BRF curve is the SAME for the backscatter side (φ = 180°) and
    the forward-scatter side (φ = 0°) of the principal plane.  This symmetry
    is broken in the collided field (which includes leaf scattering).

    Parameters
    ----------
    theta_view_deg : float
        View zenith angle [degrees] from the upward zenith.  For upward
        directions, θ_view is between 0° (nadir) and 90° (horizon).
    LAI            : float
        Total leaf area index of the canopy.
    rho_g          : float
        Ground reflectance.
    Fd_total_at_ground : float
        Total downward uncollided flux at L = LAI [W m⁻²].

    Returns
    -------
    float
        BRF (dimensionless).
    """
    # μ_view = cos(θ_view).  For upward directions, θ_view < 90° → μ_view > 0.
    mu_view = abs(np.cos(np.radians(theta_view_deg)))

    # Step 1: ground-reflected isotropic intensity
    I_ground = rho_g / np.pi * Fd_total_at_ground

    # Step 2: transmit from ground to canopy top
    I_up_at_top = I_ground * np.exp(-G * LAI / mu_view)

    # Step 3: normalise by Fin/π to get BRF
    BRF = I_up_at_top / (Fin / np.pi)

    return BRF


# ==============================================================================
# SECTION 3 – COMPUTE ALL QUANTITIES FOR BOTH BANDS
# ==============================================================================
# Now we call the functions defined above to compute fluxes and BRF at
# a fine grid of depths L/LAI ∈ [0, 1] and view angles θ ∈ [0°, 90°].

N_depth = 200    # number of depth points per LAI value (increase for smoother curves)
N_angle = 300    # number of view angle points for the BRF plot


# ==============================================================================
# SECTION 4 – PLOTTING
# ==============================================================================
# We produce one figure per spectral band.  Each figure has:
#   - A header row showing all input parameters (so the figure is self-contained)
#   - Plot A : downward fluxes vs. normalised depth L/LAI
#   - Plot B : upward fluxes   vs. normalised depth L/LAI
#   - Plot C : upward BRF at L=0 in the principal plane

# ── Colour and line style choices ─────────────────────────────────────────────
# Using contrasting colours and line styles makes it easy to distinguish
# the two LAI values and the direct vs. diffuse components.

COLORS = {
    1.5: '#1f77b4',   # blue  for LAI = 1.5
    4.0: '#d62728',   # red   for LAI = 4.0
}
LS_DIRECT  = '-'    # solid line  → direct solar component
LS_DIFFUSE = '--'   # dashed line → diffuse sky component


for band_name, params in bands.items():

    rho_g   = params['rho_g']
    omega_L = params['omega_L']

    # ── 4a. Create figure layout ───────────────────────────────────────────────
    # We use GridSpec to create two rows:
    #   Row 0 (narrow)  : header text block with input parameters
    #   Row 1 (tall)    : three side-by-side plot panels

    fig = plt.figure(figsize=(16, 7.4))
    gs  = fig.add_gridspec(
        2, 3,                    # 2 rows, 3 columns
        height_ratios=[0.20, 1.0],  # header row is 20% as tall as plot row
        hspace=0.50,             # vertical space between rows
        wspace=0.35              # horizontal space between columns
    )

    # ── 4b. Header panel ───────────────────────────────────────────────────────
    # We add a text box listing all input parameters.  This makes the figure
    # self-contained — a reader does not need to look elsewhere for the inputs.

    ax_hdr = fig.add_subplot(gs[0, :])   # span all 3 columns
    ax_hdr.axis('off')                   # no axes, just text

    theta_sun_nadir = 180.0 - theta0_deg  # solar zenith from nadir = 40°

    title_str = f'Uncollided Radiative Transfer  —  {band_name} band'
    line1_str = (f'fdir = {fdir}     '
                 f'\u03b80 = {theta0_deg:.0f}\u00b0  (\u03bc0 = {mu0:.4f})     '
                 f'\u03c60 = {phi0_deg:.0f}\u00b0     '
                 f'G = {G}  (spherical leaf distribution)     '
                 f'Fin = {Fin} W m\u207b\u00b2     '
                 f'LAI = {LAI_vals[0]}, {LAI_vals[1]}')
    line2_str = (f'\u03c1L = {params["rho_L"]}     '
                 f'\u03c4L = {params["tau_L"]}     '
                 f'\u03c9L = \u03c1L + \u03c4L = {omega_L:.3f}     '
                 f'\u03c1g = {rho_g}')

    ax_hdr.text(0.5, 1.00, title_str,
                transform=ax_hdr.transAxes, ha='center', va='top',
                fontsize=13, fontweight='bold', color='#1a1a2e')
    ax_hdr.text(0.5, 0.55, line1_str,
                transform=ax_hdr.transAxes, ha='center', va='top',
                fontsize=9.5, color='#2c3e50', fontfamily='monospace')
    ax_hdr.text(0.5, 0.10, line2_str,
                transform=ax_hdr.transAxes, ha='center', va='top',
                fontsize=9.5, color='#2c3e50', fontfamily='monospace')

    # ── 4c. Create the three plot axes ─────────────────────────────────────────
    ax_A = fig.add_subplot(gs[1, 0])   # downward flux
    ax_B = fig.add_subplot(gs[1, 1])   # upward flux
    ax_C = fig.add_subplot(gs[1, 2])   # BRF principal plane

    # ── 4d. Loop over the two LAI values ───────────────────────────────────────
    for LAI in LAI_vals:

        color = COLORS[LAI]

        # -- Depth array --
        # linspace(start, stop, N) creates N evenly-spaced values from start to stop.
        # L_arr: absolute cumulative LAI from 0 to LAI
        # x_arr: normalised depth L/LAI from 0 to 1 (same x-axis for both LAI values)
        L_arr = np.linspace(0.0, LAI, N_depth)
        x_arr = L_arr / LAI     # normalised depth [dimensionless]

        # -- Pre-compute bottom boundary fluxes (needed for upward field) --
        # We call the scalar functions once for L = LAI.
        Fd_direct_bottom  = downward_direct_flux(LAI)
        Fd_diffuse_bottom = downward_diffuse_flux(LAI)
        Fd_total_bottom   = Fd_direct_bottom + Fd_diffuse_bottom

        # ── PLOT A: Downward fluxes ────────────────────────────────────────────
        # We build numpy arrays by applying each function to every element of L_arr.
        # The list comprehension [f(l) for l in L_arr] evaluates f at each depth l.

        Fd_direct_arr  = np.array([downward_direct_flux(l)  for l in L_arr])
        Fd_diffuse_arr = np.array([downward_diffuse_flux(l) for l in L_arr])

        # Normalise by Fin before plotting
        ax_A.plot(x_arr, Fd_direct_arr  / Fin,
                  color=color, ls=LS_DIRECT,  lw=2)
        ax_A.plot(x_arr, Fd_diffuse_arr / Fin,
                  color=color, ls=LS_DIFFUSE, lw=2)

        # ── PLOT B: Upward fluxes ──────────────────────────────────────────────
        # We separate the direct and diffuse contributions to the upward field.
        # The ground receives Fd_direct_bottom + Fd_diffuse_bottom and reflects
        # them both.  Because the equation is linear, we track them separately.

        Fu_direct_arr  = np.array([
            upward_flux_from_ground(l, LAI, rho_g, Fd_direct_bottom)
            for l in L_arr
        ])
        Fu_diffuse_arr = np.array([
            upward_flux_from_ground(l, LAI, rho_g, Fd_diffuse_bottom)
            for l in L_arr
        ])

        ax_B.plot(x_arr, Fu_direct_arr  / Fin,
                  color=color, ls=LS_DIRECT,  lw=2)
        ax_B.plot(x_arr, Fu_diffuse_arr / Fin,
                  color=color, ls=LS_DIFFUSE, lw=2)

        # ── PLOT C: BRF in the principal plane ────────────────────────────────
        # The principal plane contains the sun direction.
        # Convention (x-axis of Plot C):
        #   negative angles → backscatter side  (φ_view = 180°, looking toward sun)
        #   positive angles → forward scatter    (φ_view =   0°, looking away from sun)
        # At nadir (θ_view = 0°), both sides meet.

        theta_view_arr = np.linspace(0.0, 89.9, N_angle)   # 0° to nearly 90°

        # Compute BRF for each view angle.
        # For Lambertian ground + spherical canopy, BRF is the same on both
        # sides of the principal plane (see docstring of compute_BRF).
        brf_arr = np.array([
            compute_BRF(t, LAI, rho_g, Fd_total_bottom)
            for t in theta_view_arr
        ])

        # Build the full principal-plane x-axis:
        #   left side  (backscatter): −89.9° to 0°
        #   right side (forward):       0°  to 89.9°
        x_brf   = np.concatenate([-theta_view_arr[::-1],  theta_view_arr])
        brf_all = np.concatenate([ brf_arr[::-1],          brf_arr])
        # [::-1] reverses an array (so backscatter curves mirror from 0 outward)

        ax_C.plot(x_brf, brf_all, color=color, lw=2, label=f'LAI = {LAI}')

    # ── 4e. Build shared legend handles ───────────────────────────────────────
    # We create legend entries manually so that we can combine
    # line style (direct/diffuse) and colour (LAI) information cleanly.

    legend_handles_AB = (
        # Line style entries
        [Line2D([0], [0], color='gray', ls=LS_DIRECT,  lw=2, label='Direct'),
         Line2D([0], [0], color='gray', ls=LS_DIFFUSE, lw=2, label='Diffuse')] +
        # Colour entries
        [Line2D([0], [0], color=COLORS[l], lw=2, label=f'LAI = {l}')
         for l in LAI_vals]
    )

    # ── 4f. Format Plot A ──────────────────────────────────────────────────────
    ax_A.set_xlabel('Normalised Depth,  L / LAI', fontsize=11)
    ax_A.set_ylabel('Normalised Downward Flux  (F / F$_{in}$)', fontsize=11)
    ax_A.set_title('Plot A \u2014 Downward Uncollided Flux', fontsize=11, fontweight='bold')
    ax_A.set_xlim(0, 1)
    ax_A.set_ylim(bottom=0)
    ax_A.grid(True, alpha=0.3)
    ax_A.legend(handles=legend_handles_AB, fontsize=8, ncol=2, loc='upper right')

    # ── 4g. Format Plot B ──────────────────────────────────────────────────────
    ax_B.set_xlabel('Normalised Depth,  L / LAI', fontsize=11)
    ax_B.set_ylabel('Normalised Upward Flux  (F / F$_{in}$)', fontsize=11)
    ax_B.set_title('Plot B \u2014 Upward Uncollided Flux', fontsize=11, fontweight='bold')
    ax_B.set_xlim(0, 1)
    ax_B.set_ylim(bottom=0)
    ax_B.grid(True, alpha=0.3)
    ax_B.legend(handles=legend_handles_AB, fontsize=8, ncol=2, loc='upper right')

    # ── 4h. Format Plot C ──────────────────────────────────────────────────────
    ax_C.set_xlabel(
        'View Zenith Angle  (\u00b0)\n'
        '\u2190 Backscatter (\u03c6=180\u00b0)  |  Forward scatter (\u03c6=0\u00b0) \u2192',
        fontsize=10
    )
    ax_C.set_ylabel('BRF', fontsize=11)
    ax_C.set_title('Plot C \u2014 Upward BRF at L=0  (Principal Plane)',
                   fontsize=11, fontweight='bold')
    ax_C.axvline(0, color='gray', lw=0.8, ls=':')   # nadir marker

    # Mark the solar direction in the principal plane.
    # θ₀ = 140° from the zenith ↔ 40° from nadir, on the backscatter side.
    ax_C.axvline(-theta_sun_nadir, color='orange', lw=1.2, ls='--', alpha=0.85)

    ax_C.set_xlim(-90, 90)
    ax_C.set_ylim(bottom=0)
    ax_C.set_xticks(np.arange(-90, 91, 15))
    ax_C.grid(True, alpha=0.3)

    handles_C = (
        [Line2D([0], [0], color=COLORS[l], lw=2, label=f'LAI = {l}')
         for l in LAI_vals] +
        [Line2D([0], [0], color='orange', lw=1.2, ls='--',
                label=f'Solar dir. (\u03b8 = {theta_sun_nadir:.0f}\u00b0, backscatter)')]
    )
    ax_C.legend(handles=handles_C, fontsize=8)

    # ── 4i. Save figure ────────────────────────────────────────────────────────
    out_path = f'/mnt/user-data/outputs/uncollided_{band_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# ==============================================================================
# SECTION 5 – SANITY CHECKS
# ==============================================================================
# After computing results, always verify a few key values against known
# analytical answers.  This is good scientific coding practice.

print("\n--- Sanity Checks ---")

# At L = 0, the direct flux must equal fdir × Fin
F_direct_top = downward_direct_flux(0.0)
print(f"Direct flux at L=0:  {F_direct_top:.6f}  (expected {fdir * Fin:.6f})")

# At L = 0, the diffuse flux must equal (1-fdir) × Fin
F_diffuse_top = downward_diffuse_flux(0.0)
print(f"Diffuse flux at L=0: {F_diffuse_top:.6f}  (expected {(1-fdir) * Fin:.6f})")

# Total at top must equal Fin
print(f"Total flux at L=0:   {F_direct_top + F_diffuse_top:.6f}  (expected {Fin:.6f})")

print("\nAll figures saved.  Done.")
