"""
================================================================================
dom_uncollided.py
================================================================================
Discrete Ordinates Method (DOM) — Part One
Uncollided Radiation Intensities and First Collision Sources
in a 1D Leaf Canopy

Course  : Physical Models in Remote Sensing — Chapter 04, Part 03
--------------------------------------------------------------------------------

WHAT THIS CODE COMPUTES
------------------------
We solve for radiation that travels through a leaf canopy WITHOUT ever hitting
a leaf (called "uncollided" radiation).  This is the first of two steps in the
Discrete Ordinates Method:

   Step A  (this file)  →  uncollided intensities + first collision source Q
   Step B  (next file)  →  collided intensities using Q as a source term

The uncollided problem reduces to simple Beer-Lambert exponential decay, so no
iteration is needed.  We also compute the "First Collision Source" Q, which
describes where and in what direction photons scatter the very first time they
hit a leaf.  Q feeds directly into the collided problem in Part Two.

--------------------------------------------------------------------------------
INPUTS YOU CAN CHANGE  (all in the  if __name__ == "__main__"  block at bottom)
--------------------------------------------------------------------------------

  N            Number of polar quadrature directions  (must be even; try 4, 8, 16)
  M            Number of azimuthal directions         (try 4, 8, 16)
  K            Number of canopy layers                (try 20, 50, 100)

  LAI          Leaf Area Index of the canopy          (e.g. 1.5 or 4.0)
  f_dir        Fraction of incident flux that is direct solar  (0 to 1)
  theta_o_deg  Solar zenith angle in degrees, measured from the upward zenith
               Must be > 90 for a downward sun (e.g. 140 means 40 elevation)
  phi_o_deg    Solar azimuth angle in degrees         (e.g. 0)
  rho_g        Ground reflectance                     (e.g. 0.10 for RED soil)
  F_in         Total incident flux at canopy top      (normally 1.0 W/m2)

  rho_L        Leaf reflectance                       (e.g. 0.06 RED, 0.525 NIR)
  tau_L        Leaf transmittance                     (e.g. 0.04 RED, 0.450 NIR)

--------------------------------------------------------------------------------
SIGN CONVENTION  (matches lecture slides exactly)
--------------------------------------------------------------------------------

  The -z axis points UPWARD to the zenith.
  theta = polar angle measured from the -z axis (from the zenith).
  mu = cos(theta)
        mu > 0  means UPWARD   direction  (theta < 90 degrees)
        mu < 0  means DOWNWARD direction  (theta > 90 degrees)
  The sun is a DOWNWARD direction, so mu_solar < 0.
  Example: theta_solar = 140 degrees  ->  mu_solar = cos(140) = -0.766

  L = cumulative Leaf Area Index from the top of the canopy.
        L = 0    at canopy top
        L = LAI  at canopy bottom (ground surface)

--------------------------------------------------------------------------------
QUADRATURE INDEX CONVENTION  (1-based arrays to match slide notation)
--------------------------------------------------------------------------------

  All arrays are padded with a dummy index-0 slot so that array index k
  directly matches the slide notation k = 1, 2, ..., K+1.  Index 0 is
  always zero and never used in any calculation.

  numpy's leggauss() returns ordinates sorted ascending (most negative first).
  We keep that natural order, giving:
      polar_cosines[i]  for i = 1 ... N/2      -> mu < 0  (DOWNWARD)
      polar_cosines[i]  for i = N/2+1 ... N    -> mu > 0  (UPWARD)

  This matches the lecture slide notation exactly.

--------------------------------------------------------------------------------
MODULES REQUIRED
--------------------------------------------------------------------------------
  numpy      - array maths, Gauss-Legendre quadrature
  matplotlib - plotting
--------------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# SECTION 1 -- ANGULAR DISCRETIZATION  (Lecture slides 7-9)
# ==============================================================================

def compute_gauss_legendre_quadrature(num_polar_directions):
    """
    Compute Gauss-Legendre quadrature weights and ordinates on [-1, +1].

    WHY WE NEED THIS
    ----------------
    The radiative transfer equation involves integrals over all directions.
    We cannot do those integrals analytically, so we replace them with a
    weighted sum at a finite set of chosen angles:

        integral_{-1}^{+1} f(mu) dmu  ~  sum_{i=1}^{N}  w_i * f(mu_i)

    Gauss-Legendre quadrature chooses the N angles (ordinates) and weights
    so that this approximation is EXACT for any polynomial up to degree 2N-1.
    This makes it far more efficient than using uniform angle spacing.

    HOW TO READ THE OUTPUT
    ----------------------
    The ordinates come out sorted ascending (most negative first).
    With N=4 for example:
        ordinates ~ [-0.861, -0.340,  +0.340,  +0.861]
        weights   ~ [ 0.348,  0.652,   0.652,   0.348]
    The negative half (i=1..N/2) means downward directions (mu < 0).
    The positive half (i=N/2+1..N) means upward directions (mu > 0).

    SANITY CHECK (slide 9):
        sum of all w_i              = 2.0
        sum of mu_i * w_i (upward half) = 0.5

    Parameters
    ----------
    num_polar_directions : int
        N, the order of quadrature.  Must be even.

    Returns
    -------
    gauss_ordinates : ndarray, shape (N,)
        The quadrature points mu_i, sorted ascending (most negative first).
    gauss_weights : ndarray, shape (N,)
        The corresponding weights w_i.
    """
    if num_polar_directions % 2 != 0:
        raise ValueError(
            f"num_polar_directions (N) must be even, got {num_polar_directions}."
        )
    gauss_ordinates, gauss_weights = np.polynomial.legendre.leggauss(
        num_polar_directions
    )
    return gauss_ordinates, gauss_weights


def setup_angular_quadrature_grid(num_polar_directions, num_azimuthal_directions):
    """
    Build the full 2D angular quadrature grid (mu, phi).

    We discretize the sphere into N x M discrete directions:
        N Gauss-Legendre points in the polar cosine mu in [-1, +1]
        M uniformly spaced points in the azimuthal angle phi in [0, 2*pi)

    ARRAY LAYOUT (1-based indexing to match slide notation)
    -------------------------------------------------------
    Arrays are padded with a dummy slot at index 0 (always zero, never used).

        polar_cosines[i]    for i = 1..N
            i = 1..N/2      -> mu_i < 0  (downward, into the canopy)
            i = N/2+1..N    -> mu_i > 0  (upward, toward the sky)

        azimuthal_angles[j] for j = 1..M
            uniformly spaced: phi_j = (j - 0.5) * (2*pi / M)

        polar_weights[i]    -- Gauss weight for polar direction i
        azimuthal_weight    -- uniform weight = 2*pi / M (same for all j)

    Parameters
    ----------
    num_polar_directions     : int  N (must be even)
    num_azimuthal_directions : int  M

    Returns
    -------
    polar_cosines    : ndarray (N+1,)   mu_i,   index 1..N
    azimuthal_angles : ndarray (M+1,)   phi_j [rad], index 1..M
    polar_weights    : ndarray (N+1,)   w_i,    index 1..N
    azimuthal_weight : float            2*pi / M
    """
    N = num_polar_directions
    M = num_azimuthal_directions

    raw_ordinates, raw_weights = compute_gauss_legendre_quadrature(N)

    # Pad with dummy index-0 slot (0.0, never used in calculations)
    polar_cosines = np.zeros(N + 1)
    polar_weights = np.zeros(N + 1)
    polar_cosines[1:N+1] = raw_ordinates   # mu_1 .. mu_N  (negative first)
    polar_weights[1:N+1] = raw_weights     # w_1  .. w_N

    # Azimuthal grid: M equally-spaced cell-centred angles in [0, 2*pi)
    azimuthal_weight = 2.0 * np.pi / M
    azimuthal_angles = np.zeros(M + 1)
    for j in range(1, M + 1):
        azimuthal_angles[j] = (j - 0.5) * azimuthal_weight

    return polar_cosines, azimuthal_angles, polar_weights, azimuthal_weight


def print_quadrature_sanity_checks(polar_cosines, polar_weights, N):
    """
    Print two numerical checks to verify the quadrature is set up correctly.

    Check 1:  sum_{i=1}^{N} w_i  = 2.0
              (integral of the constant 1 over [-1,+1] equals 2)

    Check 2:  sum_{i=N/2+1}^{N} mu_i * w_i  = 0.5
              (integral of mu over the upward hemisphere [0,+1] equals 0.5)

    See lecture slide 9 for derivation.
    """
    print("=" * 55)
    print("QUADRATURE SANITY CHECKS  (lecture slide 9)")
    print("=" * 55)
    sum_of_weights        = np.sum(polar_weights[1:N+1])
    sum_mu_times_w_upward = np.sum(
        polar_cosines[N//2+1:N+1] * polar_weights[N//2+1:N+1]
    )
    print(f"  Sum of all weights w_i             = {sum_of_weights:.6f}  (should be 2.0)")
    print(f"  Sum of mu_i * w_i  (upward i>N/2)  = {sum_mu_times_w_upward:.6f}  (should be 0.5)")
    print(f"  Downward directions: i = 1..{N//2}   "
          f"mu range [{polar_cosines[1]:.3f}, {polar_cosines[N//2]:.3f}]  (all < 0 check)")
    print(f"  Upward   directions: i = {N//2+1}..{N}  "
          f"mu range [{polar_cosines[N//2+1]:.3f}, {polar_cosines[N]:.3f}]  (all > 0 check)")
    print()


# ==============================================================================
# SECTION 2 -- CROSS SECTIONS  (Lecture slide 7)
# ==============================================================================

def compute_extinction_cross_section():
    """
    Return the extinction cross-section G for uniform (spherical) leaf
    normal orientation.

    PHYSICAL MEANING
    ----------------
    G(Omega) is the mean projected area of leaves per unit leaf area in
    direction Omega.  It tells us how much leaf area a photon travelling
    in direction Omega "sees" per unit depth dL of canopy.

    For a UNIFORM (spherical) leaf normal distribution -- leaves point
    equally in all directions -- G is the same for every direction:

        G(Omega) = 0.5   for all Omega

    This is an exact analytical result.  It means that regardless of the
    sun angle or view angle, leaves intercept exactly half their area.

    Returns
    -------
    float : 0.5
    """
    return 0.5


def compute_volume_scattering_phase_function(
        incident_mu, incident_phi,
        scattered_mu, scattered_phi,
        leaf_albedo, leaf_transmittance):
    """
    Compute the volume scattering phase function Gamma(Omega' -> Omega).

    PHYSICAL MEANING
    ----------------
    Gamma(Omega' -> Omega) * dOmega is the probability that a photon
    arriving from direction Omega' is scattered into the solid angle dOmega
    around direction Omega, after hitting a leaf.  It encodes both leaf
    reflectance (photon bounces back) and leaf transmittance (photon passes
    through in a new direction).

    FORMULA  (from lecture C4-P1, uniform leaf normal distribution)
    ---------------------------------------------------------------
        Gamma = (omega_L / (3*pi)) * (sin(beta) - beta*cos(beta))
              + (tau_L  / 3.0)    *  cos(beta)

    where:
        beta    = scattering angle = arccos(Omega' . Omega)
        omega_L = leaf albedo = rho_L + tau_L
        tau_L   = leaf transmittance

    The dot product of two unit direction vectors in spherical coordinates:
        Omega' . Omega = mu1*mu2 + sqrt(1-mu1^2)*sqrt(1-mu2^2)*cos(phi1-phi2)

    Parameters
    ----------
    incident_mu, incident_phi   : float  incident direction Omega'
    scattered_mu, scattered_phi : float  scattered direction Omega
    leaf_albedo                 : float  omega_L = rho_L + tau_L
    leaf_transmittance          : float  tau_L

    Returns
    -------
    float : value of Gamma(Omega' -> Omega)
    """
    sin_incident  = np.sqrt(max(1.0 - incident_mu**2,  0.0))
    sin_scattered = np.sqrt(max(1.0 - scattered_mu**2, 0.0))

    cos_scattering_angle = (
        incident_mu * scattered_mu
        + sin_incident * sin_scattered * np.cos(incident_phi - scattered_phi)
    )
    # Clamp to [-1, 1] to guard against floating-point rounding
    cos_scattering_angle = np.clip(cos_scattering_angle, -1.0, 1.0)

    scattering_angle_beta = np.arccos(cos_scattering_angle)

    volume_scattering_phase_function = (
        (leaf_albedo / (3.0 * np.pi))
        * (np.sin(scattering_angle_beta)
           - scattering_angle_beta * np.cos(scattering_angle_beta))
        + (leaf_transmittance / 3.0) * np.cos(scattering_angle_beta)
    )
    return volume_scattering_phase_function


def precompute_gamma_solar_to_all_quad_directions(
        solar_mu, solar_phi,
        polar_cosines, azimuthal_angles,
        N, M, leaf_albedo, leaf_transmittance):
    """
    Pre-compute Gamma(Omega_solar -> Omega_{ij}) for every quadrature direction.

    Used in Step 6a (first collision source from the direct solar beam).
    The solar beam has one fixed direction, so we compute one row of the
    full phase-function table.

    Returns
    -------
    gamma_solar_to_quad : ndarray, shape (N+1, M+1)
        [i,j] = Gamma(Omega_solar -> Omega_{ij}),  index 0 unused.
    """
    gamma_solar_to_quad = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            gamma_solar_to_quad[i, j] = compute_volume_scattering_phase_function(
                solar_mu, solar_phi,
                polar_cosines[i], azimuthal_angles[j],
                leaf_albedo, leaf_transmittance
            )
    return gamma_solar_to_quad


def precompute_gamma_quad_to_quad(
        polar_cosines, azimuthal_angles,
        N, M, leaf_albedo, leaf_transmittance):
    """
    Pre-compute Gamma(Omega_{nm} -> Omega_{ij}) for every pair of directions.

    Used in Steps 6b and 6c (first collision source from upward-reflected
    and diffuse sky radiation).

    This is the most expensive step: N*M*N*M evaluations.
    For N=M=16 that is 65,536 calls.  We do it once and reuse everywhere.

    Returns
    -------
    gamma_quad_to_quad : ndarray, shape (N+1, M+1, N+1, M+1)
        [n,m,i,j] = Gamma(Omega_{nm} -> Omega_{ij}),  index 0 unused.
    """
    print("  Pre-computing Gamma(Omega_nm -> Omega_ij) for all direction pairs ...")
    gamma_quad_to_quad = np.zeros((N + 1, M + 1, N + 1, M + 1))
    for n in range(1, N + 1):
        for m in range(1, M + 1):
            for i in range(1, N + 1):
                for j in range(1, M + 1):
                    gamma_quad_to_quad[n, m, i, j] = (
                        compute_volume_scattering_phase_function(
                            polar_cosines[n], azimuthal_angles[m],
                            polar_cosines[i], azimuthal_angles[j],
                            leaf_albedo, leaf_transmittance
                        )
                    )
    print("  Done.\n")
    return gamma_quad_to_quad


# ==============================================================================
# SECTION 3 -- BOUNDARY CONDITIONS  (Lecture slides 4 and 14)
# ==============================================================================

def compute_upper_boundary_direct_solar_intensity(direct_fraction, solar_mu):
    """
    Upper boundary condition for the direct solar uncollided intensity.

    The solar beam enters the canopy top (L=0) as a collimated beam.
    From the lecture boundary condition (slide 4):

        I0_dir(L=0, Omega_solar) = I_solar = f_dir / |mu_solar|

    WHY DIVIDE BY |mu_solar|?
    -------------------------
    The solar beam carries a flux of f_dir * F_in [W/m2].
    To convert flux to intensity we divide by |mu_solar|:
        Intensity = Flux / |mu| = f_dir / |mu_solar|
    This accounts for the oblique angle of the beam hitting the canopy.

    Parameters
    ----------
    direct_fraction : float   f_dir, fraction of flux that is direct
    solar_mu        : float   mu_solar < 0 (downward direction)

    Returns
    -------
    float : intensity of the direct solar beam at the canopy top
    """
    return direct_fraction / abs(solar_mu)


def compute_upper_boundary_diffuse_sky_intensity(
        direct_fraction, polar_cosines, azimuthal_angles, N, M):
    """
    Upper boundary condition for diffuse sky uncollided intensity.

    The sky is assumed isotropic: every downward direction carries the
    same intensity Id (slide 4):

        I0_dif(L=0, Omega_{ij}) = Id = (1 - f_dir) / pi

    WHY DIVIDE BY pi?
    -----------------
    An isotropic downward hemisphere with intensity Id carries a total
    downward flux of pi * Id.  We want that flux to equal (1-f_dir)*F_in,
    so Id = (1-f_dir)/pi  (with F_in = 1 W/m2).

    Applied only to DOWNWARD directions: i = 1..N/2  (mu_i < 0).
    Upward slots at L=0 are zero -- no upward radiation enters from above.

    Returns
    -------
    diffuse_sky_intensity_at_top : ndarray, shape (N+1, M+1)
        [i,j] = Id  for downward directions i=1..N/2  (mu_i < 0)
              = 0   for upward   directions i=N/2+1..N (mu_i > 0)
        Index 0 is unused.
    isotropic_sky_intensity : float   Id = (1-f_dir)/pi
    """
    isotropic_sky_intensity = (1.0 - direct_fraction) / np.pi

    diffuse_sky_intensity_at_top = np.zeros((N + 1, M + 1))
    for i in range(1, N // 2 + 1):      # downward: mu_i < 0
        for j in range(1, M + 1):
            diffuse_sky_intensity_at_top[i, j] = isotropic_sky_intensity

    return diffuse_sky_intensity_at_top, isotropic_sky_intensity


# ==============================================================================
# SECTION 4 -- UNCOLLIDED INTENSITY SWEEPS  (Lecture slides 11-16)
# ==============================================================================

def sweep_direct_solar_beam_downward(
        direct_solar_intensity_at_top,
        solar_mu, extinction_G,
        layer_thickness_delta_L, num_layers_K):
    """
    Step 3a: Sweep the direct solar beam downward through the canopy.

    PHYSICS
    -------
    Moving from cell edge k to k+1 (going deeper into the canopy):

        I0_dir(L_{k+1}) = I0_dir(L_k) * exp[-G * delta_L / |mu_solar|]

    G * delta_L / |mu_solar| is the optical depth of one layer:
        G       = leaf area projection factor (0.5)
        delta_L = layer thickness in LAI units
        |mu_solar| = path-length factor (oblique beam traverses more leaf area)

    Done ONLY for the solar direction (slide 12).

    Parameters
    ----------
    direct_solar_intensity_at_top : float   I0_dir at L=0 (upper BC)
    solar_mu                      : float   mu_solar < 0 (downward)
    extinction_G                  : float   G = 0.5
    layer_thickness_delta_L       : float   delta_L = LAI / K
    num_layers_K                  : int     K

    Returns
    -------
    direct_solar_intensity_at_cell_edges : ndarray, shape (K+2,)
        [k] = intensity at cell edge k,  k = 1..K+1
        [0] = 0 (dummy, never used)
    """
    K = num_layers_K
    direct_solar_intensity_at_cell_edges = np.zeros(K + 2)

    # Upper boundary condition at canopy top, cell edge k=1 (slide 12)
    direct_solar_intensity_at_cell_edges[1] = direct_solar_intensity_at_top

    # Beer-Lambert attenuation factor for one layer
    one_layer_attenuation_factor = np.exp(
        -extinction_G * layer_thickness_delta_L / abs(solar_mu)
    )

    # Sweep downward: compute each edge from the one above it
    # k=1 -> k=2 -> k=3 -> ... -> k=K+1
    for k in range(1, K + 1):
        direct_solar_intensity_at_cell_edges[k + 1] = (
            direct_solar_intensity_at_cell_edges[k]
            * one_layer_attenuation_factor
        )

    return direct_solar_intensity_at_cell_edges


def sweep_diffuse_sky_radiation_downward(
        diffuse_sky_intensity_at_top,
        polar_cosines, N, M,
        extinction_G, layer_thickness_delta_L, num_layers_K):
    """
    Step 3b: Sweep diffuse sky radiation downward through the canopy.

    PHYSICS
    -------
    Each downward direction Omega_{ij} (i=1..N/2, mu_i < 0) sweeps
    independently through the canopy with its own path-length factor:

        I0_dif(L_{k+1}, Omega_{ij}) = I0_dif(L_k, Omega_{ij})
                                       * exp[-G * delta_L / |mu_i|]

    A near-horizontal ray (small |mu_i|) attenuates faster because it
    crosses more leaf area per unit depth.

    Done for ALL downward directions i = 1..N/2  (mu_i < 0).  Slide 13.
    Upward slots remain zero here and are filled later by the upward sweep.

    Parameters
    ----------
    diffuse_sky_intensity_at_top : ndarray (N+1, M+1)  upper BC
    polar_cosines                : ndarray (N+1,)
    N, M                         : int
    extinction_G                 : float
    layer_thickness_delta_L      : float
    num_layers_K                 : int

    Returns
    -------
    uncollided_intensity_at_cell_edges : ndarray, shape (N+1, M+1, K+2)
        [i, j, k] = intensity in direction Omega_{ij} at cell edge k
        Index 0 in every dimension is unused.
        Upward slots (i=N/2+1..N) are zero here; filled by sweep_up.
    """
    K = num_layers_K
    # Full 3D intensity array: (N+1) polar x (M+1) azimuthal x (K+2) edges
    uncollided_intensity_at_cell_edges = np.zeros((N + 1, M + 1, K + 2))

    # Upper BC at cell edge k=1 for downward directions only
    for i in range(1, N // 2 + 1):       # downward: mu_i < 0
        for j in range(1, M + 1):
            uncollided_intensity_at_cell_edges[i, j, 1] = (
                diffuse_sky_intensity_at_top[i, j]
            )

    # Sweep downward through K layers, one direction at a time
    for i in range(1, N // 2 + 1):       # downward directions only
        one_layer_attenuation_factor = np.exp(
            -extinction_G * layer_thickness_delta_L / abs(polar_cosines[i])
        )
        for j in range(1, M + 1):
            for k in range(1, K + 1):
                # Slide 13: I0_dif(L_{k+1}) = I0_dif(L_k) * exp[-G*dL/|mu_i|]
                uncollided_intensity_at_cell_edges[i, j, k + 1] = (
                    uncollided_intensity_at_cell_edges[i, j, k]
                    * one_layer_attenuation_factor
                )

    return uncollided_intensity_at_cell_edges


def apply_lambertian_ground_boundary_condition(
        direct_solar_intensity_at_cell_edges,
        uncollided_intensity_at_cell_edges,
        solar_mu, polar_cosines, polar_weights,
        azimuthal_weight, N, M, num_layers_K, ground_reflectance):
    """
    Step 4: Compute the ground reflection boundary condition.

    PHYSICS
    -------
    The ground is a Lambertian reflector with reflectance rho_g.
    It reflects all incident radiation equally in all upward directions.

    From slide 14, the reflected intensity at L=LAI for upward Omega_{ij}:

        I0(L_{K+1}, Omega_{ij}) =
            (rho_g/pi) * |mu_solar| * I0_dir(L_{K+1}, Omega_solar)   [direct]
          + (rho_g/pi) * sum_{n=1}^{N/2} sum_{m=1}^{M}
                           w_n * w_phi * |mu_n| * I0_dif(L_{K+1}, Omega_{nm})
                                                                      [diffuse]

    The SAME reflected_intensity value applies to all upward directions
    because a Lambertian surface reflects isotropically.

    Parameters
    ----------
    direct_solar_intensity_at_cell_edges : ndarray (K+2,)
    uncollided_intensity_at_cell_edges   : ndarray (N+1, M+1, K+2)
    solar_mu                             : float   (< 0, downward)
    polar_cosines, polar_weights         : ndarray (N+1,)
    azimuthal_weight                     : float
    N, M, num_layers_K                   : int
    ground_reflectance                   : float   rho_g

    Returns
    -------
    ground_reflected_intensity_at_bottom : ndarray (N+1, M+1)
        Upward reflected intensity at L=LAI, uniform over all upward directions.
    total_downward_flux_at_ground        : float  F_down(L=LAI)
    """
    K = num_layers_K

    # Downward flux from the direct solar beam at the ground (edge k=K+1)
    direct_flux_at_ground = (
        abs(solar_mu) * direct_solar_intensity_at_cell_edges[K + 1]
    )

    # Downward flux from diffuse sky photons at the ground
    # Sum over DOWNWARD directions n=1..N/2  (mu_n < 0)  -- slide 14
    diffuse_flux_at_ground = 0.0
    for n in range(1, N // 2 + 1):        # downward: mu_n < 0
        for m in range(1, M + 1):
            diffuse_flux_at_ground += (
                polar_weights[n]
                * azimuthal_weight
                * abs(polar_cosines[n])
                * uncollided_intensity_at_cell_edges[n, m, K + 1]
            )

    total_downward_flux_at_ground = direct_flux_at_ground + diffuse_flux_at_ground

    # Lambertian reflection: uniform reflected intensity in all upward directions
    reflected_intensity = (ground_reflectance / np.pi) * total_downward_flux_at_ground

    ground_reflected_intensity_at_bottom = np.zeros((N + 1, M + 1))
    for i in range(N // 2 + 1, N + 1):    # upward: mu_i > 0
        for j in range(1, M + 1):
            ground_reflected_intensity_at_bottom[i, j] = reflected_intensity

    return ground_reflected_intensity_at_bottom, total_downward_flux_at_ground


def sweep_ground_reflected_radiation_upward(
        ground_reflected_intensity_at_bottom,
        polar_cosines, N, M,
        extinction_G, layer_thickness_delta_L, num_layers_K,
        uncollided_intensity_at_cell_edges):
    """
    Step 5: Sweep ground-reflected radiation upward through the canopy.

    PHYSICS
    -------
    Starting from the reflected intensity at L=LAI (edge k=K+1), upward
    photons are attenuated as they travel back toward the canopy top.

    Going from cell edge k+1 to k (moving upward, decreasing L):

        I0(L_k, Omega_{ij}) = I0(L_{k+1}, Omega_{ij})
                               * exp[-G * delta_L / |mu_i|]

    Done for ALL UPWARD directions: i = N/2+1..N  (mu_i > 0).  Slide 16.

    We write directly into uncollided_intensity_at_cell_edges, filling
    the upward slots left empty by the two downward sweeps.

    Parameters
    ----------
    ground_reflected_intensity_at_bottom : ndarray (N+1, M+1)  lower BC
    polar_cosines                        : ndarray (N+1,)
    N, M, num_layers_K                   : int
    extinction_G                         : float
    layer_thickness_delta_L              : float
    uncollided_intensity_at_cell_edges   : ndarray (N+1, M+1, K+2)

    Returns
    -------
    uncollided_intensity_at_cell_edges : same array, upward slots filled
    """
    K = num_layers_K

    # Set lower boundary condition at k=K+1 for all upward directions
    for i in range(N // 2 + 1, N + 1):    # upward: mu_i > 0
        for j in range(1, M + 1):
            uncollided_intensity_at_cell_edges[i, j, K + 1] = (
                ground_reflected_intensity_at_bottom[i, j]
            )

    # Sweep upward: k=K -> k=K-1 -> ... -> k=1
    for i in range(N // 2 + 1, N + 1):    # upward directions: mu_i > 0
        one_layer_attenuation_factor = np.exp(
            -extinction_G * layer_thickness_delta_L / abs(polar_cosines[i])
        )
        for j in range(1, M + 1):
            for k in range(K, 0, -1):      # K, K-1, K-2, ..., 1
                # Slide 16: I0(L_k) = I0(L_{k+1}) * exp[-G*delta_L/|mu_i|]
                uncollided_intensity_at_cell_edges[i, j, k] = (
                    uncollided_intensity_at_cell_edges[i, j, k + 1]
                    * one_layer_attenuation_factor
                )

    return uncollided_intensity_at_cell_edges


# ==============================================================================
# SECTION 5 -- CELL-CENTRE INTENSITIES  (Lecture slide 18)
# ==============================================================================

def compute_cell_centre_intensity_1d(intensity_at_cell_edges, num_layers_K):
    """
    Convert cell-edge intensities to cell-centre intensities (1D array).

    WHY WE NEED THIS
    ----------------
    The First Collision Source Q is evaluated at CELL CENTRES (midpoint
    of each layer), but we store intensities at CELL EDGES.
    We approximate the cell-centre value as the mean of the two surrounding
    edges (slide 18):

        I(L_{k+0.5}) = 0.5 * [I(L_k) + I(L_{k+1})],   k = 1..K

    This gives K cell-centre values from the K+1 cell-edge values.

    Parameters
    ----------
    intensity_at_cell_edges : ndarray (K+2,)   edge values, index 1..K+1
    num_layers_K            : int

    Returns
    -------
    intensity_at_cell_centres : ndarray (K+1,)  centre values, index 1..K
        Index 0 is unused.
    """
    K = num_layers_K
    intensity_at_cell_centres = np.zeros(K + 1)
    for k in range(1, K + 1):
        intensity_at_cell_centres[k] = 0.5 * (
            intensity_at_cell_edges[k] + intensity_at_cell_edges[k + 1]
        )
    return intensity_at_cell_centres


def compute_cell_centre_intensity_3d(
        intensity_at_cell_edges_3d, N, M, num_layers_K):
    """
    Convert cell-edge intensities to cell-centre intensities (3D array).

    Same averaging as compute_cell_centre_intensity_1d(), applied to all
    N*M directions simultaneously:

        I_centre[i,j,k] = 0.5 * [I_edge[i,j,k] + I_edge[i,j,k+1]], k=1..K

    Parameters
    ----------
    intensity_at_cell_edges_3d : ndarray (N+1, M+1, K+2)

    Returns
    -------
    intensity_at_cell_centres_3d : ndarray (N+1, M+1, K+1), index k=1..K
    """
    K = num_layers_K
    intensity_at_cell_centres_3d = np.zeros((N + 1, M + 1, K + 1))
    for k in range(1, K + 1):
        intensity_at_cell_centres_3d[:, :, k] = 0.5 * (
            intensity_at_cell_edges_3d[:, :, k]
            + intensity_at_cell_edges_3d[:, :, k + 1]
        )
    return intensity_at_cell_centres_3d


# ==============================================================================
# SECTION 6 -- FIRST COLLISION SOURCE  (Lecture slides 18-20)
# ==============================================================================

def compute_first_collision_source(
        direct_solar_intensity_at_cell_edges,
        uncollided_intensity_at_cell_edges,
        solar_mu, solar_phi,
        polar_cosines, azimuthal_angles,
        polar_weights, azimuthal_weight,
        N, M, num_layers_K,
        gamma_solar_to_quad,
        gamma_quad_to_quad):
    """
    Steps 6a-6c: Compute the First Collision Source Q at every cell centre.

    PHYSICAL MEANING
    ----------------
    Q(L_{k+0.5}, Omega_{ij}) is the rate at which uncollided photons are
    scattered INTO direction Omega_{ij} per unit volume at cell centre k+0.5.
    This is the "birth rate" of collided photons, and becomes the source term
    in the collided radiation equation solved in Part Two.

    THREE CONTRIBUTIONS  (slides 18, 19, 20)
    -----------------------------------------

    Q1  (slide 18) -- Step 6a: from the direct solar beam
        The solar beam is one collimated direction, so no quadrature sum needed:

            Q1[k+1/2, Omega_{ij}] = (1/pi) * Gamma(Omega_solar -> Omega_{ij})
                                            * I0_dir_centre[k+1/2]

        Source: solar direction Omega_solar only.
        Output: ALL directions Omega_{ij}  (i=1..N, both up and down).

    Q2  (slide 19) -- Step 6b: from DOWNWARD diffuse sky photons
        Sum over DOWNWARD source directions n=1..N/2  (mu_n < 0):

            Q2[k+1/2, Omega_{ij}] = (1/pi)
                * sum_{n=1}^{N/2} sum_{m=1}^{M}
                    w_n * w_phi * Gamma(Omega_{nm} -> Omega_{ij})
                                * I0_dif_centre[n,m,k+1/2]

        Source: downward quadrature directions n=1..N/2  (mu_n < 0).
        Output: ALL directions Omega_{ij}  (i=1..N, both up and down).
        Slide 19 note: "The integration done only for downward directions."

    Q3  (slide 20) -- Step 6c: from UPWARD ground-reflected photons
        Sum over UPWARD source directions n=N/2+1..N  (mu_n > 0):

            Q3[k+1/2, Omega_{ij}] = (1/pi)
                * sum_{n=N/2+1}^{N} sum_{m=1}^{M}
                    w_n * w_phi * Gamma(Omega_{nm} -> Omega_{ij})
                                * I0_dif_centre[n,m,k+1/2]

        Source: upward quadrature directions n=N/2+1..N  (mu_n > 0).
        Output: ALL directions Omega_{ij}  (i=1..N, both up and down).
        Slide 20 note: "The integration done for upward directions."

    TOTAL:  Q = Q1 + Q2 + Q3

    Parameters
    ----------
    direct_solar_intensity_at_cell_edges : ndarray (K+2,)
    uncollided_intensity_at_cell_edges   : ndarray (N+1, M+1, K+2)
    solar_mu, solar_phi                  : float
    polar_cosines, azimuthal_angles      : ndarray
    polar_weights                        : ndarray (N+1,)
    azimuthal_weight                     : float
    N, M, num_layers_K                   : int
    gamma_solar_to_quad                  : ndarray (N+1, M+1)
    gamma_quad_to_quad                   : ndarray (N+1, M+1, N+1, M+1)

    Returns
    -------
    first_collision_source       : ndarray (N+1, M+1, K+1)  Q  = Q1+Q2+Q3
    first_collision_source_Q1    : ndarray (N+1, M+1, K+1)  from direct solar beam
    first_collision_source_Q2    : ndarray (N+1, M+1, K+1)  from downward diffuse sky
    first_collision_source_Q3    : ndarray (N+1, M+1, K+1)  from upward ground-reflected
    """
    K = num_layers_K

    # Convert edge intensities to cell-centre values (slide 18 formula)
    direct_solar_intensity_at_cell_centres = compute_cell_centre_intensity_1d(
        direct_solar_intensity_at_cell_edges, K
    )
    uncollided_intensity_at_cell_centres = compute_cell_centre_intensity_3d(
        uncollided_intensity_at_cell_edges, N, M, K
    )

    first_collision_source_Q1 = np.zeros((N + 1, M + 1, K + 1))
    first_collision_source_Q2 = np.zeros((N + 1, M + 1, K + 1))
    first_collision_source_Q3 = np.zeros((N + 1, M + 1, K + 1))

    for k in range(1, K + 1):        # loop over K cell centres

        # ── Q1: direct solar -> all output directions  (slide 18) ────────────
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                first_collision_source_Q1[i, j, k] = (
                    (1.0 / np.pi)
                    * gamma_solar_to_quad[i, j]
                    * direct_solar_intensity_at_cell_centres[k]
                )

        # ── Q2: downward diffuse sky -> ALL output directions  (slide 19) ──────
        # Source directions: n = 1..N/2  (downward, mu_n < 0)
        # Output directions: i = 1..N   (ALL directions, slide 19: "all directions in a sphere")
        # Slide 19 note: "The integration done only for downward directions"
        #   -- this means the SOURCE sum runs over downward n only,
        #      NOT that the output is restricted to downward.
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                weighted_sum = 0.0
                for n in range(1, N // 2 + 1):        # downward source: mu_n < 0
                    for m in range(1, M + 1):
                        weighted_sum += (
                            polar_weights[n]
                            * azimuthal_weight
                            * gamma_quad_to_quad[n, m, i, j]
                            * uncollided_intensity_at_cell_centres[n, m, k]
                        )
                first_collision_source_Q2[i, j, k] = (1.0 / np.pi) * weighted_sum

        # ── Q3: upward ground-reflected -> ALL output directions  (slide 20) ──
        # Source directions: n = N/2+1..N  (upward, mu_n > 0)
        # Output directions: i = 1..N     (ALL directions, slide 20: "all directions in a sphere")
        # Slide 20 note: "The integration done for upward directions"
        #   -- this means the SOURCE sum runs over upward n only.
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                weighted_sum = 0.0
                for n in range(N // 2 + 1, N + 1):    # upward source: mu_n > 0
                    for m in range(1, M + 1):
                        weighted_sum += (
                            polar_weights[n]
                            * azimuthal_weight
                            * gamma_quad_to_quad[n, m, i, j]
                            * uncollided_intensity_at_cell_centres[n, m, k]
                        )
                first_collision_source_Q3[i, j, k] = (1.0 / np.pi) * weighted_sum

    first_collision_source = (
        first_collision_source_Q1
        + first_collision_source_Q2
        + first_collision_source_Q3
    )
    return (first_collision_source,
            first_collision_source_Q1,
            first_collision_source_Q2,
            first_collision_source_Q3)


# ==============================================================================
# SECTION 7 -- FLUX INTEGRATION
# ==============================================================================

def integrate_fluxes_over_hemisphere(
        direct_solar_intensity_at_cell_edges,
        uncollided_intensity_at_cell_edges,
        solar_mu, polar_cosines, polar_weights,
        azimuthal_weight, N, M, num_layers_K):
    """
    Compute downward and upward uncollided fluxes at every cell edge.

    PHYSICS
    -------
    Flux = integral of |mu| * I * dOmega over the relevant hemisphere.
    Approximated using Gauss quadrature:

    Downward direct flux:
        F_dir_down[k] = |mu_solar| * I0_dir[k]
        (only one direction -- the collimated solar beam)

    Downward diffuse flux:
        F_dif_down[k] = sum_{i=1}^{N/2} sum_{j=1}^{M}
                         w_i * w_phi * |mu_i| * I0_dif[i,j,k]
        (downward directions: i=1..N/2, mu_i < 0)

    Upward flux:
        F_up[k] = sum_{i=N/2+1}^{N} sum_{j=1}^{M}
                   w_i * w_phi * |mu_i| * I0_dif[i,j,k]
        (upward directions: i=N/2+1..N, mu_i > 0)

    Returns
    -------
    downward_direct_flux_at_cell_edges  : ndarray (K+2,)  index 1..K+1
    downward_diffuse_flux_at_cell_edges : ndarray (K+2,)  index 1..K+1
    upward_flux_at_cell_edges           : ndarray (K+2,)  index 1..K+1
    """
    K = num_layers_K
    downward_direct_flux_at_cell_edges  = np.zeros(K + 2)
    downward_diffuse_flux_at_cell_edges = np.zeros(K + 2)
    upward_flux_at_cell_edges           = np.zeros(K + 2)

    for k in range(1, K + 2):    # k = 1..K+1

        # Direct solar: single beam
        downward_direct_flux_at_cell_edges[k] = (
            abs(solar_mu) * direct_solar_intensity_at_cell_edges[k]
        )

        # Downward diffuse: sum over downward directions (mu_i < 0)
        for i in range(1, N // 2 + 1):
            for j in range(1, M + 1):
                downward_diffuse_flux_at_cell_edges[k] += (
                    polar_weights[i]
                    * azimuthal_weight
                    * abs(polar_cosines[i])
                    * uncollided_intensity_at_cell_edges[i, j, k]
                )

        # Upward: sum over upward directions (mu_i > 0)
        for i in range(N // 2 + 1, N + 1):
            for j in range(1, M + 1):
                upward_flux_at_cell_edges[k] += (
                    polar_weights[i]
                    * azimuthal_weight
                    * abs(polar_cosines[i])
                    * uncollided_intensity_at_cell_edges[i, j, k]
                )

    return (downward_direct_flux_at_cell_edges,
            downward_diffuse_flux_at_cell_edges,
            upward_flux_at_cell_edges)


# ==============================================================================
# SECTION 8 -- BIDIRECTIONAL REFLECTANCE FACTOR  (BRF)
# ==============================================================================

def compute_BRF_at_canopy_top_principal_plane(
        uncollided_intensity_at_cell_edges,
        polar_cosines, azimuthal_angles,
        N, M, total_incident_flux, solar_phi):
    """
    Compute the uncollided BRF at the canopy top (L=0) in the principal plane.

    DEFINITION
    ----------
    BRF is the ratio of the upward intensity to what a perfect white
    Lambertian surface would reflect under the same illumination:

        BRF(Omega_view) = I_up(L=0, Omega_view) / (F_in / pi)

    We evaluate this along the PRINCIPAL PLANE containing the sun:
        Forward scatter side: phi_view = phi_solar      (looking away from sun)
        Back    scatter side: phi_view = phi_solar + pi  (looking toward sun)

    The plot x-axis runs from -90 deg (backscatter) to +90 deg (forward).

    Upward directions: i = N/2+1..N  (mu_i > 0)
    View zenith angle: theta_view = arccos(mu_i) in [0, 90] degrees

    Returns
    -------
    view_zenith_angles_deg : ndarray  0..90 degrees, sorted ascending
    BRF_forward_scatter    : ndarray  BRF at phi = phi_solar  (forward)
    BRF_back_scatter       : ndarray  BRF at phi = phi_solar+pi (back)
    """
    upward_polar_cosines  = polar_cosines[N//2+1 : N+1]
    view_zenith_angles_deg = np.degrees(np.arccos(upward_polar_cosines))

    sort_order = np.argsort(view_zenith_angles_deg)
    view_zenith_angles_deg = view_zenith_angles_deg[sort_order]

    BRF_forward_scatter = np.zeros(N // 2)
    BRF_back_scatter    = np.zeros(N // 2)

    brf_normalisation = total_incident_flux / np.pi

    phi_forward = solar_phi % (2.0 * np.pi)
    phi_back    = (solar_phi + np.pi) % (2.0 * np.pi)

    angular_distance_to_forward = np.array(
        [abs(azimuthal_angles[j] - phi_forward) for j in range(1, M + 1)]
    )
    angular_distance_to_back = np.array(
        [abs(azimuthal_angles[j] - phi_back) for j in range(1, M + 1)]
    )
    j_forward = int(np.argmin(angular_distance_to_forward)) + 1
    j_back    = int(np.argmin(angular_distance_to_back))    + 1

    for idx, i in enumerate(range(N // 2 + 1, N + 1)):
        BRF_forward_scatter[idx] = (
            uncollided_intensity_at_cell_edges[i, j_forward, 1]
            / brf_normalisation
        )
        BRF_back_scatter[idx] = (
            uncollided_intensity_at_cell_edges[i, j_back, 1]
            / brf_normalisation
        )

    BRF_forward_scatter = BRF_forward_scatter[sort_order]
    BRF_back_scatter    = BRF_back_scatter[sort_order]

    return view_zenith_angles_deg, BRF_forward_scatter, BRF_back_scatter


# ==============================================================================
# SECTION 9 -- MASTER SOLVER
# ==============================================================================

def run_uncollided_dom_solver(canopy_params, leaf_params,
                               num_polar_directions=16,
                               num_azimuthal_directions=16,
                               num_canopy_layers=50):
    """
    Master function: run all steps of the uncollided DOM calculation.

    Calls every step in the order shown on the lecture slides and returns
    all results in a dictionary.  The plotting function uses this dictionary.

    Parameters
    ----------
    canopy_params : dict with keys:
        'LAI'         -- Leaf Area Index
        'f_dir'       -- fraction of incident flux that is direct [0..1]
        'theta_o_deg' -- solar zenith angle [degrees] from the upward zenith
                         MUST be > 90 for a downward sun
        'phi_o_deg'   -- solar azimuthal angle [degrees]
        'rho_g'       -- ground reflectance [0..1]
        'F_in'        -- total incident flux [W/m2] (use 1.0 for normalised)

    leaf_params : dict with keys:
        'rho_L'  -- leaf reflectance
        'tau_L'  -- leaf transmittance

    num_polar_directions     : int  N (must be even; default 16)
    num_azimuthal_directions : int  M (default 16)
    num_canopy_layers        : int  K (default 50)

    Returns
    -------
    dict with all computed arrays (keys listed in the return statement)
    """
    print("=" * 62)
    print("  DOM UNCOLLIDED RADIATION SOLVER")
    print("=" * 62)

    # Unpack inputs
    LAI         = canopy_params['LAI']
    f_dir       = canopy_params['f_dir']
    theta_o_deg = canopy_params['theta_o_deg']
    phi_o_deg   = canopy_params['phi_o_deg']
    rho_g       = canopy_params['rho_g']
    F_in        = canopy_params['F_in']
    rho_L       = leaf_params['rho_L']
    tau_L       = leaf_params['tau_L']
    leaf_albedo = rho_L + tau_L     # omega_L = rho_L + tau_L

    N = num_polar_directions
    M = num_azimuthal_directions
    K = num_canopy_layers

    # Solar direction: mu = cos(theta_o) measured from upward zenith
    # theta_o = 140 deg -> mu_solar = cos(140 deg) = -0.766  (downward, correct)
    solar_theta_rad = np.radians(theta_o_deg)
    solar_phi_rad   = np.radians(phi_o_deg)
    solar_mu        = np.cos(solar_theta_rad)

    if solar_mu >= 0:
        raise ValueError(
            f"solar_mu = {solar_mu:.3f} is not negative. "
            f"theta_o_deg must be > 90 for a downward sun. "
            f"Got theta_o_deg = {theta_o_deg}."
        )

    print(f"\n  Inputs:")
    print(f"    LAI={LAI},  K={K} layers,  delta_L={LAI/K:.4f}")
    print(f"    f_dir={f_dir},  theta_solar={theta_o_deg} deg,  phi_solar={phi_o_deg} deg")
    print(f"    mu_solar = cos({theta_o_deg} deg) = {solar_mu:.4f}  (< 0, downward OK)")
    print(f"    rho_g={rho_g},  rho_L={rho_L},  tau_L={tau_L},  omega_L={leaf_albedo:.4f}")
    print(f"    N={N} polar,  M={M} azimuthal\n")

    # Step 1: Angular discretization (slides 7-9)
    print("  Step 1: Angular discretization ...")
    polar_cosines, azimuthal_angles, polar_weights, azimuthal_weight = (
        setup_angular_quadrature_grid(N, M)
    )
    print_quadrature_sanity_checks(polar_cosines, polar_weights, N)

    # Step 2: Spatial discretization (slide 10)
    print("  Step 2: Spatial discretization ...")
    layer_thickness_delta_L = LAI / K
    cell_edge_LAI_values = np.zeros(K + 2)
    for k in range(1, K + 2):
        cell_edge_LAI_values[k] = (k - 1) * layer_thickness_delta_L
    print(f"    delta_L={layer_thickness_delta_L:.4f},  "
          f"L[k=1]={cell_edge_LAI_values[1]:.3f} (top),  "
          f"L[k={K+1}]={cell_edge_LAI_values[K+1]:.3f} (bottom)\n")

    # Cross sections
    extinction_G = compute_extinction_cross_section()
    print(f"  Extinction cross section G = {extinction_G}  (uniform leaf normals)\n")

    print("  Pre-computing phase functions ...")
    gamma_solar_to_quad = precompute_gamma_solar_to_all_quad_directions(
        solar_mu, solar_phi_rad,
        polar_cosines, azimuthal_angles,
        N, M, leaf_albedo, tau_L
    )
    gamma_quad_to_quad = precompute_gamma_quad_to_quad(
        polar_cosines, azimuthal_angles,
        N, M, leaf_albedo, tau_L
    )

    # Boundary condition intensities
    print("  Boundary conditions:")
    direct_solar_intensity_at_top = (
        compute_upper_boundary_direct_solar_intensity(f_dir, solar_mu)
    )
    diffuse_sky_intensity_at_top, isotropic_sky_intensity = (
        compute_upper_boundary_diffuse_sky_intensity(
            f_dir, polar_cosines, azimuthal_angles, N, M
        )
    )
    print(f"    I_solar (direct beam)   = {direct_solar_intensity_at_top:.4f} W/m2/sr")
    print(f"    I_diffuse (isotropic sky) = {isotropic_sky_intensity:.4f} W/m2/sr\n")

    # Step 3a: Direct solar sweep down (slide 12)
    print("  Step 3a: Sweeping direct solar beam downward ...")
    direct_solar_intensity_at_cell_edges = sweep_direct_solar_beam_downward(
        direct_solar_intensity_at_top,
        solar_mu, extinction_G,
        layer_thickness_delta_L, K
    )

    # Step 3b: Diffuse sky sweep down (slide 13)
    print("  Step 3b: Sweeping diffuse sky radiation downward ...")
    uncollided_intensity_at_cell_edges = sweep_diffuse_sky_radiation_downward(
        diffuse_sky_intensity_at_top,
        polar_cosines, N, M,
        extinction_G, layer_thickness_delta_L, K
    )

    # Step 4: Ground boundary condition (slide 14)
    print("  Step 4: Applying ground boundary condition ...")
    ground_reflected_intensity_at_bottom, total_downward_flux_at_ground = (
        apply_lambertian_ground_boundary_condition(
            direct_solar_intensity_at_cell_edges,
            uncollided_intensity_at_cell_edges,
            solar_mu, polar_cosines, polar_weights,
            azimuthal_weight, N, M, K, rho_g
        )
    )
    reflected_intensity_value = (rho_g / np.pi) * total_downward_flux_at_ground
    print(f"    F_down at ground = {total_downward_flux_at_ground:.4f} W/m2  "
          f"(normalised = {total_downward_flux_at_ground/F_in:.4f})")
    print(f"    Reflected intensity = {reflected_intensity_value:.6f} W/m2/sr\n")

    # Step 5: Ground-reflected sweep up (slide 16)
    print("  Step 5: Sweeping ground-reflected radiation upward ...")
    uncollided_intensity_at_cell_edges = sweep_ground_reflected_radiation_upward(
        ground_reflected_intensity_at_bottom,
        polar_cosines, N, M,
        extinction_G, layer_thickness_delta_L, K,
        uncollided_intensity_at_cell_edges
    )
    print("    Uncollided intensities at all cell edges computed\n")

    # Step 6: First Collision Source (slides 18-20)
    print("  Step 6: Computing First Collision Source Q = Q1 + Q2 + Q3 ...")
    (first_collision_source,
     first_collision_source_Q1,
     first_collision_source_Q2,
     first_collision_source_Q3) = compute_first_collision_source(
        direct_solar_intensity_at_cell_edges,
        uncollided_intensity_at_cell_edges,
        solar_mu, solar_phi_rad,
        polar_cosines, azimuthal_angles,
        polar_weights, azimuthal_weight,
        N, M, K,
        gamma_solar_to_quad,
        gamma_quad_to_quad
    )
    print("    First Collision Source Q computed\n")

    # Flux integration
    print("  Integrating fluxes ...")
    (downward_direct_flux_at_cell_edges,
     downward_diffuse_flux_at_cell_edges,
     upward_flux_at_cell_edges) = integrate_fluxes_over_hemisphere(
        direct_solar_intensity_at_cell_edges,
        uncollided_intensity_at_cell_edges,
        solar_mu, polar_cosines, polar_weights,
        azimuthal_weight, N, M, K
    )
    normalised_direct_downward_flux  = downward_direct_flux_at_cell_edges  / F_in
    normalised_diffuse_downward_flux = downward_diffuse_flux_at_cell_edges / F_in
    normalised_upward_flux           = upward_flux_at_cell_edges           / F_in

    # BRF at canopy top
    print("  Computing BRF in the principal plane ...")
    (view_zenith_angles_deg,
     BRF_forward_scatter,
     BRF_back_scatter) = compute_BRF_at_canopy_top_principal_plane(
        uncollided_intensity_at_cell_edges,
        polar_cosines, azimuthal_angles,
        N, M, F_in, solar_phi_rad
    )

    print("\n  All computations complete\n")

    return {
        'cell_edge_LAI_values'                   : cell_edge_LAI_values,
        'layer_thickness_delta_L'                : layer_thickness_delta_L,
        'polar_cosines'                          : polar_cosines,
        'azimuthal_angles'                       : azimuthal_angles,
        'polar_weights'                          : polar_weights,
        'azimuthal_weight'                       : azimuthal_weight,
        'direct_solar_intensity_at_cell_edges'   : direct_solar_intensity_at_cell_edges,
        'uncollided_intensity_at_cell_edges'     : uncollided_intensity_at_cell_edges,
        'first_collision_source'                 : first_collision_source,
        'first_collision_source_Q1'              : first_collision_source_Q1,
        'first_collision_source_Q2'              : first_collision_source_Q2,
        'first_collision_source_Q3'              : first_collision_source_Q3,
        'normalised_direct_downward_flux'        : normalised_direct_downward_flux,
        'normalised_diffuse_downward_flux'       : normalised_diffuse_downward_flux,
        'normalised_upward_flux'                 : normalised_upward_flux,
        'view_zenith_angles_deg'                 : view_zenith_angles_deg,
        'BRF_forward_scatter'                    : BRF_forward_scatter,
        'BRF_back_scatter'                       : BRF_back_scatter,
        'solar_mu'                               : solar_mu,
        'solar_phi_rad'                          : solar_phi_rad,
        'LAI'                                    : LAI,
        'K'                                      : K,
        'N'                                      : N,
        'M'                                      : M,
    }


# ==============================================================================
# SECTION 10 -- PLOTTING
# ==============================================================================

def make_three_panel_plots(results_LAI1, results_LAI2,
                            wavelength_band_label, LAI_values):
    """
    Produce the three required plots (A, B, C) for one wavelength band.

    Plot A -- Normalised downward uncollided flux vs L/LAI
    Plot B -- Normalised upward   uncollided flux vs L/LAI
    Plot C -- Uncollided BRF at canopy top, principal plane

    Parameters
    ----------
    results_LAI1, results_LAI2 : dicts from run_uncollided_dom_solver()
    wavelength_band_label       : str   e.g. 'RED' or 'NIR'
    LAI_values                  : list  e.g. [1.5, 4.0]
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Uncollided Radiation -- {wavelength_band_label} band",
        fontsize=14, fontweight='bold'
    )

    panel_colors    = ['tab:blue', 'tab:orange']
    panel_linestyle = ['-', '--']

    for idx, (res, LAI) in enumerate(
            zip([results_LAI1, results_LAI2], LAI_values)):

        K      = res['K']
        L      = res['cell_edge_LAI_values'][1:K+2]
        L_norm = L / LAI    # normalised depth: 0 (canopy top) to 1 (bottom)

        # Plot A: Normalised downward flux
        ax = axes[0]
        ax.plot(L_norm,
                res['normalised_direct_downward_flux'][1:K+2],
                color=panel_colors[idx], ls='-',
                label=f"Direct,      LAI={LAI}")
        ax.plot(L_norm,
                res['normalised_diffuse_downward_flux'][1:K+2],
                color=panel_colors[idx], ls='dotted',
                label=f"Diffuse sky, LAI={LAI}")
        ax.set_xlabel("Normalised depth  L / LAI")
        ax.set_ylabel("Normalised flux  F / F_in")
        ax.set_title("Plot A: Uncollided downward flux")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot B: Normalised upward flux
        ax = axes[1]
        ax.plot(L_norm,
                res['normalised_upward_flux'][1:K+2],
                color=panel_colors[idx], ls=panel_linestyle[idx],
                label=f"Upward reflected, LAI={LAI}")
        ax.set_xlabel("Normalised depth  L / LAI")
        ax.set_ylabel("Normalised flux  F / F_in")
        ax.set_title("Plot B: Uncollided upward flux")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Plot C: BRF in the principal plane
        ax = axes[2]
        theta_deg = res['view_zenith_angles_deg']
        ax.plot(-theta_deg[::-1], res['BRF_back_scatter'][::-1],
                color=panel_colors[idx], ls=panel_linestyle[idx],
                label=f"LAI={LAI}")
        ax.plot( theta_deg,        res['BRF_forward_scatter'],
                color=panel_colors[idx], ls=panel_linestyle[idx])
        ax.set_xlabel(
            "View zenith angle (degrees)\n<-- backscatter   |   forward scatter -->"
        )
        ax.set_ylabel("BRF")
        ax.set_title("Plot C: Uncollided BRF at L=0\n(principal plane)")
        ax.axvline(0, color='grey', lw=0.8, ls=':')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ==============================================================================
# SECTION 11 -- MAIN ENTRY POINT: SET YOUR INPUTS HERE
# ==============================================================================

if __name__ == "__main__":
    """
    =========================================================================
    STUDENT ENTRY POINT -- MODIFY INPUTS IN THIS BLOCK ONLY
    =========================================================================

    All physical inputs are defined below in clearly named variables.
    Change any of these to explore how the results change.
    Do not change anything outside this block unless you understand the code.

    QUICK-START GUIDE
    -----------------
    1. Set your desired N, M, K (resolution).  Start small to run fast.
    2. Set the solar geometry (theta, phi) and canopy properties (LAI, rho_g).
    3. Set the leaf optical properties for each spectral band.
    4. Run the script: python dom_uncollided.py
    5. Check the output PNG figures.

    RESOLUTION GUIDE
    ----------------
    Small  (fast,  ~seconds): N=4,  M=4,  K=20  -- good for testing
    Medium (balanced):         N=8,  M=8,  K=50  -- reasonable accuracy
    Large  (accurate, slow):   N=16, M=16, K=100 -- lecture-quality results
    """

    # ── Angular and spatial resolution ───────────────────────────────────────
    N = 16    # Number of Gauss-Legendre polar directions  (must be even)
    M = 16    # Number of uniform azimuthal directions
    K = 50    # Number of canopy layers

    # ── Radiation field ───────────────────────────────────────────────────────
    total_incident_flux_F_in  = 1.0   # W/m2  (normalised: all outputs are F/F_in)
    direct_fraction_f_dir     = 0.7   # 70% direct sun,  30% diffuse sky

    # ── Solar geometry ────────────────────────────────────────────────────────
    # theta_solar is measured from the UPWARD zenith (-z axis).
    # It MUST be greater than 90 degrees for a downward-going sun.
    # Example: 140 degrees from zenith = 40 degrees above the horizon.
    solar_zenith_angle_from_zenith_deg = 140.0
    solar_azimuthal_angle_deg          = 0.0

    # ── Canopy structure ──────────────────────────────────────────────────────
    # We compute results for BOTH LAI values and plot them together.
    LAI_values_to_compute = [1.5, 4.0]

    # ── Spectral bands ────────────────────────────────────────────────────────
    # Each band has its own leaf reflectance (rho_L), leaf transmittance (tau_L),
    # and ground reflectance (rho_g).
    # Leaf albedo omega_L = rho_L + tau_L is computed automatically.
    spectral_bands = {
        'RED': {
            'leaf'  : {'rho_L': 0.06,   'tau_L': 0.04},   # leaves absorb most RED
            'rho_g' : 0.10,                                 # dark soil
        },
        'NIR': {
            'leaf'  : {'rho_L': 0.525,  'tau_L': 0.45},   # leaves nearly transparent
            'rho_g' : 0.20,                                 # brighter soil in NIR
        },
    }

    # =========================================================================
    # DO NOT MODIFY BELOW THIS LINE
    # =========================================================================

    for band_label, band_data in spectral_bands.items():

        print(f"\n{'#'*62}")
        print(f"#  SPECTRAL BAND: {band_label}")
        print(f"{'#'*62}")

        results_for_each_LAI = []

        for LAI in LAI_values_to_compute:

            canopy_params = {
                'LAI'         : LAI,
                'f_dir'       : direct_fraction_f_dir,
                'theta_o_deg' : solar_zenith_angle_from_zenith_deg,
                'phi_o_deg'   : solar_azimuthal_angle_deg,
                'rho_g'       : band_data['rho_g'],
                'F_in'        : total_incident_flux_F_in,
            }

            results = run_uncollided_dom_solver(
                canopy_params,
                band_data['leaf'],
                num_polar_directions=N,
                num_azimuthal_directions=M,
                num_canopy_layers=K
            )
            results_for_each_LAI.append(results)

        figure = make_three_panel_plots(
            results_for_each_LAI[0],
            results_for_each_LAI[1],
            band_label,
            LAI_values_to_compute
        )
        output_filename = f"uncollided_dom_{band_label}.png"
        figure.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"  Figure saved: {output_filename}  (saved in current working directory)")
        plt.close(figure)

    print("\n" + "=" * 62)
    print("  All bands complete.")
    print("=" * 62)
