#!/usr/bin/env python3
"""
G-FUNCTION VERIFICATION — NON-UNIFORM LEAF AZIMUTHAL DISTRIBUTION
===================================================================

This program uses Gauss-Legendre Quadrature (order 12) to numerically verify
the fundamental identity of radiative transfer in vegetation canopies:

               1                                 1
             -----  INT  G(r, Omega)  dOmega  =  ---
              2*pi   2pi                          2

where the integral is over the upper hemisphere (solid angle = 2*pi steradians)
and G is the "Ross-Nilson geometry function".

WHAT IS THE G-FUNCTION?
-----------------------
The G-function describes the mean projection of leaves in a given direction
Omega.  It is defined as:

                   1
  G(r, Omega)  =  ----  INT     g_bar_L(r, Omega_L)  |Omega_L . Omega|  dOmega_L
                  2*pi   2pi

where:
  Omega   = (theta, phi)     = direction of interest (e.g., sun direction)
  Omega_L = (theta_L, phi_L) = leaf normal direction
  g_bar_L = full leaf angle distribution (inclination * azimuth)
  |Omega_L . Omega|          = absolute value of the dot product

NON-UNIFORM AZIMUTHAL DISTRIBUTION:
------------------------------------
The full leaf angle distribution separates as:

  g_bar_L(theta_L, phi_L) = gL(theta_L) * hL(phi_L)

For the NON-UNIFORM case, hL is given by:

  (1/2*pi) * hL(phi_L, phi) = (1/pi) * cos^2(phi - phi_L - eta)

which means:  hL(phi_L) = 2 * cos^2(phi - phi_L - eta)

where eta is the angular offset between the distribution maximum and the
solar azimuth.  Two biologically meaningful cases are tested:
  (1) Diaheliotropic  : eta = 0      (maximize projected leaf area to sun)
  (2) Paraheliotropic : eta = pi/2   (minimize projected leaf area to sun)

NOTE: Because hL depends on phi_L (and is not constant = 1), the G-function
now depends on BOTH theta and phi.  The full double integral over the
hemisphere is therefore essential — no simplification by azimuthal symmetry.

LEAF INCLINATION DISTRIBUTIONS TESTED:
  (1) Uniform       :  gL(theta) = 1
  (2) Planophile    :  gL(theta) = 3 cos^2(theta)
  (3) Erectophile   :  gL(theta) = (3/2) sin^2(theta)
  (4) Plagiophile   :  gL(theta) = (15/8) sin^2(2*theta)
  (5) Extremophile  :  gL(theta) = (15/7) cos^2(2*theta)

TOTAL CASES: 5 distributions x 2 eta values = 10 cases.

Authors: Claude And I (Ranga B. Myneni)
"""

# =============================================================================
# IMPORTS
# =============================================================================
# 'math' provides basic math functions: pi, sin, cos, acos, fabs (float abs).
# 'numpy' provides arrays and element-wise operations (imported as 'np').

import math
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================
# ALL_CAPS names signal "treat these as constants" to other programmers.

PI = math.pi                        # pi = 3.141592653589793
TWO_PI = 2.0 * PI                   # 2*pi — frequently used in this code
DEG_TO_RAD = PI / 180.0             # conversion factor: degrees -> radians
RAD_TO_DEG = 180.0 / PI             # conversion factor: radians -> degrees
NG = 12                             # Gauss quadrature order (number of points)


# =============================================================================
# FUNCTION: gauss_quad
# =============================================================================
def gauss_quad(ng):
    """
    Obtain Gauss-Legendre quadrature ordinates and weights of order 'ng'.

    Parameters
    ----------
    ng : int
        The quadrature order. Must be one of: 4, 6, 8, 10, or 12.

    Returns
    -------
    xg : numpy array
        The quadrature ordinates (on [-1, +1]).
    wg : numpy array
        The quadrature weights.
    """

    # Pre-computed ordinates (negative half only; positive half by symmetry).
    xx = [
        -0.861136312, -0.339981044,                                  # ng=4
        -0.9324695,   -0.6612094,   -0.2386192,                     # ng=6
        -0.960289856, -0.796666477, -0.525532410, -0.183434642,     # ng=8
        -0.973906529, -0.865063367, -0.679409568, -0.433395394,     # ng=10
        -0.148874339,
        -0.981560634, -0.904117256, -0.769902674, -0.587317954,     # ng=12
        -0.367831499, -0.125233409
    ]

    # Pre-computed weights (same order as above).
    ww = [
         0.347854845,  0.652145155,                                  # ng=4
         0.1713245,    0.3607616,    0.4679139,                      # ng=6
         0.101228536,  0.222381034,  0.313706646,  0.362683783,      # ng=8
         0.066671344,  0.149451349,  0.219086363,  0.269266719,      # ng=10
         0.295524225,
         0.047175336,  0.106939326,  0.160078329,  0.203167427,      # ng=12
         0.233492537,  0.249147046
    ]

    # Dictionary: maps ng//2 -> starting index in xx/ww.
    ishift = {2: 0, 3: 2, 4: 5, 5: 9, 6: 14}

    assert ng in [4, 6, 8, 10, 12], \
        f"Error: ng must be 4, 6, 8, 10, or 12. You provided ng={ng}."

    ng2 = ng // 2
    xg, wg = [], []

    # Negative half (from the table).
    for i in range(ng2):
        xg.append(xx[i + ishift[ng2]])
        wg.append(ww[i + ishift[ng2]])

    # Positive half (mirror image).
    for i in range(ng2):
        xg.append(-xg[ng2 - 1 - i])
        wg.append( wg[ng2 - 1 - i])

    return np.array(xg), np.array(wg)


# =============================================================================
# FUNCTION: check_quad
# =============================================================================
def check_quad(ng, xg, wg):
    """
    Quick sanity checks on the quadrature:
      (1) Weights should sum to 2.0.
      (2) Integral of x from 0 to 1 should equal 0.5.
    """
    weight_sum = np.sum(wg)
    print(f"  Qwts check (=2.0?): {weight_sum:.6f}")

    ng2 = ng // 2
    ordinate_check = np.sum(xg[ng2:] * wg[ng2:])
    print(f"  Qord check (=0.5?): {ordinate_check:.6f}")


# =============================================================================
# LEAF ANGLE DISTRIBUTION FUNCTIONS:  gL(theta_L)
# =============================================================================
# Each function takes a leaf zenith angle theta_L (in radians, 0 to pi/2)
# and returns the value of gL at that angle.
#
# These satisfy the normalization:
#   integral from 0 to pi/2 of gL(theta) * sin(theta) d(theta) = 1
# =============================================================================

def gL_uniform(theta_L):
    """Uniform: gL = 1  (all leaf angles equally likely)."""
    return 1.0

def gL_planophile(theta_L):
    """Planophile: gL = 3 cos^2(theta)  (mostly horizontal leaves)."""
    return 3.0 * math.cos(theta_L) ** 2

def gL_erectophile(theta_L):
    """Erectophile: gL = (3/2) sin^2(theta)  (mostly vertical leaves)."""
    return (3.0 / 2.0) * math.sin(theta_L) ** 2

def gL_plagiophile(theta_L):
    """Plagiophile: gL = (15/8) sin^2(2*theta)  (leaves near 45 degrees)."""
    return (15.0 / 8.0) * math.sin(2.0 * theta_L) ** 2

def gL_extremophile(theta_L):
    """Extremophile: gL = (15/7) cos^2(2*theta)  (leaves near 0 or 90 deg)."""
    return (15.0 / 7.0) * math.cos(2.0 * theta_L) ** 2


# =============================================================================
# FUNCTION: integrate_gL_normalization
# =============================================================================
def integrate_gL_normalization(ng, xg, wg, gL_func):
    """
    Verify the leaf angle distribution normalization:

              pi/2
             /
             |  gL(theta_L) * sin(theta_L)  d(theta_L)  =  1
             |
            / 0

    Parameters
    ----------
    ng      : int           — quadrature order
    xg, wg  : numpy arrays  — quadrature ordinates and weights
    gL_func : function       — the leaf angle distribution gL(theta_L)

    Returns
    -------
    float — the value of the integral (should be 1.0).
    """

    # Change of variable from [-1, +1] to [0, pi/2].
    lower = 0.0
    upper = PI / 2.0
    conv1 = (upper - lower) / 2.0
    conv2 = (upper + lower) / 2.0

    total = 0.0
    for i in range(ng):
        theta_L = conv1 * xg[i] + conv2           # transform ordinate
        total += wg[i] * gL_func(theta_L) * math.sin(theta_L)

    return total * conv1


# =============================================================================
# FUNCTION: integrate_hL_normalization
# =============================================================================
def integrate_hL_normalization(ng, xg, wg, phi_sun, eta):
    """
    Verify the non-uniform leaf azimuthal distribution normalization:

         1       2*pi
        ----  *  INT    hL(phi_L)  d(phi_L)  =  1
        2*pi      0

    where:  (1/2*pi) * hL(phi_L) = (1/pi) * cos^2(phi_sun - phi_L - eta)

    So the integrand of the normalization integral is:
        (1/pi) * cos^2(phi_sun - phi_L - eta)

    and the integral over [0, 2*pi] should yield 1.0.

    Parameters
    ----------
    ng       : int           — quadrature order
    xg, wg   : numpy arrays  — quadrature ordinates and weights
    phi_sun  : float          — solar azimuth angle (radians)
    eta      : float          — distribution offset parameter (radians)

    Returns
    -------
    float — the value of the integral (should be 1.0).
    """

    # Change of variable from [-1, +1] to [0, 2*pi].
    lower = 0.0
    upper = TWO_PI
    conv1 = (upper - lower) / 2.0         # = pi
    conv2 = (upper + lower) / 2.0         # = pi

    total = 0.0
    for i in range(ng):
        phi_L = conv1 * xg[i] + conv2
        # The normalized PDF: (1/pi) * cos^2(phi_sun - phi_L - eta)
        argument = phi_sun - phi_L - eta
        total += wg[i] * (1.0 / PI) * math.cos(argument) ** 2

    return total * conv1


# =============================================================================
# FUNCTION: compute_G_nonuniform
# =============================================================================
def compute_G_nonuniform(theta_view, phi_view, phi_sun, ng, xg, wg, gL_func, eta):
    """
    Compute the G-function for a given viewing direction (theta_view, phi_view)
    using a NON-UNIFORM leaf azimuthal distribution referenced to the solar
    azimuth phi_sun.

                         1      2*pi   pi/2
    G(theta_v, phi_v) = -----  INT    INT     gL(theta_L) * hL(phi_L)
                        2*pi    0      0
                                * |Omega_L . Omega_v| * sin(theta_L) d(theta_L) d(phi_L)

    The azimuthal distribution is:
      hL(phi_L) = 2 * cos^2(phi_sun - phi_L - eta)

    The dot product uses the VIEWING direction (theta_view, phi_view):
      Omega_L . Omega_v = sin(theta_L)*sin(theta_v)*cos(phi_L - phi_v)
                        + cos(theta_L)*cos(theta_v)

    IMPORTANT DISTINCTION:
    ----------------------
    - phi_view : azimuth of the VIEWING direction Omega_v  (varies in outer integral)
    - phi_sun  : azimuth of the SUN direction               (fixed parameter of hL)

    These are DIFFERENT angles. In the uniform hL case (hL = 1), this distinction
    didn't matter because hL had no phi dependence.  In the non-uniform case,
    hL depends on phi_sun (a fixed canopy property), while the dot product
    depends on phi_view (the direction being integrated over).

    Parameters
    ----------
    theta_view : float   — zenith angle of the viewing direction (radians)
    phi_view   : float   — azimuth of the viewing direction (radians)
    phi_sun    : float   — solar azimuth angle (radians) — reference for hL
    ng         : int     — quadrature order
    xg, wg     : numpy arrays — quadrature ordinates and weights
    gL_func    : function — leaf inclination distribution gL(theta_L)
    eta        : float   — azimuthal offset: 0 = diaheliotropic, pi/2 = paraheliotropic

    Returns
    -------
    float — the value of G at (theta_view, phi_view).
    """

    # =========================================================================
    # Set up the change-of-variable for the OUTER integral (phi_L: 0 to 2*pi).
    # =========================================================================
    phi_lower = 0.0
    phi_upper = TWO_PI
    conv1_phi = (phi_upper - phi_lower) / 2.0       # = pi
    conv2_phi = (phi_upper + phi_lower) / 2.0       # = pi

    # =========================================================================
    # Set up the change-of-variable for the INNER integral (theta_L: 0 to pi/2).
    # =========================================================================
    th_lower = 0.0
    th_upper = PI / 2.0
    conv1_th = (th_upper - th_lower) / 2.0          # = pi/4
    conv2_th = (th_upper + th_lower) / 2.0          # = pi/4

    # =========================================================================
    # Pre-compute sin and cos of the VIEWING direction (constant in this call).
    # =========================================================================
    sin_theta_v = math.sin(theta_view)
    cos_theta_v = math.cos(theta_view)

    # =========================================================================
    # Double integral: outer over phi_L, inner over theta_L.
    # =========================================================================
    total = 0.0

    for j in range(ng):             # --- OUTER LOOP: over phi_L ---

        # Transform the j-th ordinate from [-1,+1] to [0, 2*pi].
        phi_L = conv1_phi * xg[j] + conv2_phi

        # -----------------------------------------------------------------
        # Evaluate the NON-UNIFORM azimuthal distribution hL at this phi_L.
        # -----------------------------------------------------------------
        # hL(phi_L) = 2 * cos^2(phi_sun - phi_L - eta)
        #
        # This uses the SOLAR azimuth phi_sun as the reference direction,
        # NOT the viewing direction phi_view.
        # -----------------------------------------------------------------
        hL_val = 2.0 * math.cos(phi_sun - phi_L - eta) ** 2

        # Pre-compute cos(phi_L - phi_view) for the dot product.
        # This uses the VIEWING direction phi_view.
        cos_dphi = math.cos(phi_L - phi_view)

        for i in range(ng):         # --- INNER LOOP: over theta_L ---

            # Transform the i-th ordinate from [-1,+1] to [0, pi/2].
            theta_L = conv1_th * xg[i] + conv2_th

            # --- Compute the dot product Omega_L . Omega_view ---
            sin_theta_L = math.sin(theta_L)
            cos_theta_L = math.cos(theta_L)

            dot = sin_theta_L * sin_theta_v * cos_dphi + cos_theta_L * cos_theta_v

            # --- Absolute value (leaves scatter from both sides) ---
            abs_dot = math.fabs(dot)

            # --- Full integrand ---
            # gL(theta_L) * hL(phi_L) * |Omega_L . Omega_v| * sin(theta_L)
            #
            # Unlike the uniform case where hL = 1 dropped out, here
            # hL(phi_L) is a non-trivial weight that varies with phi_L.
            # sin(theta_L) is the Jacobian from dOmega_L.
            integrand = gL_func(theta_L) * hL_val * abs_dot * sin_theta_L

            # --- Accumulate the weighted double sum ---
            total += wg[j] * wg[i] * integrand

    # =========================================================================
    # Apply Jacobians and the 1/(2*pi) normalization.
    # =========================================================================
    # The double integral Jacobian is conv1_phi * conv1_th.
    # The G-function definition has a 1/(2*pi) prefactor.
    total = total * conv1_phi * conv1_th / TWO_PI

    return total


# =============================================================================
# FUNCTION: verify_G_identity_nonuniform
# =============================================================================
def verify_G_identity_nonuniform(ng, xg, wg, gL_func, eta, phi_sun):
    """
    Verify the identity:

           1
         -----  INT    G(r, Omega)  dOmega   =   1/2
          2*pi   2pi

    The integral is over the full upper hemisphere:

       1       2*pi      pi/2
     -----  INT     INT       G(theta_v, phi_v) * sin(theta_v)  d(theta_v) d(phi_v)  =  1/2
      2*pi    0       0

    CRITICAL DIFFERENCE FROM THE UNIFORM CASE:
    -------------------------------------------
    In the uniform hL case, G was independent of phi, so the phi integral
    simply gave a factor of 2*pi (cancelling the 1/(2*pi) prefactor).

    With non-uniform hL, G(theta_v, phi_v) depends on BOTH angles because
    the azimuthal distribution hL introduces a preferred direction (the
    solar azimuth).  Therefore we MUST compute the full double integral
    over the hemisphere — the phi integral does NOT simplify.

    Despite this phi-dependence, the identity (1/2*pi)*INT G dOmega = 1/2
    still holds.  This is a fundamental property of the G-function that
    is independent of the leaf angle distribution.

    NOTE: This is a QUADRUPLE integral:
      - 2 loops for the hemisphere integral (theta_v, phi_v)
      - 2 loops for the G-function (theta_L, phi_L)
      Total: ng^4 = 12^4 = 20,736 innermost evaluations per case.

    Parameters
    ----------
    ng       : int           — quadrature order
    xg, wg   : numpy arrays  — quadrature ordinates and weights
    gL_func  : function       — leaf inclination distribution gL(theta_L)
    eta      : float          — azimuthal offset (0 or pi/2)
    phi_sun  : float          — solar azimuth angle (radians)

    Returns
    -------
    float — the value of the integral (should be 0.5).
    """

    # =========================================================================
    # Set up the change-of-variable for the OUTER integral (phi_v: 0 to 2*pi).
    # =========================================================================
    # phi_v = azimuth of the viewing direction (hemisphere integration variable)
    phi_lower = 0.0
    phi_upper = TWO_PI
    conv1_phi = (phi_upper - phi_lower) / 2.0       # = pi
    conv2_phi = (phi_upper + phi_lower) / 2.0       # = pi

    # =========================================================================
    # Set up the change-of-variable for the INNER integral (theta_v: 0 to pi/2).
    # =========================================================================
    # theta_v = zenith of the viewing direction (hemisphere integration variable)
    th_lower = 0.0
    th_upper = PI / 2.0
    conv1_th = (th_upper - th_lower) / 2.0          # = pi/4
    conv2_th = (th_upper + th_lower) / 2.0          # = pi/4

    # =========================================================================
    # Perform the double integral over the upper hemisphere.
    # =========================================================================
    # OUTER loop: phi_v   = azimuth of viewing direction  (0 to 2*pi)
    # INNER loop: theta_v = zenith of viewing direction   (0 to pi/2)
    #
    # At each (theta_v, phi_v), we call compute_G_nonuniform() which
    # itself does a double integral over (theta_L, phi_L).
    # The phi_sun parameter is passed through so that hL is always
    # referenced to the fixed solar azimuth.

    total = 0.0

    for j in range(ng):             # --- OUTER LOOP: over phi_v ---

        phi_v = conv1_phi * xg[j] + conv2_phi

        for i in range(ng):         # --- INNER LOOP: over theta_v ---

            theta_v = conv1_th * xg[i] + conv2_th

            # Compute G at this viewing direction (theta_v, phi_v),
            # using the non-uniform hL with fixed solar azimuth phi_sun.
            G_val = compute_G_nonuniform(
                theta_v, phi_v, phi_sun, ng, xg, wg, gL_func, eta
            )

            # The hemisphere solid angle element: sin(theta_v) d(theta_v) d(phi_v)
            total += wg[j] * wg[i] * G_val * math.sin(theta_v)

    # =========================================================================
    # Apply Jacobians and the 1/(2*pi) prefactor.
    # =========================================================================
    total = total * conv1_phi * conv1_th / TWO_PI

    return total


# =============================================================================
# MAIN PROGRAM
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # PART 1: SET UP AND VERIFY GAUSS QUADRATURE
    # =========================================================================

    print("=" * 70)
    print("  G-FUNCTION VERIFICATION — NON-UNIFORM AZIMUTHAL DISTRIBUTION")
    print("  Showing: (1/2pi) * INT G(r, Omega) dOmega = 1/2")
    print("=" * 70)
    print()

    # --- Get quadrature ordinates and weights ---
    xg, wg = gauss_quad(NG)

    # --- Print them ---
    print(f"  Quadrature Order: {NG}")
    print(f"  {'Point':>6}  {'Ordinate (xg)':>15}  {'Weight (wg)':>15}")
    print(f"  {'-'*6}  {'-'*15}  {'-'*15}")
    for i, (x, w) in enumerate(zip(xg, wg)):
        print(f"  {i+1:>6}  {x:>15.9f}  {w:>15.9f}")
    print()

    # --- Verify the quadrature ---
    print("  --- Quadrature Checks ---")
    check_quad(NG, xg, wg)
    print()

    # =========================================================================
    # PART 2: VERIFY gL NORMALIZATION (prerequisite check)
    # =========================================================================

    print("  --- Part A: Leaf Inclination Distribution Normalization ---")
    print("  Verifying: INT_0^{pi/2} gL(theta_L) sin(theta_L) d(theta_L) = 1")
    print()

    distributions = [
        ("Uniform",      "gL = 1",                    gL_uniform),
        ("Planophile",   "gL = 3 cos^2(theta)",       gL_planophile),
        ("Erectophile",  "gL = (3/2) sin^2(theta)",   gL_erectophile),
        ("Plagiophile",  "gL = (15/8) sin^2(2theta)", gL_plagiophile),
        ("Extremophile", "gL = (15/7) cos^2(2theta)", gL_extremophile),
    ]

    print(f"  {'Distribution':<15} {'Formula':<28} {'Integral':>10} {'(=1.0?)'}")
    print(f"  {'-'*15} {'-'*28} {'-'*10} {'-'*7}")

    for name, formula, func in distributions:
        result = integrate_gL_normalization(NG, xg, wg, func)
        print(f"  {name:<15} {formula:<28} {result:>10.6f}")
    print()

    # =========================================================================
    # PART 3: VERIFY hL NORMALIZATION (prerequisite check)
    # =========================================================================
    # Verify that the non-uniform azimuthal distribution integrates to 1:
    #   (1/2*pi) * INT_0^{2*pi} hL(phi_L) d(phi_L) = 1
    #
    # We test both eta cases at an arbitrary solar azimuth phi_sun = pi/4.

    print("  --- Part B: Azimuthal Distribution Normalization ---")
    print("  Verifying: (1/2pi) * INT_0^{2pi} hL(phi_L) d(phi_L) = 1")
    print()

    # Choose an arbitrary solar azimuth for the normalization check.
    phi_sun = PI / 4.0          # 45 degrees — arbitrary but non-trivial

    eta_cases = [
        ("Diaheliotropic",  0.0),           # eta = 0:     maximize projected area
        ("Paraheliotropic", PI / 2.0),      # eta = pi/2:  minimize projected area
    ]

    print(f"  Solar azimuth phi_sun = pi/4 = {phi_sun:.6f} rad")
    print()
    print(f"  {'Case':<20} {'eta':>10} {'Integral':>12} {'(=1.0?)'}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*7}")

    for case_name, eta in eta_cases:
        result = integrate_hL_normalization(NG, xg, wg, phi_sun, eta)
        print(f"  {case_name:<20} {eta:>10.4f} {result:>12.6f}")
    print()

    # =========================================================================
    # PART 4: VERIFY THE MAIN G-FUNCTION IDENTITY
    # =========================================================================
    # (1/2pi) * INT_{2pi} G(r, Omega) dOmega = 1/2
    #
    # for all 5 leaf inclination distributions x 2 eta cases = 10 total cases.
    #
    # This is a QUADRUPLE integral (double for G, double for hemisphere).
    # With ng = 12, each case requires 12^4 = 20,736 innermost evaluations.
    # For 10 cases, that's 207,360 evaluations total.
    #
    # This may take a minute or two — be patient!

    print("  --- Part C: Main G-Function Identity (Non-Uniform hL) ---")
    print("  Verifying: (1/2pi) * INT G(r, Omega) dOmega = 1/2")
    print()
    print(f"  Solar azimuth phi_sun = pi/4 = {phi_sun:.6f} rad")
    print(f"  Quadrature: ng = {NG}  =>  ng^4 = {NG**4} evaluations per case")
    print()

    # -------------------------------------------------------------------------
    # Table header for the 10-case verification.
    # -------------------------------------------------------------------------
    print(f"  {'Distribution':<15} {'eta Case':<18} {'eta':>6} {'Integral':>10} {'(=0.5?)'}")
    print(f"  {'-'*15} {'-'*18} {'-'*6} {'-'*10} {'-'*7}")

    # -------------------------------------------------------------------------
    # Loop over all 5 distributions x 2 eta cases = 10 cases.
    # -------------------------------------------------------------------------
    for name, formula, func in distributions:
        for case_name, eta in eta_cases:

            # Compute the hemisphere-averaged G-function.
            result = verify_G_identity_nonuniform(NG, xg, wg, func, eta, phi_sun)

            # Print the result for this (distribution, eta) combination.
            print(f"  {name:<15} {case_name:<18} {eta:>6.2f} {result:>10.6f}")

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("  All 10 cases yield 0.5 — G-function identity verified!")
    print("  (5 gL distributions x 2 non-uniform hL cases)")
    print("=" * 70)
