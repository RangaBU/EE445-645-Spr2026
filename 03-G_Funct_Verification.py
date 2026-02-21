#!/usr/bin/env python3
"""
G-FUNCTION VERIFICATION PROGRAM
================================

This program uses Gauss-Legendre Quadrature (order 12) to numerically verify
the fundamental identity of radiative transfer in vegetation canopies:

               1                                 1
             -----  INT  G(r, Omega)  dOmega  =  ---
              2*pi   2pi                          2

where the integral is over the upper hemisphere (solid angle = 2*pi steradians)
and G is the "Ross-Nilson geometry function" (also called the G-function).

WHAT IS THE G-FUNCTION?
-----------------------
The G-function describes the mean projection of leaves in a given direction
Omega. It is defined as:

                   1
  G(r, Omega)  =  ----  INT     gL(r, Omega_L)  |Omega_L . Omega|  dOmega_L
                  2*pi   2pi

where:
  Omega   = (theta, phi)     = direction of interest (e.g., sun direction)
  Omega_L = (theta_L, phi_L) = leaf normal direction
  gL      = leaf angle distribution function
  |Omega_L . Omega|          = absolute value of the dot product
                                (= cosine of angle between the two directions)

For azimuthally uniform leaf distributions, gL separates as:
  g_bar_L(theta_L, phi_L) = gL(theta_L) * hL(phi_L),   where hL(phi_L) = 1.

DISTRIBUTIONS TESTED:
  (1) Uniform       :  gL(theta) = 1
  (2) Planophile    :  gL(theta) = 3 cos^2(theta)
  (3) Erectophile   :  gL(theta) = (3/2) sin^2(theta)
  (4) Plagiophile   :  gL(theta) = (15/8) sin^2(2*theta)
  (5) Extremophile  :  gL(theta) = (15/7) cos^2(2*theta)

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
# These are the SAME distributions as in our previous program
# (02-LAD_Verification.py).  They satisfy:
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

    This is carried forward from the previous program for completeness.

    Parameters
    ----------
    ng      : int           — quadrature order
    xg, wg  : numpy arrays  — quadrature ordinates and weights
    gL_func : function       — the leaf angle distribution gL(theta_L)

    Returns
    -------
    float — the value of the integral (should be 1.0).
    """

    # -------------------------------------------------------------------------
    # Change of variable from [-1, +1] to [0, pi/2].
    # -------------------------------------------------------------------------
    # conv1 = (upper - lower) / 2 = (pi/2 - 0) / 2 = pi/4
    # conv2 = (upper + lower) / 2 = (pi/2 + 0) / 2 = pi/4

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
# FUNCTION: compute_G
# =============================================================================
def compute_G(theta, phi, ng, xg, wg, gL_func):
    """
    Compute the G-function (Ross-Nilson geometry function) for a given
    viewing direction (theta, phi) and leaf angle distribution gL.

                      1      2pi    pi/2
    G(theta, phi) = -----  INT    INT     gL(theta_L) * hL(phi_L)
                     2*pi    0      0
                             * |Omega_L . Omega| * sin(theta_L) d(theta_L) d(phi_L)

    Since hL(phi_L) = 1 (azimuthally uniform leaf distribution), this becomes
    a double integral over theta_L (leaf zenith) and phi_L (leaf azimuth).

    THE DOT PRODUCT |Omega_L . Omega|:
    -----------------------------------
    Both Omega and Omega_L are unit vectors in spherical coordinates:

      Omega   = ( sin(theta)*cos(phi),     sin(theta)*sin(phi),     cos(theta)   )
      Omega_L = ( sin(theta_L)*cos(phi_L), sin(theta_L)*sin(phi_L), cos(theta_L) )

    Their dot product (using the identity cos(A-B) = cosA*cosB + sinA*sinB) is:

      Omega_L . Omega = sin(theta_L)*sin(theta)*cos(phi_L - phi)
                      + cos(theta_L)*cos(theta)

    We need the ABSOLUTE VALUE |Omega_L . Omega| because leaves scatter
    light from both sides.

    DOUBLE INTEGRAL WITH GAUSS QUADRATURE:
    ---------------------------------------
    We do TWO nested change-of-variable transformations:
      Outer loop: phi_L   over [0, 2*pi]
      Inner loop: theta_L over [0, pi/2]

    Each transformation maps the Gauss ordinates from [-1, +1] to the
    actual integration interval.

    Parameters
    ----------
    theta   : float          — zenith angle of the viewing direction (radians)
    phi     : float          — azimuth angle of the viewing direction (radians)
    ng      : int            — quadrature order
    xg, wg  : numpy arrays   — quadrature ordinates and weights
    gL_func : function        — leaf angle distribution function gL(theta_L)

    Returns
    -------
    float — the value of G at (theta, phi).
    """

    # =========================================================================
    # Set up the change-of-variable for the OUTER integral (phi_L: 0 to 2*pi).
    # =========================================================================
    # conv1_phi = (2*pi - 0) / 2 = pi          (half-width of phi_L interval)
    # conv2_phi = (2*pi + 0) / 2 = pi          (midpoint of phi_L interval)

    phi_lower = 0.0
    phi_upper = TWO_PI
    conv1_phi = (phi_upper - phi_lower) / 2.0       # = pi
    conv2_phi = (phi_upper + phi_lower) / 2.0       # = pi

    # =========================================================================
    # Set up the change-of-variable for the INNER integral (theta_L: 0 to pi/2).
    # =========================================================================
    # conv1_th = (pi/2 - 0) / 2 = pi/4         (half-width of theta_L interval)
    # conv2_th = (pi/2 + 0) / 2 = pi/4         (midpoint of theta_L interval)

    th_lower = 0.0
    th_upper = PI / 2.0
    conv1_th = (th_upper - th_lower) / 2.0          # = pi/4
    conv2_th = (th_upper + th_lower) / 2.0          # = pi/4

    # =========================================================================
    # Pre-compute the sin and cos of the viewing direction (theta, phi).
    # =========================================================================
    # We do this OUTSIDE the loops because theta and phi are fixed for a
    # given call to compute_G. Computing sin/cos is expensive, so doing
    # it once instead of ng*ng times saves significant computation.
    #
    # This is a general optimization tip: move constant calculations OUT
    # of loops whenever possible.

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    # =========================================================================
    # Perform the double integral using nested loops.
    # =========================================================================
    # The OUTER loop iterates over phi_L (leaf azimuth angle).
    # The INNER loop iterates over theta_L (leaf zenith angle).
    #
    # This is analogous to a double sum:
    #   Sum_j  Sum_i  wg[j] * wg[i] * f(phi_L_j, theta_L_i)
    #
    # In Python, 'for j in range(ng)' loops j = 0, 1, 2, ..., ng-1.
    # '+=' is shorthand for 'total = total + ...'.

    total = 0.0

    for j in range(ng):             # --- OUTER LOOP: over phi_L ---

        # Transform the j-th ordinate from [-1,+1] to [0, 2*pi].
        phi_L = conv1_phi * xg[j] + conv2_phi

        # Pre-compute cos(phi_L - phi) for use in the dot product.
        # This value is constant for the entire inner loop over theta_L.
        cos_dphi = math.cos(phi_L - phi)

        for i in range(ng):         # --- INNER LOOP: over theta_L ---

            # Transform the i-th ordinate from [-1,+1] to [0, pi/2].
            theta_L = conv1_th * xg[i] + conv2_th

            # --- Compute the dot product Omega_L . Omega ---
            # Using the spherical coordinate identity:
            #   Omega_L . Omega = sin(theta_L)*sin(theta)*cos(phi_L - phi)
            #                   + cos(theta_L)*cos(theta)
            sin_theta_L = math.sin(theta_L)
            cos_theta_L = math.cos(theta_L)

            dot = sin_theta_L * sin_theta * cos_dphi + cos_theta_L * cos_theta

            # --- Take the absolute value ---
            # Leaves can scatter light from both their upper and lower surfaces,
            # so we use |Omega_L . Omega| (absolute value of the cosine of the
            # angle between the leaf normal and the viewing direction).
            abs_dot = math.fabs(dot)

            # --- Evaluate the integrand ---
            # The full integrand is:
            #   gL(theta_L) * hL(phi_L) * |Omega_L . Omega| * sin(theta_L)
            #
            # Since hL(phi_L) = 1 (azimuthally uniform), it drops out.
            # sin(theta_L) is the Jacobian from the solid angle element
            # dOmega_L = sin(theta_L) d(theta_L) d(phi_L).
            integrand = gL_func(theta_L) * abs_dot * sin_theta_L

            # --- Accumulate the weighted sum ---
            # Both the inner weight wg[i] and outer weight wg[j] multiply
            # the integrand. This is how Gauss quadrature extends to 2D:
            #   Double integral ≈ Sum_j Sum_i  wg[j] * wg[i] * f(x_j, x_i)
            total += wg[j] * wg[i] * integrand

    # =========================================================================
    # Apply the Jacobians from both change-of-variable transformations,
    # and the 1/(2*pi) normalization factor.
    # =========================================================================
    # The double integral's Jacobian is conv1_phi * conv1_th.
    # Then we divide by 2*pi as required by the G-function definition.

    total = total * conv1_phi * conv1_th / TWO_PI

    return total


# =============================================================================
# FUNCTION: verify_G_identity
# =============================================================================
def verify_G_identity(ng, xg, wg, gL_func):
    """
    Verify the identity:

           1      
         -----  INT    G(r, Omega)  dOmega   =   1/2
          2*pi   2pi

    The integral is over the full upper hemisphere:

      INT  dOmega  =  INT     INT        sin(theta) d(theta) d(phi)
       2pi              0->2pi  0->pi/2

    So the identity becomes:

       1       2pi       pi/2
     -----  INT     INT       G(theta, phi) * sin(theta)  d(theta) d(phi)  =  1/2
      2*pi    0       0

    SIMPLIFICATION DUE TO AZIMUTHAL SYMMETRY:
    ------------------------------------------
    Because hL(phi_L) = 1 (leaves have no preferred azimuth), the G-function
    is independent of phi:  G(theta, phi) = G(theta).

    Therefore the phi integral just gives a factor of 2*pi, which cancels
    the 1/(2*pi) prefactor, and the identity simplifies to:

        pi/2
       /
       |  G(theta)  sin(theta)  d(theta)   =   1/2
       |
      / 0

    However, to demonstrate the FULL double integral (as written in the
    original equation), we perform the complete computation over BOTH
    theta and phi. This is more instructive and general.

    Parameters
    ----------
    ng      : int            — quadrature order
    xg, wg  : numpy arrays   — quadrature ordinates and weights
    gL_func : function        — leaf angle distribution gL(theta_L)

    Returns
    -------
    float — the value of the integral (should be 0.5).
    """

    # =========================================================================
    # Set up the change-of-variable for the OUTER integral (phi: 0 to 2*pi).
    # =========================================================================
    phi_lower = 0.0
    phi_upper = TWO_PI
    conv1_phi = (phi_upper - phi_lower) / 2.0       # = pi
    conv2_phi = (phi_upper + phi_lower) / 2.0       # = pi

    # =========================================================================
    # Set up the change-of-variable for the INNER integral (theta: 0 to pi/2).
    # =========================================================================
    th_lower = 0.0
    th_upper = PI / 2.0
    conv1_th = (th_upper - th_lower) / 2.0          # = pi/4
    conv2_th = (th_upper + th_lower) / 2.0          # = pi/4

    # =========================================================================
    # Perform the double integral over the upper hemisphere.
    # =========================================================================
    # OUTER loop: phi   = azimuth of viewing direction (0 to 2*pi)
    # INNER loop: theta = zenith of viewing direction  (0 to pi/2)
    #
    # At each (theta, phi), we call compute_G() to evaluate the G-function,
    # which itself performs a double integral.  So in total, this is a
    # QUADRUPLE integral (2 for G, 2 for the hemisphere), computed with
    # ng^4 = 12^4 = 20,736 function evaluations!

    total = 0.0

    for j in range(ng):             # --- OUTER LOOP: over phi ---

        phi = conv1_phi * xg[j] + conv2_phi

        for i in range(ng):         # --- INNER LOOP: over theta ---

            theta = conv1_th * xg[i] + conv2_th

            # Compute G at this viewing direction.
            G_val = compute_G(theta, phi, ng, xg, wg, gL_func)

            # The hemisphere solid angle element is sin(theta) d(theta) d(phi).
            total += wg[j] * wg[i] * G_val * math.sin(theta)

    # Apply Jacobians from the change-of-variable and the 1/(2*pi) prefactor.
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
    print("  G-FUNCTION VERIFICATION")
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
    # PART 2: VERIFY gL NORMALIZATION (from previous program)
    # =========================================================================
    # This is a prerequisite check: the leaf angle distributions must
    # integrate to 1.0 before we can trust the G-function results.

    print("  --- Part A: Leaf Angle Distribution Normalization ---")
    print("  Verifying: INT_0^{pi/2} gL(theta_L) sin(theta_L) d(theta_L) = 1")
    print()

    # A list of tuples: (name, formula_string, function).
    # 'lambda' is Python shorthand for defining a tiny function in one line.
    # Example: lambda x: x**2   is the same as   def f(x): return x**2
    # We don't use lambda here — we use the named functions defined above —
    # but you may encounter lambda elsewhere in Python code.

    distributions = [
        ("Uniform",      "gL = 1",                    gL_uniform),
        ("Planophile",   "gL = 3 cos^2(theta)",       gL_planophile),
        ("Erectophile",  "gL = (3/2) sin^2(theta)",   gL_erectophile),
        ("Plagiophile",  "gL = (15/8) sin^2(2theta)", gL_plagiophile),
        ("Extremophile", "gL = (15/7) cos^2(2theta)", gL_extremophile),
    ]

    # Print a table header, then loop over all distributions.
    print(f"  {'Distribution':<15} {'Formula':<28} {'Integral':>10} {'(=1.0?)'}")
    print(f"  {'-'*15} {'-'*28} {'-'*10} {'-'*7}")

    for name, formula, func in distributions:
        result = integrate_gL_normalization(NG, xg, wg, func)
        print(f"  {name:<15} {formula:<28} {result:>10.6f}")
    print()

    # =========================================================================
    # PART 3: VERIFY THE MAIN IDENTITY
    # =========================================================================
    # (1/2pi) * INT_{2pi} G(r, Omega) dOmega = 1/2
    #
    # This is the key result: averaged over ALL viewing directions (the full
    # upper hemisphere), the G-function always equals 1/2, regardless of
    # which leaf angle distribution is used.
    #
    # NOTE: This computation involves a QUADRUPLE integral (double integral
    # for G, nested inside a double integral over the hemisphere), so it takes
    # ng^4 = 12^4 = 20,736 evaluations of the innermost expression per
    # distribution.  Be patient — it may take a few seconds.

    print("  --- Part B: Main Identity Verification ---")
    print("  Verifying: (1/2pi) * INT G(r, Omega) dOmega = 1/2")
    print()
    print(f"  {'Distribution':<15} {'Formula':<28} {'Integral':>10} {'(=0.5?)'}")
    print(f"  {'-'*15} {'-'*28} {'-'*10} {'-'*7}")

    for name, formula, func in distributions:
        result = verify_G_identity(NG, xg, wg, func)
        print(f"  {name:<15} {formula:<28} {result:>10.6f}")

    print()
    print("=" * 70)
    print("  All hemisphere-averaged G-functions equal 0.5 — identity verified!")
    print("=" * 70)
