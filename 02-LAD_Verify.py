#!/usr/bin/env python3
"""
LEAF ANGLE DISTRIBUTION — NORMALIZATION VERIFICATION
=====================================================

This program uses Gauss-Legendre Quadrature (order 12) to numerically verify
that five common leaf angle distribution functions all satisfy the
normalization condition:

              pi/2
             /
             |  gL(theta_L) * sin(theta_L)  d(theta_L)  =  1
             |
            / 0

where:
  theta_L   = leaf inclination angle (0 = horizontal, pi/2 = vertical)
  gL(theta) = the leaf angle distribution function
  sin(theta) appears because we are integrating over the hemisphere

The five distributions tested are:
  (1) Uniform       :  gL(theta) = 1
  (2) Planophile    :  gL(theta) = 3 cos^2(theta)          (mostly horizontal leaves)
  (3) Erectophile   :  gL(theta) = (3/2) sin^2(theta)      (mostly vertical leaves)
  (4) Plagiophile   :  gL(theta) = (15/8) sin^2(2*theta)   (leaves near 45 degrees)
  (5) Extremophile  :  gL(theta) = (15/7) cos^2(2*theta)   (leaves near 0 or 90 degrees)

This code also verifies the normalization of a non-uniform leaf normal
azimuthal distribution function hL(phi_L, phi), which is a probability
density function (PDF) over the leaf azimuth angle phi_L:

    (1 / 2*pi) * hL(phi_L, phi) = (1 / pi) * cos^2(phi - phi_L - eta)

The normalization condition for this azimuthal PDF is:

        (1 / 2*pi) * integral from 0 to 2*pi of hL(phi_L) d(phi_L) = 1

Two cases of the parameter eta are tested:
  (1) Diaheliotropic  :  eta = 0      (maximize projected leaf area to sun)
  (2) Paraheliotropic :  eta = pi/2   (minimize projected leaf area to sun)

This code combines the Gauss Quadrature routine with the leaf angle and
leaf azimuth distribution calculations into a single file.

Authors: Claude And I (Ranga B. Myneni)
"""

# =============================================================================
# IMPORTS
# =============================================================================
# 'math' gives us basic math functions (pi, sin, cos, etc.)
# 'numpy' gives us arrays and vectorized math operations.
#   We import numpy as 'np' — a universal convention in Python.

import math
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================
# Constants are written in ALL_CAPS by Python convention.
# Python doesn't enforce this — it's a signal to other programmers.

PI = math.pi                        # pi = 3.141592653589793
DEG_TO_RAD = PI / 180.0             # multiply degrees by this to get radians
RAD_TO_DEG = 180.0 / PI             # multiply radians by this to get degrees
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
        Higher orders give more accurate results but use more points.

    Returns
    -------
    xg : numpy array of floats
        The quadrature ordinates (the specially-chosen x-positions on [-1, +1]).
    wg : numpy array of floats
        The quadrature weights (how much "importance" each point gets).

    Notes
    -----
    Gauss-Legendre quadrature is defined on [-1, +1]. To integrate over a
    different interval [A, B], you must transform the ordinates (see
    the function 'integrate_distribution' below for how this is done).
    """

    # -------------------------------------------------------------------------
    # Step 1: Pre-computed (tabulated) ordinates and weights.
    # -------------------------------------------------------------------------
    # These are the NEGATIVE-half ordinates for orders 4 through 12.
    # They are packed into one list; 'ishift' tells us where each order starts.
    #
    # In Python, lists use square brackets [].
    # IMPORTANT: Python indexing starts at 0 (not 1 like Fortran).

    xx = [
        -0.861136312, -0.339981044,                                  # ng=4  (2 values)
        -0.9324695,   -0.6612094,   -0.2386192,                     # ng=6  (3 values)
        -0.960289856, -0.796666477, -0.525532410, -0.183434642,     # ng=8  (4 values)
        -0.973906529, -0.865063367, -0.679409568, -0.433395394,     # ng=10 (5 values)
        -0.148874339,
        -0.981560634, -0.904117256, -0.769902674, -0.587317954,     # ng=12 (6 values)
        -0.367831499, -0.125233409
    ]

    ww = [
         0.347854845,  0.652145155,                                  # ng=4
         0.1713245,    0.3607616,    0.4679139,                      # ng=6
         0.101228536,  0.222381034,  0.313706646,  0.362683783,      # ng=8
         0.066671344,  0.149451349,  0.219086363,  0.269266719,      # ng=10
         0.295524225,
         0.047175336,  0.106939326,  0.160078329,  0.203167427,      # ng=12
         0.233492537,  0.249147046
    ]

    # A Python "dictionary" maps keys to values using { }.
    # ishift[ng//2] gives the starting index in xx/ww for that order.
    ishift = {
        2: 0,    # ng=4:  starts at index 0
        3: 2,    # ng=6:  starts at index 2
        4: 5,    # ng=8:  starts at index 5
        5: 9,    # ng=10: starts at index 9
        6: 14    # ng=12: starts at index 14
    }

    # -------------------------------------------------------------------------
    # Step 2: Validate the input.
    # -------------------------------------------------------------------------
    # 'assert' stops the program with an error if the condition is False.
    # The 'in' keyword checks if ng is a member of the list.

    assert ng in [4, 6, 8, 10, 12], \
        f"Error: ng must be 4, 6, 8, 10, or 12. You provided ng={ng}."

    # -------------------------------------------------------------------------
    # Step 3: Build the full set of ordinates and weights.
    # -------------------------------------------------------------------------
    # ng2 = half the order. We only store the negative side because
    # Gauss-Legendre quadrature is symmetric around zero.
    #
    # '//' is Python's integer division (12 // 2 = 6, not 6.0).

    ng2 = ng // 2

    # Start with empty lists; .append() adds one item at a time.
    xg = []
    wg = []

    # --- First half: negative side (directly from the table) ---
    # 'range(ng2)' produces 0, 1, 2, ..., ng2-1.
    for i in range(ng2):
        xg.append(xx[i + ishift[ng2]])
        wg.append(ww[i + ishift[ng2]])

    # --- Second half: positive side (mirror of the first half) ---
    for i in range(ng2):
        xg.append(-xg[ng2 - 1 - i])       # flip the sign and reverse order
        wg.append( wg[ng2 - 1 - i])        # same weight, reversed order

    # Convert Python lists to numpy arrays so we can do element-wise math.
    # Example: np.array([1,2,3]) * 2 gives array([2,4,6]).
    # A regular Python list * 2 would give [1,2,3,1,2,3] (repetition!).
    xg = np.array(xg)
    wg = np.array(wg)

    return xg, wg


# =============================================================================
# FUNCTION: check_quad
# =============================================================================
def check_quad(ng, xg, wg):
    """
    Check that the quadrature is correctly set up:
      (1) Weights should sum to 2.0
      (2) Integral of x from 0 to 1 should equal 0.5.
    """

    # --- Check 1: weights sum to 2.0? ---
    # np.sum() adds all elements of an array — like a Fortran DO loop.
    weight_sum = np.sum(wg)
    print(f"  Qwts check (=2.0?): {weight_sum:.6f}")

    # --- Check 2: integral of x over [0,1] = 0.5? ---
    # We only sum the positive-side ordinates (second half of the arrays).
    # Array slicing: xg[ng//2:] means "from index ng//2 to the end".
    ng2 = ng // 2
    ordinate_check = np.sum(xg[ng2:] * wg[ng2:])
    print(f"  Qord check (=0.5?): {ordinate_check:.6f}")


# =============================================================================
# LEAF ANGLE DISTRIBUTION FUNCTIONS
# =============================================================================
# Each function below takes theta_L (in radians) and returns gL(theta_L).
#
# WHY WRITE THEM AS SEPARATE FUNCTIONS?
# --------------------------------------
# In Python, functions are "first-class objects" — you can pass a function
# as an argument to another function, just like you'd pass a number or string.
# This lets us write ONE integration routine and use it with ANY distribution.
#
# The 'def' keyword defines a function.
# 'math.sin(x)' computes the sine of x (x must be in radians).
# 'math.cos(x)' computes the cosine of x.
# '**' is the exponentiation operator:  x**2 means x squared.
# =============================================================================

def gL_uniform(theta_L):
    """
    Uniform distribution: all leaf angles are equally likely.

    gL(theta_L) = 1

    Parameters
    ----------
    theta_L : float
        Leaf inclination angle in radians (0 to pi/2).

    Returns
    -------
    float
        The value of gL at theta_L, which is always 1.0.
    """
    return 1.0


def gL_planophile(theta_L):
    """
    Planophile distribution: leaves are predominantly horizontal.
    "Plano" = flat/plane, "phile" = loving — these plants love flat leaves.

    gL(theta_L) = 3 * cos^2(theta_L)

    Leaves cluster near theta_L = 0 (horizontal) because cos(0) = 1 is large,
    while cos(pi/2) = 0 makes gL small for vertical leaves.

    Parameters
    ----------
    theta_L : float
        Leaf inclination angle in radians.

    Returns
    -------
    float
        The value of gL at theta_L.
    """
    # math.cos() computes cosine;  ** 2 squares the result
    return 3.0 * math.cos(theta_L) ** 2


def gL_erectophile(theta_L):
    """
    Erectophile distribution: leaves are predominantly vertical (erect).
    "Erecto" = upright, "phile" = loving — these plants love vertical leaves.

    gL(theta_L) = (3/2) * sin^2(theta_L)

    Leaves cluster near theta_L = pi/2 (vertical) because sin(pi/2) = 1.

    Parameters
    ----------
    theta_L : float
        Leaf inclination angle in radians.

    Returns
    -------
    float
        The value of gL at theta_L.
    """
    return (3.0 / 2.0) * math.sin(theta_L) ** 2


def gL_plagiophile(theta_L):
    """
    Plagiophile distribution: leaves cluster around 45 degrees.
    "Plagio" = oblique, "phile" = loving — these plants love angled leaves.

    gL(theta_L) = (15/8) * sin^2(2 * theta_L)

    The function sin^2(2*theta) peaks at theta = pi/4 (= 45 degrees),
    which is exactly the angle these plants prefer.

    Parameters
    ----------
    theta_L : float
        Leaf inclination angle in radians.

    Returns
    -------
    float
        The value of gL at theta_L.
    """
    # Note the '2 * theta_L' — this is the "double angle" that shifts
    # the peak of sin^2 to 45 degrees instead of 90 degrees.
    return (15.0 / 8.0) * math.sin(2.0 * theta_L) ** 2


def gL_extremophile(theta_L):
    """
    Extremophile distribution: leaves cluster near 0 AND 90 degrees.
    "Extremo" = extreme, "phile" = loving — leaves favor the extreme angles.

    gL(theta_L) = (15/7) * cos^2(2 * theta_L)

    The function cos^2(2*theta) peaks at theta = 0 and theta = pi/2,
    giving a bimodal (two-peaked) distribution at the extremes.

    Parameters
    ----------
    theta_L : float
        Leaf inclination angle in radians.

    Returns
    -------
    float
        The value of gL at theta_L.
    """
    return (15.0 / 7.0) * math.cos(2.0 * theta_L) ** 2


# =============================================================================
# FUNCTION: integrate_distribution
# =============================================================================
def integrate_distribution(ng, xg, wg, gL_func, lower_limit, upper_limit):
    """
    Numerically integrate:
                  upper_limit
                 /
                 |  gL_func(theta) * sin(theta)  d(theta)
                 |
                / lower_limit

    using Gauss-Legendre quadrature.

    HOW THE CHANGE-OF-VARIABLE WORKS:
    ----------------------------------
    Gauss quadrature is defined on [-1, +1], but we want to integrate
    over [lower_limit, upper_limit] = [0, pi/2].

    We transform each quadrature ordinate xg[i] (which lives in [-1, +1])
    to a new point 'theta' in [0, pi/2] using:

        theta = conv1 * xg[i] + conv2

    where:
        conv1 = (upper_limit - lower_limit) / 2    (= pi/4, the half-width)
        conv2 = (upper_limit + lower_limit) / 2    (= pi/4, the midpoint)

    The Jacobian (stretching factor) of this transformation is conv1,
    so the final answer must be multiplied by conv1.

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : numpy array
        Quadrature ordinates (on [-1, +1]).
    wg : numpy array
        Quadrature weights.
    gL_func : function
        The leaf angle distribution function gL(theta_L).
        This is a Python function passed as an argument — a powerful feature!
        It allows us to write this integration routine ONCE and reuse it
        for all five distributions.
    lower_limit : float
        Lower bound of integration (in radians).
    upper_limit : float
        Upper bound of integration (in radians).

    Returns
    -------
    float
        The numerical value of the integral.
    """

    # -------------------------------------------------------------------------
    # Step 1: Compute the transformation constants.
    # -------------------------------------------------------------------------
    conv1 = (upper_limit - lower_limit) / 2.0      # half-width (scaling)
    conv2 = (upper_limit + lower_limit) / 2.0      # midpoint   (shifting)

    # -------------------------------------------------------------------------
    # Step 2: Loop over all quadrature points and accumulate the sum.
    # -------------------------------------------------------------------------
    total = 0.0                 # initialize the running sum to zero

    for i in range(ng):
        # --- Transform the ordinate from [-1,+1] to [0, pi/2] ---
        theta = conv1 * xg[i] + conv2

        # --- Evaluate the integrand: gL(theta) * sin(theta) ---
        # gL_func(theta) calls whichever distribution function was passed in.
        # math.sin(theta) gives the sine factor from the normalization integral.
        integrand = gL_func(theta) * math.sin(theta)

        # --- Accumulate: weight * function value ---
        total = total + wg[i] * integrand

    # -------------------------------------------------------------------------
    # Step 3: Multiply by the Jacobian (conv1) to account for the
    #         change of variable from [-1,+1] to [0, pi/2].
    # -------------------------------------------------------------------------
    total = total * conv1

    return total


# =============================================================================
# MAIN PROGRAM
# =============================================================================
# This block runs when you execute the file directly:
#   python 02-Leaf_Angle_Distributions.py
#
# It does NOT run if this file is imported as a module by another script.
# The 'if __name__ == "__main__"' pattern is standard Python practice.
# =============================================================================

if __name__ == "__main__":

    # =========================================================================
    # PART 1: SET UP AND VERIFY GAUSS QUADRATURE
    # =========================================================================

    print("=" * 60)
    print("  LEAF ANGLE DISTRIBUTION — NORMALIZATION VERIFICATION")
    print("=" * 60)
    print()

    # --- Get the quadrature ordinates and weights ---
    # The function returns TWO values. Python lets us "unpack" them
    # into two separate variables in one line:
    xg, wg = gauss_quad(NG)

    # --- Print the quadrature points ---
    # f-strings: f"text {variable}" embeds the variable's value in the string.
    # ':>6' means right-align in 6 characters; ':.9f' means 9 decimal places.
    print(f"  Quadrature Order: {NG}")
    print(f"  {'Point':>6}  {'Ordinate (xg)':>15}  {'Weight (wg)':>15}")
    print(f"  {'-'*6}  {'-'*15}  {'-'*15}")

    # 'zip(xg, wg)' pairs elements: (xg[0],wg[0]), (xg[1],wg[1]), ...
    # 'enumerate' adds an index:    (0, (xg[0],wg[0])), (1, (xg[1],wg[1])), ...
    for i, (x, w) in enumerate(zip(xg, wg)):
        print(f"  {i+1:>6}  {x:>15.9f}  {w:>15.9f}")
    print()

    # --- Verify the quadrature ---
    print("  --- Quadrature Checks ---")
    check_quad(NG, xg, wg)
    print()

    # =========================================================================
    # PART 2: VERIFY NORMALIZATION OF LEAF ANGLE DISTRIBUTIONS
    # =========================================================================
    # We want to show that for each distribution gL:
    #
    #       pi/2
    #      /
    #      |  gL(theta_L) * sin(theta_L) d(theta_L)  =  1
    #      |
    #     / 0
    #
    # The integration limits are 0 (horizontal) to pi/2 (vertical).
    # =========================================================================

    print("  --- Leaf Angle Distribution Normalization ---")
    print()
    print("  Verifying:  integral from 0 to pi/2 of")
    print("              gL(theta_L) * sin(theta_L) d(theta_L) = 1")
    print()

    # Define the integration limits (in radians).
    lower = 0.0                 # 0 radians = 0 degrees  (horizontal)
    upper = PI / 2.0            # pi/2 radians = 90 degrees (vertical)

    # -------------------------------------------------------------------------
    # Here we create a list of "tuples". A tuple is like a list but uses
    # parentheses () and cannot be modified after creation.
    #
    # Each tuple contains:
    #   (1) A descriptive name (string)
    #   (2) The mathematical formula as a string (for display)
    #   (3) The actual Python function to integrate
    #
    # This is a very Pythonic pattern: store related data together and
    # loop over it, instead of writing repetitive code for each case.
    # -------------------------------------------------------------------------

    distributions = [
        ("Uniform",      "gL = 1",                    gL_uniform),
        ("Planophile",   "gL = 3 cos^2(theta)",       gL_planophile),
        ("Erectophile",  "gL = (3/2) sin^2(theta)",   gL_erectophile),
        ("Plagiophile",  "gL = (15/8) sin^2(2theta)", gL_plagiophile),
        ("Extremophile", "gL = (15/7) cos^2(2theta)", gL_extremophile),
    ]

    # -------------------------------------------------------------------------
    # Loop over each distribution and compute the integral.
    #
    # 'for name, formula, func in distributions' UNPACKS each tuple:
    #   name    gets the string like "Uniform"
    #   formula gets the string like "gL = 1"
    #   func    gets the actual function like gL_uniform
    #
    # This is called "tuple unpacking" and is a very handy Python feature.
    # -------------------------------------------------------------------------

    # Print a nice table header.
    # ':<15' means left-align in 15 characters; ':>10' means right-align in 10.
    print(f"  {'Distribution':<15} {'Formula':<28} {'Integral':>10} {'(=1.0?)'}")
    print(f"  {'-'*15} {'-'*28} {'-'*10} {'-'*7}")

    for name, formula, func in distributions:

        # Call our integration function, passing the distribution function
        # as an argument. This is the key Python concept: "functions as arguments".
        result = integrate_distribution(NG, xg, wg, func, lower, upper)

        # Print the result. ':.6f' formats to 6 decimal places.
        print(f"  {name:<15} {formula:<28} {result:>10.6f}")

    print()
    print("=" * 60)
    print("  All integrals should equal 1.0 — normalization verified!")
    print("=" * 60)

    # =========================================================================
    # PART 3: VERIFY NORMALIZATION OF LEAF AZIMUTHAL DISTRIBUTION
    # =========================================================================
    # The non-uniform leaf normal azimuthal distribution function hL is a
    # probability density function (PDF) defined as:
    #
    #   (1 / 2*pi) * hL(phi_L, phi) = (1 / pi) * cos^2(phi - phi_L - eta)
    #
    # where:
    #   phi_L  = leaf normal azimuth angle (the variable of integration)
    #   phi    = solar azimuth angle (a parameter, not integrated over)
    #   eta    = difference between the azimuth of the maximum of hL
    #            and the solar azimuth
    #
    # Two biologically meaningful cases of eta:
    #   (a) Diaheliotropic  : eta = 0     — leaves orient to MAXIMIZE
    #       the projected leaf area to the incident photon stream
    #   (b) Paraheliotropic : eta = pi/2  — leaves orient to MINIMIZE
    #       the projected leaf area to the incident photon stream
    #
    # The normalization condition for this PDF is:
    #
    #        2*pi
    #       /
    #  1    |
    # ---- *|  hL(phi_L)  d(phi_L)  =  1
    # 2*pi  |
    #       / 0
    #
    # Substituting the definition of hL, this becomes:
    #
    #        2*pi
    #       /
    #  1    |                                        2
    # ---- *|  2 * cos^2(phi - phi_L - eta)  d(phi_L)  =  ---  * I
    #  pi   |                                               pi
    #       / 0
    #
    # where I = integral from 0 to 2*pi of cos^2(phi - phi_L - eta) d(phi_L).
    #
    # Analytically, this integral equals pi for any values of phi and eta,
    # so (2/pi) * pi = 2, and (1/2*pi) * 2*pi * (2/pi) ... let's just
    # verify numerically that the full normalization equals 1.
    # =========================================================================

    print()
    print()
    print("=" * 60)
    print("  LEAF AZIMUTHAL DISTRIBUTION — NORMALIZATION VERIFICATION")
    print("=" * 60)
    print()
    print("  Verifying:  (1/2*pi) * integral from 0 to 2*pi of")
    print("              hL(phi_L) d(phi_L) = 1")
    print()
    print("  where (1/2*pi) * hL = (1/pi) * cos^2(phi - phi_L - eta)")
    print()

    # -------------------------------------------------------------------------
    # Integration limits for the azimuthal distribution.
    # The leaf azimuth angle phi_L ranges from 0 to 2*pi (full circle).
    # -------------------------------------------------------------------------
    lower_az = 0.0              # lower limit: 0 radians
    upper_az = 2.0 * PI         # upper limit: 2*pi radians

    # -------------------------------------------------------------------------
    # We choose a representative solar azimuth angle phi.
    # The normalization should hold for ANY value of phi; here we pick
    # phi = pi/4 (45 degrees) as a non-trivial test value.
    # -------------------------------------------------------------------------
    phi_sun = PI / 4.0          # solar azimuth = 45 degrees (arbitrary choice)

    # -------------------------------------------------------------------------
    # Define the two eta cases to test.
    # Each tuple contains: (description, eta value)
    # -------------------------------------------------------------------------
    eta_cases = [
        ("Diaheliotropic",  0.0),           # eta = 0:     maximize projected area
        ("Paraheliotropic", PI / 2.0),      # eta = pi/2:  minimize projected area
    ]

    # -------------------------------------------------------------------------
    # Print the table header for azimuthal normalization results.
    # -------------------------------------------------------------------------
    print(f"  Solar azimuth phi = pi/4 = {phi_sun:.6f} rad")
    print()
    print(f"  {'Case':<20} {'eta':>10} {'Integral':>12} {'(=1.0?)'}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*7}")

    # -------------------------------------------------------------------------
    # Loop over each eta case and compute the normalization integral.
    #
    # For each case, we numerically evaluate:
    #
    #        2*pi
    #   1   /
    #  --- * |  hL(phi_L)  d(phi_L)
    #  2*pi  |
    #        / 0
    #
    # From the definition:  (1/2*pi) * hL = (1/pi) * cos^2(phi - phi_L - eta)
    # so:  hL = (2*pi/pi) * cos^2(phi - phi_L - eta) = 2 * cos^2(...)
    #
    # Therefore the integrand of the normalization integral
    # (1/2*pi) * hL(phi_L) is:
    #
    #   (1/pi) * cos^2(phi - phi_L - eta)
    #
    # We integrate this over phi_L from 0 to 2*pi using the same
    # Gauss-Legendre quadrature machinery already defined above.
    # -------------------------------------------------------------------------

    for case_name, eta in eta_cases:

        # --- Compute the transformation constants for [0, 2*pi] ---
        # Same change-of-variable as before, but now the interval is
        # [0, 2*pi] instead of [0, pi/2].
        conv1_az = (upper_az - lower_az) / 2.0     # half-width = pi
        conv2_az = (upper_az + lower_az) / 2.0     # midpoint   = pi

        # --- Numerical integration loop ---
        total_az = 0.0

        for i in range(NG):
            # Transform the ordinate from [-1, +1] to [0, 2*pi]
            phi_L = conv1_az * xg[i] + conv2_az

            # Evaluate the integrand: (1/pi) * cos^2(phi - phi_L - eta)
            # This is the normalized PDF (1/2*pi) * hL(phi_L).
            argument = phi_sun - phi_L - eta
            integrand_az = (1.0 / PI) * math.cos(argument) ** 2

            # Accumulate the weighted sum
            total_az = total_az + wg[i] * integrand_az

        # Multiply by the Jacobian (conv1_az = pi) for the change of variable
        total_az = total_az * conv1_az

        # --- Print the result ---
        print(f"  {case_name:<20} {eta:>10.4f} {total_az:>12.6f}")

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Azimuthal integrals should equal 1.0 — normalization verified!")
    print("=" * 60)
