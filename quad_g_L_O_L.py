#!/usr/bin/env python3
"""
PROGRAM: quad_g_L_O_L.py

This program is a Python translation of the Fortran program QUAD_g_L(O_L).

It illustrates:
  (1) How to obtain Gaussian quadrature points and weights [ng, xg, wg]
  (2) How to obtain leaf normal orientation probability density functions (PDFs):
      - PLANOPHILE leaf inclination PDF: gL(thetaL)
      - UNIFORM leaf azimuthal PDF: hL(phiL)

Gaussian quadrature is a numerical method for approximating definite integrals.
Instead of summing many tiny rectangles (like simple Riemann sums), it picks
a small number of cleverly chosen points ("ordinates") and weights, so that
the weighted sum gives an exact answer for polynomials up to a certain degree.

In vegetation science, "leaf normal orientation" describes which direction
leaves face. The "planophile" distribution means leaves tend to be horizontal
(like a flat plate facing up), which is common in many broadleaf plants.
"""

# =============================================================================
# IMPORTS
# =============================================================================

# 'math' is Python's built-in library for mathematical functions like cos, pi, etc.
import math


# =============================================================================
# CONSTANTS
# =============================================================================

# math.pi gives us the value of pi (3.141592654...) to full machine precision.
# In the original Fortran code, PI was defined manually as 3.141592654.
PI = math.pi

# Conversion factors between degrees and radians.
# Radians are the "natural" unit for angles in math; there are 2*pi radians in 360 degrees.
# To convert degrees to radians: multiply by (pi / 180)
# To convert radians to degrees: multiply by (180 / pi)
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# ng is the "order" of the Gaussian quadrature, i.e., how many sample points we use.
# Higher ng = more accurate integration, but also more computation.
# The original Fortran code uses ng = 12.
NG = 12


# =============================================================================
# FUNCTION: gauss_quad
# =============================================================================

def gauss_quad(ng):
    """
    Obtain Gauss-Legendre quadrature ordinates (xg) and weights (wg) for a
    given quadrature order (ng).

    Parameters
    ----------
    ng : int
        The quadrature order. Must be one of: 4, 6, 8, 10, or 12.
        This is the number of points at which we will evaluate the function
        we want to integrate.

    Returns
    -------
    xg : list of float
        The quadrature ordinates (also called "nodes" or "abscissas").
        These are the specific x-values where we evaluate our function.
        They live in the interval [-1, +1].
    wg : list of float
        The quadrature weights. Each weight tells us how much "importance"
        to give the function value at the corresponding ordinate.

    How Gaussian Quadrature Works (Briefly)
    ----------------------------------------
    To approximate an integral from -1 to +1 of f(x) dx, we compute:

        integral ≈ sum over i of [ wg[i] * f(xg[i]) ]

    The magic is that the xg and wg values are chosen so this sum is EXACT
    for any polynomial of degree up to (2*ng - 1). That's remarkably accurate
    for smooth functions even with just a few points.

    Note on Storage
    ----------------
    The Fortran code stores only the NEGATIVE-half ordinates and their weights,
    because Gauss-Legendre points are symmetric about zero. The positive-half
    points are simply the negatives of the negative-half points (mirrored),
    and they share the same weights. This saves storage space.
    """

    # -------------------------------------------------------------------------
    # These are the pre-computed negative-half ordinates for orders 4 through 12.
    # They come from mathematical tables of Gauss-Legendre quadrature.
    # Each group of numbers corresponds to a different quadrature order.
    # -------------------------------------------------------------------------
    xx = [
        -0.861136312, -0.339981044,                                # ng=4  (2 negative points)
        -0.9324695,   -0.6612094,   -0.2386192,                   # ng=6  (3 negative points)
        -0.960289856, -0.796666477, -0.525532410, -0.183434642,   # ng=8  (4 negative points)
        -0.973906529, -0.865063367, -0.679409568, -0.433395394,   # ng=10 (5 negative points)
        -0.148874339,
        -0.981560634, -0.904117256, -0.769902674, -0.587317954,   # ng=12 (6 negative points)
        -0.367831499, -0.125233409
    ]

    # -------------------------------------------------------------------------
    # These are the pre-computed weights corresponding to the ordinates above.
    # Each weight is paired with the ordinate at the same position.
    # -------------------------------------------------------------------------
    ww = [
        0.347854845, 0.652145155,                                  # ng=4
        0.1713245,   0.3607616,   0.4679139,                      # ng=6
        0.101228536, 0.222381034, 0.313706646, 0.362683783,       # ng=8
        0.066671344, 0.149451349, 0.219086363, 0.269266719,       # ng=10
        0.295524225,
        0.047175336, 0.106939326, 0.160078329, 0.203167427,       # ng=12
        0.233492537, 0.249147046
    ]

    # -------------------------------------------------------------------------
    # 'ishift' tells us where in the xx/ww arrays each quadrature order starts.
    # This is an "offset" or "index shift" so we can look up the right group.
    #
    # For ng=4:  ng//2 = 2, ishift[2] = 0  → start at index 0 in xx/ww
    # For ng=6:  ng//2 = 3, ishift[3] = 2  → start at index 2
    # For ng=8:  ng//2 = 4, ishift[4] = 5  → start at index 5
    # For ng=10: ng//2 = 5, ishift[5] = 9  → start at index 9
    # For ng=12: ng//2 = 6, ishift[6] = 14 → start at index 14
    #
    # Note: ishift has 7 entries (indices 0-6). Index 0 and 1 are unused placeholders.
    # -------------------------------------------------------------------------
    ishift = [0, 0, 0, 2, 5, 9, 14]

    # ng2 is half of ng. Since Gauss-Legendre points are symmetric,
    # we only need to store half the points and mirror the rest.
    ng2 = ng // 2  # '//' is Python's integer (floor) division operator

    # -------------------------------------------------------------------------
    # Initialize xg and wg as lists of zeros with 'ng' elements.
    # In Python, [0.0] * ng creates a list like [0.0, 0.0, 0.0, 0.0] for ng=4.
    # -------------------------------------------------------------------------
    xg = [0.0] * ng
    wg = [0.0] * ng

    # -------------------------------------------------------------------------
    # Fill in the first half (the negative ordinates and their weights)
    # by copying from our pre-computed tables.
    # -------------------------------------------------------------------------
    for i in range(ng2):
        # 'i + ishift[ng2]' picks the correct starting position in xx/ww.
        xg[i] = xx[i + ishift[ng2]]
        wg[i] = ww[i + ishift[ng2]]

    # -------------------------------------------------------------------------
    # Fill in the second half (the positive ordinates) by mirroring.
    # The positive ordinates are the negatives of the negative ordinates,
    # read in reverse order. The weights are the same (just mirrored).
    #
    # For ng=4:
    #   xg[0], xg[1] are negative (already filled above)
    #   xg[2] = -xg[1]  (mirror of 2nd negative point)
    #   xg[3] = -xg[0]  (mirror of 1st negative point)
    #   wg[2] =  wg[1], wg[3] = wg[0]
    # -------------------------------------------------------------------------
    for i in range(ng2, ng):
        # 'ng - 1 - i' counts backwards from the first half.
        # Example for ng=4: when i=2, ng-1-i = 1; when i=3, ng-1-i = 0.
        xg[i] = -xg[ng - 1 - i]
        wg[i] =  wg[ng - 1 - i]

    # Return the complete arrays of ordinates and weights.
    return xg, wg


# =============================================================================
# FUNCTION: check_quad
# =============================================================================

def check_quad(ng, xg, wg):
    """
    Verify that the quadrature points and weights are correct by running
    two simple tests.

    Test 1: The sum of ALL weights should equal 2.0.
            This is because Gauss-Legendre quadrature integrates over [-1, +1],
            and the integral of the constant function f(x) = 1 over [-1, +1]
            equals 2. Since the weights must reproduce this exactly:
                sum(wg) = integral from -1 to +1 of 1 dx = 2.0

    Test 2: The integral of x from 0 to +1 should equal 0.5.
            We only sum over the POSITIVE ordinates (the upper half, indices
            ng//2 through ng-1) to integrate from 0 to +1.
            integral from 0 to 1 of x dx = [x^2 / 2] from 0 to 1 = 0.5

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : list of float
        Quadrature ordinates.
    wg : list of float
        Quadrature weights.
    """

    # -------------------------------------------------------------------------
    # Test 1: Sum all the weights.
    # 'sum()' is a built-in Python function that adds up all elements in a list.
    # -------------------------------------------------------------------------
    weight_sum = sum(wg)
    print(f" Qwts check (=2.0?): {weight_sum}")
    # The 'f' before the string makes it an "f-string" (formatted string literal).
    # Inside curly braces {}, Python evaluates the expression and inserts the result.

    # -------------------------------------------------------------------------
    # Test 2: Compute integral of x from 0 to +1.
    # We sum xg[i] * wg[i] only for the positive-half ordinates.
    # In Python, range(ng//2, ng) gives indices from ng//2 up to ng-1.
    # For ng=4: range(2, 4) gives i = 2, 3.
    # -------------------------------------------------------------------------
    ordinate_sum = 0.0
    for i in range(ng // 2, ng):
        ordinate_sum += xg[i] * wg[i]
        # '+=' means "add the right side to the current value of the left side"
        # It's shorthand for: ordinate_sum = ordinate_sum + xg[i] * wg[i]

    print(f" Qord check (=0.5?): {ordinate_sum}")


# =============================================================================
# FUNCTION: example_integral
# =============================================================================

def example_integral(ng, xg, wg):
    """
    Demonstrate how to compute a definite integral with arbitrary bounds [A, B]
    using Gauss-Legendre quadrature.

    The example computes:  integral from -1 to +1 of |x| dx

    The exact answer is 1.0 (the area under |x| from -1 to +1 forms two
    right triangles, each with area 0.5).

    KEY CONCEPT: Change of Variables
    --------------------------------
    Gauss-Legendre quadrature is defined on the interval [-1, +1]. But what if
    we want to integrate over a different interval [A, B]?

    We use a linear change of variables:
        x_new = conv1 * xg[i] + conv2

    where:
        conv1 = (B - A) / 2     ... this "stretches" or "compresses" the interval
        conv2 = (B + A) / 2     ... this "shifts" the center of the interval

    This maps each quadrature ordinate xg[i] (which lives in [-1, +1]) to a
    new value x_new that lives in [A, B].

    We must also multiply the final sum by conv1 (the Jacobian of the
    transformation) to account for the change in interval width.

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : list of float
        Quadrature ordinates (in [-1, +1]).
    wg : list of float
        Quadrature weights.
    """

    # Define the limits of integration.
    upper_limit = 1.0
    lower_limit = -1.0

    # -------------------------------------------------------------------------
    # Step 1: Compute the conversion factors for the change of variables.
    # -------------------------------------------------------------------------
    conv1 = (upper_limit - lower_limit) / 2.0   # = (1 - (-1)) / 2 = 1.0
    conv2 = (upper_limit + lower_limit) / 2.0   # = (1 + (-1)) / 2 = 0.0

    # -------------------------------------------------------------------------
    # Step 2: Perform the numerical integration.
    # For each quadrature point, we:
    #   (a) Map xg[i] from [-1,+1] to the actual integration interval [A,B]
    #   (b) Evaluate the function |x| at this mapped point
    #   (c) Multiply by the weight wg[i] and accumulate the sum
    # -------------------------------------------------------------------------
    total = 0.0
    for i in range(ng):
        # Map the quadrature ordinate to the actual interval [lower_limit, upper_limit].
        new_ordinate = conv1 * xg[i] + conv2

        # Evaluate |x| at the mapped point and accumulate the weighted sum.
        # 'abs()' is Python's built-in absolute value function.
        total += abs(new_ordinate) * wg[i]

    # -------------------------------------------------------------------------
    # Step 3: Multiply by conv1 (the Jacobian factor) to complete the
    # change-of-variables correction.
    # -------------------------------------------------------------------------
    total *= conv1
    # '*=' means "multiply the left side by the right side and store the result"
    # Equivalent to: total = total * conv1

    print(f" Intg check (=1.0?): {total}")


# =============================================================================
# FUNCTION: leaf_normal_pdf
# =============================================================================

def leaf_normal_pdf(ng, xg, wg):
    """
    Compute the leaf normal orientation probability density functions (PDFs):

    1) gL(thetaL) — the PLANOPHILE leaf inclination angle PDF
    2) hL(phiL)   — the UNIFORM leaf azimuthal angle PDF

    Background
    ----------
    In plant canopy science, we describe how leaves are oriented using two angles:

    - thetaL (theta_L): the "inclination angle" — how tilted the leaf is
      from horizontal. thetaL = 0 means perfectly horizontal (flat),
      thetaL = pi/2 means perfectly vertical (standing on edge).

    - phiL (phi_L): the "azimuthal angle" — which compass direction the
      leaf faces (north, south, east, west, etc.).

    A "planophile" canopy has leaves that prefer to be HORIZONTAL (flat).
    The PDF for this distribution is:

        gL(thetaL) = (2/pi) * (1 + cos(2 * thetaL))

    This function peaks at thetaL = 0 (horizontal) and equals zero at
    thetaL = pi/2 (vertical), confirming the preference for flat leaves.

    The azimuthal distribution is UNIFORM, meaning leaves face all compass
    directions equally — there's no preferred azimuth.

        hL(phiL) = 1.0   (constant for all directions)

    Normalization Check
    -------------------
    A valid PDF must integrate to 1.0 over its domain. For gL:

        integral from 0 to pi/2 of gL(thetaL) d(thetaL) = 1.0

    We verify this using Gauss-Legendre quadrature.

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : list of float
        Quadrature ordinates (in [-1, +1]).
    wg : list of float
        Quadrature weights.

    Returns
    -------
    gL : list of float
        The planophile leaf inclination PDF evaluated at each (mapped)
        quadrature point.
    hL : list of float
        The uniform leaf azimuthal PDF (all values are 1.0).
    """

    # -------------------------------------------------------------------------
    # Set hL to 1.0 for all quadrature points (uniform azimuthal distribution).
    # [1.0] * ng creates a list of ng ones, e.g., [1.0, 1.0, 1.0, 1.0] for ng=4.
    # -------------------------------------------------------------------------
    hL = [1.0] * ng

    # -------------------------------------------------------------------------
    # Now compute the planophile gL(thetaL) = (2/pi) * (1 + cos(2*thetaL))
    #
    # The integration domain is thetaL in [0, pi/2] (0 to 90 degrees).
    # We need to map our quadrature ordinates (which live in [-1, +1]) to
    # the interval [0, pi/2] using the same change-of-variables trick.
    # -------------------------------------------------------------------------

    upper_limit = PI / 2.0   # pi/2 radians = 90 degrees
    lower_limit = 0.0        # 0 radians = 0 degrees

    # Step 1: Compute conversion factors for mapping [-1,+1] → [0, pi/2].
    conv1 = (upper_limit - lower_limit) / 2.0   # = pi/4
    conv2 = (upper_limit + lower_limit) / 2.0   # = pi/4

    # -------------------------------------------------------------------------
    # Step 2: Evaluate gL at each mapped quadrature point and simultaneously
    #         compute the normalization integral to verify it equals 1.0.
    # -------------------------------------------------------------------------
    gL = [0.0] * ng   # Initialize gL as a list of zeros.
    normalization_sum = 0.0

    for i in range(ng):
        # Map the i-th quadrature ordinate from [-1,+1] to [0, pi/2].
        new_ordinate = conv1 * xg[i] + conv2

        # Evaluate the planophile PDF at this angle.
        # math.cos() computes the cosine of an angle given in RADIANS.
        gL[i] = (2.0 / PI) * (1.0 + math.cos(2.0 * new_ordinate))

        # Accumulate the weighted sum for the normalization check.
        normalization_sum += gL[i] * wg[i]

    # Step 3: Apply the Jacobian factor (conv1) to complete the integral.
    normalization_sum *= conv1

    print(f" LNO  check (=1.0?): {normalization_sum}")

    # Return both PDFs.
    return gL, hL


# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    """
    Main function — the entry point of the program.

    This is equivalent to the "PROGRAM" block in the original Fortran code.
    It calls each subroutine in sequence:
      1. Get the quadrature points and weights.
      2. Verify them with simple checks.
      3. Demonstrate an example integral.
      4. Compute and verify the leaf normal orientation PDFs.
    """

    # Step 1: Obtain Gauss-Legendre quadrature ordinates and weights.
    print("=" * 60)
    print(" Step 1: Obtaining Gauss-Legendre quadrature (ng = {})".format(NG))
    print("=" * 60)
    xg, wg = gauss_quad(NG)
    # 'xg, wg = ...' is called "tuple unpacking". The function returns two
    # values, and Python assigns the first to xg and the second to wg.

    # Print the ordinates and weights so we can see them.
    print("\n Ordinates (xg) and Weights (wg):")
    print(" {:>5s}   {:>12s}   {:>12s}".format("i", "xg[i]", "wg[i]"))
    # '{:>12s}' means: right-align the string in a field 12 characters wide.
    for i in range(NG):
        print(" {:>5d}   {:>12.8f}   {:>12.8f}".format(i, xg[i], wg[i]))
        # '{:>12.8f}' means: right-align a floating-point number, 12 chars wide,
        # 8 digits after the decimal point.
    print()

    # Step 2: Check that the quadrature is correct.
    print("=" * 60)
    print(" Step 2: Checking the quadrature")
    print("=" * 60)
    check_quad(NG, xg, wg)
    print()

    # Step 3: Demonstrate an example integral: integral of |x| from -1 to +1.
    print("=" * 60)
    print(" Step 3: Example integral of |x| from -1 to +1")
    print("=" * 60)
    example_integral(NG, xg, wg)
    print()

    # Step 4: Compute the leaf normal orientation PDFs.
    print("=" * 60)
    print(" Step 4: Leaf normal orientation PDFs")
    print("=" * 60)
    gL, hL = leaf_normal_pdf(NG, xg, wg)

    # Print the resulting PDF values for inspection.
    print("\n Planophile gL and Uniform hL at quadrature points:")
    print(" {:>5s}   {:>12s}   {:>12s}".format("i", "gL[i]", "hL[i]"))
    for i in range(NG):
        print(" {:>5d}   {:>12.8f}   {:>12.8f}".format(i, gL[i], hL[i]))
    print()

    print("Program complete.")


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# The block below is a standard Python idiom. It means:
#   "If this file is being run directly (not imported as a module), then
#    call the main() function."
#
# When you run a Python file, Python sets the special variable __name__ to
# the string "__main__". If instead you import this file from another script,
# __name__ would be set to the module's name, and main() would NOT run
# automatically.
# =============================================================================

if __name__ == "__main__":
    main()
