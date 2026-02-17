#!/usr/bin/env python3
"""
PROGRAM: quad_g_L_O_L.py

Converted from FORTRAN to Python.

This program illustrates:
  (1) How to obtain Gaussian quadrature points and weights [ng, xg, wg]
  (2) How to obtain leaf normal orientation probability density functions (pdfs):
      - PLANOPHILE leaf inclination pdf: gL(thetaL)
      - UNIFORM leaf azimuthal pdf: hL(phiL)

=============================================================================
WHAT IS GAUSSIAN QUADRATURE?
=============================================================================
Gaussian quadrature is a numerical method for approximating definite integrals.
Instead of summing up many tiny rectangles (like the Riemann sum), Gaussian
quadrature picks a small number of cleverly-chosen points (called "ordinates"
or "nodes") and assigns each a "weight". The integral is then approximated as:

    integral ≈ sum of [ weight(i) * function_value_at_node(i) ]

This gives very accurate results with far fewer points than simpler methods.

=============================================================================
WHAT IS A LEAF NORMAL ORIENTATION PDF?
=============================================================================
In vegetation canopy science, leaves are oriented in different directions.
The "leaf normal" is a vector pointing perpendicular to the leaf surface.
The pdf gL(thetaL) describes how likely it is that a leaf has a particular
inclination angle thetaL (measured from horizontal). A "planophile" canopy
has mostly horizontal leaves (small thetaL values are more probable).

=============================================================================
PYTHON BASICS FOR BEGINNERS (used in this program):
=============================================================================

1. IMPORTS: 
   "import numpy as np" loads the NumPy library, which provides arrays 
   and math functions. We refer to it as "np" for short.

2. VARIABLES:
   In Python, you just write:  x = 5
   No need to declare types like in FORTRAN.

3. ARRAYS (lists and NumPy arrays):
   - Python lists:       my_list = [1, 2, 3]
   - NumPy arrays:       my_array = np.array([1.0, 2.0, 3.0])
   NumPy arrays are preferred for math because they support element-wise
   operations (e.g., my_array * 2 gives [2.0, 4.0, 6.0]).

4. INDEXING:
   Python arrays start at index 0 (not 1 like FORTRAN).
   So my_array[0] is the first element, my_array[1] is the second, etc.

5. FUNCTIONS:
   Defined with "def function_name(arguments):".
   They "return" values instead of modifying arguments in place (usually).

6. FOR LOOPS:
   "for i in range(n):" loops with i = 0, 1, 2, ..., n-1.
   "for i in range(a, b):" loops with i = a, a+1, ..., b-1.

7. PRINTING:
   print("text", variable) prints text and a variable's value.

8. f-STRINGS:
   print(f"The value is {x:.4f}") prints x with 4 decimal places.
   The f before the quotes means "formatted string".
"""

# =============================================================================
# IMPORT SECTION
# =============================================================================
# "import" brings in external libraries (collections of pre-written code).
# "numpy" is the fundamental package for numerical computing in Python.
# "as np" gives it a short nickname so we can write np.cos() instead of numpy.cos().

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================
# In Python, constants are just regular variables. By convention, we write
# them in ALL_CAPS to signal "don't change this value".
#
# np.pi is NumPy's built-in value of pi (3.141592653589793...), which is
# more precise than typing it out manually.

PI = np.pi                      # The mathematical constant pi ≈ 3.14159...
DEG_TO_RAD = PI / 180.0        # Multiply degrees by this to get radians
RAD_TO_DEG = 180.0 / PI        # Multiply radians by this to get degrees
NG = 8                          # Quadrature order (number of points to use)


# =============================================================================
# FUNCTION: gauss_quad
# =============================================================================
def gauss_quad(ng):
    """
    Obtain Gauss-Legendre quadrature ordinates (nodes) and weights.

    WHAT THIS FUNCTION DOES:
    ------------------------
    Given an order 'ng', this function returns two arrays:
      - xg: the quadrature nodes (x-positions where we evaluate the function)
      - wg: the quadrature weights (how much each node contributes to the sum)

    These are pre-computed values from mathematical tables. The nodes and
    weights are symmetric around zero on the interval [-1, +1].

    PARAMETERS:
    -----------
    ng : int
        The quadrature order. Must be one of: 4, 6, 8, 10, or 12.
        A higher order gives a more accurate approximation of the integral,
        but uses more evaluation points.

    RETURNS:
    --------
    xg : numpy array of length ng
        The quadrature ordinates (nodes), values between -1 and +1.
    wg : numpy array of length ng
        The quadrature weights, positive values that sum to 2.0.

    EXAMPLE USAGE:
    --------------
        xg, wg = gauss_quad(4)
        print(xg)  # prints the 4 node positions
        print(wg)  # prints the 4 weights
    """

    # -------------------------------------------------------------------------
    # PRE-COMPUTED TABLE VALUES
    # -------------------------------------------------------------------------
    # These are the "negative half" of the nodes and weights for orders 4–12.
    # Gauss-Legendre quadrature nodes are symmetric: if x is a node, so is -x,
    # and both have the same weight. So we only store the negative half and
    # mirror them to get the positive half.
    #
    # Each Python list below is written on multiple lines for readability.
    # The backslash (\) is NOT needed in Python when you're inside brackets.

    xx = [
        -0.861136312, -0.339981044,                                     # ng=4  (2 negative nodes)
        -0.9324695,   -0.6612094,   -0.2386192,                        # ng=6  (3 negative nodes)
        -0.960289856, -0.796666477, -0.525532410, -0.183434642,        # ng=8  (4 negative nodes)
        -0.973906529, -0.865063367, -0.679409568, -0.433395394,        # ng=10 (5 negative nodes)
        -0.148874339,
        -0.981560634, -0.904117256, -0.769902674, -0.587317954,        # ng=12 (6 negative nodes)
        -0.367831499, -0.125233409
    ]

    ww = [
        0.347854845, 0.652145155,                                       # ng=4
        0.1713245,   0.3607616,   0.4679139,                           # ng=6
        0.101228536, 0.222381034, 0.313706646, 0.362683783,            # ng=8
        0.066671344, 0.149451349, 0.219086363, 0.269266719,            # ng=10
        0.295524225,
        0.047175336, 0.106939326, 0.160078329, 0.203167427,            # ng=12
        0.233492537, 0.249147046
    ]

    # -------------------------------------------------------------------------
    # SHIFT TABLE
    # -------------------------------------------------------------------------
    # This dictionary maps ng/2 to the starting index in the xx and ww arrays.
    # For example, ng=4 means ng/2=2, and the ng=4 data starts at index 0 in xx.
    # ng=6 means ng/2=3, and the ng=6 data starts at index 2 in xx.
    #
    # A "dictionary" in Python is like a lookup table:  {key: value, ...}

    ishift = {
        2: 0,       # ng=4:  start at index 0
        3: 2,       # ng=6:  start at index 2
        4: 5,       # ng=8:  start at index 5
        5: 9,       # ng=10: start at index 9
        6: 14       # ng=12: start at index 14
    }

    # -------------------------------------------------------------------------
    # BUILD THE FULL NODE AND WEIGHT ARRAYS
    # -------------------------------------------------------------------------
    # We create arrays of zeros with length ng, then fill them in.

    xg = np.zeros(ng)      # np.zeros(ng) creates an array of ng zeros: [0.0, 0.0, ...]
    wg = np.zeros(ng)

    ng2 = ng // 2           # "//" is integer division in Python (drops the remainder)
                             # e.g., 4 // 2 = 2,  5 // 2 = 2

    # First, copy the negative-half nodes and their weights from the tables.
    # range(ng2) gives: 0, 1, ..., ng2-1

    for i in range(ng2):
        xg[i] = xx[i + ishift[ng2]]    # Look up the right section of the table
        wg[i] = ww[i + ishift[ng2]]

    # Now mirror to get the positive-half nodes.
    # The positive nodes are the negatives of the negative nodes, in reverse order.
    # The weights are the same (just mirrored).
    #
    # In FORTRAN this was: xg(i) = -xg(ng+1-i)
    # In Python (0-indexed): xg[i] = -xg[ng-1-i]

    for i in range(ng2, ng):            # i goes from ng2 to ng-1
        xg[i] = -xg[ng - 1 - i]        # Mirror the node (flip sign)
        wg[i] =  wg[ng - 1 - i]        # Copy the weight (same value)

    # Return both arrays. In Python, you can return multiple values separated
    # by commas. The caller receives them as a "tuple" which can be unpacked:
    #     xg, wg = gauss_quad(4)

    return xg, wg


# =============================================================================
# FUNCTION: check_quad
# =============================================================================
def check_quad(ng, xg, wg):
    """
    Check that the quadrature nodes and weights are correct.

    WHAT THIS FUNCTION DOES:
    ------------------------
    Two verification tests:

    Test 1: The weights should sum to exactly 2.0.
            (Gauss-Legendre quadrature integrates over [-1, +1], which has
             length 2, so the weights must sum to 2.)

    Test 2: The integral of x from 0 to 1 should equal 0.5.
            (Because the antiderivative of x is x²/2, so [1²/2 - 0²/2] = 0.5.)
            We approximate this by summing x(i)*w(i) for nodes in [0, 1].

    PARAMETERS:
    -----------
    ng : int
        Number of quadrature points.
    xg : numpy array
        The quadrature nodes.
    wg : numpy array
        The quadrature weights.

    RETURNS:
    --------
    Nothing. This function just prints the results of the checks.
    """

    # -------------------------------------------------------------------------
    # TEST 1: Do the weights sum to 2.0?
    # -------------------------------------------------------------------------
    # np.sum(wg) adds up all elements in the array wg.
    # This is equivalent to: wg[0] + wg[1] + wg[2] + ... + wg[ng-1]

    weight_sum = np.sum(wg)
    print(f" Qwts check (=2.0?): {weight_sum}")

    # -------------------------------------------------------------------------
    # TEST 2: Does the integral of x from 0 to 1 equal 0.5?
    # -------------------------------------------------------------------------
    # We only sum over the positive-half nodes (indices ng//2 to ng-1),
    # because those are the nodes in the interval [0, 1].
    #
    # np.sum(xg[ng//2:] * wg[ng//2:]) means:
    #   - xg[ng//2:] takes elements from index ng//2 to the end (this is "slicing")
    #   - The "*" multiplies element-by-element (not matrix multiplication)
    #   - np.sum() adds up all the products

    integral_sum = np.sum(xg[ng // 2:] * wg[ng // 2:])
    print(f" Qord check (=0.5?): {integral_sum}")


# =============================================================================
# FUNCTION: example_integral
# =============================================================================
def example_integral(ng, xg, wg):
    """
    Demonstrate how to compute a definite integral using Gaussian quadrature.

    WHAT THIS FUNCTION DOES:
    ------------------------
    Computes the integral:  ∫ from A to B of |x| dx,  where A = -1 and B = +1.

    The exact answer is 1.0 (area of two triangles, each with base 1 and height 1).

    HOW TO CHANGE THE INTEGRATION LIMITS:
    --------------------------------------
    Gauss-Legendre quadrature nodes are defined on [-1, +1]. To integrate over
    a different interval [A, B], we apply a linear change of variables:

        new_x = conv1 * xg[i] + conv2

    where:
        conv1 = (B - A) / 2       (scaling factor)
        conv2 = (B + A) / 2       (shifting factor)

    And the integral becomes:
        integral = conv1 * sum( f(new_x[i]) * wg[i] )

    Don't forget to multiply by conv1 at the end!

    PARAMETERS:
    -----------
    ng : int
        Number of quadrature points.
    xg : numpy array
        The quadrature nodes.
    wg : numpy array
        The quadrature weights.

    RETURNS:
    --------
    Nothing. Prints the computed integral for verification.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Define the integration limits and conversion factors
    # -------------------------------------------------------------------------

    upper_limit = 1.0       # B: upper bound of the integral
    lower_limit = -1.0      # A: lower bound of the integral

    conv1 = (upper_limit - lower_limit) / 2.0      # = (1 - (-1)) / 2 = 1.0
    conv2 = (upper_limit + lower_limit) / 2.0      # = (1 + (-1)) / 2 = 0.0

    # -------------------------------------------------------------------------
    # STEP 2: Evaluate the integral
    # -------------------------------------------------------------------------
    # For each quadrature node, transform it to the [A, B] interval,
    # evaluate the function |x| at that point, multiply by the weight,
    # and accumulate into the sum.

    total = 0.0                                     # Initialize the running sum to zero

    for i in range(ng):                             # Loop over all ng quadrature points
        new_ord = conv1 * xg[i] + conv2            # Transform node to [A, B] interval
        total = total + abs(new_ord) * wg[i]        # Add |x| * weight to the sum
        #
        # "abs()" is Python's built-in absolute value function.
        # abs(-3) returns 3, abs(5) returns 5.

    # -------------------------------------------------------------------------
    # STEP 3: Apply the final scaling factor
    # -------------------------------------------------------------------------
    # IMPORTANT: You must multiply the sum by conv1 to complete the
    # change of variables. Forgetting this is a common mistake!

    total = total * conv1

    print(f" Intg check (=1.0?): {total}")


# =============================================================================
# FUNCTION: leaf_normal_pdf
# =============================================================================
def leaf_normal_pdf(ng, xg, wg):
    """
    Compute the leaf normal orientation probability density functions (pdfs).

    WHAT THIS FUNCTION DOES:
    ------------------------
    Computes two pdfs that describe how leaves are oriented in a canopy:

    1. gL(thetaL) — the PLANOPHILE leaf inclination angle pdf.
       "Planophile" means the canopy has mostly horizontal leaves.
       The formula is:
           gL(thetaL) = (2/pi) * (1 + cos(2 * thetaL))
       where thetaL is the leaf inclination angle (0 = horizontal, pi/2 = vertical).

       This pdf peaks at thetaL=0 (horizontal) and equals zero at thetaL=pi/2
       (vertical), which matches the "planophile" (horizontal-loving) distribution.

    2. hL(phiL) — the UNIFORM leaf azimuthal angle pdf.
       This is simply 1.0 everywhere, meaning leaves face all compass directions
       equally (no preferred azimuthal direction).

    The function also checks that gL integrates to 1.0 over [0, pi/2],
    which is required for any valid probability density function.

    PARAMETERS:
    -----------
    ng : int
        Number of quadrature points.
    xg : numpy array
        The quadrature nodes.
    wg : numpy array
        The quadrature weights.

    RETURNS:
    --------
    gL : numpy array of length ng
        The planophile leaf inclination pdf evaluated at the quadrature nodes
        (mapped to the interval [0, pi/2]).
    hL : numpy array of length ng
        The uniform leaf azimuthal pdf (all values are 1.0).
    """

    # -------------------------------------------------------------------------
    # INITIALIZE hL: Uniform azimuthal pdf
    # -------------------------------------------------------------------------
    # np.ones(ng) creates an array of ng ones: [1.0, 1.0, 1.0, ...]
    # This represents a uniform (constant) probability density.

    hL = np.ones(ng)

    # -------------------------------------------------------------------------
    # COMPUTE gL: Planophile leaf inclination pdf
    # -------------------------------------------------------------------------
    # We need to evaluate gL at quadrature nodes mapped to [0, pi/2].
    # The Gauss quadrature nodes xg are on [-1, +1], so we transform them
    # to [0, pi/2] using the same conv1/conv2 technique as before.

    gL = np.zeros(ng)       # Create an array of zeros to hold gL values

    # STEP 1: Define integration limits and conversion factors

    upper_limit = PI / 2.0      # pi/2 radians = 90 degrees (vertical)
    lower_limit = 0.0           # 0 radians = 0 degrees (horizontal)

    conv1 = (upper_limit - lower_limit) / 2.0      # = pi/4
    conv2 = (upper_limit + lower_limit) / 2.0      # = pi/4

    # STEP 2: Evaluate gL at each transformed node and accumulate the integral

    total = 0.0

    for i in range(ng):
        # Transform the quadrature node from [-1,+1] to [0, pi/2]
        new_ord = conv1 * xg[i] + conv2

        # Evaluate the planophile pdf at this angle:
        #   gL = (2/pi) * (1 + cos(2 * thetaL))
        #
        # np.cos() computes the cosine of an angle in radians.

        gL[i] = (2.0 / PI) * (1.0 + np.cos(2.0 * new_ord))

        # Accumulate the weighted sum for the normalization check
        total = total + gL[i] * wg[i]

    # STEP 3: Apply the scaling factor to complete the integral

    total = total * conv1

    # Print the normalization check. For a valid pdf, this must equal 1.0.
    print(f" LNO  check (=1.0?): {total}")

    # Return both pdf arrays
    return gL, hL


# =============================================================================
# MAIN PROGRAM
# =============================================================================
# In Python, the code below runs when you execute this file directly
# (e.g., by typing "python quad_g_L_O_L.py" in the terminal).
#
# The special variable __name__ is set to "__main__" when the script is run
# directly. If this file were imported by another script, __name__ would be
# set to the module name instead, and this block would NOT run.
#
# This is a very common Python pattern called the "main guard".

if __name__ == "__main__":

    # Print a header so the user knows the program is running
    print("=" * 60)
    print("  QUAD_g_L(O_L) — Gaussian Quadrature & Leaf Normal PDFs")
    print("=" * 60)
    print()

    # STEP 1: Get the quadrature nodes and weights
    # The function returns two arrays, which we "unpack" into xg and wg.
    print("--- Quadrature Setup ---")
    xg, wg = gauss_quad(NG)

    # Print the nodes and weights so the user can see them
    # "enumerate()" gives both the index and the value in a loop:
    #   for index, value in enumerate(my_list):
    print()
    print(f"  Quadrature order (ng) = {NG}")
    print(f"  {'i':>3}  {'xg[i]':>12}  {'wg[i]':>12}")      # Column headers
    print(f"  {'---':>3}  {'--------':>12}  {'--------':>12}")
    for i, (x, w) in enumerate(zip(xg, wg)):
        # "zip(xg, wg)" pairs up corresponding elements: (xg[0],wg[0]), (xg[1],wg[1]), ...
        print(f"  {i:>3}  {x:>12.6f}  {w:>12.6f}")
    print()

    # STEP 2: Check the quadrature
    print("--- Quadrature Checks ---")
    check_quad(NG, xg, wg)
    print()

    # STEP 3: Example integral
    print("--- Example Integral: int_{-1}^{+1} |x| dx ---")
    example_integral(NG, xg, wg)
    print()

    # STEP 4: Compute the leaf normal pdfs
    print("--- Leaf Normal Orientation PDFs ---")
    gL, hL = leaf_normal_pdf(NG, xg, wg)

    # Print the pdf values
    print()
    print(f"  {'i':>3}  {'gL[i]':>12}  {'hL[i]':>12}")
    print(f"  {'---':>3}  {'--------':>12}  {'--------':>12}")
    for i in range(NG):
        print(f"  {i:>3}  {gL[i]:>12.6f}  {hL[i]:>12.6f}")
    print()

    # Program complete
    print("=" * 60)
    print("  Program finished successfully.")
    print("=" * 60)
