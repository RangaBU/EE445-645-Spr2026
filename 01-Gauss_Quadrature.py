#!/usr/bin/env python3
"""
GAUSS QUADRATURE PROGRAM
========================

This is a Python program that demonstrates Gauss-Legendre Quadrature 
a numerical method for approximating definite integrals (i.e., computing 
the area under a curve).

What is Gauss Quadrature?
-------------------------
Instead of dividing an interval into equal parts (like the trapezoidal rule),
Gauss Quadrature picks specially-chosen points ("ordinates") and weights
to get a much more accurate answer with fewer points.

The program does three things:
  (1) Sets up the quadrature ordinates (xg) and weights (wg) for order Ng
  (2) Checks that the quadrature is correctly set up
  (3) Performs a simple example integral to demonstrate usage

Author: Claude And I (Ranga B. Myneni)
"""

# =============================================================================
# IMPORTS
# =============================================================================
# 'math' is a built-in Python library that gives us mathematical functions
# like pi, sin, cos, abs, etc.
import math

# 'numpy' is a popular external library for numerical computing in Python.
# We import it with the shorthand 'np' so we can write np.array(...) instead
# of numpy.array(...). This is a very common convention in Python.
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================
# In Python, we define constants as regular variables (usually in ALL CAPS
# by convention). Python doesn't have a built-in "constant" type like Fortran's
# PARAMETER, but using ALL CAPS tells other programmers "don't change this."

PI = math.pi                        # pi = 3.141592653589793 
DEG_TO_RAD = PI / 180.0             # multiply degrees by this to get radians
RAD_TO_DEG = 180.0 / PI             # multiply radians by this to get degrees
NG = 12                             # quadrature order (number of points)


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
        The quadrature ordinates (the specially-chosen x-positions).
    wg : numpy array of floats
        The quadrature weights (how much "importance" each point gets).

    Notes
    -----
    - In Python, it's more natural to *return* new values from a function.
    - The ordinates and weights below are well-known tabulated values for
      Gauss-Legendre quadrature on the interval [-1, +1].
    """

    # -------------------------------------------------------------------------
    # Step 1: Define the tabulated (pre-computed) ordinates and weights.
    # -------------------------------------------------------------------------
    # These are the NEGATIVE-half ordinates for orders 4 through 12.
    # They are packed into one long list, and we use 'ishift' to find
    # where each order's values start.
    #
    # In Python, lists are created with square brackets [].
    # Indexing starts at 0, NOT 1 like in Fortran. This is very important!

    xx = [
        -0.861136312, -0.339981044,                                  # for ng=4  (2 values)
        -0.9324695,   -0.6612094,   -0.2386192,                     # for ng=6  (3 values)
        -0.960289856, -0.796666477, -0.525532410, -0.183434642,     # for ng=8  (4 values)
        -0.973906529, -0.865063367, -0.679409568, -0.433395394,     # for ng=10 (5 values)
        -0.148874339,
        -0.981560634, -0.904117256, -0.769902674, -0.587317954,     # for ng=12 (6 values)
        -0.367831499, -0.125233409
    ]

    ww = [
         0.347854845,  0.652145155,                                  # for ng=4  (2 weights)
         0.1713245,    0.3607616,    0.4679139,                      # for ng=6  (3 weights)
         0.101228536,  0.222381034,  0.313706646,  0.362683783,      # for ng=8  (4 weights)
         0.066671344,  0.149451349,  0.219086363,  0.269266719,      # for ng=10 (5 weights)
         0.295524225,
         0.047175336,  0.106939326,  0.160078329,  0.203167427,      # for ng=12 (6 weights)
         0.233492537,  0.249147046
    ]

    # 'ishift' tells us where in the xx/ww arrays each order's data begins.
    # The keys of this dictionary are ng/2 values (2, 3, 4, 5, 6).
    # In Python, a "dictionary" (dict) maps keys to values using curly braces {}.
    #
    # NOTE: Python uses 0-based indexing, so we subtract 1 from the
    # Fortran ishift values (which were 0, 0, 2, 5, 9, 14 for 1-based indexing).
    # Actually, ishift[1]=0 in Fortran corresponds to ng=2 (ng2=1), which we
    # don't use. The mapping below is adjusted for 0-based indexing.

    ishift = {
        2: 0,    # ng=4:  starts at index 0
        3: 2,    # ng=6:  starts at index 2
        4: 5,    # ng=8:  starts at index 5
        5: 9,    # ng=10: starts at index 9
        6: 14    # ng=12: starts at index 14
    }

    # -------------------------------------------------------------------------
    # Step 2: Validate the input
    # -------------------------------------------------------------------------
    # Check that ng is one of the allowed values.
    # 'assert' will stop the program with an error message if the condition is False.

    assert ng in [4, 6, 8, 10, 12], \
        f"Error: ng must be 4, 6, 8, 10, or 12. You provided ng={ng}."

    # -------------------------------------------------------------------------
    # Step 3: Extract the ordinates and weights for the requested order
    # -------------------------------------------------------------------------
    # ng2 is half the order. We only store the negative-side values because
    # Gauss-Legendre quadrature is symmetric: the positive side is just
    # the mirror image.
    #
    # '//' is integer division in Python (like dividing and rounding down).
    # Regular '/' gives a float (e.g., 12/2 = 6.0), but '//' gives an int (6).

    ng2 = ng // 2

    # Create empty lists to hold our ordinates and weights.
    # In Python, [] creates an empty list. We'll fill it with .append().
    xg = []
    wg = []

    # --- Fill in the first half (negative side) ---
    # 'range(ng2)' generates the numbers 0, 1, 2, ..., ng2-1.
    # This is like Fortran's "DO i = 1, ng2" but starting from 0.

    for i in range(ng2):
        xg.append(xx[i + ishift[ng2]])    # .append() adds a value to the end of a list
        wg.append(ww[i + ishift[ng2]])

    # --- Fill in the second half (positive side, by symmetry) ---
    # The positive ordinates are the negatives of the first half, reversed.
    # The weights are the same as the first half, reversed.
    #
    # In the Fortran code: xg(i) = -xg(ng+1-i)
    # In Python (0-based):  xg[i] = -xg[ng-1-i]  ... but we build it differently.

    for i in range(ng2):
        # 'ng2 - 1 - i' reverses the order: if ng2=6, this gives 5, 4, 3, 2, 1, 0
        xg.append(-xg[ng2 - 1 - i])
        wg.append( wg[ng2 - 1 - i])

    # -------------------------------------------------------------------------
    # Step 4: Convert lists to numpy arrays for easier math later
    # -------------------------------------------------------------------------
    # numpy arrays support element-wise operations: e.g., xg * wg multiplies
    # each element of xg by the corresponding element of wg.
    # Regular Python lists don't do this — they would concatenate instead!

    xg = np.array(xg)
    wg = np.array(wg)

    # Return both arrays. In Python, you can return multiple values separated
    # by commas. The caller receives them as a "tuple" which can be unpacked:
    #   xg, wg = gauss_quad(12)

    return xg, wg


# =============================================================================
# FUNCTION: check_quad
# =============================================================================
def check_quad(ng, xg, wg):
    """
    Check that the quadrature is correctly set up by verifying two properties:

      (1) The weights should sum to exactly 2.0
          (because Gauss-Legendre integrates over [-1, +1], and the
           integral of f(x)=1 over [-1,1] is 2.0)

      (2) The integral of x from 0 to 1 should equal 0.5
          (a simple test: area under the line y=x from 0 to 1)

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : numpy array
        Quadrature ordinates.
    wg : numpy array
        Quadrature weights.
    """

    # -------------------------------------------------------------------------
    # Check 1: Do the weights sum to 2.0?
    # -------------------------------------------------------------------------
    # np.sum() adds up all elements of an array.
    # This is equivalent to the Fortran DO loop that accumulates 'sum'.

    weight_sum = np.sum(wg)

    # 'f-strings' (formatted string literals) let you embed variables
    # directly into a string using {variable_name}. The 'f' before the
    # quote is required. Example: f"x = {x}" prints "x = 5" if x is 5.
    # The ':.6f' part formats the number to 6 decimal places.

    print(f"  Qwts check (=2.0?): {weight_sum:.6f}")

    # -------------------------------------------------------------------------
    # Check 2: Does the integral of x from 0 to 1 equal 0.5?
    # -------------------------------------------------------------------------
    # We only use the second half of the ordinates (the positive ones,
    # indices ng//2 through ng-1) because we're integrating from 0 to 1.
    #
    # In Python, array "slicing" lets us grab a portion of an array:
    #   xg[start:stop]  gives elements from index 'start' up to (but NOT
    #   including) index 'stop'.
    #
    # So xg[ng//2:] means "from index ng//2 to the end of the array".

    ng2 = ng // 2

    # Element-wise multiplication: xg[ng2:] * wg[ng2:] multiplies each
    # ordinate by its weight, then np.sum adds them all up.
    ordinate_check = np.sum(xg[ng2:] * wg[ng2:])

    print(f"  Qord check (=0.5?): {ordinate_check:.6f}")


# =============================================================================
# FUNCTION: example_integral
# =============================================================================
def example_integral(ng, xg, wg):
    """
    Demonstrate how to compute a definite integral using Gauss Quadrature.

    We compute:  integral from A to B of |x| dx,  where A = -1, B = +1.

    The exact answer is 1.0 (two triangles, each with area 0.5).

    KEY IDEA:
    ---------
    Gauss quadrature ordinates are defined on [-1, +1]. To integrate over
    a different interval [A, B], we need to transform the ordinates using:

        new_x = conv1 * xg[i] + conv2

    where:
        conv1 = (B - A) / 2      (scaling factor)
        conv2 = (B + A) / 2      (shifting factor)

    And the final integral must be multiplied by conv1.

    Parameters
    ----------
    ng : int
        Quadrature order.
    xg : numpy array
        Quadrature ordinates.
    wg : numpy array
        Quadrature weights.
    """

    # -------------------------------------------------------------------------
    # Step 1: Define the integration limits and conversion factors
    # -------------------------------------------------------------------------

    upper_limit = 1.0       # B
    lower_limit = -1.0      # A

    # conv1 scales the interval width: maps [-1,+1] to [A, B]
    conv1 = (upper_limit - lower_limit) / 2.0

    # conv2 shifts the interval center: the midpoint of [A, B]
    conv2 = (upper_limit + lower_limit) / 2.0

    # -------------------------------------------------------------------------
    # Step 2: Perform the numerical integration
    # -------------------------------------------------------------------------
    # For each quadrature point:
    #   1. Transform the ordinate to the actual integration interval
    #   2. Evaluate the function at that transformed point
    #   3. Multiply by the weight
    #   4. Add to the running sum

    total = 0.0     # This will accumulate our answer

    # 'range(ng)' generates 0, 1, 2, ..., ng-1
    for i in range(ng):
        # Transform the ordinate from [-1,+1] to [lower_limit, upper_limit]
        new_ordinate = conv1 * xg[i] + conv2

        # Evaluate our function |x| at the transformed point and multiply by weight.
        # 'abs()' is Python's built-in absolute value function.
        # In the Fortran code, this was: sum = sum + abs(neword*wg(i))
        total = total + abs(new_ordinate) * wg[i]

    # -------------------------------------------------------------------------
    # Step 3: Apply the final scaling factor
    # -------------------------------------------------------------------------
    # Don't forget to multiply by conv1! This accounts for the change of
    # variables from [-1,+1] to [A, B].

    total = total * conv1

    print(f"  Intg check (=1.0?): {total:.6f}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================
# In Python, the code below runs when you execute this file directly
# (e.g., by typing "python gauss_quadrature.py" in the terminal).
#
# The special variable __name__ is set to "__main__" when the file is
# run directly. If this file were imported by another Python file
# (using "import gauss_quadrature"), __name__ would be "gauss_quadrature"
# instead, and this block would NOT run. This is a very common Python pattern.

if __name__ == "__main__":

    print("=" * 50)              # Print a line of 50 '=' characters
    print("  GAUSS QUADRATURE PROGRAM")
    print("=" * 50)
    print()                      # Print a blank line

    # --- Step 1: Get the quadrature ordinates and weights ---
    # The function returns two values, which we "unpack" into xg and wg.
    xg, wg = gauss_quad(NG)

    # --- (Optional) Print the ordinates and weights ---
    # 'enumerate()' gives us both the index (i) and value as we loop.
    # This is the Pythonic way to loop when you need both index and value.
    print(f"  Quadrature Order: {NG}")
    print(f"  {'Point':>6}  {'Ordinate (xg)':>15}  {'Weight (wg)':>15}")
    print(f"  {'-'*6}  {'-'*15}  {'-'*15}")

    for i, (x, w) in enumerate(zip(xg, wg)):
        # 'zip(xg, wg)' pairs up corresponding elements: (xg[0], wg[0]), (xg[1], wg[1]), ...
        # 'enumerate' then adds an index: (0, (xg[0], wg[0])), (1, (xg[1], wg[1])), ...
        # ':>6' means right-align in a field 6 characters wide.
        # ':>15.9f' means right-align, 15 chars wide, 9 decimal places.
        print(f"  {i+1:>6}  {x:>15.9f}  {w:>15.9f}")

    print()

    # --- Step 2: Check the quadrature ---
    print("  --- Quadrature Checks ---")
    check_quad(NG, xg, wg)
    print()

    # --- Step 3: Run the example integral ---
    print("  --- Example Integral: int_{-1}^{+1} |x| dx ---")
    example_integral(NG, xg, wg)
    print()

    print("=" * 50)
    print("  Done!")
    print("=" * 50)
