"""Finite-difference weights on arbitrary node sets (Fornberg's algorithm).

Re-implemented here rather than imported from the wave-equation companion
(plan D2). The recursion is B. Fornberg, *Generation of finite difference
formulas on arbitrarily spaced grids*, Math. Comp. 51 (1988) 699–706, in the
form of *Calculation of weights in finite difference formulas*, SIAM Review
40 (1998) 685–691; the dissertation's MATLAB ``weights.m`` is the same
algorithm with the derivative orders down the rows.
"""

import numpy as np


def fornberg_weights(z: float, x: np.ndarray, m: int) -> np.ndarray:
    """Weights of derivatives 0..``m`` at ``z`` from samples at the nodes ``x``.

    Returns an array of shape ``(m + 1, len(x))``; row ``k`` applied to
    ``f(x)`` approximates ``f^(k)(z)`` exactly for every polynomial of degree
    below ``len(x)``. Rows with ``k >= len(x)`` are zero.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        raise ValueError("at least one node is needed")
    c = np.zeros((m + 1, n))
    c[0, 0] = 1.0
    prev_prod = 1.0  # product of (x_{i-1} - x_j), j < i - 1
    for i in range(1, n):
        mn = min(i, m)
        prod = 1.0
        for j in range(i):
            d = x[i] - x[j]
            prod *= d
            if j == i - 1:
                # The newest node gets its own column from the previous one.
                ratio = prev_prod / prod
                for k in range(mn, 0, -1):
                    c[k, i] = ratio * (
                        k * c[k - 1, i - 1] - (x[i - 1] - z) * c[k, i - 1]
                    )
                c[0, i] = -ratio * (x[i - 1] - z) * c[0, i - 1]
            # Every earlier column is updated to include node i.
            for k in range(mn, 0, -1):
                c[k, j] = ((x[i] - z) * c[k, j] - k * c[k - 1, j]) / d
            c[0, j] = (x[i] - z) * c[0, j] / d
        prev_prod = prod
    return c
