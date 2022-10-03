
def odesolver45(f, y, h):
    """Calculate the next step of an IVP of a time-invariant ODE with a RHS
    described by f, with an order 4 approx. and an order 5 approx.
    Parameters:
        f: function. RHS of ODE.
        y: float. Current position.
        h: float. Step length.
    Returns:
        q: float. Order 2 approx.
        w: float. Order 3 approx.
    """
    s1 = f(y)
    s2 = f(y + h * s1 / 4.0)
    s3 = f(y + 3.0 * h * s1 / 32.0 + 9.0 * h * s2 / 32.0)
    s4 = f(
        y
        + 1932.0 * h * s1 / 2197.0
        - 7200.0 * h * s2 / 2197.0
        + 7296.0 * h * s3 / 2197.0
    )
    s5 = f(
        y
        + 439.0 * h * s1 / 216.0
        - 8.0 * h * s2
        + 3680.0 * h * s3 / 513.0
        - 845.0 * h * s4 / 4104.0
    )
    s6 = f(
        y
        - 8.0 * h * s1 / 27.0
        + 2 * h * s2
        - 3544.0 * h * s3 / 2565
        + 1859.0 * h * s4 / 4104.0
        - 11.0 * h * s5 / 40.0
    )
    w = y + h * (
        25.0 * s1 / 216.0 + 1408.0 * s3 / 2565.0 + 2197.0 * s4 / 4104.0 - s5 / 5.0
    )
    q = y + h * (
        16.0 * s1 / 135.0
        + 6656.0 * s3 / 12825.0
        + 28561.0 * s4 / 56430.0
        - 9.0 * s5 / 50.0
        + 2.0 * s6 / 55.0
    )
    return w, q

