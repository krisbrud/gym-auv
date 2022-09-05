def observe_obstacle_fun(t, dist) -> bool:
    # TODO: Find out what this function is even meant to do.
    return t % (int(0.0025 * dist**1.7) + 1) == 0


def return_true_fun(t, dist) -> bool:
    return True
