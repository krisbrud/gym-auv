# %%
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from gym_auv.objects.vessel import Vessel
from gym_auv.objects.obstacles import CircularObstacle, VesselObstacle
from gym_auv.objects.vessel.sensor import (
    find_rays_to_simulate_for_obstacles,
    simulate_sensor,
)
import shapely
import shapely.geometry


def make_sensor_ends(sensor_range, angle):
    return np.array([sensor_range * np.cos(angle), sensor_range * np.sin(angle)]).T


def make_vessel_obstacle(own_vessel=False) -> VesselObstacle:
    if own_vessel:
        pos = [0, 0]
        heading = np.deg2rad(60)
        # heading = 0
        width = 10
        init_state = [*pos, heading]
    else:
        pos = [130, 10]
        heading = np.deg2rad(120)
        init_state = [*pos, heading]
        width = 25
    trajectory = [[0, 0] * 10]

    return VesselObstacle(
        width, trajectory, init_position=pos, init_heading=heading, init_update=False
    )


def make_circular_obstacle() -> CircularObstacle:
    pos = np.array([-80, 100])
    # pos = np.array([6, 4.8])
    radius = 40

    return CircularObstacle(pos, radius)


def make_circular_obstacle2() -> CircularObstacle:
    pos = np.array([-45, -100])
    # pos = np.array([6, 4.8])
    radius = 30

    return CircularObstacle(pos, radius)


def plot_circular_obst(obst, ax):
    edge_color = mcolors.to_rgba("#12293c")
    if isinstance(obst, CircularObstacle):
        obst_object = plt.Circle(
            obst.position,
            obst.radius,
            facecolor="#C0C0C0",
            edgecolor=edge_color,
            linewidth=0.5,
            zorder=10,
        )
        obst_object.set_hatch("////")
        obst = ax.add_patch(obst_object)


def plot_vessel_obst(vessel_obst, ax):
    edge_color = mcolors.to_rgba("#12293c")
    vessel_obst_object = plt.Polygon(
        np.array(list(vessel_obst.init_boundary.exterior.coords)),
        True,
        facecolor="#C0C0C0",
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=10,
    )
    # vessel_obst_object.set_hatch("////")
    vessel_obst_object.set_hatch("////")
    ax.add_patch(vessel_obst_object)


def plot_own_vessel_obst(vessel_obst, ax):
    edge_color = mcolors.to_rgba("#12293c")
    # orange = mcolors.to_rgba("#fa9b28")
    vessel_obst_object = plt.Polygon(
        np.array(list(vessel_obst.init_boundary.exterior.coords)),
        True,
        facecolor="#C0C0C0",
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=10,
    )
    # vessel_obst_object.set_hatch("////")
    # vessel_obst_object.set_hatch("")
    ax.add_patch(vessel_obst_object)


def plot_mrr(obst, ax):
    edge_color = mcolors.to_rgba("#12293c")
    mrr = get_minimum_rotated_rect(obst)
    mrr_object = plt.Polygon(
        np.array(list(mrr.exterior.coords)),
        False,  # True,
        facecolor="none",  # "tab:blue",
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=10,
    )
    # mrr_object.set_hatch("////")
    ax.add_patch(mrr_object)


def get_minimum_rotated_rect(obst):
    """
    Returns the minimum rotated rectangle of the obstacle
    """
    # Get the minimum rotated rectangle
    return obst._calculate_boundary().minimum_rotated_rectangle


def find_rays_for_obstacles(obstacles, own_pos, own_heading, n_rays=180):
    obstacles_per_ray = find_rays_to_simulate_for_obstacles(
        obstacles,
        shapely.geometry.Point(*own_pos),
        own_heading,
        angle_per_ray=np.deg2rad(360 / n_rays),
        n_rays=n_rays,
    )
    return obstacles_per_ray


sensor_range = 150
n_sensors = 180
sensor_starts = np.zeros((n_sensors, 2))
angles = np.deg2rad(np.linspace(-180 + 2, 180 + 2, n_sensors, endpoint=False))
circular_obst = make_circular_obstacle()
circular_obst2 = make_circular_obstacle2()

own_pos = [0, 0]
own_heading = np.deg2rad(60)
vessel_obst = make_vessel_obstacle()
own_vessel_obst = make_vessel_obstacle(own_vessel=True)
sensor_ends_raw = make_sensor_ends(sensor_range, angles + own_heading)

obstacles = [vessel_obst, circular_obst, circular_obst2]
rays_for_obstacles = find_rays_for_obstacles(
    obstacles, own_pos, own_heading, n_rays=n_sensors
)
p0_point = shapely.geometry.Point(*own_pos)

def get_simulated_endpoints(own_heading, p0_point, sensor_range, obstacles):
    simulated_ranges = []
    for angle in angles:
        result = simulate_sensor(
            own_heading + angle,
            p0_point,
            sensor_range,
            obstacles,
            # [vessel_obst],
        )
        simulated_range = result[0]
        simulated_ranges.append(simulated_range)
    simulated_rays = np.array(simulated_ranges)

    simulated_endpoints = (
        simulated_rays
        * np.array([np.cos(angles + own_heading), np.sin(angles + own_heading)])
    ).T
    return simulated_endpoints
# simulated_ranges = []
# for angle in angles:
#     result = simulate_sensor(
#         own_heading + angle,
#         p0_point,
#         sensor_range,
#         obstacles,
#         # [vessel_obst],
#     )
#     simulated_range = result[0]
#     simulated_ranges.append(simulated_range)
# simulated_rays = np.array(simulated_ranges)

# simulated_endpoints = (
#     simulated_rays
#     * np.array([np.cos(angles + own_heading), np.sin(angles + own_heading)])
# ).T

simulated_endpoints = get_simulated_endpoints(own_heading, p0_point, sensor_range, obstacles)

# print(simulated_range)


plt.style.use("ggplot")


not_activated_color = (220 / 255, 220 / 255, 220 / 255, 1.0)
colors = [not_activated_color] * n_sensors


def plot_enclosing_circle(obst, ax):
    edge_color = mcolors.to_rgba("#12293c")
    circle = obst.enclosing_circle
    x = circle.center.x
    y = circle.center.y
    circle_object = plt.Circle(
        (x, y),
        circle.radius,
        facecolor="none",
        edgecolor=edge_color,
        linewidth=0.5,
        zorder=10,
    )
    # circle_object.set_hatch("////")
    ax.add_patch(circle_object)


activated_color_hex = "#ff5c5c"
fig, ax = plt.subplots(figsize=(6, 6))


def plot_sensor_rays(sensor_starts, sensor_ends_raw, colors, ax):
    for (x1, y1), (x2, y2), color in zip(sensor_starts, sensor_ends_raw, colors):
        # plt.plot([x1, x2], [y1, y2],  f"{color}-", color=color)
        # plt.plot([x1, x2], [y1, y2], color=color)
        ax.plot([x1, x2], [y1, y2], color=color)


def plot_sensor_rays_same_color(sensor_starts, sensor_ends_raw, color, ax):
    for (x1, y1), (x2, y2) in zip(sensor_starts, sensor_ends_raw):
        # plt.plot([x1, x2], [y1, y2],  f"{color}-", color=color)
        # plt.plot([x1, x2], [y1, y2], color=color)
        ax.plot([x1, x2], [y1, y2], color=color)


def make_fig_ax():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    ax.set_aspect("equal")

    return fig, ax


mask = np.array([len(x) > 0 for x in rays_for_obstacles])


def plot0():
    # Plot gray sensor rays
    fig, ax = make_fig_ax()
    colors = [activated_color_hex] * n_sensors
    # plot_sensor_rays(sensor_starts, sensor_ends_raw, colors, ax)
    plot_own_vessel_obst(own_vessel_obst, ax)
    plot_sensor_rays(sensor_starts, simulated_endpoints, colors, ax)

    # Plot own vessel
    # plot_vessel_obst(own_vessel_obst, ax)

    # Plot obstacles
    plot_vessel_obst(vessel_obst, ax)
    plot_circular_obst(circular_obst, ax)
    plot_circular_obst(circular_obst2, ax)

    return fig, ax


def plot1():
    # Plot gray sensor rays
    fig, ax = make_fig_ax()
    colors = [not_activated_color] * n_sensors
    plot_own_vessel_obst(own_vessel_obst, ax)
    plot_sensor_rays(sensor_starts, sensor_ends_raw, colors, ax)

    # Plot own vessel
    # plot_vessel_obst(own_vessel_obst, ax)

    # Plot obstacles
    plot_vessel_obst(vessel_obst, ax)
    plot_circular_obst(circular_obst, ax)
    plot_circular_obst(circular_obst2, ax)
    plot_enclosing_circle(vessel_obst, ax)
    plot_mrr(vessel_obst, ax)



    return fig, ax


def plot2():
    # Plot gray sensor rays
    fig, ax = make_fig_ax()

    # light_blue = mcolors.to_rgba("#a0cdff")
    light_green = mcolors.to_rgba("#a0ffcd")
    accent_color = light_green

    colors = [accent_color] * n_sensors
    accent_colors = [accent_color] * int(np.sum(mask))
    plot_own_vessel_obst(own_vessel_obst, ax)
    plot_sensor_rays_same_color(
        sensor_starts[mask, :], sensor_ends_raw[mask, :], accent_color, ax
    )

    # plot_sensor_rays_same_color(
    #     sensor_starts[~mask, :], simulated_endpoints[~mask, :], not_activated_color, ax
    # )
    plot_sensor_rays_same_color(
        sensor_starts[~mask, :], simulated_endpoints[~mask, :], not_activated_color, ax
    )

    # Plot own vessel
    # plot_vessel_obst(own_vessel_obst, ax)

    # Plot obstacles
    plot_vessel_obst(vessel_obst, ax)
    plot_circular_obst(circular_obst, ax)
    plot_circular_obst(circular_obst2, ax)
    plot_enclosing_circle(vessel_obst, ax)
    plot_mrr(vessel_obst, ax)

    return fig, ax


def plot3():
    # Plot gray sensor rays
    fig, ax = make_fig_ax()

    # light_blue = mcolors.to_rgba("#a0cdff")
    light_green = mcolors.to_rgba("#a0ffcd")
    accent_color = light_green

    # colors = [accent_color] * n_sensors
    # accent_colors = [accent_color] * int(np.sum(mask))
    plot_own_vessel_obst(own_vessel_obst, ax)
    plot_sensor_rays_same_color(
        sensor_starts[mask, :], simulated_endpoints[mask, :], activated_color_hex, ax
    )

    # plot_sensor_rays_same_color(
    #     sensor_starts[~mask, :], simulated_endpoints[~mask, :], not_activated_color, ax
    # )
    plot_sensor_rays_same_color(
        sensor_starts[~mask, :], simulated_endpoints[~mask, :], not_activated_color, ax
    )

    # Plot own vessel
    # plot_vessel_obst(own_vessel_obst, ax)

    # Plot obstacles
    plot_vessel_obst(vessel_obst, ax)
    plot_circular_obst(circular_obst, ax)
    plot_circular_obst(circular_obst2, ax)
    # plot_enclosing_circle(vessel_obst, ax)
    # plot_mrr(vessel_obst, ax)

    return fig, ax


filename_prefix = "rangefinder_figs/rangefinder-plot-"

fig0, ax0 = plot0()
fig0.savefig(f"{filename_prefix}0.pdf")
fig0.show()

fig1, ax1 = plot1()
fig1.savefig(f"{filename_prefix}1.pdf")
fig1.show()

fig2, ax2 = plot2()
fig2.savefig(f"{filename_prefix}2.pdf")
fig2.show()

fig3, ax3 = plot3()
fig3.savefig(f"{filename_prefix}3.pdf")
fig3.show()


# %%
