def _render_path(viewer: Viewer2D, path: Path):
    viewer.draw_polyline(path.points, linewidth=1, color=(0.3, 1.0, 0.3))


def _render_vessel(viewer: Viewer2D, vessel: Vessel):
    viewer.draw_polyline(
        vessel.path_taken, linewidth=1, color=(0.8, 0, 0)
    )  # previous positions
    vertices = [
        (-vessel.width / 2, -vessel.width / 2),
        (-vessel.width / 2, vessel.width / 2),
        (vessel.width / 2, vessel.width / 2),
        (3 / 2 * vessel.width, 0),
        (vessel.width / 2, -vessel.width / 2),
    ]

    viewer.draw_shape(vertices, vessel.position, vessel.heading, color=(0, 0, 0.8))


def _render_sensors(viewer: Viewer2D, vessel: Vessel):
    for isensor, sensor_angle in enumerate(vessel._sensor_angles):
        distance = vessel._last_sensor_dist_measurements[isensor]
        p0 = vessel.position
        p1 = (
            p0[0] + np.cos(sensor_angle + vessel.heading) * distance,
            p0[1] + np.sin(sensor_angle + vessel.heading) * distance,
        )

        # closeness = vessel._last_sector_dist_measurements[isector]
        closeness = vessel._last_sensor_dist_measurements[isensor]
        redness = 0.5 + 0.5 * max(0, closeness)
        greenness = 1 - max(0, closeness)
        blueness = 1
        alpha = 0.5
        viewer.draw_line(p0, p1, color=(redness, greenness, blueness, alpha))


def _render_progress(viewer: Viewer2D, path: Path, vessel: Vessel):
    ref_point = path(vessel._last_navi_state_dict["vessel_arclength"]).flatten()
    viewer.draw_circle(origin=ref_point, radius=1, res=30, color=(0.8, 0.3, 0.3))

    target_point = path(vessel._last_navi_state_dict["target_arclength"]).flatten()
    viewer.draw_circle(origin=target_point, radius=1, res=30, color=(0.3, 0.8, 0.3))


def _render_obstacles(viewer: Viewer2D, obstacles: List[BaseObstacle]):
    for obst in obstacles:
        c = (0.8, 0.8, 0.8)

        if isinstance(obst, CircularObstacle):
            viewer.draw_circle(obst.position, obst.radius, color=c)

        elif isinstance(obst, PolygonObstacle):
            viewer.draw_shape(obst.points, color=c)

        elif isinstance(obst, VesselObstacle):
            viewer.draw_shape(list(obst.boundary.exterior.coords), color=c)


def render_blue_background(W=env_bg_w, H=env_bg_h):
    color = (37, 150, 190)  # "#2596be" Semi-dark blue
    background = pyglet.shapes.Rectangle(x=0, y=0, width=W, height=H, color=color)
    background.draw()


def _render_indicators(
    viewer: Viewer2D,
    W: int,
    H: int,
    last_reward: float,
    cumulative_reward: float,
    t_step: int,
    episode: int,
    lambda_tradeoff: float,
    eta: float,
):
    viewer.reward_text_field.text = "Current Reward:"
    viewer.reward_text_field.draw()
    viewer.reward_value_field.text = "{:2.3f}".format(last_reward)
    viewer.reward_value_field.draw()

    viewer.cum_reward_text_field.text = "Cumulative Reward:"
    viewer.cum_reward_text_field.draw()
    viewer.cum_reward_value_field.text = "{:2.3f}".format(cumulative_reward)
    viewer.cum_reward_value_field.draw()

    viewer.time_step_text_field.text = "Time Step:"
    viewer.time_step_text_field.draw()
    viewer.time_step_value_field.text = str(t_step)
    viewer.time_step_value_field.draw()

    viewer.episode_text_field.text = "Episode:"
    viewer.episode_text_field.draw()
    viewer.episode_value_field.text = str(episode)
    viewer.episode_value_field.draw()

    viewer.lambda_text_field.text = "Log10 Lambda:"
    viewer.lambda_text_field.draw()
    viewer.lambda_value_field.text = "{:2.2f}".format(np.log10(lambda_tradeoff))
    viewer.lambda_value_field.draw()

    viewer.eta_text_field.text = "Eta:"
    viewer.eta_text_field.draw()
    viewer.eta_value_field.text = "{:2.2f}".format(eta)
    viewer.eta_value_field.draw()


def render_objects(viewer: Viewer2D, state: RenderableState):
    t = viewer.transform
    t.enable()
    _render_sensors(viewer, vessel=state.vessel)
    # _render_interceptions(env)
    if state.path is not None:
        _render_path(viewer, path=state.path)
    _render_vessel(viewer, vessel=state.vessel)
    # _render_tiles(env, win)
    _render_obstacles(viewer=viewer, obstacles=state.obstacles)
    if state.path is not None:
        _render_progress(viewer=viewer, path=state.path, vessel=state.vessel)
    # _render_interceptions(env)

    # Visualise path error (DEBUGGING)
    # p = np.array(env.vessel.position)
    # dir = rotate(env.past_obs[-1][0:2], env.vessel.heading)
    # env._viewer2d.draw_line(p, p + 10*np.array(dir), color=(0.8, 0.3, 0.3))

    for geom in viewer.onetime_geoms:
        geom.render()

    t.disable()

    if state.show_indicators:
        _render_indicators(
            viewer=viewer,
            W=WINDOW_W,
            H=WINDOW_H,
            last_reward=state.last_reward,
            cumulative_reward=state.cumulative_reward,
            t_step=state.t_step,
            episode=state.episode,
            lambda_tradeoff=state.lambda_tradeoff,
            eta=state.eta,
        )
