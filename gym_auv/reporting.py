import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 0.5
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


matplotlib.use('pdf')

def report(env, report_dir):
    try:
        os.makedirs(report_dir, exist_ok=True)

        relevant_history = env.history[-min(100, len(env.history)):]
        collisions = np.array([obj['collisions'] for obj in relevant_history])
        collision_baselines = np.array([obj['collision_baselines'] for obj in relevant_history])
        no_collisions = collisions == 0
        cross_track_errors = np.array([obj['cross_track_error'] for obj in relevant_history])
        progresses = np.array([obj['progress'] for obj in relevant_history])
        reached_goals = np.array([obj['reached_goal'] for obj in relevant_history])
        rewards = np.array([obj['reward'] for obj in relevant_history])
        success = np.logical_and(no_collisions, reached_goals)
        timesteps = np.array([obj['timesteps'] for obj in relevant_history]) 

        with open(os.path.join(report_dir, 'report.txt'), 'w') as f:
            f.write('# PERFORMANCE METRICS (LAST 100 EPISODES AVG.)\n')
            f.write('{:<30}{:<30}\n'.format('Episodes', env.episode-1))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Reward', rewards.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Std. Reward', rewards.std()))
            f.write('{:<30}{:<30.2%}\n'.format('Avg. Progress', progresses.mean()))
            f.write('{:<30}{:<30.2%}\n'.format('Reached Goal Percentage', reached_goals.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Collisions', collisions.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Collision Baseline', collision_baselines.mean()))
            f.write('{:<30}{:<30.2%}\n'.format('No collisions', no_collisions.mean()))
            f.write('{:<30}{:<30.2%}\n'.format('Success', success.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Cross-Track Error', cross_track_errors.mean()))
            f.write('{:<30}{:<30.2f}\n'.format('Avg. Timesteps', timesteps.mean()))

        plt.style.use('ggplot')
        plt.rc('font', family='serif')
        #plt.rc('font', family='serif', serif='Times')
        #plt.rc('text', usetex=True) #RAISES FILENOTFOUNDERROR
        plt.rc('xtick', labelsize=8)
        plt.rc('ytick', labelsize=8)
        plt.rc('axes', labelsize=8)

        collisions = np.array([obj['collisions'] for obj in env.history])
        smoothed_collisions = gaussian_filter1d(collisions.astype(float), sigma=100)
        collision_baselines = np.array([obj['collision_baselines'] for obj in env.history])
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(collisions, color='blue', linewidth=0.5, alpha=0.2, label='Collisions')
        ax.plot(smoothed_collisions, color='blue', linewidth=1, alpha=0.4)
        ax.hlines(collision_baselines.mean(), 0, env.episode-2, color='black', linestyles='dashed', linewidth=1, label='Baseline')
        ax.set_title('Collisions vs. Baseline')
        ax.set_ylabel(r"Collisions")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'collisions.pdf'), format='pdf')
        plt.close(fig)

        cross_track_errors = np.array([obj['cross_track_error'] for obj in env.history])
        smoothed_cross_track_errors = gaussian_filter1d(cross_track_errors, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(cross_track_errors, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_cross_track_errors, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Avg. Cross-Track Error")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'cross_track_error.pdf'), format='pdf')
        plt.close(fig)

        rewards = np.array([obj['reward'] for obj in env.history])
        smoothed_rewards = gaussian_filter1d(rewards, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rewards, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_rewards, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Reward")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'reward.pdf'), format='pdf')
        plt.close(fig)

        progresses = np.array([obj['progress'] for obj in env.history])
        smoothed_progresses = gaussian_filter1d(progresses, sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
        ax.plot(progresses, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_progresses, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Progress [%]")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'progress.pdf'), format='pdf')
        plt.close(fig)

        timesteps = np.array([obj['timesteps'] for obj in env.history])
        smoothed_timesteps = gaussian_filter1d(timesteps.astype(float), sigma=100)
        plt.axis('scaled')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(timesteps, color='blue', linewidth=0.5, alpha=0.2)
        ax.plot(smoothed_timesteps, color='blue', linewidth=1, alpha=0.4)
        ax.set_ylabel(r"Timesteps")
        ax.set_xlabel(r"Episode")
        ax.legend()
        fig.savefig(os.path.join(report_dir, 'timesteps.pdf'), format='pdf')
        plt.close(fig)

    except PermissionError as e:
        print('Warning: Report files are open - could not update report: ' + str(repr(e)))
    except OSError as e:
        print('Warning: Ignoring OSError: ' + str(repr(e)))

def test_report(fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    class Struct(object): pass
    env = Struct()
    env.history = []
    env.episode = 1001
    for episode in range(1000):
        progress = min(1, np.random.random() + np.random.random()*episode/1000)
        env.history.append({
            'collisions': np.random.poisson(0.1 + 7*(1-episode/1000)),
            'cross_track_error': np.random.gamma(10 - 5*episode/1000, 3) ,
            'collision_baselines': np.random.poisson(7),
            'progress': progress,
            'reached_goal': int(progress > 0.9),
            'reward': np.random.normal(-1000 + 2000*progress + episode**1.1, 2000),
            'timesteps': np.random.poisson(500 + progress*500 + episode)
        })

    t = np.linspace(-400, 400, 1000)
    a = np.random.normal(0, 1)
    b = 100*np.random.random()
    c = 50*np.random.random() + 20
    d = 0.1*np.random.random()
    e = 100*np.random.random()
    f = 0.01*np.random.random() 
    g = 100*np.random.random()
    h = 0.01*np.random.random() 
    path_x = t + g*np.sin(h*t)
    path_y = a*t + b + e*np.sin(f*path_x)
    path_taken_x = path_x + e*np.sin(f*t)*np.cos(np.arctan2(path_y, path_x))
    path_taken_y = path_y + g*np.sin(h*t)*np.sin(np.arctan2(path_y, path_x))

    path = np.vstack((path_x, path_y))
    path_taken = np.vstack((path_taken_x, path_taken_y)).T

    env.last_episode = {
        'path': path,
        'path_taken': path_taken
    }
    env.obstacles = []
    for o in range(np.random.poisson(10)):
        obst = Struct()
        s = int(1000*np.clip(np.random.random(), 0.1, 0.9))
        obst.position = np.array([
            path_x[s],
            path_y[s]
        ])
        obst.radius = np.random.poisson(30) 
        env.obstacles.append(obst)

    report(env, fig_dir)
    plot_last_episode(env, fig_dir)

def plot_last_episode(env, fig_dir, fig_prefix=''):
    """
    Plots the result of a path following episode.

    Parameters
    ----------
    fig_dir : str
        Absolute path to a directory to store the plotted
        figure in.
    """

    path = env.last_episode['path']
    path_taken = env.last_episode['path_taken']

    if (fig_prefix != '' and not fig_prefix[0] == '_'):
        fig_prefix = '_' + fig_prefix

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    #plt.rc('font', family='serif', serif='Times')
    #plt.rc('text', usetex=True) #RAISES FILENOTFOUNDERROR
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.axis('scaled')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    axis_min = min(path[1, :].min(), path[0, :].min()) - 100
    axis_max = max(path[1, :].max(), path[0, :].max()) + 100

    ax.plot(path[1, :], path[0, :], dashes=[6, 2], color='black', linewidth=1.5, label=r'Path')
    ax.plot(path_taken[:, 1], path_taken[:, 0], color='tab:blue', label=r'Path taken')
    ax.set_ylabel(r"North (m)")
    ax.set_xlabel(r"East (m)")
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.legend()

    for obst in env.obstacles:
        circle = plt.Circle(
            obst.position[::-1],
            obst.radius,
            facecolor='tab:red',
            edgecolor='black',
            linewidth=0.5,
            zorder=10
        )
        obst = ax.add_patch(circle)
        obst.set_hatch('////')

    goal = plt.Circle(
        (path[1, -1], path[0, -1]),
        (axis_max - axis_min)/100,
        facecolor='green',
        linewidth=0.5,
        zorder=11
    )
    ax.add_patch(goal)
    ax.annotate("Goal", 
        xy=(path[1, -1] + (axis_max - axis_min)/25, path[0, -1]), 
        fontsize=7, ha="center", zorder=20, color='green',
    )

    fig.savefig(os.path.join(fig_dir, '{}path.pdf'.format(fig_prefix)), format='pdf')
    plt.close(fig)