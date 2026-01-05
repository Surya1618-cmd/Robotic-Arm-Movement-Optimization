# src/components/visualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def init_3d_plot():
    """Create a 3D plot for arm rendering."""
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return fig, ax

def draw_arm(ax, joints, end_effector, target, obstacles=None):
    """Draw the robotic arm, target, and spherical obstacles."""
    x = [0] + [j[0] for j in joints] + [end_effector[0]]
    y = [0] + [j[1] for j in joints] + [end_effector[1]]
    z = [0] + [j[2] for j in joints] + [end_effector[2]]

    ax.clear()
    ax.plot(x, y, z, 'o-', label='Arm', color='blue')
    ax.scatter(*target, c='red', s=100, label='Target')

    if obstacles:
        for obs in obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = 0.2 * np.cos(u) * np.sin(v) + obs[0]
            y = 0.2 * np.sin(u) * np.sin(v) + obs[1]
            z = 0.2 * np.cos(v) + obs[2]
            ax.plot_surface(x, y, z, color='gray', alpha=0.4)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    ax.legend()
    plt.draw()
    plt.pause(0.001)
