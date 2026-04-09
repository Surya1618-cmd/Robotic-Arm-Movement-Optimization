import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


class Arm3DVisualizer:
    def __init__(self, csv_path="data/training_log.csv"):
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, height_ratios=[2.2, 1.2], width_ratios=[1.3, 1])

        # --- 3D Arm (BIG)
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection="3d")

        # --- Live distance plot
        self.ax_dist = self.fig.add_subplot(gs[0, 1])

        # --- Success rate + logs
        self.ax_success = self.fig.add_subplot(gs[1, :])

        self.angle = 45
        self.traj = []
        self.distances = []

        # Load CSV once
        df = pd.read_csv(csv_path)
        success = (df["best_distance"] < 0.2).rolling(50).mean()
        self.ax_success.plot(success, color="green", linewidth=2)
        self.ax_success.set_title("Success Rate Over Time")
        self.ax_success.set_xlabel("Episode")
        self.ax_success.set_ylabel("Success Probability")
        self.ax_success.grid(True)

        # Log text area
        self.log_text = self.ax_success.text(
            0.01, 0.95, "",
            transform=self.ax_success.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace"
        )

        self.ax_success.margins(y=0.2)
        plt.tight_layout()
        plt.ion()

    def update(self, joints, target, step, episode, distance):
        # ---------- 3D ARM ----------
        self.ax3d.cla()

        xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
        self.ax3d.plot(xs, ys, zs, "-o", linewidth=4, color="blue", label="Arm")

        ee = joints[-1]
        self.traj.append(ee)
        traj = np.array(self.traj)
        self.ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color="green", linewidth=2, label="Trajectory")

        self.ax3d.scatter(*target, color="red", s=100, label="Target")

        self.ax3d.set_xlim(-2, 2)
        self.ax3d.set_ylim(-2, 2)
        self.ax3d.set_zlim(0, 2.5)

        self.ax3d.set_title(f"3D Arm | Episode {episode}")
        self.angle += 1.2
        self.ax3d.view_init(elev=25, azim=self.angle)
        self.ax3d.legend(loc="upper left")

        # ---------- LIVE DISTANCE ----------
        self.distances.append(distance)
        self.ax_dist.cla()
        self.ax_dist.plot(self.distances, color="orange", linewidth=2)
        self.ax_dist.set_title("Live Distance vs Step")
        self.ax_dist.set_xlabel("Step")
        self.ax_dist.set_ylabel("Distance")
        self.ax_dist.grid(True)

        # ---------- LOGS ----------
        log_str = (
            f"Episode: {episode}\n"
            f"Step: {step}\n"
            f"Distance: {distance:.4f}"
        )
        self.log_text.set_text(log_str)

        plt.pause(0.001)
