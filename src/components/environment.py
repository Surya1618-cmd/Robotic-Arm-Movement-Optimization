# --- environment.py (Updated for Curriculum Learning) ---

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ThreeJointArmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.L1 = 0.5
        self.L2 = 1.0
        self.L3 = 1.0

        self.joint_limits = [(-np.pi, np.pi), (-np.pi / 2, np.pi / 2), (-np.pi / 2, np.pi / 2)]
        self.action_space = spaces.Discrete(6)

        low_angles = np.array([lim[0] for lim in self.joint_limits], dtype=np.float32)
        high_angles = np.array([lim[1] for lim in self.joint_limits], dtype=np.float32)
        low_target = np.array([-2.5, -2.5, 0.0], dtype=np.float32)
        high_target = np.array([2.5, 2.5, 3.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([low_angles, low_target]),
            high=np.concatenate([high_angles, high_target]),
            dtype=np.float32
        )

        self.target_threshold = 0.1
        self.target_bonus = 10.0
        self.max_steps = 200

        self.joint_angles = np.zeros(3, dtype=np.float32)
        self.target = np.zeros(3, dtype=np.float32)
        self.current_step = 0

        self.obstacles = [np.array([1.0, -1.0, 0.5]), np.array([-1.0, 1.0, 0.5])]

        # === Curriculum Setup ===
        self.curriculum = True
        self.difficulty = "easy"  # Will be updated externally

        if self.render_mode == "human":
            self._init_render()

    def _init_render(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([0, 3])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if self.curriculum:
            if self.difficulty == "easy":
                r = self.np_random.uniform(0.5, 1.0)
            elif self.difficulty == "medium":
                r = self.np_random.uniform(1.0, 2.0)
            else:  # hard
                r = self.np_random.uniform(2.0, 2.5)
        else:
            r = self.np_random.uniform(1.5, 2.5)

        vec = self.np_random.standard_normal(3)
        vec /= np.linalg.norm(vec)
        self.target = vec * r
        self.target[2] = np.abs(self.target[2])

        self.joint_angles = np.zeros(3, dtype=np.float32)
        obs = np.concatenate([self.joint_angles, self.target]).astype(np.float32)
        info = {"distance": np.linalg.norm(self._end_effector_pos() - self.target)}
        return obs, info

    def step(self, action):
        delta = 0.1
        if action == 0: self.joint_angles[0] += delta
        elif action == 1: self.joint_angles[0] -= delta
        elif action == 2: self.joint_angles[1] += delta
        elif action == 3: self.joint_angles[1] -= delta
        elif action == 4: self.joint_angles[2] += delta
        elif action == 5: self.joint_angles[2] -= delta

        for i, (low, high) in enumerate(self.joint_limits):
            self.joint_angles[i] = np.clip(self.joint_angles[i], low, high)

        ee_pos = self._end_effector_pos()
        distance = np.linalg.norm(ee_pos - self.target)
        reward = -distance
        terminated = distance < self.target_threshold
        if terminated:
            reward += self.target_bonus

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        obs = np.concatenate([self.joint_angles, self.target]).astype(np.float32)
        info = {
            "distance": distance,
            "end_effector_pos": ee_pos,
            "target": self.target
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        self.ax.clear()
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([0, 3])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        joint1 = np.array([0, 0, self.L1])
        joint2 = joint1 + self.L2 * np.array([
            np.cos(self.joint_angles[1]) * np.cos(self.joint_angles[0]),
            np.cos(self.joint_angles[1]) * np.sin(self.joint_angles[0]),
            np.sin(self.joint_angles[1])
        ])
        ee = joint2 + self.L3 * np.array([
            np.cos(self.joint_angles[1] + self.joint_angles[2]) * np.cos(self.joint_angles[0]),
            np.cos(self.joint_angles[1] + self.joint_angles[2]) * np.sin(self.joint_angles[0]),
            np.sin(self.joint_angles[1] + self.joint_angles[2])
        ])
        arm_x = [0, joint1[0], joint2[0], ee[0]]
        arm_y = [0, joint1[1], joint2[1], ee[1]]
        arm_z = [0, joint1[2], joint2[2], ee[2]]

        self.ax.plot(arm_x, arm_y, arm_z, 'o-', color='blue', label='Arm')
        self.ax.scatter(*self.target, c='red', s=100, label='Target')

        for obs in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = 0.2 * np.cos(u) * np.sin(v) + obs[0]
            y = 0.2 * np.sin(u) * np.sin(v) + obs[1]
            z = 0.2 * np.cos(v) + obs[2]
            self.ax.plot_surface(x, y, z, color='gray', alpha=0.4)

        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def _end_effector_pos(self):
        a1, a2, a3 = self.joint_angles
        x = (self.L2 * np.cos(a2) + self.L3 * np.cos(a2 + a3)) * np.cos(a1)
        y = (self.L2 * np.cos(a2) + self.L3 * np.cos(a2 + a3)) * np.sin(a1)
        z = self.L1 + self.L2 * np.sin(a2) + self.L3 * np.sin(a2 + a3)
        return np.array([x, y, z], dtype=np.float32)


if __name__ == "__main__":
    env = ThreeJointArmEnv(render_mode="human")
    obs, _ = env.reset()

    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.render()
    input("Press Enter to close...")
