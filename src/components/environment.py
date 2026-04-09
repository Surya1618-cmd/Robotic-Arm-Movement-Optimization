
import numpy as np
import gym
from gym import spaces


class ThreeJointArmEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Arm lengths
        self.L1 = 0.8
        self.L2 = 0.8
        self.L3 = 0.6

        self.MAX_REACH = self.L1 + self.L2 + self.L3

        self.max_angle = np.pi

        # 7 actions (6 movements + NO-OP)
        self.action_space = spaces.Discrete(7)

        high = np.array([self.max_angle]*3 + [2.5, 2.5, 2.5])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.prev_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.joint_angles = np.zeros(3)
        self.prev_action = None

        # 🎯 Sample only REACHABLE targets
        while True:
            target = np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(0.1, 1.8),
            ])
            if np.linalg.norm(target) <= self.MAX_REACH:
                break

        self.target = target
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0

        # Prevent freezing
        if self.prev_action is not None and action == self.prev_action:
            reward -= 0.05
        self.prev_action = action

        delta = 0.05

        if action == 0:
            self.joint_angles[0] += delta
        elif action == 1:
            self.joint_angles[0] -= delta
        elif action == 2:
            self.joint_angles[1] += delta
        elif action == 3:
            self.joint_angles[1] -= delta
        elif action == 4:
            self.joint_angles[2] += delta
        elif action == 5:
            self.joint_angles[2] -= delta
        elif action == 6:
            pass  # NO-OP

        self.joint_angles = np.clip(
            self.joint_angles, -self.max_angle, self.max_angle
        )

        ee = self._end_effector()
        distance = np.linalg.norm(ee - self.target)

        reward -= distance

        done = distance < 0.05
        truncated = False

        info = {"distance": distance}
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.concatenate([self.joint_angles, self.target]).astype(np.float32)

    def _end_effector(self):
        a1, a2, a3 = self.joint_angles

        p2 = np.array([
            self.L2*np.cos(a2)*np.cos(a1),
            self.L2*np.cos(a2)*np.sin(a1),
            self.L1 + self.L2*np.sin(a2)
        ])

        p3 = np.array([
            p2[0] + self.L3*np.cos(a2+a3)*np.cos(a1),
            p2[1] + self.L3*np.cos(a2+a3)*np.sin(a1),
            p2[2] + self.L3*np.sin(a2+a3)
        ])

        return p3

    def get_joint_positions(self):
        a1, a2, a3 = self.joint_angles

        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.L1])

        p2 = np.array([
            self.L2*np.cos(a2)*np.cos(a1),
            self.L2*np.cos(a2)*np.sin(a1),
            self.L1 + self.L2*np.sin(a2)
        ])

        p3 = np.array([
            p2[0] + self.L3*np.cos(a2+a3)*np.cos(a1),
            p2[1] + self.L3*np.cos(a2+a3)*np.sin(a1),
            p2[2] + self.L3*np.sin(a2+a3)
        ])

        return np.vstack([p0, p1, p2, p3])