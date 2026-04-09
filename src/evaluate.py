import torch
import yaml
import copy
import numpy as np

from src.components.environment import ThreeJointArmEnv
from src.components.agent import DQNAgent
from src.components.visualizations import Arm3DVisualizer

cfg = yaml.safe_load(open("configs/dqn.yaml"))

env = ThreeJointArmEnv()
agent = DQNAgent(
    env.observation_space.shape[0],
    env.action_space.n,
    cfg
)

agent.q.load_state_dict(torch.load("artifacts/dqn_model.pth"))
agent.q.eval()

# ✅ NO csv_path
viz = Arm3DVisualizer()

episodes = 3
MAX_CANDIDATES = 4

for ep in range(1, episodes + 1):
    state, _ = env.reset()
    target = env.target

    # ✅ Correct reset
    viz.traj = []

    print(f"\nEpisode {ep} — Target: {target}")

    for step in range(1, cfg["max_steps"] + 1):

        base_action = agent.select_action(state, epsilon=0.05)

        best_action = base_action
        best_distance = float("inf")

        candidate_actions = list(
            set([base_action] + list(np.random.randint(0, env.action_space.n, MAX_CANDIDATES)))
        )

        for a in candidate_actions:
            env_copy = copy.deepcopy(env)
            _, _, _, _, info = env_copy.step(a)
            if info["distance"] < best_distance:
                best_distance = info["distance"]
                best_action = a

        next_state, reward, done, truncated, info = env.step(best_action)

        joints = env.get_joint_positions()
        viz.update(joints, target, step, ep, info["distance"])

        state = next_state

        if info["distance"] < 0.1:
            print(f"✔ Target reached in {step} steps")
            break

        if done or truncated:
            break

# ✅ keep window open
print("\nEvaluation finished. Close the window manually.")
import matplotlib.pyplot as plt
plt.ioff()
plt.show()

