# src/predict.py

import torch
import numpy as np
from src.components.agent import DQNAgent
from src.components.environment import ThreeJointArmEnv

def predict_once(model_path="best_model.pth", epsilon=0.0):
    env = ThreeJointArmEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    config = {
        "gamma": 0.99,
        "epsilon": epsilon,
        "epsilon_min": epsilon,
        "epsilon_decay": 1.0,
        "learning_rate": 1e-3,
        "use_dueling": True
    }

    agent = DQNAgent(state_dim, action_dim, config)
    agent.q_network.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.q_network.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state, epsilon)
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        env.render()

        if truncated:
            break

    env.close()
    print(f"Total reward from prediction: {total_reward:.2f}")

if __name__ == "__main__":
    predict_once()
