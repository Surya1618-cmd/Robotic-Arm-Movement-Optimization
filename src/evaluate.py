# src/evaluate.py

import torch
from src.components.environment import ThreeJointArmEnv
from src.components.agent import DQNAgent
from src.config import config

def evaluate(num_episodes=5):
    env = ThreeJointArmEnv(render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, config)
    agent.q_network.load_state_dict(torch.load("best_model.pth"))
    agent.q_network.eval()

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(config["max_steps_per_episode"]):
            action = agent.select_action(state, epsilon=0.0)  # always greedy
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break

        print(f"Episode {episode}\nFinished in {step + 1} steps, Reward: {total_reward:.2f}\n")

if __name__ == "__main__":
    evaluate()
    input("Press Enter to close...")
