# --- train.py (with Curriculum Learning Advancement) ---

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

from src.config import config
from src.logger import get_logger
from src.components.environment import ThreeJointArmEnv
from src.components.agent import DQNAgent
from src.components.reward_functions import shaped_reward

logger = get_logger("train")

def train():
    env = ThreeJointArmEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, config)

    total_rewards = []
    moving_avg_rewards = []
    eps_history = []
    best_reward = float('-inf')
    reward_window = deque(maxlen=config["moving_avg_window"])

    for episode in range(1, config["num_episodes"] + 1):
        # === Curriculum Difficulty Advancement ===
        if episode < 700:
            env.difficulty = "easy"
        elif episode < 1400:
            env.difficulty = "medium"
        else:
            env.difficulty = "hard"

        state, _ = env.reset()
        total_reward = 0
        epsilon = config["epsilon_end"] + (config["epsilon_start"] - config["epsilon_end"]) * \
            np.exp(-episode / config["epsilon_decay_steps"])

        for _ in range(config["max_steps_per_episode"]):
            action = agent.select_action(state, epsilon)
            next_state, _, terminated, truncated, info = env.step(action)

            reward, done, distance = shaped_reward(next_state[:3], next_state[3:],
                                                   threshold=config["target_threshold"],
                                                   bonus=config["target_bonus"])
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > config["warmup_steps"]:
                agent.learn(config["batch_size"])
                nn.utils.clip_grad_norm_(agent.q_network.parameters(), max_norm=config["grad_clip_norm"])

            if done or truncated:
                break

        if episode % config["target_update_freq"] == 0:
            agent.update_target_network()

        total_rewards.append(total_reward)
        reward_window.append(total_reward)
        avg_reward = np.mean(reward_window)
        moving_avg_rewards.append(avg_reward)
        eps_history.append(epsilon)

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.q_network.state_dict(), 'best_model.pth')

        if episode % 100 == 0:
            logger.info(f"Episode {episode}: Total Reward = {total_reward:.2f}, "
                        f"Avg Reward = {avg_reward:.2f}, Epsilon = {epsilon:.3f}, Difficulty = {env.difficulty}")

    df = pd.DataFrame({
        'episode': np.arange(1, config["num_episodes"] + 1),
        'total_reward': total_rewards,
        'avg_reward': moving_avg_rewards,
        'epsilon': eps_history
    })
    df.to_csv('training_log.csv', index=False)

if __name__ == "__main__":
    train()
