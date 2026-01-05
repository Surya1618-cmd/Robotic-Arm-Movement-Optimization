import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, dueling=False):
        super(QNetwork, self).__init__()
        self.dueling = dueling

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        if dueling:
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, action_dim)
        else:
            self.output = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            return value + (advantage - advantage.mean())
        else:
            return self.output(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config["gamma"]
        self.lr = config["learning_rate"]
        self.use_dueling = config["use_dueling"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]

        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_steps = config["epsilon_decay_steps"]
        self.steps_done = 0

        self.q_network = QNetwork(state_dim, action_dim, dueling=self.use_dueling)
        self.target_network = QNetwork(state_dim, action_dim, dueling=self.use_dueling)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

        self.update_target_network(hard_update=True)

        self.memory = deque(maxlen=config["replay_buffer_size"])

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-self.steps_done / self.epsilon_decay_steps)

        self.steps_done += 1

        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self, hard_update=False):
        if hard_update:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
