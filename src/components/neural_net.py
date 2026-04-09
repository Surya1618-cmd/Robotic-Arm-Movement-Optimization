import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)

        value = self.value(x)
        advantage = self.advantage(x)

        # Dueling formula
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q
