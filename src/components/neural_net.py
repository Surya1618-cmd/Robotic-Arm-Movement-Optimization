import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_dueling=False):
        super(QNetwork, self).__init__()
        self.use_dueling = use_dueling

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        if self.use_dueling:
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, action_dim)
        else:
            self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine streams: Q(s, a) = V(s) + (A(s, a) - mean(A(s, ·)))
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc3(x)
