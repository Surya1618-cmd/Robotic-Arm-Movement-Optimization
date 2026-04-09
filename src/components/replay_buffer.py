import numpy as np
import random

class PERBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        max_p = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_p)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_p
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p
