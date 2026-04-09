import torch
import torch.nn.functional as F
import numpy as np
from .neural_net import DuelingDQN
from .replay_buffer import PERBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(x):
    """
    Fast, safe conversion:
    list / tuple / np.ndarray -> torch.FloatTensor on device
    """
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.q = DuelingDQN(state_dim, action_dim).to(device)
        self.target = DuelingDQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg["learning_rate"])
        self.buffer = PERBuffer()

        self.gamma = cfg["gamma"]
        self.action_dim = action_dim

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        # ✅ FIXED (no slow list → tensor conversion)
        s = to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q(s).argmax(dim=1).item()

    def train_step(self, batch, indices, weights):
        # Unpack batch
        s, a, r, ns, d = zip(*batch)

        # ✅ FIXED conversions (FAST)
        s = to_tensor(s)
        ns = to_tensor(ns)
        a = torch.from_numpy(np.asarray(a, dtype=np.int64)).unsqueeze(1).to(device)
        r = torch.from_numpy(np.asarray(r, dtype=np.float32)).unsqueeze(1).to(device)
        d = torch.from_numpy(np.asarray(d, dtype=np.float32)).unsqueeze(1).to(device)
        w = torch.from_numpy(np.asarray(weights, dtype=np.float32)).unsqueeze(1).to(device)

        # Q(s,a)
        q = self.q(s).gather(1, a)

        # Double DQN
        with torch.no_grad():
            next_a = self.q(ns).argmax(dim=1, keepdim=True)
            q_next = self.target(ns).gather(1, next_a)
            target = r + self.gamma * (1 - d) * q_next

        # PER loss
        loss = (w * F.mse_loss(q, target, reduction="none")).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # TD-error for PER
        td = (q - target).abs().detach().cpu().numpy().flatten()
        return td
