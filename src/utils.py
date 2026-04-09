import numpy as np
import random
import torch
import os


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_q_table(q_table, path="artifacts/q_table.npy"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, q_table, allow_pickle=True)


def load_q_table(path="artifacts/q_table.npy"):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))
