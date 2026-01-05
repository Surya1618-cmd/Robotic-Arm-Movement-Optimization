# src/utils/util.py

import os
import torch
import numpy as np

def set_seed(seed):
    """Set random seeds for reproducibility."""
    import random
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, path="artifacts/best_model.pth"):
    """Save model weights to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Load model weights from file into a given model."""
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model

def compute_distance(a, b):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.array(a) - np.array(b))
