import pytest
import numpy as np
from src.components.environment import ThreeJointArmEnv

def test_environment_reset_and_step():
    env = ThreeJointArmEnv()
    obs, info = env.reset()
    assert len(obs) == 6
    assert "distance" in info

    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert len(next_obs) == 6
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "distance" in info
