import numpy as np
from src.components.agent import DQNAgent
from src.config import config


def test_agent_action_and_memory():
    state_dim = 6
    action_dim = 6
    agent = DQNAgent(state_dim, action_dim, config)
    state = np.random.rand(state_dim).astype(np.float32)

    action = agent.select_action(state, epsilon=0.5)
    assert isinstance(action, int)


    next_state = np.random.rand(state_dim).astype(np.float32)
    agent.remember(state, action, reward=-1.0, next_state=next_state, done=False)
    assert len(agent.memory) > 0