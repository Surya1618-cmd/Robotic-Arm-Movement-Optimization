import yaml
from pathlib import Path

BASE_PATH = Path("configs")
QL_PATH = BASE_PATH / "qlearning.yaml"
ENV_PATH = BASE_PATH / "environment.yaml"


def load_qlearning_config():
    with open(QL_PATH, "r") as f:
        return yaml.safe_load(f)


def load_environment_config():
    if ENV_PATH.exists():
        with open(ENV_PATH, "r") as f:
            return yaml.safe_load(f)
    return None
