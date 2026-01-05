# src/components/reward_functions.py

import numpy as np

def shaped_reward(end_effector_pos, target_pos, previous_distance=None, threshold=0.1, bonus=10.0, penalty=1.0):
    """
    Compute a shaped reward:
    - Base: negative distance to target
    - +bonus for reaching the target
    - +small improvement bonus if getting closer
    - -penalty if getting farther away
    """
    distance = np.linalg.norm(end_effector_pos - target_pos)
    reward = -distance
    done = False

    if previous_distance is not None:
        delta = previous_distance - distance
        reward += 0.1 * delta if delta > 0 else -penalty * abs(delta)

    if distance < threshold:
        reward += bonus
        done = True

    return reward, done, distance
