import numpy as np

# HARD success (final goal)
HARD_THRESHOLD = 0.35
HARD_SUCCESS_BONUS = 30.0

# SOFT success (curriculum trigger)
SOFT_THRESHOLD = 0.55
SOFT_SUCCESS_BONUS = 8.0

STEP_PENALTY = 0.01

def shaped_reward(ee, target, difficulty, previous_distance=None):
    distance = np.linalg.norm(ee - target)

    reward = -distance

    # Progress shaping
    if previous_distance is not None:
        reward += 0.5 * (previous_distance - distance)

    reward -= STEP_PENALTY

    done = False
    success = False

    # 🔹 HARD success (true success)
    if distance < HARD_THRESHOLD:
        reward += HARD_SUCCESS_BONUS
        done = True
        success = True

    # 🔹 SOFT success (curriculum)
    elif distance < SOFT_THRESHOLD:
        reward += SOFT_SUCCESS_BONUS
        done = True   # end episode early

    return reward, done, distance, success
