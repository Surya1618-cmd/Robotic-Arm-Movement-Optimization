import os
os.makedirs("artifacts", exist_ok=True)
import yaml, csv, os, torch
from src.components.environment import ThreeJointArmEnv
from src.components.agent import DQNAgent

cfg = yaml.safe_load(open("configs/dqn.yaml"))
env = ThreeJointArmEnv()

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)

os.makedirs("data", exist_ok=True)
log = csv.writer(open("data/training_log.csv","w",newline=""))
log.writerow(["episode","total_reward","best_distance","epsilon"])

epsilon = cfg["epsilon_start"]

for ep in range(1, cfg["episodes"]+1):
    s,_ = env.reset()
    total, best = 0, 1e9

    for _ in range(cfg["max_steps"]):
        a = agent.select_action(s, epsilon)
        ns,r,d,t,info = env.step(a)
        agent.buffer.push((s,a,r,ns,d))
        total += r
        best = min(best, info["distance"])
        s = ns
        if d or t: break

        if len(agent.buffer.buffer) > cfg["batch_size"]:
            batch, idx, w = agent.buffer.sample(cfg["batch_size"])
            td = agent.train_step(batch, idx, w)
            agent.buffer.update_priorities(idx, td+1e-6)

    epsilon = max(cfg["epsilon_min"], epsilon*cfg["epsilon_decay"])
    log.writerow([ep,total,best,epsilon])

    if ep % 50 == 0:
        print(f"Ep {ep} | Reward {total:.2f} | BestDist {best:.3f} | Eps {epsilon:.2f}")

torch.save(agent.q.state_dict(), "artifacts/dqn_model.pth")

