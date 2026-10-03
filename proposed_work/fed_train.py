"""
Federated DRL for Vertical Handoff -- pure Python, no ns-3/SUMO.
==================================================================
This is your proposed-work extension on top of the reproduced base paper
(env.py / agent.py). Where the base paper uses meta-transfer learning
(A2E) to generalize across scenarios, this replaces that mechanism with
federated averaging across vehicles:

  For each communication round:
    1. Every vehicle gets a copy of the current global network weights.
    2. Each vehicle trains locally (DDQN-TS, Algorithm 1) for
       `episodes_per_round` episodes on its OWN environment instance
       (own random seed -> own stretch of trajectory/channel realizations,
       i.e. non-IID data across vehicles, same as real deployments).
    3. The server averages the vehicles' online-network weights (FedAvg).
    4. The Thompson-sampling exploration head (Bayesian linear regression)
       is kept as a single SHARED object updated by every vehicle during
       local training -- i.e. only aggregated statistics cross the
       vehicle/server boundary, not raw (state, action, reward) data,
       matching the federated-learning privacy motivation from the
       literature review (Sec. IX-C).
    5. The aggregated global model is evaluated on a fresh held-out
       environment instance to produce this round's reported metrics.

Everything here is standard Python + PyTorch + NumPy. No ns-3, no SUMO --
env.py's simplified channel/mobility model (see its docstring) stands in
for both, exactly as in the base-paper reproduction.
"""

import argparse
import copy
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import V2NHandoverEnv
from agent import DDQNThompsonAgent, BayesianLinearHead, set_seed
from train import run_episode


def fedavg(state_dicts):
    """Simple (unweighted) federated averaging of a list of state_dicts."""
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0)
    return avg


def run_federated(scenario: str, num_vehicles: int, num_rounds: int,
                   episodes_per_round: int, lr: float, max_steps: int,
                   feature_dim: int = 64, gamma: float = 0.99,
                   buffer_capacity: int = 2000, batch_size: int = 64,
                   target_sync_every: int = 7, seed: int = 0):
    """Every hyperparameter a dashboard would expose is a real argument here
    -- nothing about vehicle count, episode budget, or network/RL
    hyperparameters is hardcoded. Defaults above only apply when a caller
    (CLI or otherwise) doesn't override them."""
    set_seed(seed)

    # one persistent env + agent per vehicle (non-IID: each vehicle sees its
    # own random stream of BS layouts/fading via its own seed)
    envs = [V2NHandoverEnv(scenario=scenario, seed=seed + 100 * v) for v in range(num_vehicles)]
    for e in envs:
        if max_steps is not None:
            e.max_steps = max_steps

    state_dim, action_dim = envs[0].state_dim, envs[0].action_dim
    shared_blr = BayesianLinearHead(feature_dim=feature_dim, num_actions=action_dim)

    agent_kwargs = dict(feature_dim=feature_dim, lr=lr, gamma=gamma,
                         buffer_capacity=buffer_capacity, batch_size=batch_size,
                         target_sync_every=target_sync_every, blr=shared_blr)

    # template agent purely to hold/init the global network weights
    global_agent = DDQNThompsonAgent(state_dim, action_dim, seed=seed, **agent_kwargs)
    global_weights = global_agent.online.state_dict()

    vehicle_agents = [
        DDQNThompsonAgent(state_dim, action_dim, seed=seed + v, **agent_kwargs)
        for v in range(num_vehicles)
    ]

    eval_env = V2NHandoverEnv(scenario=scenario, seed=seed + 999)
    if max_steps is not None:
        eval_env.max_steps = max_steps

    history = {"round": [], "reward": [], "ho_prob": [], "pingpong_prob": []}

    for rnd in range(1, num_rounds + 1):
        local_weights = []
        for v, agent in enumerate(vehicle_agents):
            agent.online.load_state_dict(global_weights)
            agent.target.load_state_dict(global_weights)
            for _ in range(episodes_per_round):
                run_episode(envs[v], agent, train=True)  # trains network + updates shared_blr
            local_weights.append(agent.online.state_dict())

        global_weights = fedavg(local_weights)
        global_agent.online.load_state_dict(global_weights)
        global_agent.target.load_state_dict(global_weights)

        eval_stats = run_episode(eval_env, global_agent, train=False)
        history["round"].append(rnd)
        history["reward"].append(eval_stats["reward"])
        history["ho_prob"].append(eval_stats["ho_prob"])
        history["pingpong_prob"].append(eval_stats["pingpong_prob"])

        print(f"[round {rnd:3d}/{num_rounds}] global eval: "
              f"reward={eval_stats['reward']:.4f}  HO_prob={eval_stats['ho_prob']:.3f}  "
              f"ping-pong={eval_stats['pingpong_prob']:.3f}")

    return history, global_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["urban", "rural", "highway"], default="urban")
    p.add_argument("--num_vehicles", type=int, default=10)
    p.add_argument("--num_rounds", type=int, default=20)
    p.add_argument("--episodes_per_round", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max_steps", type=int, default=200,
                    help="episode length override; full Table-I length is 3000-5000 steps")
    p.add_argument("--feature_dim", type=int, default=64, help="Q-network feature/embedding size")
    p.add_argument("--gamma", type=float, default=0.99, help="RL discount factor")
    p.add_argument("--buffer_capacity", type=int, default=2000, help="replay buffer size per vehicle")
    p.add_argument("--batch_size", type=int, default=64, help="minibatch size for each gradient step")
    p.add_argument("--target_sync_every", type=int, default=7,
                    help="steps between target-network syncs (paper uses 7)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_json", default="federated_history.json")
    p.add_argument("--out_png", default="federated_convergence.png")
    args = p.parse_args()

    history, _ = run_federated(
        args.scenario, args.num_vehicles, args.num_rounds, args.episodes_per_round,
        args.lr, args.max_steps, feature_dim=args.feature_dim, gamma=args.gamma,
        buffer_capacity=args.buffer_capacity, batch_size=args.batch_size,
        target_sync_every=args.target_sync_every, seed=args.seed)

    with open(args.out_json, "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)
    print(f"\nSaved raw metrics to {args.out_json}")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(history["round"], history["reward"], color="#4361ee", label="Global avg. reward")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Normalized reward", color="#4361ee")
    ax2 = ax1.twinx()
    ax2.plot(history["round"], history["ho_prob"], color="#e76f51", label="Handoff prob.")
    ax2.plot(history["round"], history["pingpong_prob"], color="#2a9d8f", label="Ping-pong prob.")
    ax2.set_ylabel("Probability")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    plt.title(f"Federated DRL convergence -- {args.scenario}, "
              f"{args.num_vehicles} vehicles, {args.num_rounds} rounds")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()