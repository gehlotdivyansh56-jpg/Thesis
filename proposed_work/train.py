"""
Reproduces the base-paper comparison of Sec. VI:
  - DDQN-TS (similar-scenario): trained & tested on the SAME scenario -> upper bound
  - DDQN-TS (different-scenario): trained on 'urban', tested on the target scenario
  - DDQN (epsilon-greedy, different-scenario): the paper's baseline [17]

(The paper's third method, A2E meta-transfer learning, is the paper's OWN
proposed contribution on top of this DDQN-TS base algorithm -- it is not
part of "the base paper's DDQN-TS baseline" and is intentionally left out
here. Add it once this base reproduction matches the paper's qualitative
trend, then your Federated-DRL extension replaces A2E's role.)

Usage:
    python train.py --target highway --episodes 60
    python train.py --target rural   --episodes 60
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env import V2NHandoverEnv
from agent import DDQNThompsonAgent, DDQNEpsGreedyAgent, set_seed


def run_episode(env, agent, train=True):
    s = env.reset()
    mask = env.valid_action_mask()
    total_reward, ho_count, pingpong_count, steps = 0.0, 0, 0, 0
    losses = []
    done = False
    while not done:
        a = agent.select_action(s, valid_mask=mask)
        s2, r, done, info = env.step(a)
        if train:
            agent.store(s, a, r, s2, float(done))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
        total_reward += r
        ho_count += int(info["handover"])
        pingpong_count += int(info["ping_pong"])
        steps += 1
        s = s2
    return dict(reward=total_reward / steps, ho_prob=ho_count / steps,
                pingpong_prob=pingpong_count / max(ho_count, 1),
                loss=np.mean(losses) if losses else np.nan)


def train_agent(agent_ctor, scenario, episodes, seed, max_steps=None):
    set_seed(seed)
    env = V2NHandoverEnv(scenario=scenario, seed=seed)
    if max_steps is not None:
        env.max_steps = max_steps
    agent = agent_ctor(env.state_dim, env.action_dim)
    history = []
    for ep in range(episodes):
        stats = run_episode(env, agent, train=True)
        history.append(stats)
        if (ep + 1) % 10 == 0:
            print(f"  [{scenario}] ep {ep+1}/{episodes}  "
                  f"reward={stats['reward']:.4f}  HO_prob={stats['ho_prob']:.3f}  "
                  f"pingpong={stats['pingpong_prob']:.3f}")
    return agent, history


def evaluate_cross_scenario(agent, target_scenario, seed, episodes=5, max_steps=None):
    env = V2NHandoverEnv(scenario=target_scenario, seed=seed + 1000)
    if max_steps is not None:
        env.max_steps = max_steps
    results = [run_episode(env, agent, train=False) for _ in range(episodes)]
    return dict(reward=np.mean([r["reward"] for r in results]),
                ho_prob=np.mean([r["ho_prob"] for r in results]),
                pingpong_prob=np.mean([r["pingpong_prob"] for r in results]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["highway", "rural", "urban"], default="highway",
                         help="target scenario, matching the paper's Sec. VI-A / VI-B")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None,
                         help="override episode length (steps of 200ms). Full "
                              "scenario length from Table I is 3000-5000 steps "
                              "per episode, which is slow to iterate on -- start "
                              "with e.g. --max_steps 300 while debugging, then "
                              "remove this flag for the real run.")
    args = parser.parse_args()

    print(f"=== DDQN-TS (similar-scenario) trained & tested on '{args.target}' ===")
    _, hist_similar = train_agent(DDQNThompsonAgent, args.target, args.episodes, args.seed,
                                   max_steps=args.max_steps)

    print(f"\n=== DDQN-TS (different-scenario): train on 'urban', test on '{args.target}' ===")
    agent_ts, hist_ts_urban = train_agent(DDQNThompsonAgent, "urban", args.episodes, args.seed,
                                           max_steps=args.max_steps)
    cross_ts = evaluate_cross_scenario(agent_ts, args.target, args.seed, max_steps=args.max_steps)

    print(f"\n=== DDQN (epsilon-greedy, different-scenario): train on 'urban', test on '{args.target}' ===")
    agent_eps, hist_eps_urban = train_agent(DDQNEpsGreedyAgent, "urban", args.episodes, args.seed,
                                             max_steps=args.max_steps)
    cross_eps = evaluate_cross_scenario(agent_eps, args.target, args.seed, max_steps=args.max_steps)

    print("\n================ Cross-scenario evaluation summary ================")
    print(f"{'Method':35s} {'Reward':>10s} {'HO prob':>10s} {'Ping-pong':>10s}")
    print(f"{'DDQN-TS (similar-scenario, upper bound)':35s} "
          f"{hist_similar[-1]['reward']:10.4f} {hist_similar[-1]['ho_prob']:10.3f} "
          f"{hist_similar[-1]['pingpong_prob']:10.3f}")
    print(f"{'DDQN-TS (different-scenario)':35s} {cross_ts['reward']:10.4f} "
          f"{cross_ts['ho_prob']:10.3f} {cross_ts['pingpong_prob']:10.3f}")
    print(f"{'DDQN eps-greedy (different-scenario)':35s} {cross_eps['reward']:10.4f} "
          f"{cross_eps['ho_prob']:10.3f} {cross_eps['pingpong_prob']:10.3f}")

    # --- convergence plot, mirroring the paper's Fig. 5 ---
    plt.figure(figsize=(7, 4.5))
    plt.plot([h["reward"] for h in hist_similar], label="DDQN-TS (similar-scenario)")
    plt.plot([h["reward"] for h in hist_ts_urban], label="DDQN-TS (trained on urban)")
    plt.plot([h["reward"] for h in hist_eps_urban], label="DDQN eps-greedy (trained on urban)")
    plt.xlabel("Episode")
    plt.ylabel("Normalized reward")
    plt.title(f"Convergence -- target scenario: {args.target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"convergence_{args.target}.png", dpi=150)
    print(f"\nSaved convergence plot to convergence_{args.target}.png")


if __name__ == "__main__":
    main()