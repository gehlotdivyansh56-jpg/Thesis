import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Ensure parent path can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proposed_env import ProposedIoVEnv
from attn_fed_td3 import AttnFedTD3Agent, FederatedAggregator

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def run_proposed_training():
    results_dir = os.path.join("proposed_work", "results")
    models_dir = os.path.join("proposed_work", "models")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("==================================================================================", flush=True)
    print("  PROPOSED NOVEL WORK: Attn-FedTD3 Proactive Vertical Handover Simulation", flush=True)
    print("==================================================================================", flush=True)

    num_vehicles = 12
    max_steps = 100
    episodes = 60
    eval_episodes = 20

    env = ProposedIoVEnv(num_vehicles=num_vehicles, max_steps=max_steps)
    state_dim = env.state_dim
    action_dim = env.num_actions

    # Create proposed agent with federated edge aggregator
    proposed_agent = AttnFedTD3Agent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    rewards_history = []
    pdrs_history = []
    lats_history = []
    tps_history = []
    pingpongs_history = []

    start_time = time.time()
    print("\n[+] Training Novel Proposed Attn-FedTD3 Agent...", flush=True)

    for ep in range(1, episodes + 1):
        states = env.reset()
        ep_reward = 0.0
        ep_pdrs, ep_lats, ep_tps, ep_pp = [], [], [], 0

        for step in range(max_steps):
            # Batch action selection across all vehicles
            action_list = proposed_agent.select_actions_batch(states, eval_mode=False)
            action_dict = {veh.veh_id: action_list[i] for i, veh in enumerate(env.vehicles)}

            next_states, rewards, done, info_dict = env.step(action_dict)

            # Store multi-agent vehicle transitions
            for v_idx, veh in enumerate(env.vehicles):
                r = rewards[veh.veh_id]
                ep_reward += r
                proposed_agent.memory.append((
                    states[v_idx],
                    action_dict[veh.veh_id],
                    r,
                    next_states[veh.veh_id],
                    float(done)
                ))

                info = info_dict[veh.veh_id]
                ep_pdrs.append(info['pdr'])
                ep_lats.append(info['latency'])
                ep_tps.append(info['throughput'])
                if info['is_pingpong']:
                    ep_pp += 1

            # Step-level gradient update (batch_size=64)
            proposed_agent.update(batch_size=64)
            states = [next_states[veh.veh_id] for veh in env.vehicles]

        rewards_history.append(ep_reward / num_vehicles)
        pdrs_history.append(np.mean(ep_pdrs) * 100.0)
        lats_history.append(np.mean(ep_lats))
        tps_history.append(np.mean(ep_tps))
        pingpongs_history.append(ep_pp / num_vehicles)

        if ep % 10 == 0 or ep == 1:
            print(f" Episode {ep:02d}/{episodes} | Avg Reward: {rewards_history[-1]:6.2f} | PDR: {pdrs_history[-1]:5.1f}% | Latency: {lats_history[-1]:5.2f} ms | Throughput: {tps_history[-1]:5.1f} Mbps | Ping-Pong: {pingpongs_history[-1]:.2f} | Epsilon: {proposed_agent.epsilon:.3f}", flush=True)

    print(f"[+] Proposed Work Training complete in {time.time() - start_time:.2f}s.", flush=True)

    # Save trained global model
    torch.save(proposed_agent.actor.state_dict(), os.path.join(models_dir, "attn_fed_td3_actor.pth"))
    torch.save(proposed_agent.critic.state_dict(), os.path.join(models_dir, "attn_fed_td3_critic.pth"))

    # ----------------------------------------------------
    # DETERMINISTIC EVALUATION PHASE (eval_mode=True)
    # ----------------------------------------------------
    print(f"\n[+] Running Deterministic Evaluation across {eval_episodes} test episodes (eval_mode=True)...", flush=True)
    eval_pdrs, eval_lats, eval_tps, eval_pps = [], [], [], []

    for ep in range(eval_episodes):
        states = env.reset()
        ep_pdrs, ep_lats, ep_tps, ep_pp = [], [], [], 0

        for step in range(max_steps):
            action_list = proposed_agent.select_actions_batch(states, eval_mode=True)
            action_dict = {veh.veh_id: action_list[i] for i, veh in enumerate(env.vehicles)}

            next_states, rewards, done, info_dict = env.step(action_dict)

            for veh in env.vehicles:
                info = info_dict[veh.veh_id]
                ep_pdrs.append(info['pdr'])
                ep_lats.append(info['latency'])
                ep_tps.append(info['throughput'])
                if info['is_pingpong']:
                    ep_pp += 1

            states = [next_states[veh.veh_id] for veh in env.vehicles]

        eval_pdrs.append(np.mean(ep_pdrs) * 100.0)
        eval_lats.append(np.mean(ep_lats))
        eval_tps.append(np.mean(ep_tps))
        eval_pps.append(ep_pp / num_vehicles)

    final_prop_pdr = np.mean(eval_pdrs)
    final_prop_lat = np.mean(eval_lats)
    final_prop_tp = np.mean(eval_tps)
    final_prop_pp = np.mean(eval_pps)

    # Base Paper Performance Values (from Base Paper [13] / benchmark)
    base_pdr = 82.62
    base_lat = 4.51
    base_tp = 212.85
    base_pp = 4.12

    # Comparative DataFrame
    df_compare = pd.DataFrame({
        "Scheme": ["Proposed Novel Attn-FedTD3", "Base Paper MA-DDQN [13]", "RSSI Hysteresis", "Static DSRC / ITS-G5"],
        "PDR (%)": [final_prop_pdr, base_pdr, 85.45, 56.95],
        "E2E Latency (ms)": [final_prop_lat, base_lat, 2.67, 5.80],
        "Throughput (Mbps)": [final_prop_tp, base_tp, 230.95, 28.22],
        "Ping-Pong Rate (per veh)": [final_prop_pp, base_pp, 2.68, 7.38]
    })

    print("\n" + "="*85, flush=True)
    print("          SIDE-BY-SIDE COMPARISON: PROPOSED WORK vs BASE PAPER [13]", flush=True)
    print("="*85, flush=True)
    print(df_compare.to_string(index=False), flush=True)
    print("="*85, flush=True)

    df_compare.to_csv(os.path.join(results_dir, "proposed_evaluation_metrics.csv"), index=False)

    # ----------------------------------------------------
    # GENERATE COMPARATIVE FIGURES
    # ----------------------------------------------------
    schemes = df_compare["Scheme"]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    # 1. PDR Plot
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    bars = plt.bar(schemes, df_compare["PDR (%)"], color=colors, width=0.52)
    plt.title("Packet Delivery Ratio (PDR %): Proposed vs Base Paper", fontsize=13, fontweight='bold')
    plt.ylabel("PDR (%) - [Higher is Better]", fontsize=11)
    plt.ylim(0, 115)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 1.2, f"{h:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proposed_pdr_comparison.png"))
    plt.close()

    # 2. Latency Plot
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    bars = plt.bar(schemes, df_compare["E2E Latency (ms)"], color=colors, width=0.52)
    plt.title("End-to-End Latency (ms): Proposed vs Base Paper", fontsize=13, fontweight='bold')
    plt.ylabel("Latency (ms) - [Lower is Better]", fontsize=11)
    plt.ylim(0, max(df_compare["E2E Latency (ms)"]) * 1.25)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.15, f"{h:.2f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proposed_latency_comparison.png"))
    plt.close()

    # 3. Ping-Pong Plot
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    bars = plt.bar(schemes, df_compare["Ping-Pong Rate (per veh)"], color=colors, width=0.52)
    plt.title("Ping-Pong Handovers per Vehicle: Proposed vs Base Paper", fontsize=13, fontweight='bold')
    plt.ylabel("Ping-Pong Count / Vehicle - [Lower is Better]", fontsize=11)
    plt.ylim(0, max(df_compare["Ping-Pong Rate (per veh)"]) * 1.25)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.15, f"{h:.2f}", ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proposed_pingpong_comparison.png"))
    plt.close()

    # 4. Throughput Plot
    plt.figure(figsize=(8.5, 5.2), dpi=300)
    bars = plt.bar(schemes, df_compare["Throughput (Mbps)"], color=colors, width=0.52)
    plt.title("Average System Throughput (Mbps): Proposed vs Base Paper", fontsize=13, fontweight='bold')
    plt.ylabel("Throughput (Mbps) - [Higher is Better]", fontsize=11)
    plt.ylim(0, max(df_compare["Throughput (Mbps)"]) * 1.2)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 6.0, f"{h:.1f} Mbps", ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proposed_throughput_comparison.png"))
    plt.close()

    # 5. Training Reward Curve
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(range(1, episodes + 1), rewards_history, label="Proposed Attn-FedTD3", color="#2ca02c", linewidth=2.5)
    plt.title("Proposed Attn-FedTD3 Convergence Curve", fontsize=13, fontweight='bold')
    plt.xlabel("Training Episode", fontsize=11)
    plt.ylabel("Average Cumulative Reward per Vehicle", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "proposed_reward_curve.png"))
    plt.close()

    print(f"\n[+] All proposed work experimental figures and metrics CSV saved to '{results_dir}/'.", flush=True)

if __name__ == "__main__":
    run_proposed_training()
