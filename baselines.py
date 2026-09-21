import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class RSSIHysteresisAgent:
    """
    Traditional Hysteresis-based Vertical Handover Algorithm.
    Triggers handoff only when candidate RSSI > serving RSSI + Margin (Hysteresis H).
    """
    def __init__(self, num_bs, margin_db=6.0):
        self.num_bs = num_bs
        self.margin_db = margin_db

    def select_action(self, state, current_bs_id, qos_type):
        # Extract RSSI values from state
        # State layout: [x, y, v, qos_val (1), bs_onehot (8), RSSI_all (8), SINR_all (8), load_all (8)]
        rssi_start = 4 + self.num_bs
        rssi_vals = state[rssi_start : rssi_start + self.num_bs] * 100.0  # Denormalize
        
        serving_rssi = rssi_vals[current_bs_id] if 0 <= current_bs_id < self.num_bs else -120.0
        best_candidate_id = current_bs_id
        max_rssi = serving_rssi

        for bs_id in range(self.num_bs):
            if rssi_vals[bs_id] > serving_rssi + self.margin_db and rssi_vals[bs_id] > max_rssi:
                max_rssi = rssi_vals[bs_id]
                best_candidate_id = bs_id

        # Map to action index (Slice selection: 0 for URLLC, 1 for eMBB)
        slice_idx = 0 if qos_type == 'URLLC' else 1
        return best_candidate_id * 2 + slice_idx


class StaticDSRC_Agent:
    """
    Static DSRC/ITS-G5 baseline.
    Only connects to candidate DSRC RSUs (IDs: 2, 3, 4).
    """
    def __init__(self, num_bs=8):
        self.dsrc_ids = [2, 3, 4]

    def select_action(self, state, qos_type):
        rssi_start = 4 + 8
        rssi_vals = state[rssi_start : rssi_start + 8] * 100.0
        best_dsrc_id = self.dsrc_ids[0]
        max_rssi = -120.0

        for b_id in self.dsrc_ids:
            if rssi_vals[b_id] > max_rssi:
                max_rssi = rssi_vals[b_id]
                best_dsrc_id = b_id

        slice_idx = 0 if qos_type == 'URLLC' else 1
        return best_dsrc_id * 2 + slice_idx


class SingleAgentDQN:
    """
    Standard Single-Agent Deep Q-Network without target decoupling.
    Demonstrates Q-value overestimation bias in dynamic VHO environments.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=30000)
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return torch.argmax(self.q_net(s_t), dim=1).item()

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Standard DQN Target (no decoupling)
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.q_net(next_states_t).max(1)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()
