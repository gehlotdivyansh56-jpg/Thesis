import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque

class SpatialTemporalAttention(nn.Module):
    """
    Multi-Head Spatial-Temporal Self-Attention Mechanism.
    Dynamically attends to spatial base station channels, interference, and lookahead trajectory features.
    """
    def __init__(self, input_dim, embed_dim=128, num_heads=4):
        super(SpatialTemporalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.fc_in = nn.Linear(input_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        h = self.fc_in(x).unsqueeze(1)  # (batch, 1, embed_dim)
        batch_size = h.size(0)

        q = self.query(h).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(h).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(h).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (batch, num_heads, 1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)

        out = self.layer_norm(self.proj(attn_out) + h.squeeze(1))
        return out


class AttnActor(nn.Module):
    """
    Actor Network: State -> Attention Representation -> Action Preference Logits
    """
    def __init__(self, state_dim, action_dim, embed_dim=128):
        super(AttnActor, self).__init__()
        self.attn = SpatialTemporalAttention(state_dim, embed_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        attn_feat = self.attn(state)
        x = self.relu(self.fc1(attn_feat))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


class TwinCritic(nn.Module):
    """
    Twin Critic Networks (Q1 and Q2) to prevent Q-value overestimation bias.
    """
    def __init__(self, state_dim, action_dim):
        super(TwinCritic, self).__init__()
        # Q1 Architecture
        self.q1_fc1 = nn.Linear(state_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, action_dim)

        # Q2 Architecture
        self.q2_fc1 = nn.Linear(state_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x1 = self.relu(self.q1_fc1(state))
        x1 = self.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)

        x2 = self.relu(self.q2_fc1(state))
        x2 = self.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        return q1, q2


class AttnFedTD3Agent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, tau=0.005, policy_freq=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.total_it = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor and Target Actor
        self.actor = AttnActor(state_dim, action_dim).to(self.device)
        self.actor_target = AttnActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Twin Critic and Target Critic
        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = deque(maxlen=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.992

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.actor(s_t)
            return torch.argmax(logits, dim=1).item()

    def select_actions_batch(self, states, eval_mode=False):
        states_np = np.array(states, dtype=np.float32)
        N = len(states)
        with torch.no_grad():
            s_t = torch.FloatTensor(states_np).to(self.device)
            logits = self.actor(s_t)
            actions = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        
        if not eval_mode:
            for i in range(N):
                if random.random() < self.epsilon:
                    actions[i] = random.randint(0, self.action_dim - 1)
        return actions

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return 0.0

        self.total_it += 1
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # ------------------- 1. Critic Update -------------------
        with torch.no_grad():
            next_logits = self.actor_target(next_states_t)
            next_probs = F.softmax(next_logits, dim=-1)

            target_q1, target_q2 = self.critic_target(next_states_t)
            min_target_q = torch.min(target_q1, target_q2)
            expected_next_q = (next_probs * min_target_q).sum(dim=-1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * expected_next_q

        q1_all, q2_all = self.critic(states_t)
        current_q1 = q1_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        current_q2 = q2_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        critic_loss = F.smooth_l1_loss(current_q1, target_q) + F.smooth_l1_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ------------------- 2. Delayed Actor Update (TD3 Core) -------------------
        if self.total_it % self.policy_freq == 0:
            actor_logits = self.actor(states_t)
            action_probs = F.softmax(actor_logits, dim=-1)
            log_probs = F.log_softmax(actor_logits, dim=-1)

            with torch.no_grad():
                q1_eval, q2_eval = self.critic(states_t)
                min_q = torch.min(q1_eval, q2_eval)

            # Differentiable expected Q maximization
            expected_q = (action_probs * min_q).sum(dim=-1)
            entropy = -(action_probs * log_probs).sum(dim=-1)
            actor_loss = -(expected_q + 0.01 * entropy).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Soft update target network weights
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return critic_loss.item()


class FederatedAggregator:
    """
    Federated Learning Aggregator for cooperative edge vehicle model updates.
    Performs federated parameter averaging across decentralized vehicular agents.
    """
    @staticmethod
    def aggregate_agents(agents):
        if not agents:
            return

        # 1. Aggregate Actor Parameters
        actor_state_dicts = [a.actor.state_dict() for a in agents]
        avg_actor_dict = {}
        for k in actor_state_dicts[0].keys():
            avg_actor_dict[k] = torch.stack([d[k].float() for d in actor_state_dicts], dim=0).mean(dim=0)

        # 2. Aggregate Critic Parameters
        critic_state_dicts = [a.critic.state_dict() for a in agents]
        avg_critic_dict = {}
        for k in critic_state_dicts[0].keys():
            avg_critic_dict[k] = torch.stack([d[k].float() for d in critic_state_dicts], dim=0).mean(dim=0)

        # 3. Synchronize all agents with global model
        for a in agents:
            a.actor.load_state_dict(avg_actor_dict)
            a.critic.load_state_dict(avg_critic_dict)
