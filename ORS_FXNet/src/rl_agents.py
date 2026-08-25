
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DirectionalCorrectionNetwork(nn.Module):
    def __init__(self, state_dim, hidden_units, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, n_actions),
        )

    def forward(self, state):
        return self.net(state)


class DirectionalCorrectionAgent:
    

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.actions = torch.tensor([-cfg.action_delta, 0.0, cfg.action_delta], device=device)
        self.q_net = DirectionalCorrectionNetwork(cfg.state_dim, cfg.hidden_units).to(device)
        self.target_net = DirectionalCorrectionNetwork(cfg.state_dim, cfg.hidden_units).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(cfg.replay_buffer_size)
        self.epsilon = 1.0

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            idx = random.randrange(len(self.actions))
        else:
            with torch.no_grad():
                q_values = self.q_net(state.unsqueeze(0).to(self.device))
                idx = int(torch.argmax(q_values, dim=-1).item())
        return idx, float(self.actions[idx].item())

    def store(self, state, action_idx, reward, next_state, done):
        self.buffer.push(state, action_idx, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * self.cfg.epsilon_decay)

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        states = torch.stack(batch.state).to(self.device)
        actions = torch.tensor(batch.action, device=self.device).long()
        rewards = torch.tensor(batch.reward, device=self.device).float()
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, device=self.device).float()

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + self.cfg.gamma * next_q * (1 - dones)
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())



class MagnitudeActor(nn.Module):
    def __init__(self, state_dim, hidden_units, action_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, 1), nn.Tanh(),
        )
        self.action_bound = action_bound

    def forward(self, state):
        return self.net(state) * self.action_bound


class MagnitudeCritic(nn.Module):
    def __init__(self, state_dim, hidden_units):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class MagnitudeCorrectionAgent:
   

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.actor = MagnitudeActor(cfg.state_dim, cfg.actor_hidden, cfg.action_bound).to(device)
        self.target_actor = MagnitudeActor(cfg.state_dim, cfg.actor_hidden, cfg.action_bound).to(device)
        self.critic = MagnitudeCritic(cfg.state_dim, cfg.critic_hidden).to(device)
        self.target_critic = MagnitudeCritic(cfg.state_dim, cfg.critic_hidden).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.buffer = ReplayBuffer(10_000)
        self.noise_std = 0.05

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0).to(self.device)).squeeze(0)
        if not greedy:
            action = action + torch.randn_like(action) * self.noise_std
            action = torch.clamp(action, -self.cfg.action_bound, self.cfg.action_bound)
        return float(action.item())

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def _soft_update(self, target, source):
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(self.cfg.tau * s_param.data + (1 - self.cfg.tau) * t_param.data)

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None
        batch = self.buffer.sample(batch_size)
        states = torch.stack(batch.state).to(self.device)
        actions = torch.tensor(batch.action, device=self.device).float().unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=self.device).float().unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.tensor(batch.done, device=self.device).float().unsqueeze(1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target = rewards + self.cfg.gamma * target_q * (1 - dones)
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        return float(critic_loss.item()), float(actor_loss.item())



class GatingPolicyNetwork(nn.Module):
   

    def __init__(self, state_dim, hidden_units):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_units), nn.ReLU(),
            nn.Linear(hidden_units, hidden_units), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_units, 1)
        self.log_std = nn.Parameter(torch.zeros(1) - 0.5)
        self.value_head = nn.Linear(hidden_units, 1)

    def forward(self, state):
        h = self.shared(state)
        mean = torch.sigmoid(self.mean_head(h))
        std = torch.exp(self.log_std).clamp(1e-3, 0.5)
        value = self.value_head(h)
        return mean, std, value


class StabilityGatingAgent:
    

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.policy = GatingPolicyNetwork(cfg.state_dim, cfg.hidden_units).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.memory = []

    def select_action(self, state, greedy=False):
        with torch.no_grad():
            mean, std, value = self.policy(state.unsqueeze(0).to(self.device))
        if greedy:
            w1 = mean.squeeze()
            log_prob = torch.zeros(1)
        else:
            dist = torch.distributions.Normal(mean, std)
            raw = dist.sample()
            w1 = torch.clamp(raw, 0.0, 1.0).squeeze()
            log_prob = dist.log_prob(raw).squeeze()
        return float(w1.item()), log_prob.detach(), value.detach()

    def store(self, state, action, log_prob, reward, value, done):
        self.memory.append((state, action, log_prob, reward, value, done))

    def _compute_returns(self, rewards, values, dones):
        returns = []
        running = 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            running = r + self.cfg.gamma * running * (1 - d)
            returns.insert(0, running)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t = torch.tensor(values, dtype=torch.float32, device=self.device)
        advantages = returns - values_t
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, epochs=4):
        if not self.memory:
            return None
        states, actions, old_log_probs, rewards, values, dones = zip(*self.memory)
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device).float().unsqueeze(1)
        old_log_probs = torch.stack(old_log_probs).to(self.device).squeeze()
        returns, advantages = self._compute_returns(rewards, [v.item() for v in values], dones)

        total_loss = 0.0
        for _ in range(epochs):
            mean, std, value = self.policy(states)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).squeeze()
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()
            value_loss = F.mse_loss(value.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - self.cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())

        self.memory = []
        return total_loss / epochs


def build_state_vector(y_base, h_summary, z_summary, volatility, prev_error, state_dim):

    parts = [
        np.atleast_1d(y_base).astype(np.float32),
        np.atleast_1d(h_summary).astype(np.float32),
        np.atleast_1d(z_summary).astype(np.float32),
        np.atleast_1d(volatility).astype(np.float32),
        np.atleast_1d(prev_error).astype(np.float32),
    ]
    vec = np.concatenate(parts)
    if len(vec) < state_dim:
        vec = np.pad(vec, (0, state_dim - len(vec)))
    else:
        vec = vec[:state_dim]
    return vec


def correction_reward(y_hat, y_true, y_prev_true, correction_bonus=0.05):
    """r_t = -|y_hat - y| + lambda * 1[correct direction]   -- Eq. (42)"""
    error = -abs(y_hat - y_true)
    direction_true = np.sign(y_true - y_prev_true)
    direction_pred = np.sign(y_hat - y_prev_true)
    bonus = correction_bonus if direction_true == direction_pred and direction_true != 0 else 0.0
    return error + bonus


class MultiAgentCorrectionEnsemble:
 
    def __init__(self, dqn_cfg, ddpg_cfg, ppo_cfg, device="cpu"):
        self.dqn_agent = DirectionalCorrectionAgent(dqn_cfg, device)
        self.ddpg_agent = MagnitudeCorrectionAgent(ddpg_cfg, device)
        self.ppo_agent = StabilityGatingAgent(ppo_cfg, device)
        self.device = device

    def correct(self, state_vec, y_base, greedy=False):
        state_t = torch.as_tensor(state_vec, dtype=torch.float32)
        _, delta_dqn = self.dqn_agent.select_action(state_t, greedy=greedy)
        delta_ddpg = self.ddpg_agent.select_action(state_t, greedy=greedy)
        w1, log_prob, value = self.ppo_agent.select_action(state_t, greedy=greedy)
        w2 = 1.0 - w1
        y_final = y_base + w1 * delta_dqn + w2 * delta_ddpg  # Table 4 formulation
        return y_final, {
            "delta_dqn": delta_dqn, "delta_ddpg": delta_ddpg,
            "w1": w1, "w2": w2, "log_prob": log_prob, "value": value,
        }

    def save(self, path):
       
        torch.save({
            "dqn_q_net": self.dqn_agent.q_net.state_dict(),
            "ddpg_actor": self.ddpg_agent.actor.state_dict(),
            "ddpg_critic": self.ddpg_agent.critic.state_dict(),
            "ppo_policy": self.ppo_agent.policy.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, dqn_cfg, ddpg_cfg, ppo_cfg, device="cpu"):
        ensemble = cls(dqn_cfg, ddpg_cfg, ppo_cfg, device)
        checkpoint = torch.load(path, map_location=device)
        ensemble.dqn_agent.q_net.load_state_dict(checkpoint["dqn_q_net"])
        ensemble.dqn_agent.target_net.load_state_dict(checkpoint["dqn_q_net"])
        ensemble.ddpg_agent.actor.load_state_dict(checkpoint["ddpg_actor"])
        ensemble.ddpg_agent.target_actor.load_state_dict(checkpoint["ddpg_actor"])
        ensemble.ddpg_agent.critic.load_state_dict(checkpoint["ddpg_critic"])
        ensemble.ddpg_agent.target_critic.load_state_dict(checkpoint["ddpg_critic"])
        ensemble.ppo_agent.policy.load_state_dict(checkpoint["ppo_policy"])
        return ensemble
