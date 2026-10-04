import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_size, act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu(x)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.net(state)

class PPOAgent:
    def __init__(self, obs_dim, state_dim, act_dim, lr=3e-4, gamma=0.99, clip_param=0.2, ppo_epochs=4, batch_size=128, entropy_coef=0.01):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.95)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.95)
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef

    def get_action(self, obs, deterministic=False):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor(obs)
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # Tanh squashing + log_prob correction
        action_tanh = torch.tanh(raw_action)
        log_prob = log_prob - torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
        # Rescale from [-1, 1] to [0, 1] for env
        action = (action_tanh + 1.0) / 2.0
        return action.squeeze(0).numpy(), log_prob.squeeze(0).numpy()
        
    def get_value(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state)
        return value.squeeze(0).numpy()[0]

    def update(self, rollouts):
        obs = torch.FloatTensor(np.array(rollouts['obs']))
        states = torch.FloatTensor(np.array(rollouts['states']))
        actions = torch.FloatTensor(np.array(rollouts['actions']))
        log_probs_old = torch.FloatTensor(np.array(rollouts['log_probs']))
        returns = torch.FloatTensor(np.array(rollouts['returns'])).unsqueeze(1)
        advantages = torch.FloatTensor(np.array(rollouts['advantages'])).unsqueeze(1)
        
        # 优势函数归一化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = obs.size(0)
        if dataset_size == 0:
            return 0, 0
        indices = np.arange(dataset_size)
        
        total_actor_loss, total_critic_loss = 0, 0
        steps = 0
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                b_obs = obs[batch_idx]
                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_log_probs_old = log_probs_old[batch_idx]
                b_returns = returns[batch_idx]
                b_advantages = advantages[batch_idx]
                
                dist = self.actor(b_obs)
                # Stored actions are in [0,1]; convert back through tanh to get raw actions
                actions_tanh = torch.clamp(b_actions * 2 - 1, -1 + 1e-6, 1 - 1e-6)
                raw_actions = torch.atanh(actions_tanh)
                b_log_probs = dist.log_prob(raw_actions).sum(dim=-1)
                b_log_probs = b_log_probs - torch.log(1 - actions_tanh.pow(2) + 1e-6).sum(dim=-1)
                ratio = torch.exp(b_log_probs - b_log_probs_old)

                surr1 = ratio * b_advantages.squeeze(1)
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_advantages.squeeze(1)
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus for exploration
                entropy = dist.entropy().sum(dim=-1).mean()
                actor_loss = actor_loss - self.entropy_coef * entropy

                values = self.critic(b_states)
                critic_loss = nn.MSELoss()(values, b_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                steps += 1

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        return total_actor_loss/steps, total_critic_loss/steps

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    returns = []
    advantages = []
    gae = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_val = next_value * (1 - dones[step])
        else:
            next_val = values[step + 1] * (1 - dones[step])
        delta = rewards[step] + gamma * next_val - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
    return returns, advantages