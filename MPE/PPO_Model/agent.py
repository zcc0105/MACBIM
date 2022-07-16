import numpy as np
import torch as T
import torch.optim as optim

from PPO_Model.networks import Actor, Critic
from PPO_Model.memory import PPOMemory
from infoAlg import GaussDistribution


class Agent:
    def __init__(self, state_dim, action_dim, cfg, agent_idx, high, low):
        self.gamma = cfg.gamma
        self.policy_clip = cfg.policy_clip
        self.n_epochs = cfg.n_epochs
        self.gae_lambda = cfg.gae_lambda
        self.device = cfg.device
        self.n_actions = cfg.n_actions
        self.name = 'ppo_A_%s' % agent_idx
        self.high = high[agent_idx]
        self.low = low[agent_idx]
        self.actor = Actor(state_dim, self.n_actions, cfg.hidden_dim, self.name + '_actor').to(self.device)
        self.critic = Critic(state_dim, cfg.hidden_dim, self.name + '_critic').to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PPOMemory(cfg.batch_size)
        self.loss = 0
        self.reAction = GaussDistribution(high=self.high, low=self.low)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        dist, dist_ = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        reaction = dist_[0][action].detach().numpy()
        actions = [self.reAction.get_action([reaction])[0] for _ in range(self.n_actions)]
        value = T.squeeze(value).item()
        return action, actions, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.sample()
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.device)
            values = T.tensor(values).to(self.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                dist, dist_ = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def remember(self, state, actions, probs, vals, reward, done):
        self.memory.push(state, actions, probs, vals, reward, done)

    def save(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
