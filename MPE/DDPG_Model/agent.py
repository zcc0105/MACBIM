#coding:UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from infoAlg import GaussDistribution
from MPE.DDPG_Model.networks import Actor, Critic
from MPE.DDPG_Model.memory import ReplayBuffer


class Agent:
    def __init__(self, cfg, agent_idx, high, low, state_dim, action_dim):
        self.device = cfg.device
        self.name = 'ddpg_A_%s' % agent_idx
        self.high = high[agent_idx]
        self.low = low[agent_idx]
        self.action_dim = cfg.n_actions
        self.critic = Critic(state_dim, action_dim, cfg.layer1, cfg.layer2, self.name+'_critic').to(cfg.device)
        self.actor = Actor(state_dim, action_dim, cfg.layer1, cfg.layer2, self.name+'_actor').to(cfg.device)
        self.target_critic = Critic(state_dim, action_dim, cfg.layer1, cfg.layer2, self.name+'_target_critic').to(cfg.device)
        self.target_actor = Actor(state_dim, action_dim, cfg.layer1, cfg.layer2, self.name+'_target_actor').to(cfg.device)
        self.reAction = GaussDistribution(high=self.high, low=self.low)
        # copy parameters to target net
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBuffer(cfg.replaybuffer)
        self.batch_size = cfg.batch_size
        self.tau = cfg.tau
        self.gamma = cfg.gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        # action = action[0].detach().numpy()
        # actions = self.reAction.get_action(action)
        actions = action.detach().cpu().numpy()[0]
        return actions

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(
            self.batch_size)
        # convert variables to Tensor
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

    def remember(self, state, actions, reward, new_states, done):
        self.memory.push(state, actions, reward, new_states, done)

