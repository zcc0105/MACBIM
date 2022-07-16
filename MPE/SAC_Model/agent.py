from SAC_Model.memory import ReplayBuffer
from SAC_Model.networks import ValueNet, PolicyNet, SoftQNet
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from infoAlg import GaussDistribution


class Agent:
    def __init__(self, state_dim, action_dim, high, low, cfg, agent_idx, path='D:\BCIM\MABCIM\MPE\MASB\MASB\env') -> None:
        self.batch_size = cfg.batch_size
        self.memory = ReplayBuffer(cfg.capacity)
        self.device = cfg.device
        self.path = path
        self.high = high[agent_idx]
        self.low = low[agent_idx]
        self.name = 'SAC_%s' % agent_idx
        self.value_net = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.target_value_net = ValueNet(state_dim, cfg.hidden_dim).to(self.device)
        self.soft_q_net = SoftQNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.policy_net = PolicyNet(state_dim, action_dim, cfg.hidden_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=cfg.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr)
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        self.reAction = GaussDistribution(high=self.high, low=self.low)

    def learn(self, gamma=0.99, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0, soft_tau=1e-2):
        if len(self.memory) < self.batch_size:
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        expected_q_value = self.soft_q_net(state, action)
        expected_value = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)

        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())

        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())

        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()

        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss = std_lambda * log_std.pow(2).mean()
        z_loss = z_lambda * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def choose_action(self, state):
        action = self.policy_net.get_action(state)
        actions = self.reAction.get_action(action)
        return actions

    def save(self):
        torch.save(self.value_net.state_dict(), self.path + "\SAC_value")
        torch.save(self.value_optimizer.state_dict(), self.path + "\SAC_value_optimizer")

        torch.save(self.soft_q_net.state_dict(), self.path + "\SAC_soft_q")
        torch.save(self.soft_q_optimizer.state_dict(), self.path + "\SAC_soft_q_optimizer")

        torch.save(self.policy_net.state_dict(), self.path + "\SAC_policy")
        torch.save(self.policy_optimizer.state_dict(), self.path + "\SAC_policy_optimizer")

    def load(self):
        self.value_net.load_state_dict(torch.load(self.path + "\SAC_value"))
        self.value_optimizer.load_state_dict(torch.load(self.path + "\SAC_value_optimizer"))
        self.target_value_net = copy.deepcopy(self.value_net)

        self.soft_q_net.load_state_dict(torch.load(self.path + "\SAC_soft_q"))
        self.soft_q_optimizer.load_state_dict(torch.load(self.path + "\SAC_soft_q_optimizer"))

        self.policy_net.load_state_dict(torch.load(self.path + "\SAC_policy"))
        self.policy_optimizer.load_state_dict(torch.load(self.path + "\SAC_policy_optimizer"))