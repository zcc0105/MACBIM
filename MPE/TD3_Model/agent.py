import torch
import copy
import torch.nn.functional as F
from MPE.TD3_Model.networks import Actor, Critic
from MPE.TD3_Model.memory import ReplayBuffer
from infoAlg import GaussDistribution
import os


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, high, low, cfg, agent_idx,
                 chkpt_dir='/MPE/MASB/MASB/env/'):
        self.name = 'TD3_agent%s' % agent_idx
        self.n_actions = cfg.n_actions
        self.path = os.getcwd() + chkpt_dir
        self.max_action = max_action
        self.gamma = cfg.gamma
        self.lr = cfg.lr
        self.policy_noise = cfg.policy_noise
        self.noise_clip = cfg.noise_clip
        self.policy_freq = cfg.policy_freq
        self.batch_size = cfg.batch_size
        self.device = cfg.device
        self.tau = cfg.tau
        self.total_it = 0
        self.high = high[agent_idx]
        self.low = low[agent_idx]

        self.actor = Actor(state_dim, action_dim, cfg.layer1, cfg.layer2, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim, cfg.layer1, cfg.layer2).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = ReplayBuffer(state_dim, action_dim)
        self.reAction = GaussDistribution(high=self.high, low=self.low)

    def remember(self, state, action, next_state, reward, done):
        self.memory.push(state, action, next_state, reward, done)

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state)
        action = action.detach().numpy()[0]
        actions = self.reAction.get_action(action)
        # actions = action.cpu().data.numpy().flatten()
        return actions

    def learn(self):
        if not self.memory.ready():
            return
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self):
        torch.save(self.critic.state_dict(), self.path + "\TD3_critic")
        torch.save(self.critic_optimizer.state_dict(), self.path + "\TD3_critic_optimizer")

        torch.save(self.actor.state_dict(), self.path + '\TD3_actor')
        torch.save(self.actor_optimizer.state_dict(), self.path + "\TD3_actor_optimizer")

    def load(self):
        self.critic.load_state_dict(torch.load(self.path + "\TD3_critic"))
        self.critic_optimizer.load_state_dict(torch.load(self.path + "\TD3_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.path + "\TD3_actor"))
        self.actor_optimizer.load_state_dict(torch.load(self.path + "\TD3_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)