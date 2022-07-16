import torch
import torch as T
import torch.nn.functional as F
from MADDPG_Model.agent import Agent


class MADDPG_master:
    def __init__(self, actor_dims, critic_dims, n_agents, cfg, env):
        self.agents = []
        self.n_agents = n_agents
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], critic_dims,
                                     n_agents, agent_idx, cfg,
                                     env.action_space[agent_idx].high, env.action_space[agent_idx].low))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            actions_ = agent.choose_action(raw_obs[agent_idx])
            actions.append(actions_)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        actor_obs, states, actions, rewards, \
        actor_new_obs, states_, dones = memory.sample_buffer()

        device = self.agents[0].actor.device
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_obs[agent_idx],
                                  dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_obs[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1).detach()
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()
            actor_loss = agent.critic.forward(states, mu).flatten()
            actorloss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actorloss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            x = target.to(torch.float32)
            y = critic_value.to(torch.float32)
            critic_loss = F.mse_loss(x, y)
            agent.critic.optimizer.zero_grad()
            criticloss = critic_loss.to(torch.float32)
            criticloss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            agent.update_network_parameters()
