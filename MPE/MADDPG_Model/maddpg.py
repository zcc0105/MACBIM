import torch
import torch as T
import torch.nn.functional as F
from MPE.MADDPG_Model.agent import Agent


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
        statesTensor = T.tensor(states, dtype=T.float).to(device)
        actionsTensor = T.tensor(actions, dtype=T.float).to(device)
        rewardsTensor = T.tensor(rewards).to(device)
        statesTensor_ = T.tensor(states_, dtype=T.float).to(device)
        donesTensor = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            mu_states = T.tensor(actor_obs[agent_idx],
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)

            new_states = T.tensor(actor_new_obs[agent_idx],
                                  dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            old_agents_actions.append(actionsTensor[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            # print("start learning")
            actor_loss = agent.critic.forward(statesTensor, mu)
            newActor_loss = actor_loss.flatten()
            actorloss = -T.mean(newActor_loss)
            agent.actor.optimizer.zero_grad()
            actorloss.backward(retain_graph=True)
            agent.actor.optimizer.step()

            critic_value_ = agent.target_critic.forward(statesTensor_, new_actions)
            newCritic_value_ = critic_value_.flatten()
            newCritic_value_.clone()[donesTensor[:, 0]] = 0.0
            target = rewardsTensor[:, agent_idx] + agent.gamma * newCritic_value_

            critic_value = agent.critic.forward(statesTensor, old_actions)
            newCritic_value = critic_value.flatten()
            # x = target.to(torch.float32)
            # y = critic_value.to(torch.float32)
            criticloss = F.mse_loss(target, newCritic_value)
            agent.critic.optimizer.zero_grad()
            # criticloss = critic_loss.to(torch.float32)
            criticloss.backward(retain_graph=True)
            agent.critic.optimizer.step()


            agent.update_network_parameters()
