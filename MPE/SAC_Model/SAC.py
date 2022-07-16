from SAC_Model.agent import Agent


class SAC:
    def __init__(self, actor_dims, n_agents, cfg, env):
        self.agents = []
        self.n_agents = n_agents
        self.action_dim = cfg.n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(actor_dims[agent_idx], self.action_dim,
                                     env.action_space[agent_idx].high, env.action_space[agent_idx].low,
                                     cfg, agent_idx))

    def remember(self, state, action, next_state, reward, done):
        for agent_idx, agent in enumerate(self.agents):
            agent.remember(state[agent_idx], action[agent_idx], next_state[agent_idx], reward[agent_idx], done[agent_idx])

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            actions_ = agent.choose_action(raw_obs[agent_idx])
            actions.append(actions_)
        return actions

    def learn(self):
        for agent in self.agents:
            agent.learn()

