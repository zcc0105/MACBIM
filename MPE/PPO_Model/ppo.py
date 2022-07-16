from PPO_Model.agent import Agent


class PPO:
    def __init__(self, state_dim, cfg, env, n_agents):
        self.device = cfg.device
        self.agents = []
        self.n_agents = n_agents
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(state_dim[agent_idx], cfg.action_bound[agent_idx], cfg, agent_idx,
                                     env.action_space[agent_idx].high, env.action_space[agent_idx].low))

    def choose_action(self, observation):
        action = []
        actions = []
        probs = []
        values = []
        for agent_idx, agent in enumerate(self.agents):
            action_, actions_, probs_, value_ = agent.choose_action(observation[agent_idx])
            action.append(action_)
            actions.append(actions_)
            probs.append(probs_)
            values.append(value_)
        return action, actions, probs, values

    def remember(self, state, actions, probs, vals, reward, done):
        for agent_idx, agent in enumerate(self.agents):
            agent.remember(state[agent_idx], actions[agent_idx], probs[agent_idx],
                           vals[agent_idx], reward[agent_idx], done[agent_idx])

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def save(self):
        print('...saving checkpoint...')
        for agent in self.agents:
            agent.save()

    def load(self):
        print('...loading checkpoint...')
        for agent in self.agents:
            agent.load()