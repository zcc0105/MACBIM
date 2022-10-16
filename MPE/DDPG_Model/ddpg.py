from MPE.DDPG_Model.agent import Agent

class DDPG(object):
    def __init__(self, cfg, env, state_dim, n_agents):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = cfg.n_actions
        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(cfg, agent_idx, env.action_space[agent_idx].high,
                                     env.action_space[agent_idx].low,
                                     state_dim[agent_idx], cfg.n_actions))

    def save_checkpoint(self):
        print('...saving checkpoint...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('...load checkpoint...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self):
        for agent in self.agents:
            agent.learn()

    def remember(self, state, actions, reward, new_states, done):
        for idx, agent in enumerate(self.agents):
            agent.remember(state[idx], actions[idx], reward[idx], new_states[idx], int(done[idx]))
