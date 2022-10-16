import torch as T
from MPE.MADDPG_Model.networks import ActorNetwork, CriticNetwork
from infoAlg import GaussDistribution

class Agent:
    def __init__(self, actor_dims, critic_dims, n_agents, agent_idx, cfg, high, low):
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.n_actions = cfg.n_actions
        self.agent_name = 'maddpg_A_%s' % agent_idx
        self.high = high[agent_idx]
        self.low = low[agent_idx]
        self.device = cfg.device
        self.agent_idx = agent_idx
        self.actor = ActorNetwork(cfg.actor_lr, actor_dims, cfg.layer1, cfg.layer2, cfg.n_actions,
                                  name=self.agent_name+'_actor')
        self.critic = CriticNetwork(cfg.critic_lr, critic_dims,
                            cfg.layer1, cfg.layer2, n_agents, cfg.n_actions,
                            name=self.agent_name+'_critic')
        self.target_actor = ActorNetwork(cfg.actor_lr, actor_dims, cfg.layer1, cfg.layer2, cfg.n_actions,
                                        name=self.agent_name+'_target_actor')
        self.target_critic = CriticNetwork(cfg.critic_lr, critic_dims,
                                            cfg.layer1, cfg.layer2, n_agents, cfg.n_actions,
                                            name=self.agent_name+'_target_critic')
        self.update_network_parameters(tau=1)
        self.reAction = GaussDistribution(high=self.high, low=self.low)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        action = self.actor.forward(state)
        actions = action.detach().cpu().numpy()[0]
        actions = self.reAction.get_action(actions)
        return actions

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                    (1-tau)*target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


