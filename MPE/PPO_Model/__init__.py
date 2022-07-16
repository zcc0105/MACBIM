from infoAlg import SeedsInfo

class PPOConfig(SeedsInfo):
    def __init__(self) -> None:
        super().__init__()
        self.env = 'simple_bidding'
        self.algo = 'PPO'
        self.eval_eps = 50
        self.batch_size = 5
        self.gamma = 0.99
        self.n_epochs = 4
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gae_lambda = 0.95
        self.policy_clip = 0.2
        self.hidden_dim = 256
        self.action_bound = self.budget
