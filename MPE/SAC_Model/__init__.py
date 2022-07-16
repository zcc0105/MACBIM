from infoAlg import SeedsInfo


class SACConfig(SeedsInfo):
    def __init__(self) -> None:
        super().__init__()
        self.algo = 'SAC'
        self.env = 'simple_bidding'
        self.gamma = 0.99
        self.mean_lambda = 1e-3
        self.std_lambda = 1e-3
        self.z_lambda = 0.0
        self.soft_tau = 1e-2
        self.value_lr = 3e-4
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.hidden_dim = 256
        self.batch_size = 256
