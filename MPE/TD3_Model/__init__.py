from infoAlg import SeedsInfo


class TD3Config(SeedsInfo):
    def __init__(self) -> None:
        super().__init__()
        self.algo = 'TD3'
        self.env = 'simple_bidding'
        self.layer1 = 400
        self.layer2 = 300
        self.start_timestep = 25e3
        self.eval_freq = 5e3
        self.max_timestep = 200000
        self.expl_noise = 0.1
        self.batch_size = 100
        self.gamma = 0.99
        self.lr = 0.001
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
