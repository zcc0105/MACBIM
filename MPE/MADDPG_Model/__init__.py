from infoAlg import SeedsInfo


class MADDPGConfig(SeedsInfo):
    def __init__(self) -> None:
        super().__init__()
        self.env = 'simple_bidding'
        self.algo = 'MCBIM'
        self.batch_size = 1024
        self.gamma = 0.95
        self.tau = 0.01
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.replaybuffer = 100000
        self.layer1 = 64
        self.layer2 = 64
        self.S = 1
