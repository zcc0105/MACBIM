import torch as T
from infoAlg import SeedsInfo


class DDPGConfig(SeedsInfo):
    def __init__(self) -> None:
        super().__init__()
        self.env = 'simple_bidding'
        self.algo = 'DDPG'
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.replaybuffer = 100000
        self.layer1 = 400
        self.layer2 = 300
        self.S = 1
        self.n_actions = 2
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu") # check gpu