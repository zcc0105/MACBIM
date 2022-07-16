import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch as T
import os


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, name,
                 chkpt_dir='D:\BCIM\MABCIM\MPE\MASB\MASB\env'):
        super(Actor, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist_ = self.actor(state)
        dist = Categorical(dist_)
        return dist, dist_

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, name,
                 chkpt_dir='D:\BCIM\MABCIM\MPE\MASB\MASB\env'):
        super(Critic, self).__init__()
        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))