#coding:UTF-8
import torch.nn as nn
import torch
import torch.nn.functional as F
import os


class Critic(nn.Module):
    def __init__(self, n_obs, output_dim, layer1, layer2, name, init_w=3e-3,
                 chkpt_file='D:\BCIM\MABCIM\MPE\MASB\MASB\env'):
        super(Critic, self).__init__()
        
        self.chkpt_file = os.path.join(chkpt_file, name)
        self.linear1 = nn.Linear(n_obs + output_dim, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, output_dim)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)
        
    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))


class Actor(nn.Module):
    def __init__(self, n_obs, output_dim, layer1, layer2, name, init_w=3e-3,
                 chkpt_file='D:\BCIM\MABCIM\MPE\MASB\MASB\env'):
        super(Actor, self).__init__()
        
        self.chkpt_file = os.path.join(chkpt_file, name)
        self.linear1 = nn.Linear(n_obs, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.linear3 = nn.Linear(layer2, output_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))