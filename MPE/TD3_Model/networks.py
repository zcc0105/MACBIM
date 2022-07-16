import torch.nn as nn
import torch.nn.functional as F
import torch


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, layer1, layer2, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, layer1)
        self.l2 = nn.Linear(layer1, layer2)
        self.l3 = nn.Linear(layer2, action_dim)

        self.max_action = torch.FloatTensor([max_action])

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        res = self.max_action * torch.tanh(self.l3(a))
        return res


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, layer1, layer2):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, layer1)
        self.l2 = nn.Linear(layer1, layer2)
        self.l3 = nn.Linear(layer2, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, layer1)
        self.l5 = nn.Linear(layer1, layer2)
        self.l6 = nn.Linear(layer2, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

