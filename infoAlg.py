#coding:UTF-8
import random
import torch as T
import numpy as np
from scipy.stats import norm

num_seeds = 9


class SeedsInfo:
    def __init__(self) -> None:
        self.datasets = ['D:\BCIM\MABCIM\dataset\Cit-HepPh.txt',
                         'D:\BCIM\MABCIM\dataset\dblp_sub.txt',
                         'D:\BCIM\MABCIM\dataset\\facebook_combined.txt',
                         'D:\BCIM\MABCIM\dataset\p2p-Gnutella09.txt',
                         'D:\BCIM\MABCIM\dataset\BA.txt']
        self.num_agents = 2
        self.num_seeds = num_seeds
        self.item = 2
        self.dataset_item = 2
        self.budget = [5, 5]
        self.seeds = {
            'cit': [[9207217, 9609432, 9402208, 9603336, 9411366],
                    [9207217, 9609432, 9402208, 9603336, 9411366, 9304307, 9901210],
                    [9207217, 9609432, 9402208, 9603336, 9411366, 9304307, 9901210, 9912238, 9608387]],
            'dblp': [[45479, 6737, 13884, 4799, 11457],
                     [45479, 6737, 13884, 4799, 11457, 6433, 13451],
                     [45479, 6737, 13884, 4799, 11457, 6433, 13451, 33126, 3420]],
            'facebook': [[107, 1684, 1912, 3437, 0],
                         [107, 1684, 1912, 3437, 0, 2543, 2347],
                         [107, 1684, 1912, 3437, 0, 2543, 2347, 1888, 1800]],
            'p2p': [[2436, 6611, 3548, 2784, 3846],
                    [2436, 6611, 3548, 2784, 3846, 7027, 3003],
                    [2436, 6611, 3548, 2784, 3846, 7027, 3003, 3700, 6029]],
            'BA': [[239, 8, 47, 86, 170]]
        }
        self.cost = [sum(self.budget) / self.num_seeds for _ in range(self.num_seeds)]

        self.N_GAMES = 50
        self.MAX_STEPS_times = 3
        self.MAX_STEPS = 200 * self.MAX_STEPS_times
        self.p_step = 20
        self.l_step = 40
        self.n_actions = 2
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        self.rewardPlatform = []
        self.successReward = []
        self.seedsPrice = {}
        for i in range(self.num_seeds):
            self.seedsPrice[i] = []
        self.rewardAgent = {}
        self.agentBudleft = {}
        self.successSeeds = {}
        for i in range(self.num_agents):
            self.rewardAgent[i] = []
            self.agentBudleft[i] = []
            self.successSeeds[i] = []


# 高斯噪声
class GaussDistribution(object):
    def __init__(self, high, low, sigma=0.2):
        self.mu = (high-low)/[(num_seeds+1)/2]
        self.sigma = sigma
        self.high = high
        self.low = low

    def get_action(self, action):
        actions = []
        for i in action:
            if i < 0:
                actions.append(np.clip(random.gauss(0, self.sigma), self.low, self.high))
            else:
                x_ = norm.ppf(i, loc=self.mu, scale=self.sigma)
                actions.append(np.clip(x_[0] + random.gauss(0, self.sigma), self.low, self.high))
        return actions
