import networkx as nx
import numpy as np
from MPE.MASB.MASB.core import World, Agent, Landmark
from MPE.MASB.MASB.scenario import BaseScenario
from StageOne.reward import rewardGet, spreardGet
from StageOne.utils import parse_graph_txt_file
import random


class SimBid(BaseScenario):
    def __init__(self, resCfg):
        cfg = resCfg
        self.item = cfg.item       # index of dataset
        self.dataset_item = cfg.dataset_item       # index of seed set
        self.num_agents = cfg.num_agents     # number of the competitors
        self.num_landmarks = cfg.num_seeds      # number of the seeds
        self.graph = cfg.datasets[self.item]
        self.G = parse_graph_txt_file(self.graph)
        self.dataset_name = {
            0: 'cit',
            1: 'dblp',
            2: 'facebook',
            3: 'p2p',
            4: 'BA'
        }
        self.candicateSeeds = cfg.seeds[self.dataset_name[self.item]][self.dataset_item]
        self.cost = cfg.cost
        self.agentBudget = cfg.budget
        self.rewardRecord = dict()
        self.is_undirect = False if self.item == 0 or self.item == 3 else True

    def make_world(self):
        world = World()
        world.num_agents = self.num_agents
        world.num_landmarks = self.num_landmarks
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.movable = False
            agent.silent = True
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'seed_%d' % i
            landmark.movable = False
        world.graph = parse_graph_txt_file(self.graph)
        self.reset_world(world)
        # self.rewardRecord = rewardGet(self.dataset_name[self.item], self.num_landmarks)
        return world

    # reward return
    def reward(self, agent, world):
        seeds = agent.seeds
        if len(seeds) == 0:
            return 0.0
        compSeeds = []
        candSeeds = self.candicateSeeds
        for other_agent in world.agents:
            if other_agent != agent:
                compSeeds.extend(other_agent.seeds)
        get_reward = self.reward_match(seeds, compSeeds, candSeeds)
        return get_reward

    def reward_match(self, seeds, compSeeds, candSeeds):
        if self.is_undirect:
            if self.item == 1:
                rand = random.uniform(-10, 10)
            else:
                rand = random.uniform(-20, 20)
            reward = rand + self.undirectGraphReward(seeds, compSeeds, candSeeds)
            if reward < 0:
                reward = self.undirectGraphReward(seeds, compSeeds, candSeeds)
        else:
            if self.item == 0:
                rand = random.uniform(-10, 10)
            else:
                rand = random.uniform(-5, 5)
            reward = rand + self.directGraphReward(seeds, compSeeds, candSeeds)
            if reward < 0:
                reward = self.directGraphReward(seeds, compSeeds, candSeeds)
        return reward
        # state = np.zeros(len(candSeeds), dtype=int)
        # for item in seeds:
        #     index_ = candSeeds.index(item)
        #     state[index_] = 1
        # for item in compSeeds:
        #     index_ = candSeeds.index(item)
        #     state[index_] = 2

        # state = ''.join([str(item) for item in state])
        # return self.rewardRecord[state]

        # if state in self.rewardRecord:
        #     return self.rewardRecord[state]
        # else:
        #     self.rewardRecord[state] = spreardGet(parse_graph_txt_file(self.graph), seeds, compSeeds, candSeeds)
        #     return self.rewardRecord[state]

    def directGraphReward(self, seeds, compSeeds, candSeeds):
        neighbors_seeds = []
        for v in seeds:
            neighbor_u = []
            for u in self.G.predecessors(v):
                if u in candSeeds or u in compSeeds:
                    continue
                neighbor_u.append(u)
            neighbors_seeds.extend(neighbor_u)

        neighbors_compseeds = []
        for v in compSeeds:
            neighbor_u = []
            for u in self.G.predecessors(v):
                if u in seeds or u in candSeeds:
                    continue
                neighbor_u.append(u)
            neighbors_compseeds.extend(neighbor_u)

        same = list(set(neighbors_seeds) & set(neighbors_compseeds))
        for i in same:
            neighbors_seeds.remove(i)
        return len(neighbors_seeds)

    def undirectGraphReward(self, seeds, compSeeds, candSeeds):
        neighbors_seeds = []
        for v in seeds:
            neighbor_u = []
            for u in self.G.neighbors(v):
                if u in candSeeds or u in compSeeds:
                    continue
                neighbor_u.append(u)
            neighbors_seeds.extend(neighbor_u)

        neighbors_compseeds = []
        for v in compSeeds:
            neighbor_u = []
            for u in self.G.neighbors(v):
                if u in seeds or u in candSeeds:
                    continue
                neighbor_u.append(u)
            neighbors_compseeds.extend(neighbor_u)

        same = list(set(neighbors_seeds) & set(neighbors_compseeds))
        for i in same:
            neighbors_seeds.remove(i)
        return len(neighbors_seeds)


    def reset_world(self, world):
        for i, agent in enumerate(world.agents):
            agent.budget = self.agentBudget[i]  # 设置智能体初始预算
            agent.budget_left = agent.budget
            agent.seeds = []
            agent.bid_action = 0.0
            agent.isBid = True
            agent.reward = 0
            agent.bid_price = []
            for i in range(world.num_landmarks):
                agent.bid_price.append(0.0)
            agent.state.cost = 0.0
            agent.state.revenue = 0.0
        for i, landmark in enumerate(world.landmarks):
            landmark.initial_cost = self.cost[i]
            landmark.data_index = self.candicateSeeds[i]
            landmark.success_bid_price = 0.0
            landmark.is_involve = False
            landmark.spread_value = 0.0
            landmark.bratio = 0.0
            landmark.is_success = False

    def observation(self, agent, world, first):
        seeds_info = []
        agent_info = []
        for i, landmark in enumerate(world.landmarks):
            # seeds_info.append(float(landmark.initial_cost))
            if first:
                seeds_info.append(0.0)
            else:
                seeds_info.append(float(agent.bid_price[i]))
            # seeds_info.append(float(landmark.success_bid_price))
        agent_info.append(agent.budget_left)
        return np.concatenate([seeds_info]+[agent_info])

    def observation2(self, agent, world, landmark_i):
        seed_info = []
        agent_info = []
        for i, landmark in enumerate(world.landmarks):
            if i <= landmark_i:
                seed_info.append(float(landmark.initial_cost))
                seed_info.append(landmark.success_bid_price)
                seed_info.append(float(agent.bid_price[i]))
            else:
                seed_info.append(float(landmark.initial_cost))
                seed_info.append(0.0)
                seed_info.append(0.0)
        agent_info.append(agent.budget_left)
        agent_info.append(agent.reward)
        return np.concatenate([seed_info]+[agent_info])

    def benchmark_data(self, agent, world):
        agent.state.cost = agent.budget-agent.budget_left
        agent.state.revenue = agent.reward
        return agent.state



