import numpy as np
from MASB.MASB.core import World, Agent, Landmark
from MASB.MASB.scenario import BaseScenario
from StageOne.reward import rewardGet
from StageOne.utils import parse_graph_txt_file
from infoAlg import SeedsInfo


class SimBid(BaseScenario):
    def __init__(self) -> None:
        cfg = SeedsInfo()
        self.item = cfg.item       # 表示数据集的下标
        self.dataset_item = cfg.dataset_item       # 表示数据集中种子集的下标
        self.num_agents = cfg.num_agents     # 竞争者数目
        self.num_landmarks = cfg.num_seeds      # 种子的数目
        self.graph = cfg.datasets[self.item]
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

    def make_world(self):
        world = World()
        # 设置世界属性
        world.num_agents = self.num_agents
        world.num_landmarks = self.num_landmarks
        # 添加智能体
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i
            agent.movable = False  # 仅仅有出价行为，设置为不可移动
            agent.silent = True
        # 添加地标  即候选种子集
        world.landmarks = [Landmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'seed_%d' % i
            landmark.movable = False
        world.graph = parse_graph_txt_file(self.graph)
        self.reset_world(world)
        self.rewardRecord = rewardGet(self.dataset_name[self.item],
                                      self.num_landmarks)
        return world

    # 返回奖励值
    def reward(self, agent, world):
        seeds = agent.seeds
        compSeeds = []
        candSeeds = self.candicateSeeds
        for other_agent in world.agents:
            if other_agent != agent:
                compSeeds.extend(other_agent.seeds)
        # 获取candseeds与seeds、compSeeds交集的差
        get_reward = self.reward_match(seeds, compSeeds, candSeeds)
                            # self.candicateSeeds, self.dataset_name[self.item])
        return get_reward

    def reward_match(self, seeds, compSeeds, candSeeds):
        state = np.zeros(len(candSeeds), dtype=int)
        for item in seeds:
            index_ = candSeeds.index(item)
            state[index_] = 1
        for item in compSeeds:
            index_ = candSeeds.index(item)
            state[index_] = 2
        # 竞价产生的结果，用进制表示
        state = ''.join([str(item) for item in state])
        return self.rewardRecord[state]

    def reset_world(self, world):
        # 调整种子拍卖顺序
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

    # first是用来存储是否是一轮竞价
    def observation(self, agent, world, first):
        # 对于每个竞争者来说，只能观察到每个种子的起拍价以及成交价
        seeds_info = []
        agent_info = []
        # 在seeds_info中存储种子的起拍价以及成功拍卖的价格 以及agent的拍卖价格
        # 以2个竞争者，5个种子节点为例，seeds_info中应当有15个元素
        for i, landmark in enumerate(world.landmarks):
            # seeds_info.append(float(landmark.initial_cost))
            if first:
                seeds_info.append(0.0)
            else:
                seeds_info.append(float(agent.bid_price[i]))
            # seeds_info.append(float(landmark.success_bid_price))
        # 在agent_info中存储agent的花销以及奖励
        agent_info.append(agent.budget_left)
        return np.concatenate([seeds_info]+[agent_info])

    # 每一次竞价的obs返回
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



