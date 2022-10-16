import gym
from gym import spaces
import numpy as np
from MPE.MASB.MASB.multi_discrete import MultiDiscrete

class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None, reward_callback2 = None,
                 observation_callback=None, observation2_callback=None, info_callback=None,
                 done_callback=None):

        self.world = world
        self.world_size = len(self.world.graph.nodes())
        self.agents = self.world.policy_agents
        self.landmarks = self.world.seeds_landmarks
        self.num_agents = len(world.policy_agents)
        self.num_landmarks = len(world.seeds_landmarks)
        self.reset_callback = reset_callback

        self.reward_callback = reward_callback
        self.reward_callback2 = reward_callback2

        self.observation_callback = observation_callback
        self.observation2_callback = observation2_callback

        self.info_callback = info_callback
        self.done_callback = done_callback
        self.discrete_action_space = False
        self.force_discrete_action = True
        self.time = 0

        self.action_space = []
        self.observation_space = []
        self.init_cost = []
        self.success_price = []
        self.trans_probability = {}
        for i in range(self.num_landmarks):
            name = 'seed_%s' % i
            self.trans_probability[name] = []

        for agent in self.agents:
            total_action_space = []
            if not self.discrete_action_space:
                agent_action_space = spaces.Box(low=0., high=agent.budget_left, shape=(int(self.num_landmarks),), dtype=np.float32)
                total_action_space.append(agent_action_space)
            if len(total_action_space) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            obs_dim = len(observation_callback(agent, self.world, first=True))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
        for landmark in self.landmarks:
            self.init_cost.append(landmark.initial_cost)
            self.success_price.append(landmark.success_bid_price)

    def step(self, action_n):
        isTest = False
        self.agents = self.world.policy_agents
        self.landmarks = self.world.seeds_landmarks
        if not isTest:
            for i, landmark in enumerate(self.landmarks):
                landmark.initial_cost = self.init_cost[i]
                if landmark.is_success:
                    self.success_price[i] = landmark.success_bid_price
                else:
                    self.success_price[i] = landmark.initial_cost
            self.bid(self.world, action_n)
        else:
            self.contribution(self.world)
        self._modify_cost(self.world)
        return self.infoRecord(is_test=False, lan_i=None)

    def step_test(self, actions, world, landmark, landmark_i):
        start_bid_price = self.seedsInitPrice(world, landmark_i)
        landmark.is_involve = True
        landmark.success_bid_price, results_agent_price = self.batchBid(world, landmark,landmark_i, start_bid_price[0], actions)
        if landmark.is_success:
            self.success_price[landmark_i] = landmark.success_bid_price
        else:
            self.success_price[landmark_i] = start_bid_price[0]
        for i, agent in enumerate(world.agents):
            agent.bid_price[landmark_i] = results_agent_price[i]
            agent.state.cost = agent.budget - agent.budget_left
            agent.state.revenue = agent.reward
            agent.isBid = True
        return self.infoRecord(is_test=True, lan_i=landmark_i)

    def infoRecord(self, is_test, lan_i):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        for agent in self.agents:
            agent.reward = self._get_reward(agent)
            if is_test:
                obs_n.append(self._get_obs2(agent, self.world, lan_i))
            else:
                obs_n.append(self._get_obs(agent, self.world, first=False))
            reward_n.append(agent.reward)
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self.reset_callback(self.world)
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent, self.world, first=True))
        return obs_n

    def trainReset(self):
        self.reset_callback(self.world)
        for i, landmark in enumerate(self.landmarks):
            landmark.initial_cost = self.init_cost[i]

    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    def _get_obs(self, agent, world, first):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, world, first)

    def _get_obs2(self, agent, world, landmark_i):
        if self.observation2_callback is None:
            return np.zeros(0)
        return self.observation2_callback(agent, world, landmark_i)

    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    def _get_reward2(self, seeds, compSeeds, candSeeds):
        if self.reward_callback2 is None:
            return 0.0
        return self.reward_callback2(seeds, compSeeds, candSeeds)

    def _set_action(self, action, agent, lan_i, is_rand=False):
        action_new = np.array(action, dtype=float)
        if not agent.isBid:
            return 0
        if is_rand:
            index = np.arange(len(action_new))
            rnd_index = np.random.choice(index, 1)
            res = action_new[rnd_index]
        else:
            res = action_new[lan_i]
        if res > agent.budget_left:
            return agent.budget_left
        return res

    def _modify_cost(self, world):
        range = 0.3
        for i, landmark in enumerate(world.landmarks):
            if self.success_price[i] == 0:
                b_s = 0
            else:
                b_s = landmark.success_bid_price / self.success_price[i]
            if sum(self.trans_probability[landmark.name]) == 0:
                P_E = 0.2
            else:
                P_E = landmark.spread_value / np.mean(self.trans_probability[landmark.name])

            if b_s >= 1:
                if P_E >= 1.0:
                    landmark.initial_cost = landmark.initial_cost * (min(1 + range, P_E))
            else:
                if P_E < 1.0:
                    landmark.initial_cost -= range * landmark.initial_cost
            self.init_cost[i] = float(round(landmark.initial_cost, 3))

    def contribution(self, world):
        compSeeds = []
        candseeds = []
        for landmark in world.landmarks:
            if landmark.is_success:
                compSeeds.append(landmark.data_index)
            candseeds.append(landmark.data_index)
        for landmark in world.landmarks:
            seeds = []
            if landmark.is_success:
                seeds.append(landmark.data_index)
                compSeeds = list(set(compSeeds).difference(seeds))
                landmark.spread_value = self._get_reward2(seeds, compSeeds, candseeds)
                landmark.bratio = landmark.spread_value*1.0 / self.world_size
            if not landmark.is_success:
                landmark.spread_value = 0.0
            self.trans_probability[landmark.name].append(landmark.spread_value)

    def bid(self, world, action_n):
        for i, landmark in enumerate(world.landmarks):
            start_bid_price = self.seedsInitPrice(world, i)
            landmark.is_involve = True
            landmark.success_bid_price, results_agent_price = self.batchBid(world, landmark, i, start_bid_price[0], action_n)
            for j, agent in enumerate(world.agents):
                agent.bid_price[i] = results_agent_price[j]
                agent.state.cost = agent.budget - agent.budget_left
                agent.isBid = True
        self.contribution(world)

    def batchBid(self, world, landmark, lan_i, start_bid_price, action_n):
        all_Agent_Price = []
        result_Agent_Price = []
        for i, agent in enumerate(self.agents):
            if agent.budget_left < start_bid_price:
                agent.isBid = False
            agent.bid_action = self._set_action(action_n[i], agent, lan_i)
            result_Agent_Price.append(float(agent.bid_action))
        for i, agent in enumerate(world.agents):
            if agent.bid_action < start_bid_price or not agent.isBid:
                all_Agent_Price.append(0)
            else:
                all_Agent_Price.append(float(agent.bid_action))
        max_price = max(all_Agent_Price)
        if max_price == 0:
            return 0, result_Agent_Price
        landmark.is_success = True
        if all_Agent_Price.count(max_price) > 1:
            max_index = self.maxValueIndex(all_Agent_Price, max_price)
            agent_reward = {}
            for item in max_index:
                for i, agent in enumerate(world.agents):
                    if item == i:
                        agent.seeds.append(landmark.data_index)
                        agent_reward[item] = self._get_reward(agent)
            max_index_ = max(agent_reward, key=lambda k: agent_reward[k])
            for i, agent in enumerate(world.agents):
                if landmark.data_index in agent.seeds:
                    if i != max_index_:
                        agent.seeds.remove(landmark.data_index)
                    elif i == max_index_:
                        all_Agent_Price.remove(max_price)
                        second_Price = max(all_Agent_Price)
                        agent.budget_left -= second_Price
        else:
            max_index = all_Agent_Price.index(max_price)
            all_Agent_Price.remove(max_price)
            second_Price = max(all_Agent_Price)
            if second_Price == 0: second_Price = start_bid_price
            for i, agent in enumerate(world.agents):
                if i == max_index:
                    agent.budget_left -= second_Price
                    agent.seeds.append(landmark.data_index)
                    # agent.reward = self._get_reward(agent)
                    break
        return float(second_Price), result_Agent_Price

    def seedsInitPrice(self, world, i):
        return [landmark.initial_cost for landmark in world.landmarks if landmark.name == 'seed_' + str(i)]

    def maxValueIndex(self, all_Agent_Price, max_price):
        flag = 0
        lis = []
        for i in range(all_Agent_Price.count(max_price)):
            sec = flag
            flag = all_Agent_Price[flag:].index(max_price)
            lis.append(flag + sec)
            flag = lis[-1:][0] + 1
        return lis

