import numpy as np


class AgentState(object):
    def __init__(self):
        super(AgentState, self).__init__()
        self.cost = 0.0
        self.revenue = 0.0


class Entity(object):
    def __init__(self):
        self.name = ''
        self.movable = False


class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.initial_cost = 0.4
        self.is_involve = False
        self.is_success = False
        self.data_index = 0
        self.success_bid_price = 0.0
        self.spread_value = 0.0
        self.bratio = 0.0


class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        self.state = AgentState()
        self.budget = 0.0
        self.budget_left = 0.0
        self.seeds = []
        self.isBid = False
        self.reward = 0
        self.bid_price = []
        self.bid_action = 0.0


class World(object):
    def __init__(self):
        self.agents = []
        self.landmarks = []
        self.platforms = []
        self.dt = 0.1

        self.num_agents = 0
        self.num_landmarks = 0
        self.num_platforms = 0
        self.graph = None


    @property
    def entities(self):
        return self.agents + self.landmarks

    @property
    def policy_agents(self):
        return [agent for agent in self.agents]

    @property
    def seeds_landmarks(self):
        return [landmark for landmark in self.landmarks]




