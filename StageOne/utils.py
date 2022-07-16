#coding:UTF-8
import random

import networkx as nx
import numpy as np
import time


def parse_graph_txt_file(fname):
    edge_list = []
    if 'facebook' in fname:
        separator = ' '
    else:
        separator = '\t'
    with open(fname, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            edge_list.append(list(map(int, line.strip().split(separator))))
    if 'facebook' in fname or 'dblp' in fname or 'BA' in fname:
        G = graphLoad(edge_list)
    else:
        G = digraphLoad(edge_list)
    return G


def graphLoad(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    for u in G.nodes():
        for v in G.neighbors(u):
            if v != u:
                pro = 1.0 / G.degree(u)
                G.add_edge(v, u, weight=pro)
    return G


def digraphLoad(edge_list):
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    for u in G.nodes():
        for v in G.predecessors(u):
            if v != u:
                if G.in_degree(u) != 0:
                    probability = 1.0 / G.in_degree(u)
                else:
                    probability = 0.0
                G.add_edge(v, u, weight=probability)
    return G


def rewardLimit(reward, budget):
    fairThres = 0.35
    avgB = np.mean(budget)
    avgInf = np.mean(reward)
    avgRate = avgInf / avgB
    for i in range(len(reward)):
        iRate = reward[i] / budget[i]
        if 1 + fairThres >= iRate / avgRate >= 1 - fairThres:
            continue
        else:
            return False
    return True


def budgetGet(env):
    budget = []
    for agent in env.agents:
        budget.append(agent.budget)
    return budget


def initialCostGet(env):
    seeds = {}
    for i in range(env.num_landmarks):
        seeds[i] = []
    for k, landmark in enumerate(env.landmarks):
        name = 'seed_%s' % k
        if landmark.name == name:
            seeds[k].append(landmark.initial_cost)
    return seeds


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


def budgetLeftGet(env):
    data = []
    for agent in env.agents:
        data.append(agent.budget_left)
    return data


def fairCal(reward, budget):
    avg_reward = sum(reward) / sum(budget)
    if avg_reward == 0:
        return np.zeros(len(reward))
    resRate = []
    for i in range(len(budget)):
        res = reward[i] / budget[i]
        res = round(res / avg_reward, 3)
        if res!= 0:
            resRate.append(res)
        else:
            resRate.append(0.0)
    return resRate


def seedsRewardGet(env):
    RA1 = []
    RA2 = []
    for agent in env.agents:
        if agent.name == 'agent_1':
            RA1.append(agent.seeds)
            RA1.append(agent.reward)
        elif agent.name == 'agent_2':
            RA2.append(agent.seeds)
            RA2.append(agent.reward)
    return RA1, RA2



