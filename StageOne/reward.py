import copy
import itertools

import networkx as nx
import numpy as np
import time
from StageOne import utils


def loadGraph(G, seeds, compSeeds, candSeeds):
    G.graph['free'] = []
    G.graph['1'] = list(seeds)
    G.graph['2'] = list(compSeeds)
    G.graph['3'] = list(candSeeds)
    for node in G.nodes():
        if node not in G.graph['1'] and node not in G.graph['2'] and node not in G.graph['3']:
            G.graph['free'].append(node)
    return


def activate(G, node, player):
    if node in G.graph['free']:
        G.graph['free'].remove(node)
    G.graph[str(player)].append(node)
    return


def weight(G, u, v):
    if G.has_edge(u, v):
        return G[u][v]['weight']
    else:
        return 0


def getActivate(G, u, active1, active2):
    if active1 > active2:
        activate(G, u, 1)
        return 1
    elif active1 < active2:
        activate(G, u, 2)
        return 2
    else:
        k = np.random.randint(1, 3)
        activate(G, u, k)
        return k


def compDiffuse(G, seeds, compSeeds):
    nodeThreshold = {}
    for node in G.nodes():
        nodeThreshold[node] = np.random.uniform(0, 1)

    nodeValues1, nodeValues2 = {}, {}
    for node in G.nodes():
        nodeValues1[node] = 0
        nodeValues2[node] = 0

    newActive1 = True
    newActive2 = True

    currentActiveSeeds = copy.deepcopy(seeds)
    currentActiveCompSeeds = copy.deepcopy(compSeeds)

    newActiveSeeds = set()
    newActiveCompSeeds = set()

    activeNodes = list(copy.deepcopy(seeds))
    activeNodesComSeeds = list(copy.deepcopy(compSeeds))
    if len(activeNodesComSeeds) == 0:
        newActive2 = False
    activeNodes.extend(activeNodesComSeeds)
    while newActive1 or newActive2:
        judgeUnionSeeds = set()
        judgeUnionCompSeeds = set()
        for node in currentActiveSeeds:
            for neighbor in G.neighbors(node):
                judgeUnionSeeds.add(neighbor)
        for node in currentActiveCompSeeds:
            for neighbor in G.neighbors(node):
                judgeUnionCompSeeds.add(neighbor)
        unionNode = judgeUnionSeeds & judgeUnionCompSeeds
        if np.random.randint(1, 3) == 1:
            for node in currentActiveSeeds:
                for neighbor in G.neighbors(node):
                    if neighbor not in activeNodes:
                        nodeValues1[neighbor] += weight(G, node, neighbor)
                        if nodeValues1[neighbor] >= nodeThreshold[neighbor]:
                            if neighbor not in unionNode:
                                activate(G, neighbor, 1)
                                newActiveSeeds.add(neighbor)
                                activeNodes.append(neighbor)
                            else:
                                k = getActivate(G, neighbor, nodeValues1[neighbor], nodeValues2[neighbor])
                                if k == 1:
                                    newActiveSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
                                elif k == 2:
                                    newActiveCompSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
            for node in currentActiveCompSeeds:
                for neighbor in G.neighbors(node):
                    if neighbor not in activeNodes:
                        nodeValues2[neighbor] += weight(G, node, neighbor)
                        if nodeValues2[neighbor] >= nodeThreshold[neighbor]:
                            if neighbor not in unionNode:
                                activate(G, neighbor, 2)
                                newActiveCompSeeds.add(neighbor)
                                activeNodes.append(neighbor)
                            else:
                                k = getActivate(G, neighbor, nodeValues1[neighbor], nodeValues2[neighbor])
                                if k == 2:
                                    newActiveCompSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
                                elif k == 1:
                                    newActiveSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
        else:
            for node in currentActiveCompSeeds:
                for neighbor in G.neighbors(node):
                    if neighbor not in activeNodes:
                        nodeValues2[neighbor] = nodeValues2[neighbor] + weight(G, node, neighbor)
                        if nodeValues2[neighbor] >= nodeThreshold[neighbor]:
                            if neighbor not in unionNode:
                                activate(G, neighbor, 2)
                                newActiveCompSeeds.add(neighbor)
                                activeNodes.append(neighbor)
                            else:
                                k = getActivate(G, neighbor, nodeValues1[neighbor], nodeValues2[neighbor])
                                if k == 2:
                                    newActiveCompSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
                                elif k == 1:
                                    newActiveSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
            for node in currentActiveSeeds:
                for neighbor in G.neighbors(node):
                    if neighbor not in activeNodes:
                        nodeValues1[neighbor] = nodeValues1[neighbor] + weight(G, node, neighbor)
                        if nodeValues1[neighbor] >= nodeThreshold[neighbor]:
                            if neighbor not in unionNode:
                                activate(G, neighbor, 1)
                                newActiveSeeds.add(neighbor)
                                activeNodes.append(neighbor)
                            else:
                                k = getActivate(G, neighbor, nodeValues1[neighbor], nodeValues2[neighbor])
                                if k == 1:
                                    newActiveSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
                                elif k == 2:
                                    newActiveCompSeeds.add(neighbor)
                                    unionNode.remove(neighbor)
                                    activeNodes.append(neighbor)
        activeNodes = list(set(activeNodes).difference(G.graph['3']))
        newActiveSeeds = set(newActiveSeeds).difference(G.graph['3'])
        newActiveCompSeeds = set(newActiveCompSeeds).difference(G.graph['3'])

        if newActiveSeeds:
            currentActiveSeeds = list(newActiveSeeds)
            newActiveSeeds = set()
        else:
            newActive1 = False
        if newActiveCompSeeds:
            currentActiveCompSeeds = list(newActiveCompSeeds)
            newActiveCompSeeds = set()
        else:
            newActive2 = False
    return


def get_party(item, canInit):
    seeds = []
    candseeds = []
    seeds_num = len(canInit)
    for i in range(seeds_num):
        if item[i] == 1:
            seeds.append(canInit[i])
        elif item[i] == 0:
            candseeds.append(canInit[i])
    compseeds = list(set(canInit).difference(set(seeds).union(candseeds)))
    return seeds, compseeds, candseeds


# 用CLT算法得到影响力，然后放在文件里面存储起来
def rewardStore(G, candInit, dataset_name):
    seeds_num = len(candInit)
    res_record = {}
    state_dict = np.arange(0, 3, 1)
    res = list(itertools.product(state_dict, repeat=seeds_num))
    filename = './DatasetRes/' + str(dataset_name) + '_' + str(seeds_num) + '.txt'
    for item in res:
        seeds, compseeds, candseeds = get_party(item, candInit)
        res_record[str(item)] = spreardGet(G, seeds, compseeds, candseeds)
    with open(filename, 'w') as f:
        for item in res_record:
            f.write(str(item) + '\t' + str(res_record[item]) + '\n')
        f.close()


#读取文件里面存储的CLT模型计算出来的影响力
def rewardGet(dataset_name, seeds_num):
    state_store = []
    reward_store = []
    filename = './DatasetRes/' + str(dataset_name) + '_' + str(seeds_num) + '.txt'
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            state_, reward_ = line.split('\t')
            state_ = ''.join(str(state_[1:len(state_) - 1]).split(', '))
            state_store.append(state_)
            reward_store.append(float(reward_))
        rewardRecord = dict(zip(state_store, reward_store))
        f.close()
    return rewardRecord


def string_to_float(str):
    return float(str)


def spreardGet(G, seeds, compSeeds, candSeeds):
    iterations = 4
    spreadSeeds = np.zeros(iterations)
    spreadCompSeeds = np.zeros(iterations)

    startTime = time.time()
    for i in range(iterations):
        loadGraph(G, seeds, compSeeds, candSeeds)
        compDiffuse(G, seeds, compSeeds)
        spread1 = len(G.graph['1'])
        spread2 = len(G.graph['2'])
        spreadSeeds[i] = spread1
        spreadCompSeeds[i] = spread2

    endTime = time.time()

    return np.average(spreadSeeds)


def reward_test():
    start_time = time.time()
    G = utils.parse_graph_txt_file('../Dataset/facebook_combined.txt')
    candInit = [[107, 1684, 1912, 3437, 0],
                [107, 1684, 1912, 3437, 0, 2543, 2347],
                [107, 1684, 1912, 3437, 0, 2543, 2347, 1888, 1800]]
    seeds = [107, 1684, 1912]
    candSeeds = [3437, 0, 2543]
    compSeeds = [2347]
    neighbors_seeds = []
    for v in seeds:
        neighbor_u = []
        for u in G.neighbors(v):
            # 不包含candSeeds跟compSeeds
            if u in candSeeds or u in compSeeds:
                continue
            neighbor_u.append(u)
        neighbors_seeds.extend(neighbor_u)

    neighbors_compseeds = []
    for v in compSeeds:
        neighbor_u = []
        for u in G.neighbors(v):
            # 不包含candSeeds跟compSeeds
            if u in seeds or u in candSeeds:
                continue
            neighbor_u.append(u)
        neighbors_compseeds.extend(neighbor_u)

    same = list(set(neighbors_seeds) & set(neighbors_compseeds))
    for i in same:
        neighbors_seeds.remove(i)
    print(len(neighbors_seeds))

    # dataset_name = ['facebook']
    # num_agents = 2
    # for item in candInit:
    #     rewardStore(G, item, dataset_name, num_agents)
    # end_time = time.time()
    # print('Running time:{:.5f}'.format(end_time - start_time))


if __name__ == "__main__":
    reward_test()