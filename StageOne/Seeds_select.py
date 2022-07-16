#coding:UTF-8
from StageOne import utils
import copy
import numpy as np

def LT(G, S, mc=10):
    spread = []
    for _ in range(mc):
        nodeThreshold = {}
        for node in G.nodes():
            nodeThreshold[node] = np.random.uniform(0, 1)

        nodeValues = {}
        for node in G.nodes():
            nodeValues[node] = 0
        newActive = True

        currentActiveNodes = copy.deepcopy(S)
        newActiveNodes = set()
        activeNodes = list(copy.deepcopy(S))

        influence = len(S)
        while newActive:
            for node in currentActiveNodes:
                neighbor_fn = G.neighbors(node) if not G.is_directed() else G.predecessors(node)
                for neighbor in neighbor_fn:
                    if neighbor not in activeNodes:
                        nodeValues[neighbor] += G[neighbor][node]['weight']
                        if nodeValues[neighbor] >= nodeThreshold[neighbor]:
                            newActiveNodes.add(neighbor)
                            activeNodes.append(neighbor)
            influence += len(newActiveNodes)
            if newActiveNodes:
                currentActiveNodes = list(newActiveNodes)
                newActiveNodes = set()
            else:
                newActive = False
        spread.append(influence)
    return np.mean(spread)


def degree(graph, k):
    return [x for x, y in sorted(graph.degree, key=lambda x: x[1], reverse=True)[:k]]


def degreeDiGraph(graph, k):
    degrees = {}
    seeds = []
    for node in graph.nodes:
        degrees[node] = graph.in_degree(node)
    for _ in range(k):
        node = max(degrees, key=degrees.get)
        seeds.append(node)
        degrees.pop(node)
    return seeds


def degreeGraph(graph, k):
    degrees = {}
    seeds = []
    for node in graph.nodes:
        degrees[node] = graph.degree(node)
    for _ in range(k):
        node = max(degrees, key=degrees.get)
        seeds.append(node)
        degrees.pop(node)
    return seeds


def spreadDiGraph(graph, dataset_name):
    fname = dataset_name + '_node_spread.txt'
    with open(fname, 'r') as f:
        f.write(dataset_name + '_node_spread\n')
        for node in graph.nodes():
            res = LT(graph, [node])
            f.write(str(node)+'\t'+str(res)+'\n')
    return


def seedsSelect():
    Dataset = '../dataset/dblp_sub.txt'
    G1 = utils.parse_graph_txt_file(Dataset)
    k = [5, 7, 9]
    for item in k:
        print(degree(G1, item))


