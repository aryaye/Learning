# -*- coding: utf-8 -*-

import networkx as nx
import matplotlib.pyplot as plt
import urllib.request


def combine_set(x, y):
    for comp in set_list:
        if (x in comp) and (y in comp):
            return
        if x in comp:
            comp1 = comp
        if y in comp:
            comp2 = comp
    comp3 = comp1 | comp2
    set_list.append(comp3)
    set_list.remove(comp1)
    set_list.remove(comp2)
    return


def normalized_cut(edges, cut1, cut2):
    edges1 = 0
    edges2 = 0
    cross = 0
    for edge in edges:
        (x,y) = edge
        if (x in cut1) and (y in cut1):
            edges1 += 1
        elif (x in cut1) and (y in cut2):
            cross += 1
        elif (x in cut2) and (y in cut1):
            cross += 1
        elif (x in cut2) and (y in cut2):
            edges2 += 1
    return cross*(1.0/edges1 + 1.0/edges2)/2


def greedy_algorithms(edges, cut1, cut2):
    least = normalized_cut(edges, cut1, cut2)
    change_index = 1
    change_number = 0
    for n1 in cut1:
        cut1_m = cut1.copy()
        cut1_m.remove(n1)
        cut2_m = cut2.copy()
        cut2_m.append(n1)
        nc = normalized_cut(edges, cut1_m, cut2_m)
        if nc == least and change_number:
            if n1 < change_number:
                change_number = n1
        elif nc < least:
            change_number = n1
            least = nc
    for n2 in cut2:
        cut1_m = cut1.copy()
        cut1_m.append(n2)
        cut2_m = cut2.copy()
        cut2_m.remove(n2)
        nc = normalized_cut(edges, cut1_m, cut2_m)
        if nc == least and change_number:
            if n2 < change_number:
                change_number = n2
                change_index = 2
        if nc < least:
            change_index = 2
            change_number = n2
            least = nc
    if change_number:
        if change_index == 1:
            cut1.remove(change_number)
            cut2.append(change_number)
        if change_index == 2:
            cut1.append(change_number)
            cut2.remove(change_number)
        return greedy_algorithms(edges, cut1, cut2)
    else:
        return cut1, cut2


def modularity(edges, cut1, cut2):
    e1 = 0
    e2 = 0
    a1 = 0
    a2 = 0
    for edge in edges:
        (x, y) = edge
        if (x in cut1) and (y in cut1):
            e1 += 1
            a1 += 2
        elif (x in cut2) and (y in cut2):
            e2 += 1
            a2 += 2
        elif (x in cut1) and (y in cut2):
            a1 += 1
            a2 += 1
        elif (x in cut2) and (y in cut1):
            a1 += 1
            a2 += 1
    return (e1 + e2) / len(edges) - (a1 + a2) / (2 * len(edges))


def greedy_algorithms1(edges, cut1, cut2):
    largest = modularity(edges, cut1, cut2)
    change_index = 1
    change_number = 0
    for n1 in cut1:
        cut1_m = cut1.copy()
        cut1_m.remove(n1)
        cut2_m = cut2.copy()
        cut2_m.append(n1)
        nc = modularity(edges, cut1_m, cut2_m)
        if nc == largest and change_number:
            if n1 < change_number:
                change_number = n1
        elif nc > largest:
            change_number = n1
            largest = nc
    for n2 in cut2:
        cut1_m = cut1.copy()
        cut1_m.append(n2)
        cut2_m = cut2.copy()
        cut2_m.remove(n2)
        nc = modularity(edges, cut1_m, cut2_m)
        if nc == largest and change_number:
            if n2 < change_number:
                change_number = n2
                change_index = 2
        if nc > largest:
            change_index = 2
            change_number = n2
            least = nc
    if change_number:
        if change_index == 1:
            cut1.remove(change_number)
            cut2.append(change_number)
        if change_index == 2:
            cut1.append(change_number)
            cut2.remove(change_number)
        return greedy_algorithms1(edges, cut1, cut2)
    else:
        return cut1, cut2


if __name__ == "__main__":
    # preprocessing the data
    edges = set()
    nodes = set()
    with urllib.request.urlopen("http://jmcauley.ucsd.edu/cse255/data/facebook/egonet.txt") as f:
        data = f.read().decode('utf-8').split('\n')

    for edge in data[:len(data) - 1]:
        x = int(edge.split(' ')[0])
        y = int(edge.split(' ')[1])
        edges.add((x, y))
        edges.add((y, x))
        nodes.add(x)
        nodes.add(y)

    # draw the whole community
    G = nx.Graph()
    for e in edges:
        G.add_edge(e[0], e[1])
    nx.draw(G)
    plt.show()
    plt.clf()

    # find connected components:
    set_list = []
    for n in nodes:
        set_list.append({n})

    for edge in edges:
        (x, y) = edge
        combine_set(x, y)

    print('there are', len(set_list), 'connected components')
    print('there are', max(len(set_list[0]), len(set_list[1]), len(set_list[2])), 'nodes in the largest components')

    # greedy algorithms by min normalized cut cost
    mnodes = list(set_list[1])
    mnodes.sort()
    cut1 = mnodes[:20]
    cut2 = mnodes[20:]
    medges = set()
    for edge in edges:
        (x, y) = edge
        if (x in mnodes) and (y in mnodes):
            medges.add(edge)
    print("50/50 normolized_cut is ", normalized_cut(medges, cut1, cut2))
    greedy_algorithms(medges, cut1, cut2)

    # greedy algorithms by max modularity
    cut1 = mnodes[:20]
    cut2 = mnodes[20:]
    greedy_algorithms1(medges, cut1, cut2)

    # show the largest component
    G = nx.Graph()
    for e in medges:
        G.add_edge(e[0], e[1])
    nx.draw(G)
    plt.show()
    plt.clf()