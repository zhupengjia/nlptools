#!/usr/bin/env python3
from collections import defaultdict

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class Graph(object):
    '''
        Create a graph structure with node and edge. Support directed graph and undirected graph

        Input:
            - connections: one can input existed connections, default is [], format is [[node1, node2], ...]
            - directed: True or False, directed or undirected, default is False

    '''
    def __init__(self, connections=None, directed=False):
        if connections is None:
            connections = []
        self._graph = defaultdict(dict)
        self._direct = directed
        self.add_connections(connections)


    def add_connections(self, connections):
        '''
            add connections
            
            Input:
                - connections: [[node1, node2], ...]
        '''
        for node1, node2 in connections:
            self.add(node1, node2)


    def add(self, node1, node2):
        '''
            add a connection for two nodes

            Input:
                - node1: string
                - node2: string
        '''
        if not node2 in self._graph[node1]:
            self._graph[node1][node2] = 1
        else:
            self._graph[node1][node2] += 1
        if not self._direct:
            if not node1 in self._graph[node2]:
                self._graph[node2][node1] = 1
            else:
                self._graph[node2][node1] += 1



