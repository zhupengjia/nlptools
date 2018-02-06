#!/usr/bin/env python3
from collections import defaultdict

class Graph(object):
    def __init__(self, connections=[], directed=False):
        self._graph = defaultdict(dict)
        self._direct = directed
        self.add_connections(connections)

    def add_connections(self, connections):
        for node1, node2 in connections:
            self.add(node1, node2)

    def add(self, node1, node2):
        if not node2 in self._graph[node1]:
            self._graph[node1][node2] = 1
        else:
            self._graph[node1][node2] += 1
        if not self._direct:
            if not node1 in self._graph[node2]:
                self._graph[node2][node1] = 1
            else:
                self._graph[node2][node1] += 1



