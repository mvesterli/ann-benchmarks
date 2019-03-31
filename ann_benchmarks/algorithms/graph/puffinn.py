from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
from _puffinnwrapper import build_graph

class PlainLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._hash_function = params['hash_function']
        self._recall = params['recall']
        self._hash_length = params['hash_length']

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'plain',
            hash_function = self._hash_function,
            recall = self._recall,
            hash_length = self._hash_length)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'PlainLSH(hash=%s, hash_length=%d, recall=%.2f)' % (self._hash_function, self._hash_length, self._recall)

class ProjectionLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._hash_function = params['hash_function']
        self._repetitions = params['repetitions']
        self._hash_length = params['hash_length']
        self._block_size = params['block_size']

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'projection',
            hash_function = self._hash_function,
            repetitions = self._repetitions,
            hash_length = self._hash_length,
            block_size = self._block_size)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'ProjectionLSH(hash=%s, hash_length=%d, repetitions=%d, block_size=%d)' % (self._hash_function, self._hash_length, self._repetitions, self._block_size)

class VariableHashLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._hash_function = params['hash_function']
        self._recall = params['recall']

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'variable_hash',
            hash_function = self._hash_function,
            recall = self._recall)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'VariableHashLSH(hash=%s, recall: %.2f)' % (self._hash_function, self._recall)

