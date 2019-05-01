from __future__ import absolute_import
from ann_benchmarks.algorithms.base import BaseANN
from _puffinnwrapper import build_graph

class PlainLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._params = params

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'plain',
            **self._params)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'PlainLSH(params=%s)' % self._params

class ProjectionLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._params = params

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'projection',
            **self._params)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'ProjectionLSH(params=%s)' % self._params

class FixedBucketsLSH(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self._params = params

    def fit(self, X):
        self._graph = build_graph(
            self._metric,
            X,
            self._count,
            method = 'fixed_buckets',
            **self._params)

    def query(self, idx, n):
        return self._graph[idx]

    def builds_graph(self):
        return True

    def set_count(self, count):
        self._count = count

    def __str__(self):
        return 'VariableHashLSH(params=%s)' % self._params

