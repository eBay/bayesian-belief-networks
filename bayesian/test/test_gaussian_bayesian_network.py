from __future__ import division
import pytest

import os

from bayesian.gaussian import MeansVector, CovarianceMatrix
from bayesian.gaussian_bayesian_network import *
from bayesian.examples.gaussian_bayesian_networks.river import f_a, f_b, f_c, f_d


def pytest_funcarg__river_graph(request):
    g = build_graph(f_a, f_b, f_c, f_d)
    return g


class TestGBN():

    def test_get_joint_parameters(self, river_graph):
        mu, sigma = river_graph.get_joint_parameters()
        assert mu == MeansVector(
            [[3],
             [9],
             [4],
             [14]],
            names = ['a','c','b','d'])
        assert sigma == CovarianceMatrix(
            [[4, 8, 4, 12],
             [8, 20, 8, 28],
             [4, 8, 5, 13],
             [12, 28, 13, 42]],
            names = ['a','c','b','d'])

    def test_query(self, river_graph):
        import ipdb; ipdb.set_trace()
        result = river_graph.query(a=7)
