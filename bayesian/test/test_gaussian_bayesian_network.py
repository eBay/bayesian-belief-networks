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
             [4],
             [9],
             [14]],
            names = ['a','b','c','d'])
        assert sigma == CovarianceMatrix(
            [[4, 4, 8, 12],
             [4, 5, 8, 13],
             [8, 8, 20, 28],
             [12, 13, 28, 42]],
            names = ['a','b','c','d'])

    def test_query(self, river_graph):

        result = river_graph.query(a=7)
        mu = result['joint']['mu']
        sigma = result['joint']['sigma']
        assert mu == Matrix([
            [8],
            [17],
            [26]])
        assert sigma == CovarianceMatrix(
            [[1, 0, 1],
             [0, 4, 4],
             [1, 4, 6]],
            names = ['b','c','d'])

        result = river_graph.query(a=7, c=17)
        mu = result['joint']['mu']
        sigma = result['joint']['sigma']
        assert mu == Matrix([
            [8],
            [26]])
        assert sigma == CovarianceMatrix(
            [[1, 1],
             [1, 2]],
            names = ['b','d'])

        result = river_graph.query(a=7, c=17, b=8)
        mu = result['joint']['mu']
        sigma = result['joint']['sigma']
        assert mu == Matrix([
            [26]])
        assert sigma == CovarianceMatrix(
            [[1]],
            names = ['d'])
