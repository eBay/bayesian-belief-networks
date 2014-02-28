'''Tests for the examples in examples/gaussian_bayesian_networks'''
from bayesian.gaussian_bayesian_network import build_graph
from bayesian.examples.gaussian_bayesian_networks.river import (
    f_a, f_b, f_c, f_d)
from bayesian.linear_algebra import zeros, Matrix
from bayesian.gaussian import MeansVector, CovarianceMatrix


def pytest_funcarg__river_graph(request):
    g = build_graph(f_a, f_b, f_c, f_d)
    return g


class TestRiverExample():

    def test_get_joint_parameters(self, river_graph):
        mu, sigma = river_graph.get_joint_parameters()
        assert mu == MeansVector([
            [3],
            [4],
            [9],
            [14]], names=['a', 'b', 'c', 'd'])
        assert sigma == CovarianceMatrix([
            [4, 4, 8, 12],
            [4, 5, 8, 13],
            [8, 8, 20, 28],
            [12, 13, 28, 42]], names=['a', 'b', 'c', 'd'])

    def test_query(self, river_graph):
        r = river_graph.query(a=7)
        print r
