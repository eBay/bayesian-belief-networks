'''Test the Earthquake Example as a Factor Graph.'''
from bayesian.factor_graph import build_graph
from bayesian.examples.factor_graphs.earthquake import *


def pytest_funcarg__earthquake_factor_graph(request):
    g = build_graph(
        f_burglary, f_earthquake, f_alarm,
        f_johncalls, f_marycalls)
    return g


def close_enough(x, y, r=6):
    return round(x, r) == round(y, r)


class TestEarthQuakeBBN():

    def test_no_evidence(self, earthquake_factor_graph):
        result = earthquake_factor_graph.query()

        assert close_enough(result[('alarm', True)], 0.016114)
        assert close_enough(result[('alarm', False)], 0.983886)
        assert close_enough(result[('burglary', True)], 0.010000)
        assert close_enough(result[('burglary', False)], 0.990000)
        assert close_enough(result[('earthquake', True)], 0.020000)
        assert close_enough(result[('earthquake', False)], 0.980000)
        assert close_enough(result[('johncalls', True)], 0.063697)
        assert close_enough(result[('johncalls', False)], 0.936303)
        assert close_enough(result[('marycalls', True)], 0.021119)
        assert close_enough(result[('marycalls', False)], 0.978881)
