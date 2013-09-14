'''Test the Huang-Darwiche example as a BBN.'''
from bayesian.bbn import build_bbn
from bayesian.examples.bbns.huang_darwiche import *


def pytest_funcarg__huang_darwiche_bbn(request):
    g = build_bbn(
        f_a, f_b, f_c, f_d,
        f_e, f_f, f_g, f_h)
    return g


def close_enough(x, y, r=3):
    return round(x, r) == round(y, r)


class TestHuangeDarwicheBBN():

    def test_no_evidence(self, huang_darwiche_bbn):
        result = huang_darwiche_bbn.query()
        assert close_enough(result[('a', True)], 0.5)
        assert close_enough(result[('a', False)], 0.5)
        assert close_enough(result[('d', True)], 0.68)
        assert close_enough(result[('d', False)], 0.32)
        assert close_enough(result[('b', True)], 0.45)
        assert close_enough(result[('b', False)], 0.55)
        assert close_enough(result[('c', True)], 0.45)
        assert close_enough(result[('c', False)], 0.55)
        assert close_enough(result[('e', True)], 0.465)
        assert close_enough(result[('e', False)], 0.535)
        assert close_enough(result[('f', True)], 0.176)
        assert close_enough(result[('f', False)], 0.824)
        assert close_enough(result[('g', True)], 0.415)
        assert close_enough(result[('g', False)], 0.585)
        assert close_enough(result[('h', True)], 0.823)
        assert close_enough(result[('h', False)], 0.177)
