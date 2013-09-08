'''Test the cancer example as a BBN.'''
from bayesian.bbn import build_bbn
from bayesian.examples.bbns.cancer import fP, fS, fC, fX, fD


def pytest_funcarg__cancer_graph(request):
    g = build_bbn(
        fP, fS, fC, fX, fD,
        domains={
            'P': ['low', 'high']})
    return g


def close_enough(x, y, r=3):
    return round(x, r) == round(y, r)


class TestCancerGraph():

    '''
    See table 2.2 of BAI_Chapter2.pdf
    For verification of results.
    (Note typo in some values)
    '''

    def test_no_evidence(self, cancer_graph):
        '''Column 2 of upper half of table'''

        result = cancer_graph.query()
        assert close_enough(result[('P', 'high')], 0.1)
        assert close_enough(result[('P', 'low')], 0.9)
        assert close_enough(result[('S', True)], 0.3)
        assert close_enough(result[('S', False)], 0.7)
        assert close_enough(result[('C', True)], 0.012)
        assert close_enough(result[('C', False)], 0.988)
        assert close_enough(result[('X', True)], 0.208)
        assert close_enough(result[('X', False)], 0.792)
        assert close_enough(result[('D', True)], 0.304)
        assert close_enough(result[('D', False)], 0.696)

    def test_D_True(self, cancer_graph):
        '''Column 3 of upper half of table'''
        result = cancer_graph.query(D=True)
        import pytest; pytest.set_trace()
        assert close_enough(result[('P', 'high')], 0.102)
        assert close_enough(result[('P', 'low')], 0.898)
        assert close_enough(result[('S', True)], 0.307)
        assert close_enough(result[('S', False)], 0.693)
        assert close_enough(result[('C', True)], 0.025)
        assert close_enough(result[('C', False)], 0.975)
        assert close_enough(result[('X', True)], 0.217)
        assert close_enough(result[('X', False)], 0.783)
        assert close_enough(result[('D', True)], 1)
        assert close_enough(result[('D', False)], 0)

    def test_S_True(self, cancer_graph):
        '''Column 4 of upper half of table'''
        import pytest; pytest.set_trace()
        result = cancer_graph.query(S=True)
        assert round(result[('P', 'high')], 3) == 0.1
        assert round(result[('P', 'low')], 3) == 0.9
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 0.032
        assert round(result[('C', False)], 3) == 0.968
        assert round(result[('X', True)], 3) == 0.222
        assert round(result[('X', False)], 3) == 0.778
        assert round(result[('D', True)], 3) == 0.311
        assert round(result[('D', False)], 3) == 0.689

    def test_C_True(self, cancer_graph):
        '''Column 5 of upper half of table'''
        result = cancer_graph.query(C=True)
        assert round(result[('P', 'high')], 3) == 0.249
        assert round(result[('P', 'low')], 3) == 0.751
        assert round(result[('S', True)], 3) == 0.825
        assert round(result[('S', False)], 3) == 0.175
        assert round(result[('C', True)], 3) == 1
        assert round(result[('C', False)], 3) == 0
        assert round(result[('X', True)], 3) == 0.9
        assert round(result[('X', False)], 3) == 0.1
        assert round(result[('D', True)], 3) == 0.650
        assert round(result[('D', False)], 3) == 0.350

    def test_C_True_S_True(self, cancer_graph):
        '''Column 6 of upper half of table'''
        result = cancer_graph.query(C=True, S=True)
        assert round(result[('P', 'high')], 3) == 0.156
        assert round(result[('P', 'low')], 3) == 0.844
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 1
        assert round(result[('C', False)], 3) == 0
        assert round(result[('X', True)], 3) == 0.9
        assert round(result[('X', False)], 3) == 0.1
        assert round(result[('D', True)], 3) == 0.650
        assert round(result[('D', False)], 3) == 0.350

    def test_D_True_S_True(self, cancer_graph):
        '''Column 7 of upper half of table'''
        result = cancer_graph.query(D=True, S=True)
        assert round(result[('P', 'high')], 3) == 0.102
        assert round(result[('P', 'low')], 3) == 0.898
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 0.067
        assert round(result[('C', False)], 3) == 0.933
        assert round(result[('X', True)], 3) == 0.247
        assert round(result[('X', False)], 3) == 0.753
        assert round(result[('D', True)], 3) == 1
        assert round(result[('D', False)], 3) == 0
