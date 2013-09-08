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
        result = cancer_graph.query(S=True)
        assert close_enough(result[('P', 'high')], 0.1)
        assert close_enough(result[('P', 'low')], 0.9)
        assert close_enough(result[('S', True)], 1)
        assert close_enough(result[('S', False)], 0)
        assert close_enough(result[('C', True)], 0.032)
        assert close_enough(result[('C', False)], 0.968)
        assert close_enough(result[('X', True)], 0.222)
        assert close_enough(result[('X', False)], 0.778)
        assert close_enough(result[('D', True)], 0.311)
        assert close_enough(result[('D', False)], 0.689)

    def test_C_True(self, cancer_graph):
        '''Column 5 of upper half of table'''
        result = cancer_graph.query(C=True)
        assert close_enough(result[('P', 'high')], 0.249)
        assert close_enough(result[('P', 'low')], 0.751)
        assert close_enough(result[('S', True)], 0.825)
        assert close_enough(result[('S', False)], 0.175)
        assert close_enough(result[('C', True)], 1)
        assert close_enough(result[('C', False)], 0)
        assert close_enough(result[('X', True)], 0.9)
        assert close_enough(result[('X', False)], 0.1)
        assert close_enough(result[('D', True)], 0.650)
        assert close_enough(result[('D', False)], 0.350)

    def test_C_True_S_True(self, cancer_graph):
        '''Column 6 of upper half of table'''
        result = cancer_graph.query(C=True, S=True)
        assert close_enough(result[('P', 'high')], 0.156)
        assert close_enough(result[('P', 'low')], 0.844)
        assert close_enough(result[('S', True)], 1)
        assert close_enough(result[('S', False)], 0)
        assert close_enough(result[('C', True)], 1)
        assert close_enough(result[('C', False)], 0)
        assert close_enough(result[('X', True)], 0.9)
        assert close_enough(result[('X', False)], 0.1)
        assert close_enough(result[('D', True)], 0.650)
        assert close_enough(result[('D', False)], 0.350)

    def test_D_True_S_True(self, cancer_graph):
        '''Column 7 of upper half of table'''
        result = cancer_graph.query(D=True, S=True)
        assert close_enough(result[('P', 'high')], 0.102)
        assert close_enough(result[('P', 'low')], 0.898)
        assert close_enough(result[('S', True)], 1)
        assert close_enough(result[('S', False)], 0)
        assert close_enough(result[('C', True)], 0.067)
        assert close_enough(result[('C', False)], 0.933)
        assert close_enough(result[('X', True)], 0.247)
        assert close_enough(result[('X', False)], 0.753)
        assert close_enough(result[('D', True)], 1)
        assert close_enough(result[('D', False)], 0)
