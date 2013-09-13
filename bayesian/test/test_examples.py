'''Unit tests for the examples in the examples dir.'''
from bayesian.factor_graph import build_graph
from bayesian.examples.factor_graphs.cancer import fP, fS, fC, fX, fD


'''
Since one of the goals of this package
are to have many working examples its
very important that the examples work
correctly "out of the box".
Please add unit tests for all examples
and give references to their sources.

Note that the identical graph also
appears in test_graph where many more
lower level tests are run. These tests
however import the code directly from
the examples directory.
'''


def pytest_funcarg__cancer_graph(request):
    g = build_graph(
        fP, fS, fC, fX, fD,
        domains={
            'P': ['low', 'high']})
    return g


class TestCancerGraph():

    '''
    See table 2.2 of BAI_Chapter2.pdf
    For verification of results.
    (Note typo in some values)
    '''

    def test_no_evidence(self, cancer_graph):
        '''Column 2 of upper half of table'''
        result = cancer_graph.query()
        assert round(result[('P', 'high')], 3) == 0.1
        assert round(result[('P', 'low')], 3) == 0.9
        assert round(result[('S', True)], 3) == 0.3
        assert round(result[('S', False)], 3) == 0.7
        assert round(result[('C', True)], 3) == 0.012
        assert round(result[('C', False)], 3) == 0.988
        assert round(result[('X', True)], 3) == 0.208
        assert round(result[('X', False)], 3) == 0.792
        assert round(result[('D', True)], 3) == 0.304
        assert round(result[('D', False)], 3) == 0.696

    def test_D_True(self, cancer_graph):
        '''Column 3 of upper half of table'''
        result = cancer_graph.query(D=True)
        assert round(result[('P', 'high')], 3) == 0.102
        assert round(result[('P', 'low')], 3) == 0.898
        assert round(result[('S', True)], 3) == 0.307
        assert round(result[('S', False)], 3) == 0.693
        assert round(result[('C', True)], 3) == 0.025
        assert round(result[('C', False)], 3) == 0.975
        assert round(result[('X', True)], 3) == 0.217
        assert round(result[('X', False)], 3) == 0.783
        assert round(result[('D', True)], 3) == 1
        assert round(result[('D', False)], 3) == 0

    def test_S_True(self, cancer_graph):
        '''Column 4 of upper half of table'''
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
