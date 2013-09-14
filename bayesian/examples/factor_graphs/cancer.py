'''This is the example from Chapter 2 BAI'''
from bayesian.factor_graph import *


def fP(P):
    '''Pollution'''
    if P == 'high':
        return 0.1
    elif P == 'low':
        return 0.9


def fS(S):
    '''Smoker'''
    if S is True:
        return 0.3
    elif S is False:
        return 0.7


def fC(P, S, C):
    '''Cancer'''
    table = dict()
    table['ttt'] = 0.05
    table['ttf'] = 0.95
    table['tft'] = 0.02
    table['tff'] = 0.98
    table['ftt'] = 0.03
    table['ftf'] = 0.97
    table['fft'] = 0.001
    table['fff'] = 0.999
    key = ''
    key = key + 't' if P == 'high' else key + 'f'
    key = key + 't' if S else key + 'f'
    key = key + 't' if C else key + 'f'
    return table[key]


def fX(C, X):
    '''X-ray'''
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if C else key + 'f'
    key = key + 't' if X else key + 'f'
    return table[key]


def fD(C, D):
    '''Dyspnoeia'''
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if C else key + 'f'
    key = key + 't' if D else key + 'f'
    return table[key]


if __name__ == '__main__':
    g = build_graph(
        fP, fS, fC, fX, fD,
        domains={
            'P': ['low', 'high']})
    g.q()
    g.q(P='high')
    g.q(D=True)
    g.q(S=True)
    g.q(C=True, S=True)
    g.q(D=True, S=True)
