'''This is the example from Chapter 2 BAI'''
from bayesian.factor_graph import *


def fP(P):
    '''Pollution'''
    if P.value == 'high':
        return 0.1
    elif P.value == 'low':
        return 0.9

fP.domains = dict(P=['high', 'low'])


def fS(S):
    '''Smoker'''
    if S.value is True:
        return 0.3
    elif S.value is False:
        return 0.7

fS.domains = dict(S=[True, False])


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
    key = key + 't' if P.value == 'high' else key + 'f'
    key = key + 't' if S.value else key + 'f'
    key = key + 't' if C.value else key + 'f'
    return table[key]


fC.domains = dict(P=['high', 'low'], S=[True, False], C=[True, False])


def fX(C, X):
    '''X-ray'''
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if C.value else key + 'f'
    key = key + 't' if X.value else key + 'f'
    return table[key]


fX.domains = dict(C=[True, False], X=[True, False])


def fD(C, D):
    '''Dyspnoeia'''
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if C.value else key + 'f'
    key = key + 't' if D.value else key + 'f'
    return table[key]


fD.domains = dict(C=[True, False], D=[True, False])


# Build the network

fP_node = FactorNode('fP', fP)
fS_node = FactorNode('fS', fS)
fC_node = FactorNode('fC', fC)
fX_node = FactorNode('fX', fX)
fD_node = FactorNode('fD', fD)

P = VariableNode('P')
S = VariableNode('S')
C = VariableNode('C')
X = VariableNode('X')
D = VariableNode('D')

connect(fP_node, P)
connect(fS_node, S)
connect(fC_node, [P, S, C])
connect(fX_node, [C, X])
connect(fD_node, [C, D])

graph = FactorGraph(
    [P, S, C, X, D,
     fP_node, fS_node,
     fC_node, fX_node, fD_node])


def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)


if __name__ == '__main__':
    graph.propagate()
    print marg(P, 'high')
    print marg(S, True)
    print marg(C, True)
    print marg(X, True)
    print marg(D, True)
