'''This is the earthquake example from 2.5.1 in BAI'''
from bayesian.factor_graph import *


def fBurglary(B):
    if B.value is True:
        return 0.01
    return 0.99

fBurglary.domains = dict(B=[True, False])


def fEarthquake(E):
    if E.value is True:
        return 0.02
    return 0.98

fEarthquake.domains = dict(E=[True, False])


def fAlarm(B, E, A):
    table = dict()
    table['ttt'] = 0.95
    table['ttf'] = 0.05
    table['tft'] = 0.94
    table['tff'] = 0.06
    table['ftt'] = 0.29
    table['ftf'] = 0.71
    table['fft'] = 0.001
    table['fff'] = 0.999
    key = ''
    key = key + 't' if B.value else key + 'f'
    key = key + 't' if E.value else key + 'f'
    key = key + 't' if A.value else key + 'f'
    return table[key]


fAlarm.domains = dict(
    B=[True, False],
    E=[True, False],
    A=[True, False])


def fJohnCalls(A, J):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.05
    table['ff'] = 0.95
    key = ''
    key = key + 't' if A.value else key + 'f'
    key = key + 't' if J.value else key + 'f'
    return table[key]

fJohnCalls.domains = dict(A=[True, False], J=[True, False])


def fMaryCalls(A, M):
    table = dict()
    table['tt'] = 0.7
    table['tf'] = 0.3
    table['ft'] = 0.01
    table['ff'] = 0.99
    key = ''
    key = key + 't' if A.value else key + 'f'
    key = key + 't' if M.value else key + 'f'
    return table[key]

fMaryCalls.domains = dict(
    A=[True, False],
    M=[True, False])

# Build the network

fBurglary_node = FactorNode('fBurglary', fBurglary)
fEarthquake_node = FactorNode('fEarthquake', fEarthquake)
fAlarm_node = FactorNode('fAlarm', fAlarm)
fJohnCalls_node = FactorNode('fJohnCalls', fJohnCalls)
fMaryCalls_node = FactorNode('fMaryCalls', fMaryCalls)

B = VariableNode('B')
E = VariableNode('E')
A = VariableNode('A')
J = VariableNode('J')
M = VariableNode('M')

connect(fBurglary_node, B)
connect(fEarthquake_node, E)
connect(fAlarm_node, [B, E, A])
connect(fJohnCalls_node, [A, J])
connect(fMaryCalls_node, [A, M])

graph = FactorGraph(
    [B, E, A, J, M,
     fBurglary_node, fEarthquake_node,
     fAlarm_node, fJohnCalls_node,
     fMaryCalls_node])


def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)


if __name__ == '__main__':
    graph.propagate()
    print marg(B, True)
    print marg(E, True)
    print marg(A, True)
    print marg(J, True)
    print marg(M, True)
