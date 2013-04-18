'''This is the example from Chapter 2 BAI'''
from bayesian.factor_graph import *


def fP(P):
    if P.value == True:
        return 0.1
    elif P.value == False:
        return 0.9

def fS(S):
    if S.value == True:
        return 0.3
    elif S.value == False:
        return 0.7


def fC(P, S, C):
    ''' 
    This needs to be a joint probability distribution
    over the inputs and the node itself
    '''
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
    key = key + 't' if P.value else key + 'f'
    key = key + 't' if S.value else key + 'f'
    key = key + 't' if C.value  else key + 'f'
    return table[key]


def fX(C, X):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if C.value else key + 'f'
    key = key + 't' if X.value else key + 'f'
    return table[key]


def fD(C, D):
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if C.value else key + 'f'
    key = key + 't' if D.value else key + 'f'
    return table[key]


# Build the network

fP_node = FactorNode('fP', fP)
fS_node = FactorNode('fS', fS)
fC_node = FactorNode('fC', fC)
fX_node = FactorNode('fX', fX)
fD_node = FactorNode('fD', fD)

P = VariableNode('P', parents=[fP_node])
S = VariableNode('S', parents=[fS_node])
C = VariableNode('C', parents=[fC_node])
X = VariableNode('X', parents=[fX_node])
D = VariableNode('D', parents=[fD_node])

# Now set the parents for the factor nodes...
fP_node.parents = []
fS_node.parents = []
fC_node.parents = [P, S]
fX_node.parents = [C]
fD_node.parents = [C]

# Now set children for Variable Nodes...
P.children = [fC_node]
S.children = [fC_node]
C.children = [fX_node, fD_node]
X.children = []
D.children = []

# Now set the children for the factor nodes...
fP_node.children = [P]
fS_node.children = [S]
fC_node.children = [C]
fX_node.children = [X]
fD_node.children = [D]

graph = FactorGraph([P, S, C, X, D, fP_node, fS_node, fC_node, fX_node, fD_node])

def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)



if __name__ == '__main__':
    graph.propagate()

    print marg(P, True)
    print marg(S, True)
    print marg(C, True)
    print marg(X, True)
    print marg(D, True)

def test_add_evidence():
    ''' 
    We will set x5=True, this
    corresponds to variable D in BAI
    '''
    graph.reset()
    add_evidence(x5, True)
    graph.propagate()
    normalizer = marg(x5, True)
    assert normalizer == 0.304
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 0.307
    m = marg(x2, False, normalizer)
    assert m == 0.693
    m = marg(x3, True, normalizer)
    assert m == 0.025
    m = marg(x3, False, normalizer)
    assert m == 0.975
    m = marg(x4, True, normalizer)
    assert m == 0.217
    m = marg(x4, False, normalizer)
    assert m == 0.783
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0
    
def test_add_evidence_x2_true():
    '''
    x2 = S in BAI
    '''
    graph.reset()
    add_evidence(x2, True)
    graph.propagate()
    normalizer = marg(x2, True)
    m = marg(x1, True, normalizer)
    assert m == 0.1
    m = marg(x1, False, normalizer)
    assert m == 0.9
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.032
    m = marg(x3, False, normalizer)
    assert m == 0.968
    m = marg(x4, True, normalizer)
    assert m == 0.222
    m = marg(x4, False, normalizer)
    assert m == 0.778
    m = marg(x5, True, normalizer)
    assert m == 0.311
    m = marg(x5, False, normalizer)
    assert m == 0.689
    

def test_add_evidence_x3_true():
    '''
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.249
    m = marg(x1, False, normalizer)
    assert m == 0.751
    m = marg(x2, True, normalizer)
    assert m == 0.825
    m = marg(x2, False, normalizer)
    assert m == 0.175
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350


def test_add_evidence_x2_true_and_x3_true():
    '''
    x2 = True in BAI this is Smoker = True
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x2, True)
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.156
    m = marg(x1, False, normalizer)
    assert m == 0.844
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350
    
def test_add_evidence_x5_true_x2_true():
    graph.reset()
    add_evidence(x5, True)
    add_evidence(x2, True)
    graph.propagate()
    normalizer = x5.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.067
    m = marg(x3, False, normalizer)
    assert m == 0.933
    m = marg(x4, True, normalizer)
    assert m == 0.247
    m = marg(x4, False, normalizer)
    assert m == 0.753
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0
    

