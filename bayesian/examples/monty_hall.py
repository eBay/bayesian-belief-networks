'''The Monty Hall Problem Modelled as a Bayesian Belief Network'''
from bayesian.factor_graph import *

'''

As BBN:


         GuestDoor        ActualDoor
           |    \          /
           |     MontyDoor    p(M|G,A)
           \       /
            SecondDoor p(S|G,M)


             fGuestDoor               fActualDoor
                  |                        |
              GuestDoor                ActualDoor
                  |                        |
                  +------fMontyDoor--------+           
                             |               
                         MontyDoor           
                             |               
                     fGuestSecondDoor  
                             |
                      GuestSecondDoor

'''

domains = (
    ActualDoor = ['A', 'B', 'C'],
    GuestDoor = ['A', 'B', 'C'],
    MontyDoor = ['A', 'B', 'C'],
    GuestSecondDoor = ['A', 'B', 'C'])


def fActualDoor(ActualDoor):
    if ActualDoor.value == 'A':
        return 1.0 / 3
    elif ActualDoor.value == 'B':
        return 1.0 / 3
    elif ActualDoor.value == 'C':
        return 1.0 / 3

fActualDoor.domains = domains

def fGuestDoor(GuestDoor):
    if GuestDoor.value == 'A':
        return 1.0 / 3
    elif GuestDoor.value == 'B':
        return 1.0 / 3
    elif GuestDoor.value == 'C':
        return 1.0 / 3

fGuestDoor.domains = domains

def fMontyDoor(ActualDoor, GuestDoor, MontyDoor):
    if ActualDoor == GuestDoor:
        return 0.5
    if ActualDoor == MontyDoor:
        return 0
    return 0.5

fMontyDoor.domains = domains

def fGuestSecondDoor(MontyDoor, GuestSecondDoor):
    if GuestSecondDoor == MontyDoor:
        return 0
    else:
        return 0.5

fGuestSecondDoor.domains = domains




# Build the network

fGuestDoor_node = FactorNode('fGuestDoor', fGuestDoor)
fMontyDoor_node = FactorNode('fMontyDoor', fMontyDoor)
fGuestSecondDoor_node = FactorNode('fGuestSecondDoor', fGuestSecondDoor)

GuestDoor = VariableNode('GuestDoor', parents=[fGuestDoor_node])
MontyDoor = VariableNode('MontyDoor', parents=[fMontyDoor_node])
GuestSecondDoor = VariableNode('GuestSecondDoor', parents=[fGuestSecondDoor_node])


# Now set the parents for the factor nodes...
fGuestDoor_node.parents = []
fMontyDoor_node.parents = [GuestDoor]
fGuestSecondDoor_node.parents = [MontyDoor]

# Now set children for Variable Nodes...
GuestDoor.children = [fMontyDoor_node]
MontyDoor.children = [fGuestSecondDoor_node]
GuestSecondDoor.children = []

# Now set the children for the factor nodes...
fGuestDoor_node.children = [GuestDoor]
fMontyDoor_node.children = [MontyDoor]
fGuestSecondDoor_node.children = [GuestSecondDoor]

graph = FactorGraph([
        GuestDoor,
        MontyDoor,
        GuestSecondDoor,
        fGuestDoor_node,
        fMontyDoor_node,
        fGuestSecondDoor_node])

def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)



if __name__ == '__main__':
    graph.propagate()

    print marg(GuestDoor, 'A')
    print marg(GuestDoor, 'B')
    print marg(GuestDoor, 'C')

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
    

