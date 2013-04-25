'''The Monty Hall Problem Modelled as a Bayesian Belief Network'''
from bayesian.factor_graph import *

'''
As BBN:


         GuestDoor     ActualDoor
               \         /
                MontyDoor
                p(M|G,A)


As Factor Graph:

        fGuestDoor               fActualDoor
             |                        |
         GuestDoor                ActualDoor
             |                        |
             +------fMontyDoor--------+
                         |
                     MontyDoor

Now Query: Given Guest chooses door A
and Monty chooses door B, should guest
switch to C or stay with A?

'''
def fActualDoor(ActualDoor):
    return 1.0 / 3


def fGuestDoor(GuestDoor):
    return 1.0 / 3


def fMontyDoor(ActualDoor, GuestDoor, MontyDoor):
    # AA
    if ActualDoor.value == GuestDoor.value:
        # AAA
        if GuestDoor.value == MontyDoor.value:
            return 0
        # AAB AAC
        else:
            return 0.5
    # ABB
    if GuestDoor.value == MontyDoor.value:
        return 0
    # ABA
    if ActualDoor.value == MontyDoor.value:
        return 0
    # ABC
    return 1


# Build the network
fActualDoor_node = FactorNode('fActualDoor', fActualDoor)
fGuestDoor_node = FactorNode('fGuestDoor', fGuestDoor)
fMontyDoor_node = FactorNode('fMontyDoor', fMontyDoor)

ActualDoor = VariableNode('ActualDoor', domain=['A', 'B', 'C'])
GuestDoor = VariableNode('GuestDoor', domain=['A', 'B', 'C'])
MontyDoor = VariableNode('MontyDoor', domain=['A', 'B', 'C'])

connect(fActualDoor_node, ActualDoor)
connect(fGuestDoor_node, GuestDoor)
connect(fMontyDoor_node, [ActualDoor, GuestDoor, MontyDoor])

graph = FactorGraph(
    [ActualDoor,
     GuestDoor,
     MontyDoor,
     fActualDoor_node,
     fGuestDoor_node,
     fMontyDoor_node])


def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)

if __name__ == '__main__':

    graph.propagate()

    # Initial Marginals without any knowledge.
    # Observe that the likelihood for 
    # all three doors is 1/3.
    print 'Initial Marginal Probabilities:'
    graph.status()

    # Now suppose the guest chooses
    # door A and Monty chooses door B. 
    # Should we switch our choice from
    # A to C or not? 
    # To answer this we "query" the 
    # graph with instantiation of the
    # observed variables as "evidence".
    # The likelihood for door C has
    # indeed increased to 2/3 therefore
    # we should switch to door C.
    print 'Marginals after knowing Guest chose A and Monty chose B.'
    graph.q(GuestDoor='A', MontyDoor='B')

    
