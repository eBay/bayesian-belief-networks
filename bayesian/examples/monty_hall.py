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

#domains = dict(
#    ActualDoor=['A', 'B', 'C'],
#    GuestDoor=['A', 'B', 'C'],
#    MontyDoor=['A', 'B', 'C'])


def fActualDoor(ActualDoor):
    return 1.0 / 3

#fActualDoor.domains = domains


def fGuestDoor(GuestDoor):
    return 1.0 / 3

#fGuestDoor.domains = domains


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

#fMontyDoor.domains = domains

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
    # Initial Marginals without any knowledge...
    print marg(ActualDoor, 'A')
    print marg(ActualDoor, 'B')
    print marg(ActualDoor, 'C')

    # Now we suppose the Guest chooses A
    # and Monty chooses B. Should we
    # switch or not? To answer this
    # we look at the likelyhood for
    # the actual door given the evidence...
    graph.reset()
    add_evidence(GuestDoor, 'A')
    add_evidence(MontyDoor, 'B')
    graph.propagate()
    normalizer = GuestDoor.marginal('A')
    print marg(ActualDoor, 'A', normalizer)
    print marg(ActualDoor, 'B', normalizer)
    print marg(ActualDoor, 'C', normalizer)

    # As you will see the likelihood of C is
    # twice that of A, given the evidence above
    # so we should switch.
    graph.reset()
    add_evidence(MontyDoor, 'B')
    graph.propagate()
    normalizer = MontyDoor.marginal('B')
    print marg(ActualDoor, 'A', normalizer)
    print marg(ActualDoor, 'B', normalizer)
    print marg(ActualDoor, 'C', normalizer)
    print marg(GuestDoor, 'A', normalizer)
    print marg(GuestDoor, 'B', normalizer)
    print marg(GuestDoor, 'C', normalizer)

    
