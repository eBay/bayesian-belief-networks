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

domains = dict(
    ActualDoor = ['A', 'B', 'C'],
    GuestDoor = ['A', 'B', 'C'],
    MontyDoor = ['A', 'B', 'C'])


def fActualDoor(ActualDoor):
    return 1.0 / 3

fActualDoor.domains = domains

def fGuestDoor(GuestDoor):
    return 1.0 / 3

fGuestDoor.domains = domains

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

fMontyDoor.domains = domains

# Build the network

fActualDoor_node = FactorNode('fActualDoor', fActualDoor)
fGuestDoor_node = FactorNode('fGuestDoor', fGuestDoor)
fMontyDoor_node = FactorNode('fMontyDoor', fMontyDoor)

ActualDoor = VariableNode('ActualDoor')
GuestDoor = VariableNode('GuestDoor')
MontyDoor = VariableNode('MontyDoor')

import ipdb; ipdb.set_trace()

print 'Neighbours before first call to connect:'
for node in [ActualDoor,
             GuestDoor,
             MontyDoor,
             fActualDoor_node,
             fGuestDoor_node,
             fMontyDoor_node]:
    print node.name, node.neighbours
connect(fActualDoor_node, ActualDoor)
print 'Neighbours after first call to connect:'
for node in [ActualDoor,
             GuestDoor,
             MontyDoor,
             fActualDoor_node,
             fGuestDoor_node,
             fMontyDoor_node]:
    print node.name, node.neighbours


connect(fGuestDoor_node, GuestDoor)
connect(fMontyDoor_node, ActualDoor)
connect(fMontyDoor_node, GuestDoor)
connect(fMontyDoor_node, MontyDoor)


#ActualDoor = VariableNode('ActualDoor', neighbours=[fActualDoor_node, fMontyDoor_node])
#GuestDoor = VariableNode('GuestDoor', neighbours=[fGuestDoor_node, fMontyDoor_node])
#MontyDoor = VariableNode('MontyDoor', neighbours=[fMontyDoor_node])

# Now set the parents for the factor nodes...
#fActualDoor_node.neighbours = [ActualDoor]
#fGuestDoor_node.neighbours = [GuestDoor]
#fMontyDoor_node.neighbours = [GuestDoor, ActualDoor, MontyDoor]

import ipdb; ipdb.set_trace()

graph = FactorGraph([
        ActualDoor,
        GuestDoor,
        MontyDoor,
        fActualDoor_node,
        fGuestDoor_node,
        fMontyDoor_node])

def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)


if __name__ == '__main__':
    graph.propagate()

    print marg(ActualDoor, 'A')
    print marg(ActualDoor, 'B')
    print marg(ActualDoor, 'C')

    graph.reset()
    add_evidence(GuestDoor, 'A')
    add_evidence(MontyDoor, 'B')
    graph.propagate()
    normalizer = GuestDoor.marginal('A')
    print marg(ActualDoor, 'A', normalizer)
    print marg(ActualDoor, 'B', normalizer)
    print marg(ActualDoor, 'C', normalizer)



