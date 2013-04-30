'''The Monty Hall Problem Modelled as a Bayesian Belief Network'''
from __future__ import division
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

fActualDoor.domains = dict(
    ActualDoor=['A', 'B', 'C'])


def fGuestDoor(GuestDoor):
    return 1.0 / 3

fGuestDoor.domains = dict(
    GuestDoor=['A', 'B', 'C'])


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

fMontyDoor.domains = dict(
    ActualDoor=['A', 'B', 'C'],
    GuestDoor=['A', 'B', 'C'],
    MontyDoor=['A', 'B', 'C'])

# Build the network

fActualDoor_node = FactorNode('fActualDoor', fActualDoor)
fGuestDoor_node = FactorNode('fGuestDoor', fGuestDoor)
fMontyDoor_node = FactorNode('fMontyDoor', fMontyDoor)

ActualDoor = VariableNode('ActualDoor')
GuestDoor = VariableNode('GuestDoor')
MontyDoor = VariableNode('MontyDoor')

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

def discover_sample_ordering(graph):
    '''
    Try to get the order of variable nodes
    for sampling. This would be easier in 
    the underlying BBN but lets try on
    the factor graph.
    '''
    iterations = 0
    ordering = []
    accounted_for = set()
    variable_nodes = [n for n in graph.nodes if isinstance(n, VariableNode)]
    factor_nodes = [n for n in graph.nodes if isinstance(n, FactorNode)]
    required = len([n for n in graph.nodes if isinstance(n, VariableNode)])
    # Firstly any leaf factor nodes will
    # by definition only have one variable
    # node connection, therefore these
    # variables can be set first.
    for node in graph.get_leaves():
        if isinstance(node, FactorNode):
            ordering.append(node.neighbours[0].name)
            accounted_for.add(node.neighbours[0].name)

    # Now for each factor node whose variables
    # all but one are already in the ordering,
    # we can add that one variable. This is
    # actuall
    import ipdb; ipdb.set_trace()
    while len(ordering) < required:
        for node in factor_nodes:
            args = set(get_args(node.func))
            new_args = args.difference(accounted_for)
            if len(new_args) == 1:
                ordering += list(new_args)
                accounted_for = set(ordering)
    return ordering


def get_sample():
    sample = []
    # ActualDoor
    r = random.random()
    total = 0
    r = random.random()
    for val in ['A', 'B', 'C']:
        actual_door = VariableNode('ActualDoor')
        actual_door.value = val
        total += fActualDoor(actual_door)
        if total > r:
            break
    sample.append(val)
    # GuestDoor
    total = 0
    r = random.random()
    for val in ['A', 'B', 'C']:
        guest_door = VariableNode('GuestDoor')
        guest_door.value = val
        total += fGuestDoor(guest_door)
        if total > r:
            break
    sample.append(val)
    # MontyDoor
    total = 0
    r = random.random()
    for val in ['A', 'B', 'C']:
        monty_door = VariableNode('MontyDoor')
        monty_door.value = val
        total += fMontyDoor(actual_door, guest_door, monty_door)
        if total > r:
            break
    sample.append(val)
    return sample

    

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

    # Now lets test the gibbs sampler...
    #sampler = graph.gibbs_sample()
    for i in range(0, 20):
        sample = get_sample()
        print sample
        print '=================='
    
    # Now suppose we want to estimate the
    # likelihood of C given that
    # guest has chosen A and Monty has chosen
    # B. 
    # We will take only those gibbs samples which
    # match the pattern of GuestDoor:A and MontyDoor:B
    # and then look at the percentage of times that
    # the Actual is set to C....
    valid_samples = []
    c_count = 0
    a_count = 0
    total_count = 0
    #import ipdb; ipdb.set_trace()
    for i in range(0, 100):
        sample = get_sample()
        # only use every 100th sample...
        #if i % 100 == 0:
        #    continue
        #for k, v in sample.items():
        #    print k, v.value
        print sample
        print '=================='
        if sample[1] == 'A':
            if sample[2] == 'B':
                print sample
                print '=================='
                total_count += 1
                if sample[0] == 'C':
                    c_count += 1
                if sample[0] == 'A':
                    a_count += 1
                    
    print a_count, c_count, total_count, float(c_count) / total_count
    print discover_sample_ordering(graph)
