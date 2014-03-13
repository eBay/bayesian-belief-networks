'''See if we can create an unidrected equivalent of the Monty BBN'''
from bayesian.bbn import *
from bayesian.utils import make_key
from bayesian.factor_graph import build_graph as build_factor_graph

from itertools import product as xproduct
'''
We will start by copying the BBN version.
This example shows how we can created undirected
graphical models without much more explicit support.
Essentially all the machinary necessary for undirected
models is already present in the support for
directed graphs.
We will eventually add explicit support for
undirected graphs which will male building models
much easier.
'''


def f_prize_door(prize_door):
    return 1.0 / 3


def f_guest_door(guest_door):
    return 1.0 / 3


def f_monty_door(prize_door, guest_door, monty_door):
    if prize_door == guest_door:
        if prize_door == monty_door:
            return 0
        else:
            return 0.5
    elif prize_door == monty_door:
        return 0
    elif guest_door == monty_door:
        return 0
    return 1


# Now lets try and define the nodes of the
# Notice that because we have additional edges
# in the undirected graph, each conditional
# probability distribution has additional
# variables.
def f_u_prize_door(u_guest_door, u_monty_door, u_prize_door):
    return 1.0 / 3


def f_u_guest_door(u_guest_door, u_monty_door, u_prize_door):
    return 1.0 / 3


# Essentially these will all actually be the same as the BBN
# except that we have extra edges.
def f_u_monty_door(u_prize_door, u_guest_door, u_monty_door):
    if u_prize_door == u_guest_door:
        if u_prize_door == u_monty_door:
            return 0
        else:
            return 0.5
    elif u_prize_door == u_monty_door:
        return 0
    elif u_guest_door == u_monty_door:
        return 0
    return 1


def build_join_tree_from_ug(ug, clique_priority_func=priority_func):
    # This is essentially the same as the build_join_tree
    # from BBNs except that we already have a ug so
    # we dont need to first convert the dag to a moralized
    # ug

    # Now we create a copy of the undirected graph
    # and connect all pairs of parents that are
    # not already parents called the 'moralized' graph.
    #gm = make_moralized_copy(gu, dag)
    # Its unclear if we need to do this for the
    # ug...

    # Now we triangulate the moralized graph...
    cliques, elimination_ordering = triangulate(ug, clique_priority_func)

    # And at this point the rest should be the same....

    # Now we initialize the forest and sepsets
    # Its unclear from Darwiche Huang whether we
    # track a sepset for each tree or whether its
    # a global list????
    # We will implement the Join Tree as an undirected
    # graph for now...

    # First initialize a set of graphs where
    # each graph initially consists of just one
    # node for the clique. As these graphs get
    # populated with sepsets connecting them
    # they should collapse into a single tree.
    forest = set()
    for clique in cliques:
        jt_node = JoinTreeCliqueNode(clique)
        # Track a reference from the clique
        # itself to the node, this will be
        # handy later... (alternately we
        # could just collapse clique and clique
        # node into one class...
        clique.node = jt_node
        tree = JoinTree([jt_node])
        forest.add(tree)

    # Initialize the SepSets
    S = set()  # track the sepsets
    for X, Y in combinations(cliques, 2):
        if X.nodes.intersection(Y.nodes):
            S.add(SepSet(X, Y))
    sepsets_inserted = 0
    while sepsets_inserted < (len(cliques) - 1):
        # Adding in name to make this sort deterministic
        deco = [(s, -1 * s.mass, s.cost, s.__repr__()) for s in S]
        deco.sort(key=lambda x: x[1:])
        candidate_sepset = deco[0][0]
        for candidate_sepset, _, _, _ in deco:
            if candidate_sepset.insertable(forest):
                # Insert into forest and remove the sepset
                candidate_sepset.insert(forest)
                S.remove(candidate_sepset)
                sepsets_inserted += 1
                break

    assert len(forest) == 1
    jt = list(forest)[0]
    return jt


def get_Z(
        fg, observed_vars, observed_vals, unobserved_vars,
        potential_func):
    # Basically this is the normalization
    # by the Z partition function
    # this is slightly different to
    # the marginlization in the original
    # factor graph.
    # For now we will do this seperately
    totals = defaultdict(float)
    #import ipdb; ipdb.set_trace()
    for guest_door in ['A', 'B', 'C']:
        res = fg.query(u_guest_door=guest_door)
        for k, v in res.iteritems():
            totals[k] += v
            totals[k[0]] += v
    return totals


if __name__ == '__main__':

    g = build_bbn(
        f_prize_door,
        f_guest_door,
        f_monty_door,
        domains=dict(
            prize_door=['A', 'B', 'C'],
            guest_door=['A', 'B', 'C'],
            monty_door=['A', 'B', 'C']))
    # Initial Marginals without any knowledge.
    # Observe that the likelihood for
    # all three doors is 1/3.
    print 'Initial Marginal Probabilities:'
    g.q()

    # Now we will initially manually
    # create the Undirected Graph until we have a
    # build_graph method...
    # First we manually created the 'nodes'
    prize_door_node = UndirectedNode('prize_door')
    guest_door_node = UndirectedNode('guest_door')
    monty_door_node = UndirectedNode('monty_door')

    # Now attach the python functions to the nodes
    prize_door_node.func = f_u_prize_door
    guest_door_node.func = f_u_guest_door
    monty_door_node.func = f_u_monty_door

    # Set up the variable names, again
    # this will all be done automatically
    # eventually...
    prize_door_node.variable_name = 'u_prize_door'
    guest_door_node.variable_name = 'u_guest_door'
    monty_door_node.variable_name = 'u_monty_door'

    # And now set up the edges
    prize_door_node.neighbours = [
        guest_door_node,
        monty_door_node ]

    guest_door_node.neighbours = [
        prize_door_node,
        monty_door_node ]

    monty_door_node.neighbours = [
        guest_door_node,
        prize_door_node ]

    # After that we will also have to build the
    # junction tree.... This is where it differs...
    ug = UndirectedGraph(
        [prize_door_node,
         guest_door_node,
         monty_door_node])

    # At this point we essentially have a moralized
    # graph ie we are at step 3 in the build_join_tree
    # process

    #cliques, elimination_ordering = triangulate(ug)

    # mmmm from here its unclear whether to go
    # down the factor graph path or to
    # go down the join tree path

    # Lets try the factor graph approach first...
    # This should be now equivalent to
    # creating a variable node for each variable
    # and a factor node out of the single clique...
    # The only problem is the potential function...

    # Lets try and manually create it...

    def clique_potential_func(u_prize_door, u_guest_door, u_monty_door):
        # Basically the clique potentials are the
        # product of the individual factors...
        # Note that this will only work now since
        # I have taken care to keep the parameters in
        # the same order. Normally the build_graph will
        # see to this...
        res = 1.0
        for node in ug.nodes:
            res *= node.func(u_prize_door, u_guest_door, u_monty_door)
        return res

    # Now for the single clique we need to attach this potential function...
    # Or do we need to build the join tree too?????
    # I think to guarantee that the factor graph is built out
    # of a tree we need to build the join tree...
    jt = build_join_tree_from_ug(ug)

    # Now since the inference method on
    # the join tree actually works by
    # populating truth tables we want to rather
    # use the factor graph implementation from here on...

    # This means that after we have an inference
    # result we still need to marginalize out the
    # non-observed variables, we can probably again
    # re-use code from factor graph implementation
    # for this...
    fg = build_factor_graph(
        clique_potential_func,
        domains = dict(
            u_prize_door = ['A', 'B', 'C'],
            u_guest_door = ['A', 'B', 'C'],
            u_monty_door = ['A', 'B', 'C']))
