'''See if we can create an unidrected equivalent of the Monty BBN'''
from bayesian.bbn import *
from bayesian.utils import make_key, get_args
from bayesian.factor_graph import (
    build_graph as build_factor_graph)
from bayesian.bbn import (
    JoinTreeSepSetNode, JoinTreeCliqueNode)
from bayesian.bbn import Clique
from bayesian.factor_graph import make_product_func, make_not_sum_func

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

As an undirected Graph the Monty Hall problem looks like
this:

           ----------          ----------
          |Prize Door|        |Guest Door|
           ----------          ----------
                |                   |
                |                   |
                +-----+      +------+
                      |      |
                     ----------
                    |Monty Door|
                     ----------
'''


# The factors are specified in
# exactly the same way as in the
# directed version

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
def f_u_prize_door(u_monty_door, u_prize_door):
    # How we specify these potential functions
    # is interesting...
    if u_prize_door == u_monty_door:
        # Since monty will never choose
        # the same door as the prize
        return 0
    # Now given that this is *only*
    # the potential func of these two
    # vars we return 0.5...
    return 1.0 / 2.0


def f_u_guest_door(u_guest_door, u_monty_door):
    if u_guest_door == u_monty_door:
        return 0
    return 1.0 / 2.0


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
    cliques, elimination_ordering = triangulate(
        ug, clique_priority_func)

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


def build_potential_functions(ug, jt):
    '''
    Given a Junction Tree built from
    a ug, and the original ug, we want to
    assign potential functions to each clique
    that are build from the original
    factor functions.'''
    for node in jt.nodes:
        if isinstance(node, JoinTreeSepSetNode):
            import ipdb; ipdb.set_trace()
            # For the sepsets we need a not sum
            # to exclude all variables that are
            # not in the sepset. See page 10 of
            # H & D
            # For now I will just
            # manually take the factor funcs
            # from the other nodes.
            # (Actually I think we just
            # have to choose one neighbour..)
            ns_func = make_not_sum_func(
                jt.nodes[0].clique.node.func, 'u_monty_door')
            # Now we also have to get rid of Prize door...
            #ns_func = make_not_sum_func(
            #    ns_func, 'u_prize_door')
            node.func = ns_func
            ns_func('A')
            continue
        funcs = []
        for clique_member in list(node.clique.nodes):
            print get_args(clique_member.func)
            funcs.append(clique_member.func)
        # Now create the closure over these functions...
        potential_func = make_product_func(funcs)
        node.func = potential_func
    # Also return the functions created

def dummyize(f):
    '''Given a potential function
    which is a product function of
    multiple variables in the original
    graph, convert it to a function
    which takes only one parameter
    so that it can be used as a 'root'
    factor node in a factor graph.
    '''
    args = get_args(f)
    dummy_arg = '_'.join(args)
    def dummyized(dummy_arg):
        # lets require that the single
        # arg is a list of the
        # underlying args
        retval = f(*dummy_arg)
        return retval
    dummyized.argspec = [dummy_arg]
    return dummyized


def build_factor_graph_from_jt(jt):
    '''These should be equivalent to the
    factor graph constructed explicitely
    from 3 cliques and sepsets
    which I still need to work out...
    What is unclear is whether sepset
    nodes also have potential functions...
    '''
    # First we will construct the variable nodes...
    # What if we just called the build_factor graph????
    # We will assume that clique nodes are
    # turned into variable nodes
    # and that sepset nodes are turned into
    # factor nodes. If this does not
    # work we will try the other way around!
    #
    # For the undirected Monty case we
    # have the following clusters:
    # [ Guest, Monty ] ----- [ Monty, Prize]
    # Including the 'SepSet' this looks like:
    # [ Guest, Monty ] --{Monty}--- [ Monty, Prize]
    # If we consider this as a simple
    # two variable graph with dummy variabels:
    # GM and MP then the factor graph could look like this:
    # Variable Nodes are indicated by () and
    # factors by []
    #
    # [GM] --- (GM) --- [PGM] --- (MP) --- [MP]
    #                     |
    #                   (PGM)
    # I think???
    # So factor [GM] corresponds to the potential
    # function made up of f(M)xf(G) and takes
    # the single 'dummy' variable GM.
    # The question is how do we initialize these?
    # Lets consider the domains for each of these
    # 'dummy' variables.
    # For GM we have ['AA', 'AB', 'AC',
    #                 'BA', 'BB', 'BC'
    #                 'CA', 'CB', 'CC']
    # Likewise for MP
    # What about PGM? What do we dummyize for
    # that? Maybe this is the sepset node
    # potential? by Huglin this is just unity...
    #
    #
    # Now how do we resolve the potential functions into
    # actual calls???
    # Strictly [GM]
    # Ok a simpler alternative....
    # We already have two clusters and one sepset node
    # what if we just build the factor grapj like this:
    #
    # (GM) --- [PGM] --- (MP)
    #            |
    #          (PGM)???
    # Do we need (PGM???)
    # if we conside [PGM] to in fact be
    # [GM.MP] then we dont?
    # Now for any value (P, G, M) of
    # GM and MP we can compute the marginals
    # by using the potential funcs on GM
    # as the priors...
    # So what exactly is the potentual func
    # on [PGM]????
    #


    factor_graph_funcs = [node.func for node in jt.nodes \
                          if isinstance(node, JoinTreeCliqueNode)]
    # Now dummyize these...
    factor_graph_funcs = [dummyize(f) for f in factor_graph_funcs]
    # Now add the [PGM] func... what would its arg be???
    # I think it should be a not_sum func of NOT monty
    factor_graph_funcs.append(jt.nodes[1].func)
    domains = dict(
        u_prize_door_u_monty_door_u_guest_door=xproduct(
            ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C']),
        u_guest_door_u_monty_door_u_prize_door=xproduct(
            ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C']),
        u_monty_door=['A', 'B', 'C'])
    import ipdb; ipdb.set_trace()
    fg = build_factor_graph(
        *factor_graph_funcs,
        domains=domains)
    return fg


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

    # We also need to have the domains of
    # each factor func:
    prize_door_node.func.domains = dict(
            u_prize_door=['A', 'B', 'C'],
            u_guest_door=['A', 'B', 'C'],
            u_monty_door=['A', 'B', 'C'])

    guest_door_node.func.domains = dict(
            u_prize_door=['A', 'B', 'C'],
            u_guest_door=['A', 'B', 'C'],
            u_monty_door=['A', 'B', 'C'])

    monty_door_node.func .domains = dict(
            u_prize_door=['A', 'B', 'C'],
            u_guest_door=['A', 'B', 'C'],
            u_monty_door=['A', 'B', 'C'])


    # Set up the variable names, again
    # this will all be done automatically
    # eventually...
    prize_door_node.variable_name = 'u_prize_door'
    guest_door_node.variable_name = 'u_guest_door'
    monty_door_node.variable_name = 'u_monty_door'

    # And now set up the edges
    prize_door_node.neighbours = [
        monty_door_node ]

    guest_door_node.neighbours = [
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

    # Now how to proceed...
    import ipdb; ipdb.set_trace()
    cliques, elimination_ordering = triangulate(ug)
    print cliques
    print elimination_ordering
    jt = build_join_tree_from_ug(ug)
    build_potential_functions(ug, jt)


    # Now we have the jt which is essentially a factor graph...
    # The only difference seems to be that the jt as originally
    # coded works by constructing potential truth tables,
    # while the factor graph just uses make_product_func.
    # Still need to test more to ensure that they are
    # equivalent. For now I will build a function
    # that can take a junction tree with assigned
    # potential functions and convert it into
    # a factor graph...
    fg = build_factor_graph_from_jt(jt)

    # Ok doing more research I am certain this is the way
    # to go:
    # Step 1: Convert the undirected graph to
    #         a cluster graph
    # Step 2: Convert cluster graph to spanning tree
    #         either using chords or using variable elimination
    #         (I think chords are better see Rameshs' presentation)
    # Step 3: Now each sepset and each clique correspond to
    #         nodes in the factor graph
    # Still unclear whether the sepsets themselves become
    # the factor nodes or whether the entire junction
    # tree is treated as input to the factor graph....
    # given there are only 3 nodes in the jt the latter
    # seems likely...
    # From http://ai.stanford.edu/~paskin/gm-short-course/lec3.pdf
    # it seems that the sepsets ARE the factor nodes
