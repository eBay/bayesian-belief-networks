'''See if we can create an unidrected equivalent of the Monty BBN'''
from bayesian.bbn import *
from bayesian.utils import make_key, get_args
from bayesian.factor_graph import (
    build_graph as build_factor_graph)
from bayesian.bbn import (
    JoinTreeSepSetNode, JoinTreeCliqueNode)
from bayesian.graph import Clique
from bayesian.factor_graph import (
    make_product_func, make_not_sum_func, VariableNode,
    FactorNode, FactorGraph)
from bayesian.factor_graph import connect as fg_connect
from bayesian.exceptions import *

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


def f_u_prize_door(u_prize_door):
    return 1.0 / 3


def f_u_guest_door(u_guest_door):
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
            ns_func.original_args = ['u_monty_door']
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
        retval = f(*tuple(dummy_arg))
        return retval
    dummyized.argspec = [dummy_arg]
    dummyized.original_args = f.argspec

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


    #factor_graph_funcs = [node.func for node in jt.nodes \
    #                      if isinstance(node, JoinTreeCliqueNode)]
    # Also include the sepsets...
    factor_graph_funcs = [node.func for node in jt.nodes]

    # Now dummyize these...
    factor_graph_funcs = [dummyize(f) for f in factor_graph_funcs]
    # Now add the [PGM] func... what would its arg be???
    # I think it should be a not_sum func of NOT monty
    #sepset_func
    #factor_graph_funcs.append(jt.nodes[1].func)
    # Note that because we have dummyized all factors
    # the domains need to all be tuples...
    domains = dict(
        u_prize_door_u_monty_door_u_guest_door=list(xproduct(
            ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C'])),
        u_guest_door_u_monty_door_u_prize_door=list(xproduct(
            ['A', 'B', 'C'], ['A', 'B', 'C'], ['A', 'B', 'C'])),
        u_monty_door=list(xproduct(['A', 'B', 'C'])))
    fg = build_factor_graph(
        *factor_graph_funcs,
        domains=domains)
    return fg


def dispatch_query(ug, fg, **query_kwds):
    '''
    Dispatch queries on the Undirected
    Graphical Model to the Factor Graph
    Representation.
    To do this we need a mapping from
    the original ug factor arguments to
    the cluster arguments...
    Lets say the query has var_x in the UG observed.
    We need to now ensure that in the factor graph
    every cluster function that contains the variable
    var_x is also instantiated to this value.
    '''
    # First lets ensure that the instantiated variable
    # is actually in the model and that the
    # value is actually in the domain for that
    # functions argument.
    original_vars = [n.variable_name for n in ug.nodes]
    for k, v in query_kwds.items():
        if k not in original_vars:
            raise VariableNotInGraphError(k)
    # Now we need to find all factors in the
    # fg representation that contain the
    # variable var_x
    fg_factors = []
    fg_query = dict()
    for node in fg.factor_nodes():
        for k, v in query_kwds.items():
            if k in node.func.original_args:
                fg_factors.append(node)
                fg_query[k] = v


    print fg_factors
    print fg_query
    #import ipdb; ipdb.set_trace()
    fg.q(**fg_query)
    print 'Yes!!!!'


def build_monty_factor_graph_from_bbn(g):
    # These next 3 lines are from bbn.build_join_tree
    gu = make_undirected_copy(g)
    gm = make_moralized_copy(gu, g)
    # First we manually created the 'nodes'

    prize_door_node = UndirectedNode('prize_door')
    guest_door_node = UndirectedNode('guest_door')
    monty_door_node = UndirectedNode('monty_door')

    # Now attach the python functions to the nodes
    # Actually what we really need here is
    # the *potential* functions lets
    # adopt the convention of defining
    # these functions as p_*** to distinguish
    # from factor functions.

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

    monty_door_node.func.domains = dict(
            u_prize_door=['A', 'B', 'C'],
            u_guest_door=['A', 'B', 'C'],
            u_monty_door=['A', 'B', 'C'])


    # Set up the variable names, again
    # this will all be done automatically
    # eventually...
    prize_door_node.variable_name = 'u_prize_door'
    guest_door_node.variable_name = 'u_guest_door'
    monty_door_node.variable_name = 'u_monty_door'

    # And now set up the edges, remember that
    # we are creating the fully connected
    # graph as in gm above. It is not
    # possible to represent Monty as
    # a 3 variable undirected graph
    # that is NOT fully connected.
    # This is because the conditional for
    # Monty is a function of all 3
    # nodes and thus we need a clique of
    # all 3.
    prize_door_node.neighbours = [
        monty_door_node,
        guest_door_node]

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

    # Now how to proceed...
    cliques, elimination_ordering = triangulate(ug)
    print cliques
    print elimination_ordering
    jt = build_join_tree_from_ug(ug)

    # Now see page 400 and 401 of Bishop...
    # To get to a factor graph we
    # create variable nodes of all the
    # original variables and factor
    # nodes of the *CLUSTERS*
    # (Its unclear at this point whether
    # or not we also create factor nodes
    # out of the sepsets.)

    # Lets explicitely create the Factor
    # Graph variable nodes here
    fg_variable_nodes = []
    for node in ug.nodes:
        variable_node = VariableNode(node.variable_name)
        variable_node.domain = ['A', 'B', 'C']
        fg_variable_nodes.append(variable_node)


    # Now lets create the factor nodes out
    # of the junction tree cliques...
    fg_factor_nodes = []
    for clique in jt.nodes:
        # Note we will go and fix the function later for
        # now just set to unity.
        fg_factor_node = FactorNode(clique.name, lambda x: 1)
        fg_factor_nodes.append(fg_factor_node)

    # Now we can build the factor graph
    # for this particular case its easy
    # since all the variables are just
    # in one clique so we just connect
    # every VariableNode to the one FactorNode
    # and assign the potential func to
    # the factor node....
    for variable_node in fg_variable_nodes:
        fg_connect(variable_node, fg_factor_nodes[0])

    potential_func = make_product_func(
        [n.func for n in ug.nodes])
    potential_func.domains = dict(
            u_prize_door=['A', 'B', 'C'],
            u_guest_door=['A', 'B', 'C'],
            u_monty_door=['A', 'B', 'C'])

    fg_factor_nodes[0].func = potential_func

    # This should work now!

    fg = FactorGraph(fg_variable_nodes + fg_factor_nodes)

    # Yes! This works properly now. Seems like we
    # dont even need any of that query dispatch stuff...
    # Now to test it with several others....

    # Monty is very simple though...
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

    fg = build_monty_factor_graph_from_bbn(g)
    fg2 = g.convert_to_factor_graph()
    fg.q()
    import ipdb; ipdb.set_trace()
    fg2.q()
