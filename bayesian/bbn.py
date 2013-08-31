from __future__ import division
'''Data Structures to represent a BBN as a DAG.'''
import sys
import copy
import heapq

from StringIO import StringIO
from itertools import combinations

from bayesian.utils import get_args

class Node(object):

    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents[:]
        self.children = children[:]

    def __repr__(self):
        return '<Node %s>' % self.name


class BBNNode(Node):

    def __init__(self, factor):
        super(BBNNode, self).__init__(factor.__name__)
        self.func = factor
        self.argspec = get_args(factor)

    def __repr__(self):
        return '<BBNNode %s (%s)>' % (
            self.name,
            self.argspec)


class UndirectedNode(object):

    def __init__(self, name, neighbours=[]):
        self.name = name
        self.neighbours = neighbours[:]


    def __repr__(self):
        return '<Node %s>' % self.name


class Graph(object):

    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())



class UndirectedGraph(object):

    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name


    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edges.add(tuple(sorted(edge)))
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()



class BBN(Graph):
    '''A Directed Acyclic Graph'''

    def __init__(self, nodes, name=None):
        self.nodes = nodes

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('digraph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for child in node.children:
                edge = (node.name, child.name)
                edges.add(edge)
        for source, target in edges:
            fh.write('  %s -> %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def moralize(self):
        '''Marry the unmarried parents'''
        pass
        #for node in self.nodes:
        #    unmarried = set


def connect(parent, child):
    '''
    Make an edge between a parent
    node and a child node.
    a - parent
    b - child
    '''
    parent.children.append(child)
    child.parents.append(parent)


def get_original_factors(factors):
    '''
    For a set of factors, we want to
    get a mapping of the variables to
    the factor which first introduces the
    variable to the set.
    To do this without enforcing a special
    naming convention such as 'f_' for factors,
    or a special ordering, such as the last
    argument is always the new variable,
    we will have to discover the 'original'
    factor that introduces the variable
    iteratively.
    '''
    original_factors = dict()
    while len(original_factors) < len(factors):
        for factor in factors:
            args = get_args(factor)
            unaccounted_args = [a for a in args if a not in original_factors]
            if len(unaccounted_args) == 1:
                original_factors[unaccounted_args[0]] = factor
    return original_factors


def build_bbn(*args, **kwds):
    '''Builds a BBN Graph from
    a list of functions and domains'''
    variables = set()
    domains = kwds.get('domains', {})
    name = kwds.get('name')
    variable_nodes = dict()
    factor_nodes = dict()
    if isinstance(args[0], list):
        # Assume the functions were all
        # passed in a list in the first
        # argument. This makes it possible
        # to build very large graphs with
        # more than 255 functions.
        args = args[0]


    for factor in args:
        factor_args = get_args(factor)
        variables.update(factor_args)
        bbn_node = BBNNode(factor)
        factor_nodes[factor.__name__] = bbn_node

    # Now lets create the connections
    # To do this we need to find the
    # factor node representing the variables
    # in a child factors argument and connect
    # it to the child node

    # Note that calling original_factors
    # here can break build_bbn if the
    # factors do not correctly represent
    # a BBN.
    original_factors = get_original_factors(factor_nodes.values())
    for factor_node in factor_nodes.values():
        factor_args = get_args(factor_node)
        parents = [original_factors[arg] for arg in factor_args if original_factors[arg] != factor_node]
        for parent in parents:
            connect(parent, factor_node)
    bbn = BBN(factor_nodes.values(), name=name)
    return bbn


def make_undirected_copy(dag):
    '''Returns an exact copy of the dag
    except that direction of edges are dropped.'''
    nodes = dict()
    for node in dag.nodes:
        undirected_node = UndirectedNode(
            name = node.name)
        undirected_node.func = node.func
        undirected_node.argspec = node.argspec
        nodes[node.name] = undirected_node
    # Now we need to traverse the original
    # nodes once more and add any parents
    # or children as neighbours.
    for node in dag.nodes:
        for parent in node.parents:
            nodes[node.name].neighbours.append(
                nodes[parent.name])
            nodes[parent.name].neighbours.append(
                nodes[node.name])

    g = UndirectedGraph(nodes.values())
    return g


def make_moralized_copy(gu, dag):
    '''gu is an undirected graph being
    a copy of dag.'''
    gm = copy.deepcopy(gu)
    gm_nodes = dict(
        [(node.name, node) for node in gm.nodes])
    for node in dag.nodes:
        for parent_1, parent_2 in combinations(
                node.parents, 2):
            if gm_nodes[parent_1.name] not in \
               gm_nodes[parent_2.name].neighbours:
                gm_nodes[parent_2.name].neighbours.append(
                    gm_nodes[parent_1.name])
            if gm_nodes[parent_2.name] not in \
               gm_nodes[parent_1.name].neighbours:
                gm_nodes[parent_1.name].neighbours.append(
                    gm_nodes[parent_2.name])
    return gm


def priority_func(node):
    '''Specify the rules for computing
    priority of a node. See Harwiche and Wang pg 12.
    '''
    # We need to calculate the number of edges
    # that would be added.
    # For each node, we need to connect all
    # of the nodes in itself and its neighbours
    # (the "cluster") which are not already
    # connected. This will be the primary
    # key value in the heap.
    # We need to fix the secondary key, right
    # now its just 2 (because mostly the variables
    # will be discrete binary)
    introduced_arcs = 0
    cluster = [node] + node.neighbours
    for node_a, node_b in combinations(cluster, 2):
        if node_a not in node_b.neighbours:
            assert node_b not in node_a.neighbours
            introduced_arcs += 1
    return [introduced_arcs, 2] # TODO: Fix this to look at domains


def construct_priority_queue(nodes, priority_func=priority_func):
    pq = []
    for node_name, node in nodes.iteritems():
        entry = priority_func(node) + [node.name]
        heapq.heappush(pq, entry)
    return pq


def record_cliques(cliques, cluster):
    '''We only want to save the cluster
    if it is not a subset of any clique
    already saved.
    Argument cluster must be a set'''
    if any([cluster.issubset(c) for c in cliques]):
        return
    cliques.append(cluster)


def triangulate(gm, priority_func=priority_func):
    '''Triangulate the moralized Graph. (in Place)
    and return the cliques of the triangulated
    graph as well as the elimination ordering.'''

    # First we will make a copy of gm...
    gm_ = copy.deepcopy(gm)

    # Now we will construct a priority q using
    # the standard library heapq module.
    # See docs for example of priority q tie
    # breaking. We will use a 3 element list
    # with entries as follows:
    #   - Number of edges added if V were selected
    #   - Weight of V (or cluster)
    #   - Pointer to node in gm_
    # Note that its unclear from Huang and Darwiche
    # what is meant by the "number of values of V"
    gmnodes = dict([(node.name, node) for node in gm.nodes])
    elimination_ordering = []
    cliques = []
    while True:
        gm_nodes = dict([(node.name, node) for node in gm_.nodes])
        if not gm_nodes:
            break
        pq = construct_priority_queue(gm_nodes, priority_func)
        # Now we select the first node in
        # the priority q and any arcs that
        # should be added in order to fully connect
        # the cluster should be added to both
        # gm and gm_
        v = gm_nodes[pq[0][2]]
        cluster = [v] + v.neighbours
        for node_a, node_b in combinations(cluster, 2):
            if node_a not in node_b.neighbours:
                print 'Adding edhe from %s to %s' % (
                    node_a.name, node_b.name)
                node_b.neighbours.append(node_a)
                node_a.neighbours.append(node_b)
                # Now also add this new arc to gm...
                gmnodes[node_b.name].neighbours.append(
                    gmnodes[node_a.name])
                gmnodes[node_a.name].neighbours.append(
                    gmnodes[node_b.name])
        gmcluster = set([gmnodes[c.name] for c in cluster])
        record_cliques(cliques, gmcluster)
        # Now we need to remove v from gm_...
        # This means we also have to remove it from all
        # of its neighbours that reference it...
        for neighbour in v.neighbours:
            neighbour.neighbours.remove(v)
        gm_.nodes.remove(v)
        elimination_ordering.append(v.name)
    return cliques, elimination_ordering





def build_join_tree(dag):

    # First we will create an undirected copy
    # of the dag
    gu = make_undirected_copy(dag)

    # Now we create a copy of the undirected graph
    # and connect all pairs of parents that are
    # not already parents called the 'moralized' graph.
    gm = make_moralized_copy(gu, dag)

    # Now we triangulate the moralized graph...
    elimination_ordering = triangulate(gm)
