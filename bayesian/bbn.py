from __future__ import division
'''Data Structures to represent a BBN as a DAG.'''
import sys
import copy
import heapq

from StringIO import StringIO
from itertools import combinations, product
from collections import defaultdict

from bayesian.utils import get_args, named_base_type_factory

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
        return '<UndirectedNode %s>' % self.name


class JoinTreeCliqueNode(UndirectedNode):

    def __init__(self, clique):
        super(JoinTreeCliqueNode, self).__init__(
            clique.__repr__())
        self.clique = clique

    @property
    def variable_names(self):
        '''Return the set of variable names
        that this clique represents'''
        var_names = []
        for node in self.clique.nodes:
            var_names.append(node.name[2:])
        return set(var_names)

    @property
    def neighbouring_cliques(self):
        '''Return the neighbouring cliques
        this is used during the propagation algorithm.

        '''
        neighbours = set()
        for sepset_node in self.neighbours:
            # All *immediate* neighbours will
            # be sepset nodes, its the neighbours of
            # these sepsets that form the nodes
            # clique neighbours (excluding itself)
            for clique_node in sepset_node.neighbours:
                if clique_node is not self:
                    neighbours.add(clique_node)
        return neighbours

    def __repr__(self):
        return '<JoinTreeCliqueNode: %s>' % self.clique



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


    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())


class BBN(Graph):
    '''A Directed Acyclic Graph'''

    def __init__(self, nodes, name=None, domains={}):
        self.nodes = nodes
        self.domains = domains

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


class JoinTree(UndirectedGraph):

    def __init__(self, nodes, name=None):
        super(JoinTree, self).__init__(
            nodes, name)

    @property
    def sepset_nodes(self):
        return [n for n in self.nodes if isinstance(n, JoinTreeSepSetNode)]

    @property
    def clique_nodes(self):
        return [n for n in self.nodes if isinstance(n, JoinTreeCliqueNode)]

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            if isinstance(node, JoinTreeSepSetNode):
                fh.write('  %s [ shape="box" color="blue"];\n' % node.name)
            else:
                fh.write('  %s [ shape="ellipse" color="red"];\n' % node.name)
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edges.add(tuple(sorted(edge)))
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def initialize_potentials(self, assignments, bbn):

        # Step 1, assign 1 to each cluster and sepset
        for node in self.sepset_nodes:
            node.potential = 1

        # Step 2 for each v assign it a parent cluster...

        # now for each assignment we want to
        # generate a truth-table from the
        # values of the bbn truth-tables that are
        # assigned to the clusters...

        for clique, bbn_nodes in assignments.iteritems():
            tt = dict()
            vals = []
            variables = list(clique.variable_names)
            for variable in variables:
                domain = bbn.domains.get(variable, [True, False])
                vals.append(list(product(variable, domain)))
            permutations = product(*vals)
            for permutation in permutations:
                argvals = dict(permutation)
                potential = 1
                for bbn_node in bbn_nodes:
                    arg_list = []
                    for arg_name in get_args(bbn_node.func):
                        arg = Binding(
                            arg_name,
                            argvals[arg_name])
                        arg_list.append(arg)
                    potential *= bbn_node.func(*arg_list)
                tt[permutation] = potential
            clique.potential_tt = tt

    def assign_clusters(self, bbn):
        # This is unclear why in the H&D example A does not
        # get assigned the parent ACE.
        # Perhaps its because A doesnt have a conditional
        # probability table (ie it has no parents in the BBN)
        # An alternate way to think of step 2 could then
        # be for each *CPT* assign it to a cluster and then
        # build the potential truth table from that.
        # We will proceed this way and see what transpires
        # since we have the factor graph version with sampling
        # of the Huang Darwiche BBN we can compare the
        # marginals.

        assignments_by_family = dict()
        assignments_by_clique = defaultdict(list)
        assigned = set()
        for node in bbn.nodes:
            args = get_args(node.func)
            if len(args) == 1:
                # If the func has only 1 arg
                # it means that it does not
                # specify a conditional probability
                # This is where H&D is a bit vague
                # but it seems to imply that we
                # do not assign it to any
                # clique
                continue
            # Now we need to find a cluster that
            # is a superset of the Family(v)
            # Family(v) is defined by D&H to
            # be the union of v and parents(v)
            family = set(args)
            # At this point we need to know which *variable*
            # a BBN node represents. Up to now we have
            # not *explicitely* specified this, however
            # we have been following some conventions
            # so we could just use this convention for
            # now. Need to come back to this to
            # perhaps establish the variable at
            # build bbn time...
            containing_cliques = [clique_node for clique_node in
                                  self.clique_nodes if
                                  clique_node.variable_names.issuperset(family)]
            assert len(containing_cliques) >= 1
            for clique in containing_cliques:
                if node in assigned:
                    # Make sure we assign all original
                    # PMFs only once each
                    continue
                assignments_by_clique[clique].append(node)
                assigned.add(node)
            assignments_by_family[tuple(family)] = containing_cliques
        return assignments_by_clique

    def propagate(self, starting_clique=None):
        '''Refer to H&D pg. 20'''

        # Step 1 is to choose an arbitrary clique cluster
        # as starting cluster
        if starting_clique is None:
            starting_clique = self.clique_nodes[0]

        # Step 2: Unmark all clusters, call collect_evidence(X)
        for node in self.clique_nodes:
            node.marked = False
        self.collect_evidence(sender=starting_clique)

        # Step 3: Unmark all clusters, call distribute_evidence(X)
        for node in self.clique_nodes:
            self.marked = False
        self.distribute_evidence(starting_clique)

    def collect_evidence(self, sender=None, receiver=None):

        # Step 1, Mark X

        sender.marked = True

        # Step 2, call collect_evidence on Xs unmarked
        # neighbouring clusters.
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                self.collect_evidence(
                    sender=neighbouring_clique,
                    receiver=sender)
        # Step 3, pass message from sender to receiver
        if receiver is not None:
            print 'm from %s ----> %s' % (
                sender, receiver)



    def distribute_evidence(self, clique_node):
        pass


class Clique(object):

    def __init__(self, cluster):
        self.nodes = cluster

    def __repr__(self):
        vars = sorted([n.name[2:].upper() for n in self.nodes])
        return 'Clique_%s' % ''.join(vars)

class SepSet(object):

    def __init__(self, X, Y):
        '''X and Y are cliques represented as sets.'''
        #self.cliques = [X, Y]
        self.X = X
        self.Y = Y
        self.label = X.nodes.intersection(Y.nodes)

    @property
    def mass(self):
        return len(self.label)

    @property
    def cost(self):
        '''Since cost is used as a tie-breaker
        and is an optimization for inference time
        we will punt on it for now. Instead we
        will just use the assumption that all
        variables in X and Y are binary and thus
        use a weight of 2.
        TODO: come back to this and compute
        actual weights
        '''
        return 2 ** len(self.X.nodes) + 2 ** len(self.Y.nodes)

    def insertable(self, forest):
        '''A sepset can only be inserted
        into the JT if the cliques it
        separates are NOT already on
        the same tree.
        NOTE: For efficiency we should
        add an index that indexes cliques
        into the trees in the forest.'''
        X_trees = [t for t in forest if self.X in \
                   [n.clique for n in t.clique_nodes]]
        Y_trees = [t for t in forest if self.Y in \
                   [n.clique for n in t.clique_nodes]]
        assert len(X_trees) == 1
        assert len(Y_trees) == 1
        if X_trees[0] is not Y_trees[0]:
            return True
        return False

    def insert(self, forest):
        '''Inserting this sepset into
        a forest, providing the two
        cliques are in different trees,
        means that effectively we are
        collapsing the two trees into
        one. We will explicitely perform
        this collapse by adding the
        sepset node into the tree
        and adding edges between itself
        and its clique node neighbours.
        Finally we must remove the
        second tree from the forest
        as it is now joined to the
        first.
        '''
        X_tree = [t for t in forest if self.X in \
                  [n.clique for n in t.clique_nodes]][0]
        Y_tree = [t for t in forest if self.Y in \
                  [n.clique for n in t.clique_nodes]][0]

        # Now create and insert a sepset node into the Xtree
        ss_node = JoinTreeSepSetNode(self, self)
        X_tree.nodes.append(ss_node)

        # And connect them
        self.X.node.neighbours.append(ss_node)
        ss_node.neighbours.append(self.X.node)

        # Now lets keep the X_tree and drop the Y_tree
        # this means we need to copy all the nodes
        # in the Y_tree that are not already in the X_tree
        for node in Y_tree.nodes:
            if node in X_tree.nodes:
                continue
            X_tree.nodes.append(node)

        # Now connect the sepset node to the
        # Y_node (now residing in the X_tree)
        self.Y.node.neighbours.append(ss_node)
        ss_node.neighbours.append(self.Y.node)

        # And finally we must remove the Y_tree from
        # the forest...
        forest.remove(Y_tree)


    def __repr__(self):
        return 'SepSet_%s' % ''.join([x.name[2:].upper() for x in list(self.label)])


class JoinTreeSepSetNode(UndirectedNode):

    def __init__(self, name, sepset):
        super(JoinTreeSepSetNode, self).__init__(name)
        self.sepset = sepset

    def __repr__(self):
        return '<JoinTreeSepSetNode: %s>' % self.sepset

class Binding(object):

    def __init__(self, name, value):
        self.name = name
        self.value = value


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
    if any([cluster.issubset(c.nodes) for c in cliques]):
        return
    cliques.append(Clique(cluster))


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





def build_join_tree(dag, clique_priority_func=priority_func):

    # First we will create an undirected copy
    # of the dag
    gu = make_undirected_copy(dag)

    # Now we create a copy of the undirected graph
    # and connect all pairs of parents that are
    # not already parents called the 'moralized' graph.
    gm = make_moralized_copy(gu, dag)

    # Now we triangulate the moralized graph...
    cliques, elimination_ordering = triangulate(gm, clique_priority_func)

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
    S = set() # track the sepsets
    for X, Y in combinations(cliques, 2):
        if X.nodes.intersection(Y.nodes):
            S.add(SepSet(X, Y))
    sepsets_inserted = 0
    while sepsets_inserted < (len(cliques) - 1):
        # Adding in name to make this sort deterministic
        deco = [(s, -1 * s.mass, s.cost, s.__repr__()) for s in S]
        deco.sort(key=lambda x: x[1:])
        candidate_sepset = deco[0][0]
        if candidate_sepset.insertable(forest):
            # Insert into forest and remove the sepset
            candidate_sepset.insert(forest)
            S.remove(candidate_sepset)
            sepsets_inserted += 1

    assert len(forest) == 1
    jt = list(forest)[0]
    return jt
