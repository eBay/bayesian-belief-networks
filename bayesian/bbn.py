from __future__ import division
'''Data Structures to represent a BBN as a DAG.'''
import sys
import copy

from itertools import combinations, product
from StringIO import StringIO
from collections import defaultdict

from prettytable import PrettyTable

from bayesian import GREEN, NORMAL
from bayesian.graph import Node, UndirectedNode, connect
from bayesian.graph import Graph, UndirectedGraph, priority_func
from bayesian.graph import triangulate
from bayesian.factor_graph import VariableNode, FactorNode
from bayesian.factor_graph import connect as fg_connect
from bayesian.factor_graph import FactorGraph, unity, make_unity
from bayesian.factor_graph import make_product_func

from bayesian.utils import get_args, named_base_type_factory
from bayesian.utils import get_original_factors


class BBNNode(Node):

    def __init__(self, factor):
        super(BBNNode, self).__init__(factor.__name__)
        self.func = factor
        self.argspec = get_args(factor)

    def __repr__(self):
        return '<BBNNode %s (%s)>' % (
            self.name,
            self.argspec)


class BBN(Graph):
    '''A Directed Acyclic Graph'''

    def __init__(self, nodes_dict, name=None, domains={}):
        self.nodes = nodes_dict.values()
        self.vars_to_nodes = nodes_dict
        self.domains = domains
        # For each node we want
        # to explicitly record which
        # variable it 'introduced'.
        # Note that we cannot record
        # this duing Node instantiation
        # becuase at that point we do
        # not yet know *which* of the
        # variables in the argument
        # list is the one being modeled
        # by the function. (Unless there
        # is only one argument)
        for variable_name, node in nodes_dict.items():
            node.variable_name = variable_name

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('digraph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in sorted(self.nodes, key=lambda x:x.name):
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % (
                node.name.replace('-','_')))
            for child in node.children:
                edge = (node.name.replace('-','_'),
                        child.name.replace('-','_'))
                edges.add(edge)
        for source, target in sorted(edges, key=lambda x:(x[0], x[1])):
            fh.write('  %s -> %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def build_join_tree(self):
        jt = build_join_tree(self)
        return jt

    def query(self, **kwds):
        jt = self.build_join_tree()
        assignments = jt.assign_clusters(self)
        jt.initialize_potentials(assignments, self, kwds)

        jt.propagate()
        marginals = dict()
        normalizers = defaultdict(float)

        for node in self.nodes:
            for k, v in jt.marginal(node).items():
                # For a single node the
                # key for the marginal tt always
                # has just one argument so we
                # will unpack it here
                marginals[k[0]] = v
                # If we had any evidence then we
                # need to normalize all the variables
                # not evidenced.
                if kwds:
                    normalizers[k[0][0]] += v

        if kwds:
            for k, v in marginals.iteritems():
                if normalizers[k[0]] != 0:
                    marginals[k] /= normalizers[k[0]]

        return marginals

    def q(self, **kwds):
        '''Interactive user friendly wrapper
        around query()
        '''
        result = self.query(**kwds)
        tab = PrettyTable(['Node', 'Value', 'Marginal'], sortby='Node')
        tab.align = 'l'
        tab.align['Marginal'] = 'r'
        tab.float_format = '%8.6f'
        for (node, value), prob in result.items():
            if kwds.get(node, '') == value:
                tab.add_row(['%s*' % node,
                             '%s%s*%s' % (GREEN, value, NORMAL),
                             '%8.6f' % prob])
            else:
                tab.add_row([node, value, '%8.6f' % prob])
        print tab

    def convert_to_factor_graph(self):
        '''Convert to a factor graph
        representation. '''
        jt = build_join_tree(self, clique_priority_func=priority_func)

        # First lets create the variable nodes...
        # Mmm maybe the variable nodes are the SepSet nodes???
        # Lets try that...
        fg_variable_nodes = {}
        #for node in self.nodes:
        #    variable_node = VariableNode(node.variable_name)
        #    variable_node.domain = self.domains.get(
        #        node.variable_name, [True, False])
        #    fg_variable_nodes[node.variable_name] = variable_node

        assigned = set()
        # We should record for each variable
        # the clique nodes that invole it
        # as we will need these later to
        # make the edges
        variable_index = defaultdict(set)
        factor_index = defaultdict(set)
        # Now for the factor nodes...
        fg_factor_nodes = {}
        for clique in jt.nodes:
            # The original variables from the BBN
            # are recorded in list(clique.clique.nodes)[0].variable_name
            # lets attach them as well as this will
            # be useful for assigning the variable nodes...

            # Alternately what if we made *every* clique
            # a variable? Then we can easily build a factor
            # graph and assign potentials to the factors???

            if isinstance(clique, JoinTreeSepSetNode):
                variable_node = VariableNode(clique.name)
                variable_node.original_nodes = clique.sepset.label
                variable_node.domain = expand_domains(
                    self.domains, clique.name)
                fg_variable_nodes[str(clique.name)] = variable_node
            else:
                original_nodes = clique.clique.nodes
                factor_node = FactorNode(clique.name, make_unity([clique.name]))

                factor_node.original_nodes = original_nodes
                # Track which clique this node
                # came from
                factor_node.clique = clique
                # Track which factor node was
                # created from this clique
                clique.factor_node = factor_node
                fg_factor_nodes[clique.name] = factor_node

        # We will use the original graph neighbours to
        # connect them...
        for sepset in jt.sepset_nodes:
            for neighbour in sepset.neighbours:
                fg_connect(
                    fg_factor_nodes[neighbour.name],
                    fg_variable_nodes[str(sepset.name)])


        # TODO: We need to create the potential functions
        # and assign them. We also need to attach the
        # original variables.
        # We may nned to modify the algorithm to
        # prevent assignment to SepSet nodes.
        assignments = jt.assign_clusters(self)

        # Now  for each assignment we want to build
        # the potentional functions.
        for node, assignments in assignments.items():
            bbn_funcs = [bbn_node.func for bbn_node in assignments]
            # Since the factor nodes created above
            # were assigned the function unity()
            # we do not need to include them
            # in the new potential func.
            # TODO: We should perform a check here
            # to make sure that the assignments
            # are all within the family (F(v))
            if len(bbn_funcs) > 1:
                new_potential = make_product_func(bbn_funcs)
                new_potential.domains = {}
                for func in bbn_funcs:
                    new_potential.domains.update(func.domains)
            else:
                new_potential = bbn_funcs[0]
            node.factor_node.func = new_potential

            # Now we need to add back the
            # original bbn variables into
            # the factor graph.
            # WARNING: This is unchartered territory
            # not appearing in the literature,
            # however it seems an obvious extension
            # to the inference algorithm.
            # We will thus have *two* types
            # of variable nodes in the factor graph.
            # 1) Those arising from the sepsets
            # 2) Those arising from the BBN variables
            for bbn_node in assignments:
                variable_node = VariableNode(
                    bbn_node.variable_name)
                variable_node.domain = self.domains.get(
                    variable_node.name, [True, False])
                fg_variable_nodes[variable_node.name] = variable_node
                fg_connect(variable_node, node.factor_node)

        # Now create the factor graph...
        fg = FactorGraph(
            fg_variable_nodes.values() +
            fg_factor_nodes.values())

        # Maintain a link to the jt and the BBN
        fg.jt = jt
        fg.bbn = self

        return fg




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
                fh.write('  %s [ shape="box" color="blue"];\n' % (
                    node.name.replace('-','_')))
            else:
                fh.write('  %s [ shape="ellipse" color="red"];\n' % (
                    node.name.replace('-','_')))
            for neighbour in node.neighbours:
                edge = [node.name.replace('-','_'),
                        neighbour.name.replace('-','_')]
                edges.add(tuple(sorted(edge)))
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def initialize_potentials(self, assignments, bbn, evidence={}):
        # Step 1, assign 1 to each cluster and sepset
        for node in self.nodes:
            tt = dict()
            vals = []
            variables = node.variable_names
            # Lets sort the variables here so that
            # the variable names in the keys in
            # the tt are always sorted.
            variables.sort()
            for variable in variables:
                domain = bbn.domains.get(variable, [True, False])
                vals.append(list(product([variable], domain)))
            permutations = product(*vals)
            for permutation in permutations:
                tt[permutation] = 1
            node.potential_tt = tt

        # Step 2: Note that in H&D the assignments are
        # done as part of step 2 however we have
        # seperated the assignment algorithm out and
        # done these prior to step 1.
        # Now for each assignment we want to
        # generate a truth-table from the
        # values of the bbn truth-tables that are
        # assigned to the clusters...

        for clique, bbn_nodes in assignments.iteritems():

            tt = dict()
            vals = []
            variables = list(clique.variable_names)
            variables.sort()
            for variable in variables:
                domain = bbn.domains.get(variable, [True, False])
                vals.append(list(product([variable], domain)))
            permutations = product(*vals)
            for permutation in permutations:
                argvals = dict(permutation)
                potential = 1
                for bbn_node in bbn_nodes:
                    bbn_node.clique = clique
                    # We could handle evidence here
                    # by altering the potential_tt.
                    # This is slightly different to
                    # the way that H&D do it.

                    arg_list = []
                    for arg_name in get_args(bbn_node.func):
                        arg_list.append(argvals[arg_name])

                    potential *= bbn_node.func(*arg_list)
                tt[permutation] = potential
            clique.potential_tt = tt

        if not evidence:
            # We dont need to deal with likelihoods
            # if we dont have any evidence.
            return

        # Step 2b: Set each liklihood element ^V(v) to 1
        likelihoods = self.initial_likelihoods(assignments, bbn)
        for clique, bbn_nodes in assignments.iteritems():
            for node in bbn_nodes:
                if node.variable_name in evidence:
                    for k, v in clique.potential_tt.items():
                        # Encode the evidence in
                        # the clique potential...
                        for variable, value in k:
                            if (variable == node.variable_name):
                                if value != evidence[variable]:
                                    clique.potential_tt[k] = 0


    def initial_likelihoods(self, assignments, bbn):
        # TODO: Since this is the same every time we should probably
        # cache it.
        l = defaultdict(dict)
        for clique, bbn_nodes in assignments.iteritems():
            for node in bbn_nodes:
                for value in bbn.domains.get(node.variable_name, [True, False]):
                    l[(node.variable_name, value)] = 1
        return l


    def assign_clusters(self, bbn):
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
                # clique.
                # Revising this for now as I dont
                # think its correct, I think
                # all CPTs need to be assigned
                # once and once only. The example
                # in H&D just happens to be a clique
                # that f_a could have been assigned
                # to but wasnt presumably because
                # it got assigned somewhere else.
                pass
                #continue
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
                                  (set(clique_node.variable_names).
                                   issuperset(family))]
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
            node.marked = False

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
            sender.pass_message(receiver)

    def distribute_evidence(self, sender=None, receiver=None):

        # Step 1, Mark X
        sender.marked = True

        # Step 2, pass a messagee from X to each of its
        # unmarked neighbouring clusters
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                sender.pass_message(neighbouring_clique)

        # Step 3, call distribute_evidence on Xs unmarked neighbours
        for neighbouring_clique in sender.neighbouring_cliques:
            if not neighbouring_clique.marked:
                self.distribute_evidence(
                    sender=neighbouring_clique,
                    receiver=sender)

    def marginal(self, bbn_node):
        '''Remember that the original
        variables that we are interested in
        are actually in the bbn. However
        when we constructed the JT we did
        it out of the moralized graph.
        This means the cliques refer to
        the nodes in the moralized graph
        and not the nodes in the BBN.
        For efficiency we should come back
        to this and add some pointers
        or an index.
        '''

        # First we will find the JT nodes that
        # contain the bbn_node ie all the nodes
        # that are either cliques or sepsets
        # that contain the bbn_node
        # Note that for efficiency we
        # should probably have an index
        # cached in the bbn and/or the jt.
        containing_nodes = []

        for node in self.clique_nodes:
            if bbn_node.name in [n.name for n in node.clique.nodes]:
                containing_nodes.append(node)
                # In theory it doesnt matter which one we
                # use so we could bale out after we
                # find the first one
                # TODO: With some better indexing we could
                # avoid searching for this node every time...

        clique_node = containing_nodes[0]
        tt = defaultdict(float)
        for k, v in clique_node.potential_tt.items():
            entry = transform(
                k,
                clique_node.variable_names,
                [bbn_node.variable_name]) # XXXXXX
            tt[entry] += v

        # Now if this node was evidenced we need to normalize
        # over the values...
        # TODO: It will be safer to copy the defaultdict to a regular dict
        return tt




def transform(x, X, R):
    '''Transform a Potential Truth Table
    Entry into a different variable space.
    For example if we have the
    entry [True, True, False] representing
    values of variable [A, B, C] in X
    and we want to transform into
    R which has variables [C, A] we
    will return the entry [False, True].
    Here X represents the argument list
    for the clique set X and R represents
    the argument list for the sepset.
    This implies that R is always a subset
    of X'''
    entry = []
    for r in R:
        pos = X.index(r)
        entry.append(x[pos])
    return tuple(entry)


class JoinTreeCliqueNode(UndirectedNode):

    def __init__(self, clique):
        super(JoinTreeCliqueNode, self).__init__(
            clique.__repr__())
        self.clique = clique
        # Now we create a pointer to
        # this clique node as the "parent" clique
        # node of each node in the cluster.
        #for node in self.clique.nodes:
        #    node.parent_clique = self
        # This is not quite correct, the
        # parent cluster as defined by H&D
        # is *a* cluster than is a superset
        # of Family(v)

    @property
    def variable_names(self):
        '''Return the set of variable names
        that this clique represents'''
        var_names = []
        for node in self.clique.nodes:
            var_names.append(node.variable_name)
        return sorted(var_names)

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

    def pass_message(self, target):
        '''Pass a message from this node to the
        recipient node during propagation.

        NB: It may turnout at this point that
        after initializing the potential
        Truth table on the JT we could quite
        simply construct a factor graph
        from the JT and use the factor
        graph sum product propagation.
        In theory this should be the same
        and since the semantics are already
        worked out it would be easier.'''

        # Find the sepset node between the
        # source and target nodes.
        sepset_node = list(set(self.neighbours).intersection(
            target.neighbours))[0]

        # Step 1: projection
        self.project(sepset_node)

        # Step 2 absorbtion
        self.absorb(sepset_node, target)

    def project(self, sepset_node):
        '''See page 20 of PPTC.
        We assign a new potential tt to
        the sepset which consists of the
        potential of the source node
        with all variables not in R marginalized.
        '''
        assert sepset_node in self.neighbours
        # First we make a copy of the
        # old potential tt
        sepset_node.potential_tt_old = copy.deepcopy(
            sepset_node.potential_tt)

        # Now we assign a new potential tt
        # to the sepset by marginalizing
        # out the variables from X that are not
        # in the sepset
        tt = defaultdict(float)
        for k, v in self.potential_tt.items():
            entry = transform(k, self.variable_names,
                              sepset_node.variable_names)
            tt[entry] += v
        sepset_node.potential_tt = tt

    def absorb(self, sepset, target):
        # Assign a new potential tt to
        # Y (the target)
        tt = dict()

        for k, v in target.potential_tt.items():
            # For each entry we multiply by
            # sepsets new value and divide
            # by sepsets old value...
            # Note that nowhere in H&D is
            # division on potentials defined.
            # However in Barber page 12
            # an equation implies that
            # the the division is equivalent
            # to the original assignment.
            # For now we will assume entry-wise
            # division which seems logical.
            entry = transform(k, target.variable_names,
                              sepset.variable_names)
            if target.potential_tt[k] == 0:
                tt[k] = 0
            else:
                tt[k] = target.potential_tt[k] * (sepset.potential_tt[entry] /
                                                  sepset.potential_tt_old[entry])
        target.potential_tt = tt

    def __repr__(self):
        return '<JoinTreeCliqueNode: %s>' % self.clique


class SepSet(object):

    def __init__(self, X, Y):
        '''X and Y are cliques represented as sets.'''
        self.X = X
        self.Y = Y
        self.label = list(X.nodes.intersection(Y.nodes))

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
        X_trees = [t for t in forest if self.X in
                   [n.clique for n in t.clique_nodes]]
        Y_trees = [t for t in forest if self.Y in
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
        X_tree = [t for t in forest if self.X in
                  [n.clique for n in t.clique_nodes]][0]
        Y_tree = [t for t in forest if self.Y in
                  [n.clique for n in t.clique_nodes]][0]

        # Now create and insert a sepset node into the Xtree
        sepset_name = '%s-%s' % (self.X.node.name, self.Y.node.name)
        ss_node = JoinTreeSepSetNode(sepset_name, self)
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
        return 'SepSet_%s' % ''.join(
            #[x.name[2:].upper() for x in list(self.label)])
            [x.variable_name.upper() for x in list(self.label)])


class JoinTreeSepSetNode(UndirectedNode):

    def __init__(self, name, sepset):
        super(JoinTreeSepSetNode, self).__init__(name)
        self.sepset = sepset

    @property
    def variable_names(self):
        '''Return the set of variable names
        that this sepset represents'''
        # TODO: we are assuming here
        # that X and Y are each separate
        # variables from the BBN which means
        # we are assuming that the sepsets
        # always contain only 2 nodes.
        # Need to check whether this is
        # the case.
        return sorted([x.variable_name for x in self.sepset.label])

    def __repr__(self):
        return '<JoinTreeSepSetNode: %s>' % self.sepset


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
        # more than 255 functions, since
        # Python functions are limited to
        # 255 arguments.
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
    # it to the child node.

    # Note that calling original_factors
    # here can break build_bbn if the
    # factors do not correctly represent
    # a BBN.
    original_factors = get_original_factors(factor_nodes.values())
    for factor_node in factor_nodes.values():
        factor_args = get_args(factor_node)
        parents = [original_factors[arg] for arg in
                   factor_args if original_factors[arg] != factor_node]
        for parent in parents:
            connect(parent, factor_node)
        if not hasattr(factor_node.func, 'domains'):
            factor_node.func.domains = dict()
            for arg in factor_args:
                factor_node.func.domains[arg] = domains.get(arg, [True, False])
    bbn = BBN(original_factors, name=name)
    bbn.domains = domains

    return bbn


def make_undirected_copy(dag):
    '''Returns an exact copy of the dag
    except that direction of edges are dropped.'''
    nodes = dict()
    for node in dag.nodes:
        undirected_node = UndirectedNode(
            name=node.name)
        undirected_node.func = node.func
        undirected_node.argspec = node.argspec
        undirected_node.variable_name = node.variable_name
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


def expand_domains(variable_names, domains, new_variable_name):
    vals = []
    variable_names.sort()
    for variable in variable_names:
        domain = domains.get(variable, [True, False])
        vals.append(list(product([variable], domain)))
    permutations = product(*vals)
    new_domain = defaultdict(list)
    for permutation in permutations:
        new_domain[new_variable_name].append(
            [p[1] for p in permutation])
    return new_domain
