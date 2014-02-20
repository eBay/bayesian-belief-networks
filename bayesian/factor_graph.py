from __future__ import division
'''Implements Sum-Product Algorithm and Sampling over Factor Graphs'''
import os
import csv
import sys
import copy
import inspect
import random

from collections import defaultdict
from itertools import product as iter_product
from Queue import Queue

import sqlite3
from prettytable import PrettyTable

from bayesian.persistance import SampleDB, ensure_data_dir_exists
from bayesian.exceptions import *
from bayesian.utils import get_args

DEBUG = False
GREEN = '\033[92m'
NORMAL = '\033[0m'

class Node(object):

    def is_leaf(self):
        if len(self.neighbours) == 1:
            return True
        return False

    def send(self, message):
        recipient = message.destination
        if DEBUG:
            print '%s ---> %s' % (
                self.name, recipient.name), message
        recipient.received_messages[
            self.name] = message

    def get_sent_messages(self):
        sent_messages = {}
        for neighbour in self.neighbours:
            if neighbour.received_messages.get(self.name):
                sent_messages[neighbour.name] = \
                    neighbour.received_messages.get(self.name)
        return sent_messages

    def message_report(self):
        '''
        List out all messages Node
        currently has received.
        '''
        print '------------------------------'
        print 'Messages at Node %s' % self.name
        print '------------------------------'
        for k, v in self.received_messages.iteritems():
            print '%s <-- Argspec:%s' % (v.source.name, v.argspec)
            v.list_factors()
        print '--'

    def get_target(self):
        '''
        A node can only send to a neighbour if
        it has not already sent to that neighbour
        and it has received messages from all other
        neighbours.
        '''
        neighbours = self.neighbours
        #if len(neighbours) - len(self.received_messages) > 1:
        #    return None
        needed_to_send = defaultdict(int)
        for target in neighbours:
            needed_to_send[target] = len(neighbours) - 1
        for _, message in self.received_messages.items():
            for target in neighbours:
                if message.source != target:
                    needed_to_send[target] -= 1
        for k, v in needed_to_send.items():
            if v == 0 and not self.name in k.received_messages:
                return k

    def get_neighbour_by_name(self, name):
        for node in self.neighbours:
            if node.name == name:
                return node


class VariableNode(Node):

    def __init__(self, name, domain=[True, False]):
        self.name = name
        self.domain = domain
        self.neighbours = []
        self.received_messages = {}
        self.value = None

    def construct_message(self):
        target = self.get_target()
        message = make_variable_node_message(self, target)
        return message

    def __repr__(self):
        return '<VariableNode: %s:%s>' % (self.name, self.value)

    def marginal(self, val, normalizer=1.0):
        '''
        The marginal function in a Variable
        Node is the product of all incoming
        messages. These should all be functions
        of this nodes variable.
        When any of the variables in the
        network are constrained we need to
        normalize.
        '''
        product = 1
        for _, message in self.received_messages.iteritems():
            product *= message(val)
        return product / normalizer

    def reset(self):
        self.received_messages = {}
        self.value = None

    def verify_neighbour_types(self):
        '''
        Check that all neighbours are of VariableNode type.
        '''
        for node in self.neighbours:
            if not isinstance(node, FactorNode):
                return False
        return True


class FactorNode(Node):

    def __init__(self, name, func, neighbours=[]):
        self.name = name
        self.func = func
        self.neighbours = neighbours[:]
        self.received_messages = {}
        self.func.value = None
        self.cached_functions = []

    def construct_message(self):
        target = self.get_target()
        message = make_factor_node_message(self, target)
        return message

    def verify_neighbour_types(self):
        '''
        Check that all neighbours are of VariableNode type.
        '''
        for node in self.neighbours:
            if not isinstance(node, VariableNode):
                return False
        return True

    def __repr__(self):
        return '<FactorNode %s %s(%s)>' % \
            (self.name,
             self.func.__name__,
             get_args(self.func))

    def marginal(self, val_dict):
        # The Joint marginal of the
        # neighbour variables of a factor
        # node is given by the product
        # of the incoming messages and the factor
        product = 1
        neighbours = self.neighbours
        for neighbour in neighbours:
            message = self.received_messages[neighbour.name]
            call_args = []
            for arg in get_args(message):
                call_args.append(val_dict[arg])
            if not call_args:
                call_args.append('dummy')
            product *= message(*call_args)
        # Finally we also need to multiply
        # by the factor itself
        call_args = []
        for arg in get_args(self.func):
            call_args.append(val_dict[arg])
        if not call_args:
            call_args.append('dummy')
        product *= self.func(*call_args)
        return product

    def add_evidence(self, node, value):
        '''
        Here we modify the factor function
        to return 0 whenever it is called
        with the observed variable having
        a value other than the observed value.
        '''
        args = get_args(self.func)
        pos = args.index(node.name)
        # Save the old func so that we
        # can remove the evidence later
        old_func = self.func
        self.cached_functions.insert(0, old_func)

        def evidence_func(*args):
            if args[pos] != value:
                return 0
            return old_func(*args)

        evidence_func.argspec = args
        evidence_func.domains = old_func.domains
        self.func = evidence_func

    def reset(self):
        self.received_messages = {}
        if self.cached_functions:
            self.func = self.cached_functions[-1]
            self.cached_functions = []


class Message(object):

    def list_factors(self):
        print '---------------------------'
        print 'Factors in message %s -> %s' % \
            (self.source.name, self.destination.name)
        print '---------------------------'
        for factor in self.factors:
            print factor

    def __call__(self, var):
        '''
        Evaluate the message as a function
        '''
        if getattr(self.func, '__name__', None) == 'unity':
            return 1
        assert not isinstance(var, VariableNode)
        # Now check that the name of the
        # variable matches the argspec...
        #assert var.name == self.argspec[0]
        return self.func(var)


class VariableMessage(Message):

    def __init__(self, source, destination, factors, func):
        self.source = source
        self.destination = destination
        self.factors = factors
        self.argspec = get_args(func)
        self.func = func

    def __repr__(self):
        return '<V-Message from %s -> %s: %s factors (%s)>' % \
            (self.source.name, self.destination.name,
             len(self.factors), self.argspec)


class FactorMessage(Message):

    def __init__(self, source, destination, factors, func):
        self.source = source
        self.destination = destination
        self.factors = factors
        self.func = func
        self.argspec = get_args(func)
        self.domains = func.domains

    def __repr__(self):
        return '<F-Message %s -> %s: ~(%s) %s factors.>' % \
            (self.source.name, self.destination.name,
             self.argspec,
             len(self.factors))


def connect(a, b):
    '''
    Make an edge between two nodes
    or between a source and several
    neighbours.
    '''
    if not isinstance(b, list):
        b = [b]
    for b_ in b:
        a.neighbours.append(b_)
        b_.neighbours.append(a)



def eliminate_var(f, var):
    '''
    Given a function f return a new
    function which sums over the variable
    we want to eliminate

    This may be where we have the opportunity
    to remove the use of .value....

    '''
    arg_spec = get_args(f)
    pos = arg_spec.index(var)
    new_spec = arg_spec[:]
    new_spec.remove(var)
    # Lets say the orginal argspec is
    # ('a', 'b', 'c', 'd') and they
    # are all Booleans
    # Now lets say we want to eliminate c
    # This means we want to sum over
    # f(a, b, True, d) and f(a, b, False, d)
    # Seems like all we have to do is know
    # the positionn of c and thats it???
    # Ok so its not as simple as that...
    # this is because when the *call* is made
    # to the eliminated function, as opposed
    # to when its built then its only
    # called with ('a', 'b', 'd')
    eliminated_pos = arg_spec.index(var)

    def eliminated(*args):
        template = arg_spec[:]
        total = 0
        call_args = template[:]
        i = 0
        for arg in args:
            # To be able to remove .value we
            # first need to also be able to
            # remove .name in fact .value is
            # just a side effect of having to
            # rely on .name. This means we
            # probably need to construct a
            # a list containing the names
            # of the args based on the position
            # they are being called.
            if i == eliminated_pos:
                # We need to increment i
                # once more to skip over
                # the variable being marginalized
                call_args[i] = 'marginalize me!'
                i += 1
            call_args[i] = arg
            i += 1

        for val in f.domains[var]:
            #v = VariableNode(name=var)
            #v.value = val
            #call_args[pos] = v
            call_args[pos] = val
            total += f(*call_args)
        return total

    eliminated.argspec = new_spec
    eliminated.domains = f.domains
    #eliminated.__name__ = f.__name__
    return eliminated


def memoize(f):
    '''
    The goal of message passing
    is to re-use results. This
    memoise is slightly modified from
    usual examples in that it caches
    the values of variables rather than
    the variables themselves.
    '''
    cache = {}

    def memoized(*args):
        #arg_vals = tuple([arg.value for arg in args])
        arg_vals = tuple(args)
        if not arg_vals in cache:
            cache[arg_vals] = f(*args)
        return cache[arg_vals]

    if hasattr(f, 'domains'):
        memoized.domains = f.domains
    if hasattr(f, 'argspec'):
        memoized.argspec = f.argspec
    return memoized


def make_not_sum_func(product_func, keep_var):
    '''
    Given a function with some set of
    arguments, and a single argument to keep,
    construct a new function only of the
    keep_var, summarized over all the other
    variables.

    For this branch we are trying to
    get rid of the requirement to have
    to use .value on arguments....
    Looks like its actually in the
    eliminate var...
    '''
    args = get_args(product_func)
    new_func = copy.deepcopy(product_func)
    for arg in args:
        if arg != keep_var:
            new_func = eliminate_var(new_func, arg)
            new_func = memoize(new_func)
    return new_func


def make_factor_node_message(node, target_node):
    '''
    The rules for a factor node are:
    take the product of all the incoming
    messages (except for the destination
    node) and then take the sum over
    all the variables except for the
    destination variable.
    >>> def f(x1, x2, x3): pass
    >>> node = object()
    >>> node.func = f
    >>> target_node = object()
    >>> target_node.name = 'x2'
    >>> make_factor_node_message(node, target_node)
    '''

    if node.is_leaf():
        not_sum_func = make_not_sum_func(node.func, target_node.name)
        message = FactorMessage(node, target_node, [node.func], not_sum_func)
        return message

    args = set(get_args(node.func))

    # Compile list of factors for message
    factors = [node.func]

    # Now add the message that came from each
    # of the non-destination neighbours...
    neighbours = node.neighbours
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        # When we pass on a message, we unwrap
        # the original payload and wrap it
        # in new headers, this is purely
        # to verify the procedure is correct
        # according to usual nomenclature
        in_message = node.received_messages[neighbour.name]
        if in_message.destination != node:
            out_message = VariableMessage(
                neighbour, node, in_message.factors,
                in_message.func)
            out_message.argspec = in_message.argspec
        else:
            out_message = in_message
        factors.append(out_message)

    product_func = make_product_func(factors)
    not_sum_func = make_not_sum_func(product_func, target_node.name)
    message = FactorMessage(node, target_node, factors, not_sum_func)
    return message


def make_variable_node_message(node, target_node):
    '''
    To construct the message from
    a variable node to a factor
    node we take the product of
    all messages received from
    neighbours except for any
    message received from the target.
    If the source node is a leaf node
    then send the unity function.
    '''
    if node.is_leaf():
        message = VariableMessage(
            node, target_node, [1], unity)
        return message
    factors = []
    neighbours = node.neighbours
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        factors.append(
            node.received_messages[neighbour.name])

    product_func = make_product_func(factors)
    message = VariableMessage(
        node, target_node, factors, product_func)
    return message


def make_product_func(factors):
    '''
    Return a single callable from
    a list of factors which correctly
    applies the arguments to each
    individual factor.

    The challenge here is to return a function
    whose argument list we know and ensure that
    when this function is called, its always
    called with the correct arguments.
    Since the correct argspec is attached
    to the built function it seems that
    it should be up to the caller to
    get the argument list correct.
    So we need to determine when and where its called...

    '''
    args_map = {}
    all_args = []
    domains = {}
    for factor in factors:
        #if factor == 1:
        #    continue
        args_map[factor] = get_args(factor)
        all_args += args_map[factor]
        if hasattr(factor, 'domains'):
            domains.update(factor.domains)
    args = list(set(all_args))
    # Perhaps if we sort the


    def product_func(*product_func_args):
        #import pytest; pytest.set_trace()
        #arg_dict = dict([(a.name, a) for a in product_func_args])
        arg_dict = dict(zip(args, product_func_args))
        #import pytest; pytest.set_trace()
        result = 1
        for factor in factors:
            #domains.update(factor.domains)
            # We need to build the correct argument
            # list to call this factor with.
            factor_args = []
            for arg in get_args(factor):
                if arg in arg_dict:
                    factor_args.append(arg_dict[arg])
            if not factor_args:
                # Since we always require
                # at least one argument we
                # insert a dummy argument
                # so that the unity function works.
                factor_args.append('dummy')
            result *= factor(*factor_args)

        return result

    product_func.argspec = args
    product_func.factors = factors
    product_func.domains = domains
    return memoize(product_func)


def make_unity(argspec):
    def unity(x):
        return 1
    unity.argspec = argspec
    unity.__name__ = '1'
    return unity


def unity():
    return 1


def expand_args(args):
    if not args:
        return []
    return


def dict_to_tuples(d):
    '''
    Convert a dict whose values
    are lists to a list of
    tuples of the key with
    each of the values
    '''
    retval = []
    for k, vals in d.iteritems():
        retval.append([(k, v) for v in vals])
    return retval


def expand_parameters(arg_vals):
    '''
    Given a list of args and values
    return a list of tuples
    containing all possible sequences
    of length n.
    '''
    arg_tuples = dict_to_tuples(arg_vals)
    return [dict(args) for args in iter_product(*arg_tuples)]


def add_evidence(node, value):
    '''
    Set a variable node to an observed value.
    Note that for now this is achieved
    by modifying the factor functions
    which this node is connected to.
    After updating the factor nodes
    we need to re-run the sum-product
    algorithm. We also need to normalize
    all marginal outcomes.
    '''
    node.value = value
    neighbours = node.neighbours
    for factor_node in neighbours:
        if node.name in get_args(factor_node.func):
            factor_node.add_evidence(node, value)


def discover_sample_ordering(graph):
    '''
    Try to get the order of variable nodes
    for sampling. This would be easier in
    the underlying BBN but lets try on
    the factor graph.
    '''
    iterations = 0
    ordering = []
    pmf_ordering = []
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
            ordering.append(node.neighbours[0])
            accounted_for.add(node.neighbours[0].name)
            pmf_ordering.append(node.func)

    # Now for each factor node whose variables
    # all but one are already in the ordering,
    # we can add that one variable. This is
    # actuall
    while len(ordering) < required:
        for node in factor_nodes:
            args = set(get_args(node.func))
            new_args = args.difference(accounted_for)
            if len(new_args) == 1:
                arg_name = list(new_args)[0]
                var_node = node.get_neighbour_by_name(arg_name)
                ordering.append(var_node)
                accounted_for.add(var_node.name)
                pmf_ordering.append(node.func)
    return zip(ordering, pmf_ordering)


def get_sample(ordering, evidence={}):
    '''
    Given a valid ordering, sample the network.
    '''
    sample = []
    sample_dict = dict()
    for var, func in ordering:
        r = random.random()
        total = 0
        for val in var.domain:
            test_var = VariableNode(var.name)
            test_var.value = val
            # Now we need to build the
            # argument list out of any
            # variables already in the sample
            # and this new test value in
            # the order required by the function.
            args = []
            for arg in get_args(func):
                if arg == var.name:
                    #args.append(test_var)
                    args.append(val)
                else:
                    args.append(sample_dict[arg].value)

            total += func(*args)
            if total > r:
                # We only want to use this sample
                # if it corresponds to the evidence value...
                if var.name in evidence:
                    if test_var.value == evidence[var.name]:
                        sample.append(test_var)
                        sample_dict[var.name] = test_var
                else:
                    sample.append(test_var)
                    sample_dict[var.name] = test_var
                break
        if not var.name in sample_dict:
            print 'Iterated through all values for %s and %s but no go...' \
                % (var.name, func.__name__)
            # This seems to mean that we have never seen this combination
            # of variables before, we can either discard it as irrelevant or
            # use some type of +1 smoothing???
            # What if we just randomly select some value for var????
            # lets try that as it seems the easiest....
            raise InvalidSampleException
    return sample


class FactorGraph(object):

    def __init__(self, nodes, name=None, n_samples=100):
        self.nodes = nodes
        self._inference_method = 'sumproduct'
        # We need to divine the domains for Factor nodes here...
        # First compile a mapping of factors to variables
        # from the arg spec...
        function_args = dict()
        arg_domains = dict()
        for node in self.nodes:
            if isinstance(node, VariableNode):
                #if not hasattr(node, 'domain'):
                #    node.domain = [True, False]
                arg_domains[node.name] = node.domain
            elif isinstance(node, FactorNode):
                function_args[node.func.__name__] = get_args(node.func)
        # Now if the domains for the
        # factor functions have not been explicitely
        # set we create them based on the variable
        # values it can take.
        for node in self.nodes:
            if isinstance(node, FactorNode):
                if hasattr(node.func, 'domains'):
                    continue
                domains = dict()
                for arg in get_args(node.func):
                    if not arg in arg_domains:
                        print 'WARNING: missing variable for arg:%s' % arg
                    else:
                        domains.update({arg: arg_domains[arg]})
                node.func.domains = domains
        self.name = name
        self.n_samples = n_samples
        # Now try to set the mode of inference..
        try:
            if self.has_cycles():
                # Currently only sampling
                # is supported for cyclic graphs
                self.inference_method = 'sample'
            else:
                # The sumproduct method will
                # give exact likelihoods but
                # only of the graph contains
                # no cycles.
                self.inference_method = 'sumproduct'
        except:
            print 'Failed to determine if graph has cycles, '
            'setting inference to sample.'
            self.inference_method = 'sample'
        self.enforce_minimum_samples = False

    @property
    def inference_method(self):
        return self._inference_method

    @inference_method.setter
    def inference_method(self, value):
        # If the value is being set to 'sample_db'
        # we need to make sure that the sqlite file
        # exists.
        if value == 'sample_db':
            ensure_data_dir_exists(self.sample_db_filename)
            sample_ordering = self.discover_sample_ordering()
            domains = dict([(var, var.domain) for var, _ in sample_ordering])
            if not os.path.isfile(self.sample_db_filename):
                # This is a new file so we need to
                # initialize the db...
                self.sample_db = SampleDB(
                    self.sample_db_filename,
                    domains,
                    initialize=True)
            else:
                self.sample_db = SampleDB(
                    self.sample_db_filename,
                    domains,
                    initialize=False)
        self._inference_method = value

    @property
    def sample_db_filename(self):
        '''
        Get the name of the sqlite sample
        database for external sample
        generation and querying.
        The default location for now
        will be in the users home
        directory under ~/.pypgm/data/[name].sqlite
        where [name] is the name of the
        model. If the model has
        not been given an explict name
        it will be "default".

        '''
        home = os.path.expanduser('~')
        return os.path.join(
            home, '.pypgm',
            'data',
            '%s.sqlite' % (self.name or 'default'))

    def reset(self):
        '''
        Reset all nodes back to their initial state.
        We should do this before or after adding
        or removing evidence.
        '''
        for node in self.nodes:
            node.reset()

    def has_cycles(self):
        '''
        Check if the graph has cycles or not.
        We will do this by traversing starting
        from any leaf node and recording
        both the edges traversed and the nodes
        discovered. From stackoverflow, if
        an unexplored edge leads to a
        previously found node then it has
        cycles.
        '''
        discovered_nodes = set()
        traversed_edges = set()
        q = Queue()
        for node in self.nodes:
            if node.is_leaf():
                start_node = node
                break
        q.put(start_node)
        while not q.empty():
            current_node = q.get()
            if DEBUG:
                print "Current Node: ", current_node
                print "Discovered Nodes before adding Current Node: ", \
                    discovered_nodes
            if current_node.name in discovered_nodes:
                # We have a cycle!
                if DEBUG:
                    print 'Dequeued node already processed: %s', current_node
                return True
            discovered_nodes.add(current_node.name)
            if DEBUG:
                print "Discovered Nodes after adding Current Node: ", \
                    discovered_nodes
            for neighbour in current_node.neighbours:
                edge = [current_node.name, neighbour.name]
                # Since this is undirected and we want
                # to record the edges we have traversed
                # we will sort the edge alphabetically
                edge.sort()
                edge = tuple(edge)
                if edge not in traversed_edges:
                    # This is a new edge...
                    if neighbour.name in discovered_nodes:
                        return True
                # Now place all neighbour nodes on the q
                # and record this edge as traversed
                if neighbour.name not in discovered_nodes:
                    if DEBUG:
                        print 'Enqueuing: %s' % neighbour
                    q.put(neighbour)
                traversed_edges.add(edge)
        return False

    def verify(self):
        '''
        Check several properties of the Factor Graph
        that should hold.
        '''
        # Check that all nodes are either
        # instances of classes derived from
        # VariableNode or FactorNode.
        # It is a very common error to instantiate
        # the graph with the factor function
        # instead of the corresponding factor
        # node.
        for node in self.nodes:
            if not isinstance(node, VariableNode) and \
                    not isinstance(node, FactorNode):
                bases = node.__class__.__bases__
                if not VariableNode in bases and not FactorNode in bases:
                    print ('Factor Graph does not '
                           'support nodes of type: %s' % node.__class__)
                    raise InvalidGraphException
        # First check that for each node
        # only connects to nodes of the
        # other type.
        print 'Checking neighbour node types...'
        for node in self.nodes:
            if not node.verify_neighbour_types():
                print '%s has invalid neighbour type.' % node
                return False
        print 'Checking that all factor functions have domains...'
        for node in self.nodes:
            if isinstance(node, FactorNode):
                if not hasattr(node.func, 'domains'):
                    print '%s has no domains.' % node
                    raise InvalidGraphException
                elif not node.func.domains:
                    # Also check for an empty domain dict!
                    print '%s has empty domains.' % node
                    raise InvalidGraphException
        print 'Checking that all variables are accounted for' + \
            ' by at least one function...'
        variables = set([vn.name for vn in self.nodes
                         if isinstance(vn, VariableNode)])

        largs = [get_args(fn.func) for fn in
                 self.nodes if isinstance(fn, FactorNode)]

        args = set(reduce(lambda x, y: x + y, largs))

        if not variables.issubset(args):
            print 'These variables are not used in any factors nodes: '
            print variables.difference(args)
            return False
        print 'Checking that all arguments have matching variable nodes...'
        if not args.issubset(variables):
            print 'These arguments have missing variables:'
            print args.difference(variables)
            return False
        print 'Checking that graph has at least one leaf node...'
        leaf_nodes = filter(
            lambda x: x.is_leaf(),
            self.nodes)
        if not leaf_nodes:
            print 'Graph has no leaf nodes.'
            raise InvalidGraphException
        return True

    def get_leaves(self):
        return [node for node in self.nodes if node.is_leaf()]

    def get_eligible_senders(self):
        '''
        Return a list of nodes that are
        eligible to send messages at this
        round. Only nodes that have received
        messages from all but one neighbour
        may send at any round.
        '''
        eligible = []
        for node in self.nodes:
            if node.get_target():
                eligible.append(node)
        return eligible

    def propagate(self):
        '''
        This is the heart of the sum-product
        Message Passing Algorithm.
        '''
        step = 1
        while True:
            eligible_senders = self.get_eligible_senders()
            #print 'Step: %s %s nodes can send.' \
            # % (step, len(eligible_senders))
            #print [x.name for x in eligible_senders]
            if not eligible_senders:
                break
            for node in eligible_senders:
                message = node.construct_message()
                node.send(message)
            step += 1

    def variable_nodes(self):
        return [n for n in self.nodes if isinstance(n, VariableNode)]

    def factor_nodes(self):
        return [n for n in self.nodes if isinstance(n, FactorNode)]

    def get_normalizer(self):
        for node in self.variable_nodes():
            if node.value is not None:
                normalizer = node.marginal(node.value)
                return normalizer
        return 1

    def status(self, omit=[False, 0]):
        normalizer = self.get_normalizer()
        retval = dict()
        for node in self.variable_nodes():
            for value in node.domain:
                m = node.marginal(value, normalizer)
                retval[(node.name, value)] = m
        return retval

    def query_by_propagation(self, **kwds):
        self.reset()
        for k, v in kwds.items():
            for node in self.variable_nodes():
                if node.name == k:
                    add_evidence(node, v)
        self.propagate()
        return self.status()

    def query(self, **kwds):
        if self.inference_method == 'sample_db':
            return self.query_by_external_samples(**kwds)
        elif self.inference_method == 'sample':
            return self.query_by_sampling(**kwds)
        elif self.inference_method == 'sumproduct':
            return self.query_by_propagation(**kwds)
        raise InvalidInferenceMethod

    def q(self, **kwds):
        '''Wrapper around query

        This method formats the query
        result in a nice human readable format
        for interactive use.
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

    def discover_sample_ordering(self):
        return discover_sample_ordering(self)

    def get_sample(self, evidence={}):
        '''
        We need to allow for setting
        certain observed variables and
        discarding mismatching
        samples as we generate them.
        '''
        if not hasattr(self, 'sample_ordering'):
            self.sample_ordering = self.discover_sample_ordering()
        return get_sample(self.sample_ordering, evidence)

    def query_by_sampling(self, **kwds):
        counts = defaultdict(int)
        valid_samples = 0
        while valid_samples < self.n_samples:
            print "%s of %s" % (valid_samples, self.n_samples)
            try:
                sample = self.get_sample(kwds)
                valid_samples += 1
            except:
                print 'Failed to get a valid sample...'
                print 'continuing...'
                continue
            for var in sample:
                key = (var.name, var.value)
                counts[key] += 1
        # Now normalize
        normalized = dict(
            [(k, v / valid_samples) for k, v in counts.items()])
        return normalized

    def generate_samples(self, n):
        '''
        Generate and save samples to
        the SQLite sample db for this
        model.
        '''
        if self.inference_method != 'sample_db':
            raise IncorrectInferenceMethodError(
                'generate_samples() not support for inference method: %s' % \
                self.inference_method)
        valid_samples = 0
        if not hasattr(self, 'sample_ordering'):
            self.sample_ordering = self.discover_sample_ordering()
        fn = [x[0].name for x in self.sample_ordering]
        sdb = self.sample_db
        while valid_samples < n:
            try:
                sample = self.get_sample()
            except InvalidSampleException:
                # TODO: Need to figure
                # out why we get invalid
                # samples.
                continue
            sdb.save_sample([(v.name, v.value) for v in sample])
            valid_samples += 1
        sdb.commit()
        print '%s samples stored in %s' % (n, self.sample_db_filename)

    def query_by_external_samples(self, **kwds):
        counts = defaultdict(int)
        samples = self.sample_db.get_samples(self.n_samples, **kwds)
        if len(samples) == 0:
            raise NoSamplesInDB(
                'There are no samples in the database. '
                'Generate some with graph.generate_samples(N).')
        if len(samples) < self.n_samples and self.enforce_minimum_samples:
            raise InsufficientSamplesException(
                'There are less samples in the sampling '
                'database than are required by this graph. '
                'Either generate more samples '
                '(graph.generate_samples(N) or '
                'decrease the number of samples '
                'required for querying (graph.n_samples). ')
        for sample in samples:
            for name, val in sample.items():
                key = (name, val)
                counts[key] += 1
        normalized = dict(
            [(k, v / len(samples)) for k, v in counts.items()])
        return normalized


    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            if isinstance(node, FactorNode):
                fh.write('  %s [ shape="rectangle" color="red"];\n' % node.name)
            else:
                fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
        for node in self.nodes:
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edge = tuple(sorted(edge))
                edges.add(edge)
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')


def build_graph(*args, **kwds):
    '''
    Automatically create all the
    variable and factor nodes
    using only function definitions.
    Since its cumbersome to supply
    the domains for variable nodes
    via the factor domains perhaps
    we should allow a domains dict?
    '''
    # Lets start off identifying all the
    # variables by introspecting the
    # functions.
    variables = set()
    domains = kwds.get('domains', {})
    name = kwds.get('name')
    variable_nodes = dict()
    factor_nodes = []
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
        factor_node = FactorNode(factor.__name__, factor)
        #factor_node.func.domains = domains
        # Bit of a hack for now we should actually exclude variables that
        # are not parameters of this function
        factor_nodes.append(factor_node)
    for variable in variables:
        node = VariableNode(
            variable,
            domain=domains.get(variable, [True, False]))
        variable_nodes[variable] = node
    # Now we have to connect each factor node
    # to its variable nodes
    for factor_node in factor_nodes:
        factor_args = get_args(factor_node.func)
        connect(factor_node, [variable_nodes[x] for x in factor_args])
    graph = FactorGraph(variable_nodes.values() + factor_nodes, name=name)
    #print domains
    return graph
