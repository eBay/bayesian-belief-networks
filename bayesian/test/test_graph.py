import pytest

import os

from bayesian.factor_graph import *
from bayesian.gaussian_bayesian_network import build_gbn
from bayesian.examples.gaussian_bayesian_networks.river import (
    f_a, f_b, f_c, f_d)


def pytest_funcarg__river_graph(request):
    g = build_gbn(f_a, f_b, f_c, f_d)
    return g


def fA(x1):
    if x1 is True:
        return 0.1
    elif not x1:
        return 0.9

fA.domains = dict(x1=[True, False])


def fB(x2):
    if x2 is True:
        return 0.3
    elif not x2:
        return 0.7

fB.domains = dict(x2=[True, False])


def pytest_funcarg__eliminate_var_factor(request):
    '''Get a factor to test variable elimination'''

    def factor(x1, x2, x3):
        table = dict()
        table['ttt'] = 0.05
        table['ttf'] = 0.95
        table['tft'] = 0.02
        table['tff'] = 0.98
        table['ftt'] = 0.03
        table['ftf'] = 0.97
        table['fft'] = 0.001
        table['fff'] = 0.999
        key = ''
        key = key + 't' if x1 else key + 'f'
        key = key + 't' if x2 else key + 'f'
        key = key + 't' if x3 else key + 'f'
        return table[key]

    factor.domains = dict(
        x1=[True, False],
        x2=[True, False],
        x3=[True, False])

    return factor


def fC(x1, x2, x3):
    '''
    This needs to be a joint probability distribution
    over the inputs and the node itself
    '''
    table = dict()
    table['ttt'] = 0.05
    table['ttf'] = 0.95
    table['tft'] = 0.02
    table['tff'] = 0.98
    table['ftt'] = 0.03
    table['ftf'] = 0.97
    table['fft'] = 0.001
    table['fff'] = 0.999
    key = ''
    key = key + 't' if x1 else key + 'f'
    key = key + 't' if x2 else key + 'f'
    key = key + 't' if x3 else key + 'f'
    return table[key]


fC.domains = dict(
    x1=[True, False],
    x2=[True, False],
    x3=[True, False])


def fD(x3, x4):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if x3 else key + 'f'
    key = key + 't' if x4 else key + 'f'
    return table[key]

fD.domains = dict(
    x3=[True, False],
    x4=[True, False])


def fE(x3, x5):
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if x3 else key + 'f'
    key = key + 't' if x5 else key + 'f'
    return table[key]

fE.domains = dict(
    x3=[True, False],
    x5=[True, False])

# Build the network

fA_node = FactorNode('fA', fA)
fB_node = FactorNode('fB', fB)
fC_node = FactorNode('fC', fC)
fD_node = FactorNode('fD', fD)
fE_node = FactorNode('fE', fE)

x1 = VariableNode('x1')
x2 = VariableNode('x2')
x3 = VariableNode('x3')
x4 = VariableNode('x4')
x5 = VariableNode('x5')

connect(fA_node, x1)
connect(fB_node, x2)
connect(fC_node, [x1, x2, x3])
connect(fD_node, [x3, x4])
connect(fE_node, [x3, x5])


def test_connect():
    assert fA_node.neighbours == [x1]
    assert fB_node.neighbours == [x2]
    assert fC_node.neighbours == [x1, x2, x3]
    assert fD_node.neighbours == [x3, x4]
    assert fE_node.neighbours == [x3, x5]
    assert x1.neighbours == [fA_node, fC_node]
    assert x2.neighbours == [fB_node, fC_node]
    assert x3.neighbours == [fC_node, fD_node, fE_node]
    assert x4.neighbours == [fD_node]
    assert x5.neighbours == [fE_node]


graph = FactorGraph([x1, x2, x3, x4, x5,
                     fA_node, fB_node, fC_node, fD_node, fE_node])


def test_variable_node_is_leaf():
    assert not x1.is_leaf()
    assert not x2.is_leaf()
    assert not x3.is_leaf()
    assert x4.is_leaf()
    assert x5.is_leaf()


def test_factor_node_is_leaf():
    assert fA_node.is_leaf()
    assert fB_node.is_leaf()
    assert not fC_node.is_leaf()
    assert not fD_node.is_leaf()
    assert not fE_node.is_leaf()


def test_graph_get_leaves():
    assert graph.get_leaves() == [x4, x5, fA_node, fB_node]


# Tests at step 1
def test_graph_get_step_1_eligible_senders():
    eligible_senders = graph.get_eligible_senders()
    assert eligible_senders == [x4, x5, fA_node, fB_node]


def test_node_get_step_1_target():
    assert x1.get_target() is None
    assert x2.get_target() is None
    assert x3.get_target() is None
    assert x4.get_target() == fD_node
    assert x5.get_target() == fE_node
    assert fA_node.get_target() == x1
    assert fB_node.get_target() == x2
    assert fC_node.get_target() is None
    assert fD_node.get_target() is None
    assert fE_node.get_target() is None


def test_construct_message():
    message = x4.construct_message()
    assert message.source.name == 'x4'
    assert message.destination.name == 'fD'
    assert message.argspec == []
    assert message.factors == [1]
    message = x5.construct_message()
    assert message.source.name == 'x5'
    assert message.destination.name == 'fE'
    assert message.argspec == []
    assert message.factors == [1]
    message = fA_node.construct_message()
    assert message.source.name == 'fA'
    assert message.destination.name == 'x1'
    assert message.argspec == ['x1']
    assert message.factors == [fA_node.func]
    message = fB_node.construct_message()
    assert message.source.name == 'fB'
    assert message.destination.name == 'x2'
    assert message.argspec == ['x2']
    assert message.factors == [fB_node.func]


def test_send_message():
    message = x4.construct_message()
    x4.send(message)
    assert message.destination.received_messages['x4'] == message
    message = x5.construct_message()
    x5.send(message)
    assert message.destination.received_messages['x5'] == message
    message = fA_node.construct_message()
    fA_node.send(message)
    assert message.destination.received_messages['fA'] == message
    message = fB_node.construct_message()
    fB_node.send(message)
    assert message.destination.received_messages['fB'] == message


def test_sent_messages():
    sent = x4.get_sent_messages()
    assert sent['fD'] == fD_node.received_messages['x4']
    sent = x5.get_sent_messages()
    assert sent['fE'] == fE_node.received_messages['x5']
    sent = fA_node.get_sent_messages()
    assert sent['x1'] == x1.received_messages['fA']
    sent = fB_node.get_sent_messages()
    assert sent['x2'] == x2.received_messages['fB']


# Step 2
def test_node_get_step_2_target():
    assert x1.get_target() == fC_node
    assert x2.get_target() == fC_node


def test_graph_reset():
    graph.reset()
    for node in graph.nodes:
        assert node.received_messages == {}


def test_propagate():
    graph.reset()
    graph.propagate()
    for node in graph.nodes:
        node.message_report()


def marg(x, val, normalizer=1.0):
    return round(x.marginal(val, normalizer), 3)


def test_marginals():
    m = marg(x1, True)
    assert m == 0.1
    m = marg(x1, False)
    assert m == 0.9
    m = marg(x2, True)
    assert m == 0.3
    m = marg(x2, False)
    assert m == 0.7
    m = marg(x3, True)
    assert m == 0.012  # Note slight rounding difference to BAI
    m = marg(x3, False)
    assert m == 0.988
    m = marg(x4, True)
    assert m == 0.208
    m = marg(x4, False)
    assert m == 0.792
    m = marg(x5, True)
    assert m == 0.304
    m = marg(x5, False)
    assert m == 0.696


def test_add_evidence():
    '''
    We will set x5=True, this
    corresponds to variable D in BAI
    '''
    graph.reset()
    add_evidence(x5, True)
    graph.propagate()
    normalizer = marg(x5, True)
    assert normalizer == 0.304
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 0.307
    m = marg(x2, False, normalizer)
    assert m == 0.693
    m = marg(x3, True, normalizer)
    assert m == 0.025
    m = marg(x3, False, normalizer)
    assert m == 0.975
    m = marg(x4, True, normalizer)
    assert m == 0.217
    m = marg(x4, False, normalizer)
    assert m == 0.783
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0


def test_add_evidence_x2_true():
    '''
    x2 = S in BAI
    '''
    graph.reset()
    add_evidence(x2, True)
    graph.propagate()
    normalizer = marg(x2, True)
    m = marg(x1, True, normalizer)
    assert m == 0.1
    m = marg(x1, False, normalizer)
    assert m == 0.9
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.032
    m = marg(x3, False, normalizer)
    assert m == 0.968
    m = marg(x4, True, normalizer)
    assert m == 0.222
    m = marg(x4, False, normalizer)
    assert m == 0.778
    m = marg(x5, True, normalizer)
    assert m == 0.311
    m = marg(x5, False, normalizer)
    assert m == 0.689


def test_add_evidence_x3_true():
    '''
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.249
    m = marg(x1, False, normalizer)
    assert m == 0.751
    m = marg(x2, True, normalizer)
    assert m == 0.825
    m = marg(x2, False, normalizer)
    assert m == 0.175
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350


def test_add_evidence_x2_true_and_x3_true():
    '''
    x2 = True in BAI this is Smoker = True
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x2, True)
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.156
    m = marg(x1, False, normalizer)
    assert m == 0.844
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350


def test_add_evidence_x5_true_x2_true():
    graph.reset()
    add_evidence(x5, True)
    add_evidence(x2, True)
    graph.propagate()
    normalizer = x5.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.067
    m = marg(x3, False, normalizer)
    assert m == 0.933
    m = marg(x4, True, normalizer)
    assert m == 0.247
    m = marg(x4, False, normalizer)
    assert m == 0.753
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0


# Now we are going to test based on the second
# half of table 2.2 where the prior for prior
# for the Smoking parameter (x2=True) is
# set to 0.5. We start by redefining the
# PMF for fB and then rebuilding the factor
# graph


def test_marginals_table_22_part_2_x2_prior_change():
    def fB(x2):
        if x2 is True:
            return 0.5
        elif not x2:
            return 0.5
    fB.domains = dict(x2=[True, False])

    # Build the network
    fA_node = FactorNode('fA', fA)
    fB_node = FactorNode('fB', fB)
    fC_node = FactorNode('fC', fC)
    fD_node = FactorNode('fD', fD)
    fE_node = FactorNode('fE', fE)

    x1 = VariableNode('x1')
    x2 = VariableNode('x2')
    x3 = VariableNode('x3')
    x4 = VariableNode('x4')
    x5 = VariableNode('x5')

    connect(x1, [fA_node, fC_node])
    connect(x2, [fB_node, fC_node])
    connect(x3, [fC_node, fD_node, fE_node])
    connect(x4, fD_node)
    connect(x5, fE_node)

    nodes = [x1, x2, x3, x4, x5, fA_node, fB_node, fC_node, fD_node, fE_node]

    graph = FactorGraph(nodes)
    graph.propagate()
    m = marg(x1, True)
    assert m == 0.1
    m = marg(x1, False)
    assert m == 0.9
    m = marg(x2, True)
    assert m == 0.5
    m = marg(x2, False)
    assert m == 0.5
    m = marg(x3, True)
    assert m == 0.017
    m = marg(x3, False)
    assert m == 0.983
    m = marg(x4, True)
    assert m == 0.212
    m = marg(x4, False)
    assert m == 0.788
    m = marg(x5, True)
    assert m == 0.306
    m = marg(x5, False)
    assert m == 0.694

    # Now set D=T (x5=True)
    graph.reset()
    add_evidence(x5, True)
    graph.propagate()
    normalizer = marg(x5, True)
    assert normalizer == 0.306
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 0.508
    m = marg(x2, False, normalizer)
    assert m == 0.492
    m = marg(x3, True, normalizer)
    assert m == 0.037
    m = marg(x3, False, normalizer)
    assert m == 0.963
    m = marg(x4, True, normalizer)
    assert m == 0.226
    m = marg(x4, False, normalizer)
    assert m == 0.774
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0

    graph.reset()
    add_evidence(x2, True)
    graph.propagate()
    normalizer = marg(x2, True)
    m = marg(x1, True, normalizer)
    assert m == 0.1
    m = marg(x1, False, normalizer)
    assert m == 0.9
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.032
    m = marg(x3, False, normalizer)
    assert m == 0.968
    m = marg(x4, True, normalizer)
    # Note that in Table 2.2 x4 and x5 marginals are reversed:
    assert m == 0.222
    m = marg(x4, False, normalizer)
    assert m == 0.778
    m = marg(x5, True, normalizer)
    assert m == 0.311
    m = marg(x5, False, normalizer)
    assert m == 0.689

    '''
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.201
    m = marg(x1, False, normalizer)
    assert m == 0.799
    m = marg(x2, True, normalizer)
    assert m == 0.917
    m = marg(x2, False, normalizer)
    assert m == 0.083
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350

    '''
    x2 = True in BAI this is Smoker = True
    x3 = True in BAI this is Cancer = True
    '''
    graph.reset()
    add_evidence(x2, True)
    add_evidence(x3, True)
    graph.propagate()
    normalizer = x3.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.156
    m = marg(x1, False, normalizer)
    assert m == 0.844
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 1.0
    m = marg(x3, False, normalizer)
    assert m == 0.0
    m = marg(x4, True, normalizer)
    assert m == 0.9
    m = marg(x4, False, normalizer)
    assert m == 0.1
    m = marg(x5, True, normalizer)
    assert m == 0.650
    m = marg(x5, False, normalizer)
    assert m == 0.350

    graph.reset()
    add_evidence(x5, True)
    add_evidence(x2, True)
    graph.propagate()
    normalizer = x5.marginal(True)
    m = marg(x1, True, normalizer)
    assert m == 0.102
    m = marg(x1, False, normalizer)
    assert m == 0.898
    m = marg(x2, True, normalizer)
    assert m == 1.0
    m = marg(x2, False, normalizer)
    assert m == 0.0
    m = marg(x3, True, normalizer)
    assert m == 0.067
    m = marg(x3, False, normalizer)
    assert m == 0.933
    m = marg(x4, True, normalizer)
    assert m == 0.247
    m = marg(x4, False, normalizer)
    assert m == 0.753
    m = marg(x5, True, normalizer)
    assert m == 1.0
    m = marg(x5, False, normalizer)
    assert m == 0.0


def test_verify_node_neighbour_type():

    def fA(x1):
        return 0.5

    fA_node = FactorNode('fA', fA)

    x1 = VariableNode('x1')

    connect(fA_node, x1)
    assert fA_node.verify_neighbour_types() is True
    assert x1.verify_neighbour_types() is True

    x2 = VariableNode('x2')
    x3 = VariableNode('x3')
    connect(x2, x3)
    assert x2.verify_neighbour_types() is False
    assert x3.verify_neighbour_types() is False


def test_verify_graph():
    def fA(x1):
        return 0.5

    def fB(x2):
        return 0.5

    fA_node = FactorNode('fA', fA)
    fB_node = FactorNode('fB', fB)

    x1 = VariableNode('x1')
    x2 = VariableNode('x2')

    connect(fA_node, x1)
    graph = FactorGraph([fA_node, x1])
    assert graph.verify() is True

    connect(fA_node, fB_node)
    graph = FactorGraph([fA_node, fB_node])
    assert graph.verify() is False

    connect(x1, x2)
    graph = FactorGraph([x1, x2])
    assert graph.verify() is False


def test_set_func_domains_from_variable_domains():
    def fA(x1):
        return 0.5

    def fB(x2):
        return 0.5

    x1 = VariableNode('x1', domain=['high', 'low'])
    fA_node = FactorNode('fA', fA)
    connect(x1, fA_node)
    graph = FactorGraph([x1, fA_node])
    assert fA_node.func.domains == dict(x1=['high', 'low'])

    x2 = VariableNode('x2')
    fB_node = FactorNode('fB', fB)
    connect(x2, fB_node)
    graph = FactorGraph([x2, fB_node])
    assert fB_node.func.domains == dict(x2=[True, False])


def test_discover_sample_ordering():

    def fActualDoor(ActualDoor):
        return 1.0 / 3

    def fGuestDoor(GuestDoor):
        return 1.0 / 3

    def fMontyDoor(ActualDoor, GuestDoor, MontyDoor):
        if ActualDoor == GuestDoor:
            if GuestDoor == MontyDoor:
                return 0
            else:
                return 0.5
        if GuestDoor == MontyDoor:
            return 0
        if ActualDoor == MontyDoor:
            return 0
        return 1

    # Build the network
    fActualDoor_node = FactorNode('fActualDoor', fActualDoor)
    fGuestDoor_node = FactorNode('fGuestDoor', fGuestDoor)
    fMontyDoor_node = FactorNode('fMontyDoor', fMontyDoor)

    ActualDoor = VariableNode('ActualDoor', ['A', 'B', 'C'])
    GuestDoor = VariableNode('GuestDoor', ['A', 'B', 'C'])
    MontyDoor = VariableNode('MontyDoor', ['A', 'B', 'C'])

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

    assert graph.verify() is True
    ordering = graph.discover_sample_ordering()
    assert len(ordering) == 3
    assert ordering[0][0].name == 'ActualDoor'
    assert ordering[0][1].__name__ == 'fActualDoor'
    assert ordering[1][0].name == 'GuestDoor'
    assert ordering[1][1].__name__ == 'fGuestDoor'
    assert ordering[2][0].name == 'MontyDoor'
    assert ordering[2][1].__name__ == 'fMontyDoor'


def test_sample_db_filename():
    graph = FactorGraph([], name='model_1')
    home = os.path.expanduser('~')
    expected_filename = os.path.join(
        home,
        '.pypgm',
        'data',
        'model_1.sqlite')
    assert graph.sample_db_filename == expected_filename


def test_eliminate_var(eliminate_var_factor):

    eliminated = eliminate_var(eliminate_var_factor, 'x2')
    assert eliminated.argspec == ['x1', 'x3']
    assert eliminated(True, True) == 0.07


class TestGraphModule(object):

    def test_get_topological_sort(self, river_graph):
        ordering = river_graph.get_topological_sort()
        assert len(ordering) == 4
        assert ordering[0].name == 'f_a'
        assert ordering[1].name == 'f_b'
        assert ordering[2].name == 'f_c'
        assert ordering[3].name == 'f_d'
