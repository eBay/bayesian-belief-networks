import pytest
from bayesian.factor_graph import *


def fA(x1):
    if x1.value == True:
        return 0.1
    elif x1.value == False:
        return 0.9


def fB(x2):
    if x2.value == True:
        return 0.3
    elif x2.value == False:
        return 0.7


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
    key = key + 't' if x1.value else key + 'f'
    key = key + 't' if x2.value else key + 'f'
    key = key + 't' if x3.value  else key + 'f'
    return table[key]


def fD(x3, x4):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if x3.value else key + 'f'
    key = key + 't' if x4.value else key + 'f'
    return table[key]


def fE(x3, x5):
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if x3.value else key + 'f'
    key = key + 't' if x5.value else key + 'f'
    return table[key]


# Build the network

fA_node = FactorNode('fA', fA)
fB_node = FactorNode('fB', fB)
fC_node = FactorNode('fC', fC)
fD_node = FactorNode('fD', fD)
fE_node = FactorNode('fE', fE)

x1 = VariableNode('x1', parents=[fA_node])
x2 = VariableNode('x2', parents=[fB_node])
x3 = VariableNode('x3', parents=[fC_node])
x4 = VariableNode('x4', parents=[fD_node])
x5 = VariableNode('x5', parents=[fE_node])

# Now set the parents for the factor nodes...
fA_node.parents = []
fB_node.parents = []
fC_node.parents = [x1, x2]
fD_node.parents = [x3]
fE_node.parents = [x3]

# Now set children for Variable Nodes...
x1.children = [fC_node]
x2.children = [fC_node]
x3.children = [fD_node, fE_node]
x4.children = []
x5.children = []

# Now set the children for the factor nodes...
fA_node.children = [x1]
fB_node.children = [x2]
fC_node.children = [x3]
fD_node.children = [x4]
fE_node.children = [x5]

graph = FactorGraph([x1, x2, x3, x4, x5, fA_node, fB_node, fC_node, fD_node, fE_node])

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
    

if __name__ == '__main__':

    # Note we need to set some of the  parents and children afterwards
    fA_node = FactorNode('fA', fA)
    fB_node = FactorNode('fB', fB)
    fC_node = FactorNode('fC', fC)
    fD_node = FactorNode('fD', fD)
    fE_node = FactorNode('fE', fE)

    x1 = VariableNode('x1', parents=[fA_node])
    x2 = VariableNode('x2', parents=[fB_node])
    x3 = VariableNode('x3', parents=[fC_node])
    x4 = VariableNode('x4', parents=[fD_node])
    x5 = VariableNode('x5', parents=[fE_node])

    # Now set the parents for the factor nodes...
    fA_node.parents = []
    fB_node.parents = []
    fC_node.parents = [x1, x2]
    fD_node.parents = [x3]
    fE_node.parents = [x3]

    # Now set children for Variable Nodes...
    x1.children = [fC_node]
    x2.children = [fC_node]
    x3.children = [fD_node, fE_node]
    x4.children = []
    x5.children = []

    # Now set the children for the factor nodes...
    fA_node.children = [x1]
    fB_node.children = [x2]
    fC_node.children = [x3]
    fD_node.children = [x4]
    fE_node.children = [x5]

    # Now we will start the algorithm to compute the marginals
    # Step 1 
    # fA -> x1
    step = 1
    print 'Step 1'
    message = make_factor_node_message(fA_node, x1)
    fA_node.send_to(x1, message)

    # fB -> x2
    message = make_factor_node_message(fB_node, x2)
    fB_node.send_to(x2, message)

    # x4 -> fD
    message = make_variable_node_message(x4, fD_node)
    x4.send_to(fD_node, message)

    # x5 -> fE
    message = make_variable_node_message(x5, fE_node)
    x5.send_to(fE_node, message)

    from pprint import pprint
    #pprint(x1.received_messages)
    #pprint(x2.received_messages)
    #pprint(fD_node.received_messages)
    #pprint(fE_node.received_messages)

    print 'End of Step 1.'
    print '----------------------------------------------------------------------'

    # ----------- end of step 1

    print 'Step 2'
    step = 2

    # Step 2
    # x1 already has a message so it just passes that along since
    # apart from the destination it has no other neighbours
    message = x1.received_messages['fA']
    x1.send_to(fC_node, message)

    message = x2.received_messages['fB']
    x2.send_to(fC_node, message)

    # Now when fd sends to x3 it needs to *add itself* as a factor too.
    # and them sum over the product of all factors.
    message = make_factor_node_message(fD_node, x3)
    fD_node.send_to(x3, message)

    # Now 
    message = make_factor_node_message(fE_node, x3)
    fE_node.send_to(x3, message)
    #print 'At fC_node:'
    #pprint(fC_node.received_messages)

    #print 'At x3 node:'
    #pprint(x3.received_messages)

    print 'End of Step 2.'
    print '----------------------------------------------------------------------'

    # ----------- end of step 2

    step = 3
    # Step 3
    print 'Step 3'


    message = make_factor_node_message(fC_node, x3)
    fC_node.send_to(x3, message)

    message = make_variable_node_message(x3, fC_node)
    x3.send_to(fC_node, message)

    #print 'At x3 node:'
    #pprint(x3.received_messages)
    
    #print 'At fC_node:'
    #pprint(fC_node.received_messages)


    print 'End of Step 3.'
    print '-----------------------------------------------------------------------'


    step = 4

    # Step 4
    print 'Step 4'

    ###### To make it easiser to see I will print the messages as I go along

    message = make_factor_node_message(fC_node, x1)
    fC_node.send_to(x1, message)

    message = make_factor_node_message(fC_node, x2)
    fC_node.send_to(x2, message)

    
    message = make_variable_node_message(x3, fD_node)
    x3.send_to(fD_node, message)

    message = make_variable_node_message(x3, fE_node)
    x3.send_to(fE_node, message)


    #print 'At x1 node:'
    #pprint(x1.received_messages)


    #print 'At x2 node:'
    #pprint(x2.received_messages)

    
    #print 'At fD_node:'
    #pprint(fD_node.received_messages)

    #print 'At fE_node:'
    #pprint(fE_node.received_messages)


    print 'End of Step 4.'
    print '-----------------------------------------------------------------------'

    step = 5
    # Step 5
    print 'Step 5'



    message = make_factor_node_message(fD_node, x4)
    fD_node.send_to(x4, message)

    message = make_factor_node_message(fE_node, x5)
    fE_node.send_to(x5, message)

    
    message = make_variable_node_message(x1, fA_node)
    x1.send_to(fA_node, message)

    message = make_variable_node_message(x2, fB_node)
    x2.send_to(fB_node, message)

    print 'End of Step 5.'
    print '-----------------------------------------------------------------------'


    print 'Messages at all nodes:'
    for node in [x1, x2, x3, x4, x5, fA_node, fB_node, fC_node, fD_node, fE_node]:
        node.message_report()
    x1.message_report()

    # Create a test variable to play with
    t1 = VariableNode('x1')
    t1.value = True

    #v2 = VariableNode('v2')
    #v2.value = False

    res = x1.received_messages['fC'](t1)
    print '-----------------------------------------------------------------------'
    print 'Un-normalized Marginals'
    print '-----------------------------------------------------------------------'
    marginals = dict()
    for node in [x1, x2, x3, x4, x5]:
        for value in [True]:
            marginals[node.name] = round(node.marginal(value), 3)
            print node.name, value, marginals[node.name]

    # Now since we added evidence for x5 we need to 
    # use the un-normalized marginal of x5 as the normalizer
    normalizer = marginals['x2']

    print '-----------------------------------------------------------------------'
    print 'Normalized Marginals'
    print '-----------------------------------------------------------------------'
    marginals = dict()
    for node in [x1, x2, x3, x4, x5]:
        for value in [True]:
            marginals[node.name] = round(node.marginal(value, normalizer), 3)
            print node.name, value, marginals[node.name]

    # See BAI_Chapter2 Table 2.2
    #assert marginals['x1'] == 0.100
    #assert marginals['x2'] == 0.300
    #assert marginals['x3'] == 0.012  # Note difference in rounding from Table 2.2
    #assert marginals['x4'] == 0.208
    #assert marginals['x5'] == 0.304

    # These are the assertions for the normalized marginals for D=T (x5=True)
    #assert marginals['x1']  == 0.102
    #assert marginals['x2']  == 0.307
    #assert marginals['x3']  == 0.025
    #assert marginals['x4']  == 0.217
    #assert marginals['x5']  == 1.000

    # These are the assertions for the normalized marginals for S=T (x2=True)
    assert marginals['x1']  == 0.100
    assert marginals['x2']  == 1.000
    assert marginals['x3']  == 0.032
    assert marginals['x4']  == 0.222
    assert marginals['x5']  == 0.311


    print 'Yippee! All assertions past!'

    # Now we will add evidence by setting D=T. (x5=True) remember to normalize!



