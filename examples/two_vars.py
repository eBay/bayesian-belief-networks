from factor_graph import *


# sunny
def fA(x1):
    if x1.value == True:
        return 0.7
    elif x1.value == False:
        return 0.3

# go cycling
def fB(x1, x2):
    table = dict()
    table['tt'] = 0.20
    table['tf'] = 0 #0.80
    table['ft'] = 0.05
    table['ff'] = 0 #0.95
    key = ''
    key = key + 't' if x1.value else key + 'f'
    key = key + 't' if x2.value else key + 'f'
    return table[key]


# One way to add 'evidence' is to rewrite all
# factor functions with the non observed discrete values set to 0 
def fB(x1, x2):
    table = dict()
    table['tt'] = 0.20 / 0.5
    table['tf'] = 0 #0.80
    table['ft'] = 0.05 / 0.5
    table['ff'] = 0 #0.95
    key = ''
    key = key + 't' if x1.value else key + 'f'
    key = key + 't' if x2.value else key + 'f'
    return table[key]



if __name__ == '__main__':

    # Note we need to set some of the  parents and children afterwards
    fA_node = FactorNode('fA', fA)
    fB_node = FactorNode('fB', fB)

    x1 = VariableNode('x1', parents=[fA_node])
    x2 = VariableNode('x2', parents=[fB_node])

    # Now set the parents for the factor nodes...
    fA_node.parents = []
    fB_node.parents = [x1]

    # Now set children for Variable Nodes...
    x1.children = [fB_node]
    x2.children = []

    # Now set the children for the factor nodes...
    fA_node.children = [x1]
    fB_node.children = [x2]

    # Now we will start the algorithm to compute the prior

    # Step 1 
    # fA -> x1
    step = 1
    print 'Step 1'
    message = make_factor_node_message(fA_node, x1)
    fA_node.send_to(x1, message)

    # x2 -> fB
    message = make_variable_node_message(x2, fB_node)
    x2.send_to(fB_node, message)

    print 'End of Step 1.'
    print '----------------------------------------------------------------------'

    # ----------- end of step 1

    print 'Step 2'
    step = 2

    # Step 2
    # x1 already has a message so it just passes that along since
    # apart from the destination it has no other neighbours
    message = x1.received_messages['fA']
    x1.send_to(fB_node, message)

    # Now when fd sends to x3 it needs to *add itself* as a factor too.
    # and them sum over the product of all factors.
    message = make_factor_node_message(fB_node, x1)
    fB_node.send_to(x1, message)

    print 'End of Step 2.'
    print '----------------------------------------------------------------------'

    # ----------- end of step 2

    step = 3
    # Step 3
    print 'Step 3'


    message = make_variable_node_message(x1, fA_node)
    x1.send_to(fA_node, message)

    message = make_factor_node_message(fB_node, x2)
    fB_node.send_to(x2, message)

    print 'End of Step 3.'
    print '-----------------------------------------------------------------------'

    print 'Messages at all nodes:'
    for node in [x1, x2, fA_node, fB_node]:
        node.message_report()

    print '-----------------------------------------------------------------------'
    print 'Marginals'
    print '-----------------------------------------------------------------------'
    for node in [x1, x2]:
        for value in [True, False]:
            print node.name, value, node.marginal(value)


    print '-------------------------'
    print 'Joint Marginals at fB'
    print '-------------------------'
    v1 = VariableNode('x1')
    v2 = VariableNode('x2')
    for a in [True, False]:
        for b in [True, False]:
            v1.value = a
            v2.value = b
            val_dict = dict(x1=v1, x2=v2)
            print a, b, fB_node.marginal(val_dict)


    # Now according to Bishop to get 
    # the marginal probability given some evidence
    # we divide the graph into two portions h and v
    # where h represents hidden nodes and v observed

    # So lets say I am biking ie x2=True
    # What is the likelihood its sunny?

    # So we start by multiplying p(x) by I(v, v')
    # where I(x,x') = 1 if x=x' 0 otherwise

    # so p(x) * 1 ???? where do we plug this in????

    # Alternately lets alter the function FB_node





