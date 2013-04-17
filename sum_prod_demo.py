from factor_graph import *


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


if __name__ == '__main__':

    # Now lets try and set x5=True as an observed value
    # One way to do this is to replace all functions 
    # taking x5 with a new function replacing the x5
    # with True and therefore eliminating it...
    # Lets try....
    #def fE_(x3):
    #    return fE(x3, True)
    #fE_.argspec = ['x3']


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

    # Now we will start the algorithm to compute the prior

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

    import ipdb; ipdb.set_trace()
    res = x1.received_messages['fC'](t1)
    print '-----------------------------------------------------------------------'
    print 'Marginals'
    print '-----------------------------------------------------------------------'
    for node in [x1, x2, x3, x4, x5]:
        for value in [True]:
            print node.name, value, node.marginal(value)


    



