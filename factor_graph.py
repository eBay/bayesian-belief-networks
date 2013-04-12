'''From SumProd.pdf'''
import sys
import inspect

'''
The example in SumProd.pdf has exactly the same shape 
as the cancer example:

(Note arrows are from x1->x3, x2->x3, x3->x4 and x3->x5)

      x1      x2
       \      /
        \    /
         \  /
          x3
         /  \
        /    \
       /      \
      x4      x5


The equivalent Factor Graph is:


     fA        fB
     |          |
     x1---fC---x2
           |
     fD---x3---fE
     |          |
     x4         x5


fA(x1) = p(x1)
fB(x2) = p(x2)
fC(x1,x2,x3) = p(x3|x1,x2)
fD(x3,x4) = p(x4|x3)
fE(x3,x5) = p(x5|x3)

Lets simulate this, from SumProd.pdf:

Step1:
Start at all the leaf nodes ie fA, fB, x4, x5

mfA -> x1 = fA(x1)  ie this is passing along a function and a parameter
mfB -> x2 = fB(x2) 
mx4 -> fD = 1       when starting from a variable node we always send a 1 (Not sure for the constrained version)
mx5 -> fE = 1

So at this point we have recorded the message *at* the recipient

Step 2:
mx1 -> fC = fA(x1)  This is the same message that fA passed to x1 in step 1
mx2 -> fC = fB(x2)  This is the same message that fB passed to x2 in step 1
mfD -> x3 = sum(fD(x3,x4) * mx4 -> fD (Note that the sum should *exclude* x3 and that the mx4->fd messages is 1
mfE -> x3 = sum(fE(x3,x5) * mx5 -> fE

???? Parts I dont understand is *when* is anything actually evaluated?
It seems that messages are either a 1 (if it originiated at a leaf variable) or they are a sum of functions
with unbound variables or do the factor nodes substitute the value they got into the equation?????

Only thing to do is try it!

Converting the cancer example to a factor graph..

'''

class Message(object):

    def __init__(self, argspec, func):
        self.args = args
        self.func = func

    


def make_product_func(factors):
    '''
    Return a single callable from
    a list of factors which correctly
    applies the arguments to each 
    individual factor
    '''
    args_map = {}
    all_args = []
    for factor in factors:
        args_map[factor] = inspect.getargspec(factor).args
        all_args += args_map[factor]
    # Now we need to make a callable that
    # will take all the arguments and correctly
    # apply them to each factor...
    args = list(set(all_args))
    args.sort()
    def product_func(*args):
        print args_map
        return 1
    product_func.argspec = args
    return product_func
        
    

    


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
    args = set(inspect.getargspec(node.func))
    
    # Compile list of factors for message
    factors = [node.func]
    
    # Now add the message that came from each
    # of the non-destination neighbours...
    neighbours = node.children + node.parents
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        factors.append(node.received_messages[neighbour.name])

    # Now we need to add any other variables 
    # that were added from the other factors
    for factor in factors:
        args = args.union(
            inpect.getargspec(factor).args)

    args = args.difference(set([target_node.name]))

    # Now we sum over every variable in every factor
    # that will comprise this message except for 
    # the target node...
    summands = []
    args_list = expand_parameters(args, [True, False])
    
    for bindings in args_list:
        summands.append(bind_apply)
        

    return args




class FactorNode(object):

    def __init__(self, func, parents=[], children=[]):
        self.func = func
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}
        self.bindings = dict()

    def send_to(self, recipient):
        recipient.received_messages.append((self.func, self.bindings))
        

    def make_sum(self, exclude_var):
        '''
        Sum of factors so far excludeing the
        variable that this sum will be sent 
        to. This needs to return a function
        '''
        args = [x for x in self.func_parameters if x!=exclude_var]
        
        for x in []:
            pass


def expand_parameters(args, vals):
    '''
    Given a list of args and values
    we return a list of tuples
    containing all possible n length
    sequences of vals where n is the
    length of args.
    '''

    result = []
    if not args:
        return [result]
    rest = expand_parameters(args[1:], vals)
    for r in rest:
        result.append([True] + r)
        result.append([False] + r)
    return result



class VariableNode(object):
    
    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents
        self.children = children
        self.received_messages = []
        self.sent_messages = []
        self.current_value = 1

    def send_to(self, recipient):
        recipient.received_messages.append({self.name:self.current_value})


def pollution_func(p):
    if p == True:
        return 0.1
    elif p == False:
        return 0.9
    raise 'pdf cannot resolve for %s' % x


def smoker_func(s):
    if s == True:
        return 0.3
    if s == False:
        return 0.7


def cancer_func(p, s, c):
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
    key = key + 't' if p else key + 'f'
    key = key + 't' if s else key + 'f'
    key = key + 't' if c else key + 'f'
    return table[key]


def xray_func(c, x):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if c else key + 'f'
    key = key + 't' if x else key + 'f'
    return table[key]


def dyspnoea_func(c, d):
    table = dict()
    table['tt'] = 0.65
    table['tf'] = 0.35
    table['ft'] = 0.3
    table['ff'] = 0.7
    key = ''
    key = key + 't' if c else key + 'f'
    key = key + 't' if d else key + 'f'
    return table[key]


if __name__ == '__main__':
    # Note we need to set some of the  parents and children afterwards
    pollution_fac = FactorNode(pollution_func)
    smoker_fac = FactorNode(smoker_func)
    cancer_fac = FactorNode(cancer_func)
    xray_fac = FactorNode(xray_func)
    dyspnoea_fac = FactorNode(dyspnoea_func)

    pollution_var = VariableNode('P', parents=[pollution_fac])
    smoker_var = VariableNode('S', parents=[smoker_fac])
    cancer_var = VariableNode('C', parents=[cancer_fac])
    xray_var = VariableNode('X', parents=[xray_fac])
    dyspnoea_var = VariableNode('D', parents=[dyspnoea_fac])

    # Now set the parents for the factor nodes...
    pollution_fac.parents = []
    smoker_fac.parents = []
    cancer_fac.parents = [pollution_var, smoker_var]
    xray_fac.parents = [cancer_var]
    dyspnoea_fac.parents = [dyspnoea_var]

    # Now set children for Variable Nodes...
    pollution_var.children = [cancer_fac]
    smoker_var.children = [cancer_fac]
    cancer_var.children = [xray_fac, dyspnoea_fac]
    xray_var.children = []
    dyspnoea_var.children = []

    # Now set the children for the factor nodes...
    pollution_fac.children = [pollution_var]
    smoker_fac.children = [smoker_var]
    cancer_fac.children = [cancer_var]
    xray_fac.children = [xray_var]
    dyspnoea_fac.children = [dyspnoea_var]

    # Now we will start the algorithm to compute the prior

    print make_factor_node_message(pollution_fac, pollution_var)
    sys.exit(0)


    # Step 1 
    pollution_fac.send_to(pollution_var) # The empty dict means nothing is bound so far...
    smoker_fac.send_to(smoker_var)
    
    xray_var.send_to(xray_fac)
    dyspnoea_var.send_to(dyspnoea_fac)
    import ipdb; ipdb.set_trace()
    # ----------- end of step 1

    # Step 2
    smoker_var.send_to(cancer_fac)
    pollution_var.send_to(cancer_fac)
    
    #xray_fac.send_to(cancer_var, 
    































