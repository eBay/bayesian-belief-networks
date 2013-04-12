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


     fA P      fB
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

def get_args(func):
    '''
    Return the names of the arguments
    of a function as a list of strings.
    This is so that we can omit certain
    variables when we marginalize.
    Note that functions created by
    make_product_func do not return
    an argspec, so we add a argspec
    attribute at creation time.
    '''
    if hasattr(func, 'argspec'):
        return func.argspec
    return inspect.getargspec(func).args


class Message(object):

    def __init__(self, source, destination, func):
        self.source = source
        self.destination = destination
        self.func = func

    def __repr__(self):
        return '<Message from %s -> %s: %s(%s)>' % \
            (self.source, self.destination, 
             self.func, get_args(self.func))

    
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
        if factor == 1:
            continue
        args_map[factor] = get_args(factor)
        all_args += args_map[factor]
    if not all_args:
        return 1
    # Now we need to make a callable that
    # will take all the arguments and correctly
    # apply them to each factor...
    args = list(set(all_args))
    #args.sort()
    args_list = expand_parameters(args, [True, False])
    def product_func(*args):
        return factors
    product_func.argspec = args
    return product_func


def make_not_sum(exclude_var, product_func):
    '''
    Given the variable to exclude from
    the summation and a product_func
    we create a function that marginalizes
    over all other variables and
    return it as a new callable.
    '''
    args = set(get_args(product_func))
    args = list(args.difference(
            set([exclude_var])))
    args_list = expand_parameters(
        args, [True, False])
    def sum_func(exclude_var):
        summands = []
        for bindings in args_list:
            summands.append((bindings, product_func))
        return summands
    sum_func.argspec = [exclude_var]
    return sum_func
    

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
    args = set(get_args(node.func))
    
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
            get_args(factor))
    #args = list(args.difference(set([target_node.name])))

    product_func = make_product_func(factors)
    sum_func = make_not_sum(
        target_node.name, product_func)
    message = Message(node, target_node, sum_func)
    return message



def make_unity(args):
    def unity(x):
        return 1
    unity.argspec = args
    return unity


def make_variable_node_message(node, target_node):
    '''
    The rules for a variable node are:
    pass on the product of
    all neighbours including 
    itself, but excluding the
    destination node. If this
    is a leaf node then send the 
    unity function.
    '''
    factors = [1]
    neighbours = node.children + node.parents
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        message = node.received_messages[neighbour.name]
        factors.append(message.func)

    product_func = make_product_func(factors)
    message = Message(node, target_node, product_func)
    return message


class FactorNode(object):

    def __init__(self, name, func, parents=[], children=[]):
        self.name = name
        self.func = func
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}

    def send_to(self, recipient, message):
        recipient.received_messages[
            self.name] = message
        
    def make_sum(self, exclude_var):
        '''
        Sum of factors so far excludeing the
        variable that this sum will be sent 
        to. This needs to return a function
        '''
        args = [x for x in self.func_parameters if x!=exclude_var]
        
        for x in []:
            pass

    def __repr__(self):
        return '<FactorNode %s %s(%s)>' % \
            (self.name,
             self.func.__name__,
             get_args(self.func))


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
        self.received_messages = {}
        self.sent_messages = {}

    def send_to(self, recipient, message):
        recipient.received_messages[
            self.name] = message

    def __repr__(self):
        return '<VariableNode: %s>' % self.name


def pollution_func(P):
    if P == True:
        return 0.1
    elif P == False:
        return 0.9
    raise 'pdf cannot resolve for %s' % x


def smoker_func(S):
    if S == True:
        return 0.3
    if S == False:
        return 0.7


def cancer_func(P, S, C):
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
    key = key + 't' if P else key + 'f'
    key = key + 't' if S else key + 'f'
    key = key + 't' if C else key + 'f'
    return table[key]


def xray_func(C, X):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.2
    table['ff'] = 0.8
    key = ''
    key = key + 't' if c else key + 'f'
    key = key + 't' if x else key + 'f'
    return table[key]


def dyspnoea_func(C, D):
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
    pollution_fac = FactorNode('fP', pollution_func)
    smoker_fac = FactorNode('fS', smoker_func)
    cancer_fac = FactorNode('fC', cancer_func)
    xray_fac = FactorNode('fX', xray_func)
    dyspnoea_fac = FactorNode('fD', dyspnoea_func)

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

    # Step 1 
    # fP -> P
    message = make_factor_node_message(pollution_fac, pollution_var)
    pollution_fac.send_to(pollution_var, message)

    # fS -> S
    message = make_factor_node_message(smoker_fac, smoker_var)
    smoker_fac.send_to(smoker_var, message)

    # X -> fX
    message = make_variable_node_message(xray_var, xray_fac)
    xray_var.send_to(xray_fac, message)

    # D -> fD
    message = make_variable_node_message(dyspnoea_var, dyspnoea_fac)
    dyspnoea_var.send_to(dyspnoea_fac, message)

    from pprint import pprint
    pprint(pollution_var.received_messages)
    pprint(smoker_var.received_messages)
    pprint(xray_fac.received_messages)
    pprint(dyspnoea_fac.received_messages)

    # ----------- end of step 1

    import ipdb; ipdb.set_trace()

    # Step 2
    message = make_variable_node_message(pollution_var, cancer_fac)
    pollution_var.send_to(cancer_fac, message)

    message = make_variable_node_message(smoker_var, cancer_fac)
    smoker_var.send_to(cancer_fac, message)

    message = make_factor_node_message(dyspnoea_fac, cancer_var)
    dyspnoea_fac.send_to(cancer_var, message)

    message = make_factor_node_message(xray_fac, cancer_var)
    xray_fac.send_tp(cancer_var, message)

    pprint(cancer_fac.received_messages)
    pprint(cancer_var.received_messages)

    # ----------- end of step 2

    sys.exit(0)
    
    xray_var.send_to(xray_fac)
    dyspnoea_var.send_to(dyspnoea_fac)

    # Step 2
    smoker_var.send_to(cancer_fac)
    pollution_var.send_to(cancer_fac)
    
    #xray_fac.send_to(cancer_var, 
    































