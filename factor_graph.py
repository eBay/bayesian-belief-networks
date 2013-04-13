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


class Node(object):


    def is_leaf(self):
        if not (self.parents and self.children):
            return True
        return False

    def send_to(self, recipient, message):
        print message
        recipient.received_messages[
            self.name] = message


class VariableNode(Node):
    
    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}


    def __repr__(self):
        return '<VariableNode: %s>' % self.name


class FactorNode(Node):

    def __init__(self, name, func, parents=[], children=[]):
        self.name = name
        self.func = func
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}

        
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


class Message(object):

    def __init__(self, source, destination, func):
        self.source = source
        self.destination = destination
        self.func = func
        self.argspec = get_args(self.func)

    def __call__(self, *args):
        return self.func(*args)


    def __repr__(self):
        return '<Message from %s -> %s: %s(%s)>' % \
            (self.source, self.destination, 
             self.func, get_args(self.func))


class MessageProduct(object):
    '''
    Represents a product of messages
    that a Variable Node sends to
    a factor node
    '''

    def __init__(self, source, destination, messages):
        self.source = source
        self.destination = destination
        self.messages = messages


    def __repr__(self):
        return '<MessageProduct: %s -> %s %s>' % (
            self.source.name,
            self.destination.name,
            self.messages)



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
        message = Message(node, target_node, node.func)
        return message

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
        unity_func = make_unity([node.name])
        message = Message(node, target_node, unity_func)
        return message
    factors = []
    neighbours = node.children + node.parents
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        factors.append(node.received_messages[neighbour.name])

    product_func = make_product_func(factors)
    message = Message(node, target_node, product_func)
    return message

        

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
        #if factor == 1:
        #    continue
        args_map[factor] = get_args(factor)
        all_args += args_map[factor]
    #if not all_args:
    #    return 1
    # Now we need to make a callable that
    # will take all the arguments and correctly
    # apply them to each factor...
    args = list(set(all_args))
    #args.sort()
    args_list = expand_parameters(args, [True, False])
    def product_func(*args):
        result = 1
        for factor in factors:
            result *= factor(args)
        return result
    product_func.argspec = args
    product_func.factors = factors
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
    def not_sum_func(exclude_var):
        summands = []
        for bindings in args_list:
            summands.append((bindings, product_func))
        return summands
    not_sum_func.argspec = product_func.argspec
    not_sum_func.exclude_var = exclude_var
    not_sum_func.factors = product_func.factors
    return not_sum_func
    

def make_unity(argspec):
    def unity():
        return 1
    unity.argspec = argspec
    unity.__name__ = '1'
    return unity


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



































