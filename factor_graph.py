'''From SumProd.pdf'''
import sys
import inspect

from itertools import product as iter_product

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
        print '%s ---> %s' % (self.name, recipient.name), message
        recipient.received_messages[
            self.name] = message

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


class VariableNode(Node):
    
    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}


    def __repr__(self):
        return '<VariableNode: %s>' % self.name

    def marginal(self, val):
        '''
        The marginal function is
        the product of all incoming
        messages which should be
        functions of this nodes variable
        '''
        product = 1
        self.val = val
        for _, message in self.received_messages.iteritems():
            product *= message(self)
        return product


class FactorNode(Node):

    def __init__(self, name, func, parents=[], children=[]):
        self.name = name
        self.func = func
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}

    def __repr__(self):
        return '<FactorNode %s %s(%s)>' % \
            (self.name,
             self.func.__name__,
             get_args(self.func))


class Message(object):

    def list_factors(self):
        print '---------------------------'
        print 'Factors in message %s -> %s' % (self.source.name, self.destination.name)
        print '---------------------------'
        for factor in self.factors:
            print factor


class FactorMessage(Message):

    def __init__(self, source, destination, not_sum):
        self.source = source
        self.destination = destination
        self.factors = not_sum.factors
        self.not_sum = not_sum
        self.argspec = [destination.name]

    def __repr__(self):
        return '<F-Message %s -> %s: ~(%s) %s factors (%s)>' % \
            (self.source.name, self.destination.name,
             self.not_sum.exclude_var,
             len(self.factors), self.argspec)


    def __call__(self, var):
        '''
        Evaluate the message as a function
        '''
        assert isinstance(var, VariableNode)
        # Now check that the name of the
        # variable matches the argspec...
        assert var.name == self.argspec[0]
        product = 1
        for factor in self.factors:
            product *= factor(var.val)
        return product
        
    def summarize(self):
        '''
        For all variables not in
        the argspec, we want to
        sum over all possible values
        '''
        args_map = {}
        all_args = set()
        for factor in self.factors:
            args_map[factor] = get_args(factor)
            all_args = all_args.union(
                set(args_map[factor]))
        all_args = all_args.difference(
            set(self.argspec))
        if not all_args:
            # There are no variables to
            # summarize so we are done
            return 
        import ipdb; ipdb.set_trace()
        args = list(all_args)
        arg_vals = dict()
        for arg in args:
            arg_vals[arg] = [True, False]
        args_list = list(expand_parameters(arg_vals))
        sum = 0
        # Now loop through the args_list and for
        # each factor that we can apply a binding
        # to we do so and add to the sum so far
        for factor in self.factors:
            argspec = args_map[factor]
            for arg_list in args_list:
                # We need to construct a binding
                # in the same order as the argspec
                


class VariableMessage(Message):

    def __init__(self, source, destination, factors):
        self.source = source
        self.destination = destination
        self.factors = factors
        self.argspec = [source.name]

    def __repr__(self):
        return '<V-Message from %s -> %s: %s factors (%s)>' % \
            (self.source.name, self.destination.name, 
             len(self.factors), self.argspec)


class NotSum(object):
    
    def __init__(self, exclude_var, factors):
        self.exclude_var = exclude_var
        self.factors = factors
        self.argspec = [exclude_var]

    def __repr__(self):
        return '<NotSum(%s, %s)>' % (self.exclude_var, '*'.join([repr(f) for f in self.factors]))
                                     


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
        not_sum = NotSum(target_node.name, [node.func])
        message = FactorMessage(node, target_node, not_sum)
        message.summarize()
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
        # When we pass on a message, we unwrap
        # the original payload and wrap it
        # in new headers, this is purely
        # to verify the procedure is correct
        # according to usual nomenclature
        in_message = node.received_messages[neighbour.name]
        if in_message.destination != node:
            out_message = VariableMessage(neighbour, node, in_message.factors)
            out_message.argspec = in_message.argspec
        else:
            out_message = in_message
        factors.append(out_message)

    # Now we need to add any other variables 
    # that were added from the other factors
    #for factor in factors:
    #    args = args.union(
    #        get_args(factor))
    #args = list(args.difference(set([target_node.name])))
    not_sum = NotSum(target_node.name, factors)
    message = FactorMessage(node, target_node, not_sum)
    # For efficieny we marginalize the message
    # at the time of creation. This is the whole
    # purpose of the sum product algorithm!
    message.summarize()
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
        #unity_func = make_unity([node.name])
        message = VariableMessage(node, target_node, [unity])
        return message
    factors = []
    neighbours = node.children + node.parents
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        factors.append(node.received_messages[neighbour.name])

    #product_func = make_product_func(factors)
    message = VariableMessage(node, target_node, factors)
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


def make_unity(argspec):
    def unity():
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

class Cycler(object):
    '''
    Like itertools.cycler
    except it caches the
    most recent value for 
    repeat use.
    '''

    def __init__(self, vals):
        self.cycle = cycle(vals)
        self.current_val = None

    def current(self):
        if self.current_val is None:
            self.next()
        return self.current_val

    def next(self):
        self.current_val = self.cycle.next()
        return self.current_val


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
    return iter_product(*arg_tuples)
































