'''From SumProd.pdf'''
import sys
import copy
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
        v = VariableNode(self.name)
        v.value = val
        for _, message in self.received_messages.iteritems():
            product *= message(v)
        return product


class FactorNode(Node):

    def __init__(self, name, func, parents=[], children=[]):
        self.name = name
        self.func = func
        self.parents = parents
        self.children = children
        self.received_messages = {}
        self.sent_messages = {}
        self.func.value = None

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

    def __call__(self, var):
        '''
        Evaluate the message as a function
        '''
        if getattr(self.func, '__name__', None) == 'unity':
            return 1
        assert isinstance(var, VariableNode)
        # Now check that the name of the
        # variable matches the argspec...
        #assert var.name == self.argspec[0]
        return self.func(var)


class FactorMessage(Message):

    def __init__(self, source, destination, factors, func):
        self.source = source
        self.destination = destination
        self.factors = factors
        self.func = func
        self.argspec = get_args(func)

    def __repr__(self):
        return '<F-Message %s -> %s: ~(%s) %s factors.>' % \
            (self.source.name, self.destination.name,
             self.argspec,
             len(self.factors))


        
    def summarize(self):
        '''
        For all variables not in
        the argspec, we want to
        sum over all possible values.
        summarize should *replace*
        the current factors with
        a new list of factors 
        where each factor has been
        marginalized.
        
        '''
        new_factors = []
        args_map = {}
        all_args = set()
        for factor in self.factors:
            if factor.value is not None:
                continue
            args_map[factor] = get_args(factor)
            all_args = all_args.union(
                set(args_map[factor]))
        all_args = all_args.difference(
            set(self.argspec))
        if not all_args:
            # There are no variables to
            # summarize so we are done
            return 
        args = list(all_args)
        arg_vals = dict()
        for arg in args:
            arg_vals[arg] = [True, False]
        args_list = list(expand_parameters(arg_vals))
        # Now loop through the args_list and for
        # each factor that we can apply a binding
        # to we do so and add to the sum so far
        for factor in self.factors:
            if isinstance(factor, FactorMessage) and not factor.value is None:
                factor.summarize()
                #new_factors.append(factor)
                continue
            if factor.value is not None:
                continue
            arg_spec = args_map[factor]
            # If the not_sum exclude_var is in the
            # arg spec of this factor then we cannot
            # summarize this particular factor
            if self.not_sum.exclude_var in arg_spec:
                continue
            
            arg_vals = dict()
            for arg in arg_spec:
                arg_vals[arg] = [True, False]
            args_list = list(expand_parameters(arg_vals))
            if len(args_list[0]) != len(arg_spec):
                continue
            total = 0
            for arg_list in args_list:
                bindings = build_bindings(arg_spec, arg_list)
                total += factor(*bindings)
            factor.value = total
        if all(map(lambda x: isinstance(x.value, (int, float)), self.factors)):
            self.value = reduce(lambda x, y: x.value * y.value, factors)
            

def eliminate_var(f, var):
    '''
    Given a function f return a new
    function which sums over the variable
    we want to eliminate
    '''
    arg_spec = get_args(f)
    pos = arg_spec.index(var)
    new_spec = arg_spec[:]
    new_spec.remove(var)
    
    def eliminated(*args):
        template = arg_spec[:]
        total = 0
        for val in [True, False]:
            v = VariableNode(name=var)
            v.value = val
            
            template[pos] = v
            call_args = template[:]
            for arg in args:
                arg_pos = call_args.index(arg.name)
                call_args[arg_pos] = arg
                
            total += f(*call_args)
        return total

    eliminated.argspec = new_spec
    return eliminated

    
class VariableMessage(Message):

    def __init__(self, source, destination, factors, func):
        self.source = source
        self.destination = destination
        self.factors = factors
        self.argspec = get_args(func)
        self.func = func
        #self.value = None
        #if all(map(lambda x: isinstance(x, (int, float)), factors)):
        #    self.value = reduce(lambda x, y: x * y, factors)


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
                                     


def make_not_sum_func(product_func, keep_var):
    '''
    Given a function with some set of
    arguments, and a single argument to keep,
    construct a new function only of the
    keep_var, summarized over all the other
    variables.
    '''
    args = get_args(product_func)
    new_func = copy.deepcopy(product_func)
    for arg in args:
        if arg != keep_var:
            new_func = eliminate_var(new_func, arg)
    return new_func

def build_bindings(arg_spec, arg_vals):
    '''
    Build a list of values from 
    a dict that matches the arg_spec
    '''
    assert len(arg_spec) == len(arg_vals)
    bindings = []
    #bindings = [arg_vals[arg] for arg in arg_spec]
    for arg in arg_spec:
        var = VariableNode(arg)
        var.value = arg_vals[arg]
        bindings.append(var)
    return bindings


def m1ake_factor_node_message(node, target_node):
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
            out_message = VariableMessage(neighbour, node, in_message.factors, in_message.func)
            out_message.argspec = in_message.argspec
        else:
            out_message = in_message
        factors.append(out_message)

    product_func = make_product_func(factors)
    not_sum_func = make_not_sum_func(product_func, target_node.name)
    message = FactorMessage(node, target_node, factors, not_sum_func)
    return message




def m1ake_variable_node_message(node, target_node):
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
        message = VariableMessage(node, target_node, [1])
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
        message = VariableMessage(node, target_node, [1], unity)
        return message
    factors = []
    neighbours = node.children + node.parents
    for neighbour in neighbours:
        if neighbour == target_node:
            continue
        factors.append(node.received_messages[neighbour.name])

    product_func = make_product_func(factors)
    message = VariableMessage(node, target_node, factors, product_func)
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
    args = list(set(all_args))

    def product_func(*args):
        arg_dict = dict([(a.name, a) for a in args])
        result = 1
        for factor in factors:
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
            try:
                result *= factor(*factor_args)
            except:
                import ipdb; ipdb.set_trace()
                print factor, factor_args, get_args(factor)
                
        return result

    product_func.argspec = args
    product_func.factors = factors
    return product_func


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
    return [dict(args) for args in iter_product(*arg_tuples)]
































