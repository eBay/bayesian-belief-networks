
class EqualityOverideMixin(object):

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Division(object):

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __repr__(self):
        return '<Division: (%s) / (%s)>' % (
            self.numerator,
            self.denominator)


class Mult(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '<Mult: (%s) * (%s)>' % (
            self.x,
            self.y)


class RandomVariable(EqualityOverideMixin):

    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    def __repr__(self):
        if self.val is not None:
            return '%s=%s' % (self.name, self.val)
        return self.name



class Conditional(EqualityOverideMixin):
    
    def __init__(self, conditioned, dependees):
        if not type(dependees) == type([]):
            raise "Dependees must be a list"
        self.conditioned = conditioned
        self.dependees = dependees

    def __repr__(self):
        if type(self.dependees) == type([]):
            return 'cP(%s|%s)' % (self.conditioned, ','.join([repr(x) for x in self.dependees]))
        return 'cP(%s|%s)' % (self.conditioned, self.dependees)


    def __mul__(self, other):
        return Mult(self, other)


class Prob(EqualityOverideMixin):
    '''
    Not strictly needed as we 
    could just use Joint with a single
    variable in the vars list
    but this makes it slightly easier
    to write cleaner code
    '''

    def __init__(self, x):
        self.x = x

    def __repr__(self):
        return 'P(%s)' % self.x
        
    def __hash__(self):
        return ('%s=%s' % (self.x.name, self.x.val)).__hash__()

    

class Joint(object):
    
    def __init__(self, vars):
        if not type(vars) == type([]):
            raise "vars must be a list."
        self.vars = vars

    def __repr__(self):
        return 'jP(%s)' % ','.join([repr(x) for x in self.vars])

    def __len__(self):
        return len(self.vars)

    def __mul__(self, other):
        return Mult(self, other)

    def __div__(self, other):
        return Division(self, other)




class Expression(object):
    '''
    Need to find a way to represent expressions easily
    say as lists....
    '''
    



def log(f):

    def wrapped(*args, **kwds):
        print 'Entering %s with args: %s' % (f.__name__, args)
        retval = f(*args, **kwds)
        return retval
    return wrapped

    
def make_prob(vars):
    if len(vars) == 1:
        return Prob(vars[0])
    return Joint(vars)


def joint_to_conditional(j):
    '''
    Convert a Joint object into a conditional
    using Chain rule
    vars is a list of variables
    '''
    if len(j) == 1:
        return Prob(RandomVariable(j.vars[0].name, j.vars[0].val))
    return Conditional(j.vars[0], j.vars[1:]) * joint_to_conditional(Joint(j.vars[1:]))


def conditional_to_joint(c):
    '''
    Convert a joint representation to a conditional
    e.g. working backword from Bayes Formula:
    jP(X, Y) = cP(X|Y) * P(Y)
    jP(X, Y) / P(Y) = cP(X|Y)
    therefore:
    cP(X|Y) = jP(X, Y) / P(Y)
    '''
    numerator = Joint([c.conditioned] +  c.dependees)
    denominator = make_prob(c.dependees)
    return numerator / denominator


def marginizile_head(arguments):
    return 

def expand_arguments(arguments):
    '''
    For a list of arguments with
    unkown value represented as
    None we want to return a
    list containing all combinations
    of fully constrained arguments
    in order to marginalize by summing
    >>> expand_arguments([True])
    [[True]]
    >>> expand_arguments([None])
    [[True], [False]]
    >>> expand_arguments([True, True])
    [[True, True]]
    >>> expand_arguments([True, None])
    [[True, True], [True, False]]
    >>> expand_arguments([None, False])
    [[True, False], [False, False]]
    >>> expand_arguments([True, None, False])
    [[True, True, False], [True, False, False]]
    >>> expand_arguments([None, None])
    [[True, True], [False, True], [True, False], [False, False]]
    '''
    result = []
    if not arguments:
        return [result]
    rest = expand_arguments(arguments[1:])
    for r in rest:
        if arguments[0] is None:
            result.append([True] + r)
            result.append([False] + r)
        else:
            result.append([arguments[0]] + r)
    return result


def grass_wet_func(s, r, g):
    ''' 
    This needs to be a joint probability distribution
    over the inputs and the node itself
    '''
    table = dict()
    table['fft'] = 0.0
    table['fff'] = 1.0
    table['ftt'] = 0.8
    table['ftf'] = 0.2
    table['tft'] = 0.9
    table['tff'] = 0.1
    table['ttt'] = 0.99
    table['ttf'] = 0.01
    key = ''
    key = key + 't' if s else key + 'f'
    key = key + 't' if r else key + 'f'
    key = key + 't' if g else key + 'f'
    return table[key]



def marginalize(func, arguments):
    '''
    For the given func, the arguments
    should be a list containing one
    value for each argument that the
    function takes. None will be
    taken to mean an unknown value.
    We will the marginalize by summing over
    all possible values ie [T, F]
    for the unknown arguments.
    >>> marginalize(grass_wet_func, [None, True, True])
    .3577
    >>> marginalize(grass_wet_func, [None, None, True])
    .3577
    '''
    total = 0
    all_combos = expand_arguments(arguments)
    for combo in all_combos:
        val = func(*combo)
        print combo, val
        total += val
    return total

    


class Node(object):

    def __init__(self, random_variable, func, parents=[]):
        self.random_variable = random_variable
        self.func = func
        self.name = random_variable.name
        self.parents = parents

    def can_answer(self):
        '''
        What type of questions can this node answer?
        If it has no parents then it can only
        answer the discrete probabilities
        ie p(R=True) and p(R=False)
        >>> rain_var = RandomVariable('Rain')
        >>> rain_node = Node(rain_var, lambda x: 'dummy', parents=[])
        >>> rain_node.can_answer()
        [P(Rain=True), P(Rain=False)]
        >>> sprinkler_var = RandomVariable('Sprinkler')
        >>> sprinkler_node = Node(sprinkler_var, None, parents=[rain_node])
        >>> sprinkler_node.can_answer()
        [cP(Sprinkler=True|Rain), cP(Sprinkler=False|Rain)]
        >>> grass_wet_var = RandomVariable('Grass Wet')
        >>> grass_wet_node = Node(grass_wet_var, None, parents=[rain_node, sprinkler_node])
        >>> grass_wet_node.can_answer()
        [cP(Grass Wet=True|Rain,Sprinkler), cP(Grass Wet=False|Rain,Sprinkler)]
        '''
        if not self.parents:
            return [Prob(RandomVariable(self.name, True)), Prob(RandomVariable(self.name, False))]
        else:
            parent_vars = [x.random_variable for x in self.parents]
            return [Conditional(RandomVariable(self.name, True), parent_vars),
                    Conditional(RandomVariable(self.name, False), parent_vars)]
            
    def __repr__(self):
        return '<Node: %s>' % self.name





def evaluate(node_list, evidence):
    '''
    Naive approach here is to cycle through
    the nodes repeatedly and fill in values we know...
    '''
    # Lets first see what expressions are evaluatable...
    # We can look at the parents to answer this...
    # for the Grass Wet node we know the parents are Sprinkler and Rain
    # so therefore we can evaluate the following:
    # - if we have values for grass is wet
    # 


    

def who_can_answer(p, nodes):
    '''
    >>> rain_var = RandomVariable('Rain', True)
    >>> rain_node = Node(rain_var, lambda x: 'dummy', parents=[])
    >>> p = Prob(rain_var)
    >>> who_can_answer(p, [rain_node])
    <Node: Rain>
    >>> rain_var = RandomVariable('Rain')
    >>> rain_node = Node(rain_var, lambda x: 'dummy', parents=[])
    >>> sprinkler_var = RandomVariable('Sprinkler', True)
    >>> sprinkler_node = Node(sprinkler_var, None, parents=[rain_node])
    >>> c = Conditional(sprinkler_var, [rain_var])
    >>> who_can_answer(c, [sprinkler_node, rain_node])
    <Node: Sprinkler>
    >>> who_can_answer(c, [rain_node])
    '''
    for node in nodes:
        if p in node.can_answer():
            return node
        
    
    


if __name__ == '__main__':
    import doctest
    doctest.testmod()

