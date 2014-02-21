'''Some Useful Helper Functions'''
import inspect

from prettytable import PrettyTable

# TODO: Find a better location for get_args
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


def make_key(*args):
    '''Handy for short truth table keys'''
    key = ''
    for a in args:
        if hasattr(a, 'value'):
            raise "value attribute deprecated"
        else:
            key += str(a).lower()[0]
    return key


def named_base_type_factory(v, l):
    '''Note this does not work
    for bool since bool is not
    subclassable'''
    return type(
        'labeled_{}'.format(type(v).__name__),
        (type(v), ),
        {'label': l, 'value': v})(v)


def get_original_factors(factors):
    '''
    For a set of factors, we want to
    get a mapping of the variables to
    the factor which first introduces the
    variable to the set.
    To do this without enforcing a special
    naming convention such as 'f_' for factors,
    or a special ordering, such as the last
    argument is always the new variable,
    we will have to discover the 'original'
    factor that introduces the variable
    iteratively.
    '''
    original_factors = dict()
    while len(original_factors) < len(factors):
        for factor in factors:
            args = get_args(factor)
            unaccounted_args = [a for a in args if a not in original_factors]
            if len(unaccounted_args) == 1:
                original_factors[unaccounted_args[0]] = factor
    return original_factors


def shrink_matrix(x):
    '''Remove Nulls'''
    while True:
        if len([x for x in m[0] if x is None]) == x.cols:
            x.pop()
            x = x.tr()
            continue
    return x
