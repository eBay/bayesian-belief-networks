'''Some Useful Helper Functions'''
import inspect

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
        key += str(a.value).lower()[0]
    return key
