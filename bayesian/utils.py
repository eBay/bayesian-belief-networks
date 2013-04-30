def memoize(f):
    '''
    The goal of message passing
    is to re-use results. This
    memoise is slightly modified from
    usual examples in that it caches
    the values of variables rather than
    the variables themselves.
    '''

    cache = {}

    def memoized(*args):
        arg_vals = tuple([arg.value for arg in args])
        if not arg_vals in cache:
            cache[arg_vals] = f(*args)
        return cache[arg_vals]

    if hasattr(f, 'domains'):
        memoized.domains = f.domains
    if hasattr(f, 'argspec'):
        memoized.argspec = f.argspec
    #memoized.__name__ == f.__name__
    return memoized
