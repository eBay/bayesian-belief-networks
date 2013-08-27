'''Some Useful Helper Functions'''


def make_key(*args):
    '''Handy for short truth table keys'''
    key = ''
    for a in args:
        key += str(a.value).lower()[0]
    return key
