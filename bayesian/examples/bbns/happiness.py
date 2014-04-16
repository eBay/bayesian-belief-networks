from bayesian.bbn import build_bbn
from bayesian.utils import make_key

'''
NOTE: I havent yet seen the actual probability
distributions for the variables so for now
we will just use mass functions on two
values: True, False

We will use the following conventions here:

Probability distributions will be named
f_XXXXXX.
The parameters to the f_XXXXXX will be
a single character for exampled:

def f_coherence(c):
    pass

'''

def f_coherence(c):
    return 0.5


def f_difficulty(c, d):
    tt = dict(
        ff = 0.2,
        ft = 0.8,
        tf = 0.7,
        tt = 0.3)
    return tt[make_key(c, d)]


def f_intelligence(i):
    return 0.5


def f_grade(d, i, g):
    tt = dict(
        fff = 0.5,
        fft = 0.5,
        ftf = 0.1,
        ftt = 0.9,
        tff = 0.9,
        tft = 0.1,
        ttf = 0.4,
        ttt = 0.6)
    return tt[make_key(d, i, g)]


def f_sat(i, s):
    tt = dict(
        ff = 0.9,
        ft = 0.1,
        tf = 0.2,
        tt = 0.8)
    return tt[make_key(i, s)]


def f_letter(g, l):
    tt = dict(
        ff = 0.8,
        ft = 0.2,
        tf = 0.1,
        tt = 0.9)
    return tt[make_key(g, l)]


def f_job(l, s, j):
    tt = dict(
        fff = 0.5,
        fft = 0.5,
        ftf = 0.1,
        ftt = 0.9,
        tff = 0.9,
        tft = 0.1,
        ttf = 0.4,
        ttt = 0.6)
    return tt[make_key(l, s, j)]


def f_happy(g, j, h):
    tt = dict(
        fff = 0.5,
        fft = 0.5,
        ftf = 0.1,
        ftt = 0.9,
        tff = 0.9,
        tft = 0.1,
        ttf = 0.4,
        ttt = 0.6)
    return tt[make_key(g, j, h)]


g = build_bbn(
    f_coherence,
    f_difficulty,
    f_intelligence,
    f_grade,
    f_sat,
    f_letter,
    f_job,
    f_happy)


if __name__ == '__main__':
    g = build_bbn(
    f_coherence,
    f_difficulty,
    f_intelligence,
    f_grade,
    f_sat,
    f_letter,
    f_job,
    f_happy)
    import ipdb; ipdb.set_trace()
    g.q(**{'c': True, 'd': True, 'g': True, 'i': True, 'h': True, 'j': True, 'l': True})
    g.inference_method = 'clique_tree_sum_product'
    g.q(**{'c': True, 'd': True, 'g': True, 'i': True, 'h': True, 'j': True, 'l': True})
