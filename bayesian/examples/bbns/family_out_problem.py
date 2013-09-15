'''This example is from http://www.cs.ubc.ca/~murphyk/Bayes/Charniak_91.pdf'''
from bayesian.bbn import build_bbn
from bayesian.utils import make_key

'''
This problem is also sometimes referred to
as "the Dog Problem"
'''


def family_out(fo):
    if fo:
        return 0.15
    return 0.85


def bowel_problem(bp):
    if bp:
        return 0.01
    return 0.99


def light_on(fo, lo):
    tt = dict(
        tt=0.6,
        tf=0.4,
        ft=0.05,
        ff=0.96)
    return tt[make_key(fo, lo)]


def dog_out(fo, bp, do):
    tt = dict(
        ttt=0.99,
        tft=0.9,
        ftt=0.97,
        fft=0.3)   # Note typo in article!
    key = make_key(fo, bp, do)
    if key in tt:
        return tt[key]
    key = make_key(fo, bp, not do)
    return 1 - tt[key]


def hear_bark(do, hb):
    tt = dict(
        tt=0.7,
        ft=0.01)
    key = make_key(do, hb)
    if key in tt:
        return tt[key]
    key = make_key(do, not hb)
    return 1 - tt[key]


if __name__ == '__main__':
    g = build_bbn(
        family_out,
        bowel_problem,
        light_on,
        dog_out,
        hear_bark)
    g.q()
