'''The Example from Huang and Darwiche's Procedural Guide'''
from __future__ import division
from bayesian.bbn import *
from bayesian.utils import make_key


def f_a(a):
    return 1 / 2


def f_b(a, b):
    tt = dict(
        tt=0.5,
        ft=0.4,
        tf=0.5,
        ff=0.6)
    return tt[make_key(a, b)]


def f_c(a, c):
    tt = dict(
        tt=0.7,
        ft=0.2,
        tf=0.3,
        ff=0.8)
    return tt[make_key(a, c)]


def f_d(b, d):
    tt = dict(
        tt=0.9,
        ft=0.5,
        tf=0.1,
        ff=0.5)
    return tt[make_key(b, d)]


def f_e(c, e):
    tt = dict(
        tt=0.3,
        ft=0.6,
        tf=0.7,
        ff=0.4)
    return tt[make_key(c, e)]


def f_f(d, e, f):
    tt = dict(
        ttt=0.01,
        ttf=0.99,
        tft=0.01,
        tff=0.99,
        ftt=0.01,
        ftf=0.99,
        fft=0.99,
        fff=0.01)
    return tt[make_key(d, e, f)]


def f_g(c, g):
    tt = dict(
        tt=0.8, tf=0.2,
        ft=0.1, ff=0.9)
    return tt[make_key(c, g)]


def f_h(e, g, h):
    tt = dict(
        ttt=0.05, ttf=0.95,
        tft=0.95, tff=0.05,
        ftt=0.95, ftf=0.05,
        fft=0.95, fff=0.05)
    return tt[make_key(e, g, h)]


if __name__ == '__main__':
    g = build_bbn(
        f_a, f_b, f_c, f_d,
        f_e, f_f, f_g, f_h)
    g.q()
