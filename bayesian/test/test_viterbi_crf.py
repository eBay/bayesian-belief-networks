from __future__ import division
from math import exp
import pytest

from bayesian.viterbi_crf import *


def pytest_funcarg__g_(request):
    def f_0(y_prev, y, x_seq, j):
        if y == 'NAME' and x_seq[j].istitle():
            return 1
        return 0

    def f_1(y_prev, y, x_seq, j):
        if y == 'OTHER' and not x_seq[j].istitle():
            return 1
        return 0

    feature_funcs = [f_0, f_1]
    w = [0.75, -0.5]
    x_seq = ['John', 'said']
    y_seq = ['NAME', 'OTHER']

    g_0 = make_g_func(w, feature_funcs, x_seq, 0)
    g_1 = make_g_func(w, feature_funcs, x_seq, 1)

    return {
        0: g_0,
        1: g_1
    }


def pytest_funcarg__G_(request):
    g_ = pytest_funcarg__g_(request)
    G_ = dict()
    for k, v in g_.items():
        G_[k] = make_G_func(v)
    return G_


def test_make_g_func():

    def f_0(y_prev, y, x_seq, j):
        if y == 'NAME' and x_seq[j].istitle():
            return 1
        return 0

    def f_1(y_prev, y, x_seq, j):
        if y == 'OTHER' and not x_seq[j].istitle():
            return 1
        return 0

    feature_funcs = [f_0, f_1]
    w = [0.75, -0.5]
    x_seq = ['John', 'said']
    y_seq = ['NAME', 'OTHER']

    g_0 = make_g_func(w, feature_funcs, x_seq, 0)
    g_1 = make_g_func(w, feature_funcs, x_seq, 1)

    assert g_0('__START__', 'NAME') == 0.75
    assert g_0('__START__', 'OTHER') == 0
    assert g_1('NAME', 'OTHER') == -0.5
    assert g_1('NAME', 'NAME') == 0


def test_make_G_func(g_):
    G_0 = make_G_func(g_[0])
    assert G_0('__START__', 'NAME') == exp(g_[0]('__START__', 'NAME'))
    G_1 = make_G_func(g_[1])
    assert G_1('NAME', 'OTHER') == exp(g_[1]('NAME', 'OTHER'))


def test_viterbi_bail(g_):
    x_seq = ['John', 'said']
    import ipdb; ipdb.set_trace()
    v = viterbi(x_seq, 0, g_, ['NAME', 'OTHER'], {})
    assert viterbi(x_seq, 0, g_, ['NAME', 'OTHER'], {}) == ['NAME']


def test_viterbi_recursion(g_):
    x_seq = ['John', 'said']
    import ipdb; ipdb.set_trace()
    v = viterbi(x_seq, 1, g_, ['NAME', 'OTHER'], {})
    assert viterbi(x_seq, 1, g_, ['NAME', 'OTHER'], {}) == ['NAME', 'NAME']


def test_forward_bail(G_):
    x_seq = ['John', 'said']
    assert round(forward(x_seq, 0, G_, ['NAME', 'OTHER'], {}), 3) == 3.117


def test_forward_recursion(G_):
    x_seq = ['John', 'said']
    assert round(forward(x_seq, 1, G_, ['NAME', 'OTHER'], {}), 3) == 9.447
