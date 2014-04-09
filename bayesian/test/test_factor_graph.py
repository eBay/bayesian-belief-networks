from __future__ import division
import pytest

import os

from bayesian.factor_graph import *
from bayesian.bbn import build_bbn
from bayesian.utils import make_key
from bayesian.examples.bbns.cancer import g as cancer_bbn


def pytest_funcarg__cancer_bbn(request):
    return cancer_bbn


def pytest_funcarg__huang_darwiche_nodes(request):
    '''The nodes for the Huang Darwich example'''
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

    return [f_a, f_b, f_c, f_d,
            f_e, f_f, f_g, f_h]


def pytest_funcarg__huang_darwiche_dag(request):
    nodes = pytest_funcarg__huang_darwiche_nodes(request)
    return build_bbn(nodes)


def test_has_cycles(cancer_bbn):
    '''Test fg.has_cycles() on a converted BBN.'''
    fg = cancer_bbn.convert_to_factor_graph()
    assert not fg.has_cycles()


def test_construct_message(huang_darwiche_dag):
    """Test constructed messages on
    converted huang_darwiche bbn"""
    fg = huang_darwiche_dag.convert_to_factor_graph()
    node_a = fg.variable_nodes()[0]
    assert node_a.name == 'a'
    message = node_a.construct_message()
    assert message() == 1
    assert len([n for n in fg.factor_nodes() if n == message.destination]) == 1
    destination = [n for n in fg.factor_nodes() if n == message.destination][0]
    assert message.destination == destination
    assert message.source == node_a
    assert get_args(message) == []


def test_send_message(huang_darwiche_dag):
    """Test constructed messages on
    converted huang_darwiche bbn"""
    import ipdb; ipdb.set_trace()
    fg = huang_darwiche_dag.convert_to_factor_graph()
    node_a = fg.variable_nodes()[0]
    assert node_a.name == 'a'
    message = node_a.construct_message()
    import ipdb; ipdb.set_trace()
    assert len([n for n in fg.factor_nodes() if n.name == 'f_c_f_a_f_b']) == 1
    destination = [n for n in fg.factor_nodes() if n.name == 'f_c_f_a_f_b'][0]
    assert destination.recieved_messages['a'] == message
