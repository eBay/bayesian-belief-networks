import pytest

import os

from bayesian.bbn import *
from bayesian.utils import make_key

def pytest_funcarg__sprinkler_graph(request):
    '''The Sprinkler Example as a BBN
    to be used in tests.
    '''
    cloudy = Node('Cloudy')
    sprinkler = Node('Sprinkler')
    rain = Node('Rain')
    wet_grass = Node('WetGrass')
    cloudy.children = [
        sprinkler, rain ]
    sprinkler.parents = [ cloudy ]
    sprinkler.children = [ wet_grass ]
    rain.parents = [ cloudy ]
    rain.children = [ wet_grass ]
    wet_grass.parents = [
        sprinkler,
        rain ]
    bbn = BBN([
        cloudy,
        sprinkler,
        rain,
        wet_grass])
    return bbn


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


def pytest_funcarg__huang_darwiche_moralized(request):

    dag = pytest_funcarg__huang_darwiche_dag(request)
    gu = make_undirected_copy(dag)
    gm = make_moralized_copy(gu, dag)

    return gm



class TestBBN():

    def test_get_graphviz_source(self, sprinkler_graph):
        gv_src = '''digraph G {
  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];
  Cloudy [ shape="ellipse" color="blue"];
  Sprinkler [ shape="ellipse" color="blue"];
  Rain [ shape="ellipse" color="blue"];
  WetGrass [ shape="ellipse" color="blue"];
  Rain -> WetGrass;
  Sprinkler -> WetGrass;
  Cloudy -> Sprinkler;
  Cloudy -> Rain;
}
'''
        assert sprinkler_graph.get_graphviz_source() == gv_src



    def test_get_original_factors(self, huang_darwiche_nodes):
        original_factors = get_original_factors(
            huang_darwiche_nodes)
        assert original_factors['a'] == huang_darwiche_nodes[0]
        assert original_factors['b'] == huang_darwiche_nodes[1]
        assert original_factors['c'] == huang_darwiche_nodes[2]
        assert original_factors['d'] == huang_darwiche_nodes[3]
        assert original_factors['e'] == huang_darwiche_nodes[4]
        assert original_factors['f'] == huang_darwiche_nodes[5]
        assert original_factors['g'] == huang_darwiche_nodes[6]
        assert original_factors['h'] == huang_darwiche_nodes[7]

    def test_build_graph(self, huang_darwiche_nodes):
        bbn = build_bbn(huang_darwiche_nodes)
        nodes = dict([(node.name, node) for node in bbn.nodes])
        assert nodes['f_a'].parents == []
        assert nodes['f_b'].parents == [nodes['f_a']]
        assert nodes['f_c'].parents == [nodes['f_a']]
        assert nodes['f_d'].parents == [nodes['f_b']]
        assert nodes['f_e'].parents == [nodes['f_c']]
        assert nodes['f_f'].parents == [nodes['f_d'], nodes['f_e']]
        assert nodes['f_g'].parents == [nodes['f_c']]
        assert nodes['f_h'].parents == [nodes['f_e'], nodes['f_g']]

    def test_make_undirecred_copy(self, huang_darwiche_dag):
        ug = make_undirected_copy(huang_darwiche_dag)
        nodes = dict([(node.name, node) for node in ug.nodes])
        assert set(nodes['f_a'].neighbours) == set([
            nodes['f_b'], nodes['f_c']])
        assert set(nodes['f_b'].neighbours) == set([
            nodes['f_a'], nodes['f_d']])
        assert set(nodes['f_c'].neighbours) == set([
            nodes['f_a'], nodes['f_e'], nodes['f_g']])
        assert set(nodes['f_d'].neighbours) == set([
            nodes['f_b'], nodes['f_f']])
        assert set(nodes['f_e'].neighbours) == set([
            nodes['f_c'], nodes['f_f'], nodes['f_h']])
        assert set(nodes['f_f'].neighbours) == set([
            nodes['f_d'], nodes['f_e']])
        assert set(nodes['f_g'].neighbours) == set([
            nodes['f_c'], nodes['f_h']])
        assert set(nodes['f_h'].neighbours) == set([
            nodes['f_e'], nodes['f_g']])

    def test_make_moralized_copy(self, huang_darwiche_dag):
        gu = make_undirected_copy(huang_darwiche_dag)
        gm = make_moralized_copy(gu, huang_darwiche_dag)
        nodes = dict([(node.name, node) for node in gm.nodes])
        assert set(nodes['f_a'].neighbours) == set([
            nodes['f_b'], nodes['f_c']])
        assert set(nodes['f_b'].neighbours) == set([
            nodes['f_a'], nodes['f_d']])
        assert set(nodes['f_c'].neighbours) == set([
            nodes['f_a'], nodes['f_e'], nodes['f_g']])
        assert set(nodes['f_d'].neighbours) == set([
            nodes['f_b'], nodes['f_f'], nodes['f_e']])
        assert set(nodes['f_e'].neighbours) == set([
            nodes['f_c'], nodes['f_f'], nodes['f_h'],
            nodes['f_d'], nodes['f_g']])
        assert set(nodes['f_f'].neighbours) == set([
            nodes['f_d'], nodes['f_e']])
        assert set(nodes['f_g'].neighbours) == set([
            nodes['f_c'], nodes['f_h'], nodes['f_e']])
        assert set(nodes['f_h'].neighbours) == set([
            nodes['f_e'], nodes['f_g']])

    def test_construct_priority_queue(self, huang_darwiche_moralized):
        nodes = dict(
            [(node.name, node) for node in \
             huang_darwiche_moralized.nodes])
        pq = construct_priority_queue(nodes, priority_func)
        assert pq == [[0, 2, 'f_f'], [0, 2, 'f_h'], [1, 2, 'f_b'], [1, 2, 'f_a'], [1, 2, 'f_g'], [2, 2, 'f_d'], [2, 2, 'f_c'], [7, 2, 'f_e']]

        # Note that for this test we want to ensure
        # the same elimination ordering as on page 13
        # of Darwiche and Wang. The first two entries
        # in the priority queue are actually a tie
        # so we will manually manipulate them here
        # by specifying an alternative priority func:
        def priority_func_override(node):
            introduced_arcs = 0
            cluster = [node] + node.neighbours
            for node_a, node_b in combinations(cluster, 2):
                if node_a not in node_b.neighbours:
                    assert node_b not in node_a.neighbours
                    introduced_arcs += 1
            if node.name == 'f_h':
                return [introduced_arcs, 0] # Force f_h tie breaker
            return [introduced_arcs, 2]
        pq = construct_priority_queue(
            nodes,
            priority_func_override)
        assert pq[0] == [0, 0, 'f_h']

    def test_triangulate(self, huang_darwiche_moralized):

        # Because of ties in the priority q we will
        # override the priority function here to
        # insert tie breakers to ensure the same
        # elimination ordering as Darwich Huang.
        def priority_func_override(node):
            introduced_arcs = 0
            cluster = [node] + node.neighbours
            for node_a, node_b in combinations(cluster, 2):
                if node_a not in node_b.neighbours:
                    assert node_b not in node_a.neighbours
                    introduced_arcs += 1
            if node.name == 'f_h':
                return [introduced_arcs, 0] # Force f_h tie breaker
            if node.name == 'f_g':
                return [introduced_arcs, 1] # Force f_g tie breaker
            if node.name == 'f_c':
                return [introduced_arcs, 2] # Force f_c tie breaker
            if node.name == 'f_b':
                return [introduced_arcs, 3]
            if node.name == 'f_d':
                return [introduced_arcs, 4]
            if node.name == 'f_e':
                return [introduced_arcs, 5]
            return [introduced_arcs, 10]
        cliques, elimination_ordering = triangulate(
            huang_darwiche_moralized, priority_func_override)
        nodes = dict([(node.name, node) for node in \
                      huang_darwiche_moralized.nodes])
        assert len(cliques) == 6
        assert cliques[0].nodes == set(
            [nodes['f_e'], nodes['f_g'], nodes['f_h']])
        assert cliques[1].nodes == set(
            [nodes['f_c'], nodes['f_e'], nodes['f_g']])
        assert cliques[2].nodes == set(
            [nodes['f_d'], nodes['f_e'], nodes['f_f']])
        assert cliques[3].nodes == set(
            [nodes['f_a'], nodes['f_c'], nodes['f_e']])
        assert cliques[4].nodes == set(
            [nodes['f_a'], nodes['f_b'], nodes['f_d']])
        assert cliques[5].nodes == set(
            [nodes['f_a'], nodes['f_d'], nodes['f_e']])

        assert elimination_ordering == [
            'f_h',
            'f_g',
            'f_f',
            'f_c',
            'f_b',
            'f_d',
            'f_e',
            'f_a']
        # Now lets ensure the triangulated graph is
        # the same as Darwiche Huang fig. 2 pg. 13
        nodes = dict([(node.name, node) for node in \
                      huang_darwiche_moralized.nodes])
        assert set(nodes['f_a'].neighbours) == set([
            nodes['f_b'], nodes['f_c'],
            nodes['f_d'], nodes['f_e']])
        assert set(nodes['f_b'].neighbours) == set([
            nodes['f_a'], nodes['f_d']])
        assert set(nodes['f_c'].neighbours) == set([
            nodes['f_a'], nodes['f_e'], nodes['f_g']])
        assert set(nodes['f_d'].neighbours) == set([
            nodes['f_b'], nodes['f_f'], nodes['f_e'],
            nodes['f_a']])
        assert set(nodes['f_e'].neighbours) == set([
            nodes['f_c'], nodes['f_f'], nodes['f_h'],
            nodes['f_d'], nodes['f_g'], nodes['f_a']])
        assert set(nodes['f_f'].neighbours) == set([
            nodes['f_d'], nodes['f_e']])
        assert set(nodes['f_g'].neighbours) == set([
            nodes['f_c'], nodes['f_h'], nodes['f_e']])
        assert set(nodes['f_h'].neighbours) == set([
            nodes['f_e'], nodes['f_g']])


    def test_triangulate_no_tie_break(self, huang_darwiche_moralized):
        # Now lets see what happens if
        # we dont enforce the tie-breakers...
        # It seems the triangulated graph is
        # different adding edges from d to c
        # and b to c
        # Will be interesting to see whether
        # inference will still be correct.
        cliques, elimination_ordering = triangulate(
            huang_darwiche_moralized)
        nodes = dict([(node.name, node) for node in \
                      huang_darwiche_moralized.nodes])
        assert set(nodes['f_a'].neighbours) == set([
            nodes['f_b'], nodes['f_c']])
        assert set(nodes['f_b'].neighbours) == set([
            nodes['f_a'], nodes['f_d'], nodes['f_c']])
        assert set(nodes['f_c'].neighbours) == set([
            nodes['f_a'], nodes['f_e'], nodes['f_g'],
            nodes['f_b'], nodes['f_d']])
        assert set(nodes['f_d'].neighbours) == set([
            nodes['f_b'], nodes['f_f'], nodes['f_e'],
            nodes['f_c']])
        assert set(nodes['f_e'].neighbours) == set([
            nodes['f_c'], nodes['f_f'], nodes['f_h'],
            nodes['f_d'], nodes['f_g']])
        assert set(nodes['f_f'].neighbours) == set([
            nodes['f_d'], nodes['f_e']])
        assert set(nodes['f_g'].neighbours) == set([
            nodes['f_c'], nodes['f_h'], nodes['f_e']])
        assert set(nodes['f_h'].neighbours) == set([
            nodes['f_e'], nodes['f_g']])

    def test_build_join_tree(self, huang_darwiche_dag):
        jt = build_join_tree(huang_darwiche_dag)
