from __future__ import division
import pytest

import os

from collections import Counter
from bayesian.bbn import *
from bayesian.utils import make_key


def r3(x):
    return round(x, 3)


def r5(x):
    return round(x, 5)


def pytest_funcarg__sprinkler_graph(request):
    '''The Sprinkler Example as a BBN
    to be used in tests.
    '''
    cloudy = Node('Cloudy')
    sprinkler = Node('Sprinkler')
    rain = Node('Rain')
    wet_grass = Node('WetGrass')
    cloudy.children = [sprinkler, rain]
    sprinkler.parents = [cloudy]
    sprinkler.children = [wet_grass]
    rain.parents = [cloudy]
    rain.children = [wet_grass]
    wet_grass.parents = [
        sprinkler,
        rain]
    bbn = BBN(
        dict(
            cloudy=cloudy,
            sprinkler=sprinkler,
            rain=rain,
            wet_grass=wet_grass)
        )
    return bbn


def pytest_funcarg__sprinkler_bbn(request):
    '''Sprinkler BBN built with build_bbn.'''
    def f_rain(rain):
        if rain is True:
            return 0.2
        return 0.8


    def f_sprinkler(rain, sprinkler):
        if rain is False and sprinkler is True:
            return 0.4
        if rain is False and sprinkler is False:
            return 0.6
        if rain is True and sprinkler is True:
            return 0.01
        if rain is True and sprinkler is False:
            return 0.99


    def f_grass_wet(sprinkler, rain, grass_wet):
        table = dict()
        table['fft'] = 0.0
        table['fff'] = 1.0
        table['ftt'] = 0.8
        table['ftf'] = 0.2
        table['tft'] = 0.9
        table['tff'] = 0.1
        table['ttt'] = 0.99
        table['ttf'] = 0.01
        return table[make_key(sprinkler, rain, grass_wet)]

    return build_bbn(f_rain, f_sprinkler, f_grass_wet)

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


def pytest_funcarg__huang_darwiche_jt(request):
    def priority_func_override(node):
        introduced_arcs = 0
        cluster = [node] + node.neighbours
        for node_a, node_b in combinations(cluster, 2):
            if node_a not in node_b.neighbours:
                assert node_b not in node_a.neighbours
                introduced_arcs += 1
        if node.name == 'f_h':
            return [introduced_arcs, 0]  # Force f_h tie breaker
        if node.name == 'f_g':
            return [introduced_arcs, 1]  # Force f_g tie breaker
        if node.name == 'f_c':
            return [introduced_arcs, 2]  # Force f_c tie breaker
        if node.name == 'f_b':
            return [introduced_arcs, 3]
        if node.name == 'f_d':
            return [introduced_arcs, 4]
        if node.name == 'f_e':
            return [introduced_arcs, 5]
        return [introduced_arcs, 10]
    dag = pytest_funcarg__huang_darwiche_dag(request)
    jt = build_join_tree(dag, priority_func_override)
    return jt


def pytest_funcarg__monty_bbn(request):
    def f_prize_door(prize_door):
        return 1.0 / 3


    def f_guest_door(guest_door):
        return 1.0 / 3


    def f_monty_door(prize_door, guest_door, monty_door):
        if prize_door == guest_door:
            if prize_door == monty_door:
                return 0
            else:
                return 0.5
        elif prize_door == monty_door:
            return 0
        elif guest_door == monty_door:
            return 0
        return 1

    g = build_bbn(
        f_prize_door,
        f_guest_door,
        f_monty_door,
        domains=dict(
            prize_door=['A', 'B', 'C'],
            guest_door=['A', 'B', 'C'],
            monty_door=['A', 'B', 'C']))
    return g

class TestBBN():

    def test_get_graphviz_source(self, sprinkler_graph):
        gv_src = '''digraph G {
  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];
  Cloudy [ shape="ellipse" color="blue"];
  Rain [ shape="ellipse" color="blue"];
  Sprinkler [ shape="ellipse" color="blue"];
  WetGrass [ shape="ellipse" color="blue"];
  Cloudy -> Rain;
  Cloudy -> Sprinkler;
  Rain -> WetGrass;
  Sprinkler -> WetGrass;
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
            [(node.name, node) for node in
             huang_darwiche_moralized.nodes])
        pq = construct_priority_queue(nodes, priority_func)
        assert pq == [[0, 2, 'f_f'], [0, 2, 'f_h'],
                      [1, 2, 'f_b'], [1, 2, 'f_a'],
                      [1, 2, 'f_g'], [2, 2, 'f_d'],
                      [2, 2, 'f_c'], [7, 2, 'f_e']]

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
                return [introduced_arcs, 0]  # Force f_h tie breaker
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
                return [introduced_arcs, 0]  # Force f_h tie breaker
            if node.name == 'f_g':
                return [introduced_arcs, 1]  # Force f_g tie breaker
            if node.name == 'f_c':
                return [introduced_arcs, 2]  # Force f_c tie breaker
            if node.name == 'f_b':
                return [introduced_arcs, 3]
            if node.name == 'f_d':
                return [introduced_arcs, 4]
            if node.name == 'f_e':
                return [introduced_arcs, 5]
            return [introduced_arcs, 10]
        cliques, elimination_ordering = triangulate(
            huang_darwiche_moralized, priority_func_override)
        nodes = dict([(node.name, node) for node in
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
        nodes = dict([(node.name, node) for node in
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
        nodes = dict([(node.name, node) for node in
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
        def priority_func_override(node):
            introduced_arcs = 0
            cluster = [node] + node.neighbours
            for node_a, node_b in combinations(cluster, 2):
                if node_a not in node_b.neighbours:
                    assert node_b not in node_a.neighbours
                    introduced_arcs += 1
            if node.name == 'f_h':
                return [introduced_arcs, 0]  # Force f_h tie breaker
            if node.name == 'f_g':
                return [introduced_arcs, 1]  # Force f_g tie breaker
            if node.name == 'f_c':
                return [introduced_arcs, 2]  # Force f_c tie breaker
            if node.name == 'f_b':
                return [introduced_arcs, 3]
            if node.name == 'f_d':
                return [introduced_arcs, 4]
            if node.name == 'f_e':
                return [introduced_arcs, 5]
            return [introduced_arcs, 10]

        jt = build_join_tree(huang_darwiche_dag, priority_func_override)
        for node in jt.sepset_nodes:
            assert set([n.clique for n in node.neighbours]) == \
                set([node.sepset.X, node.sepset.Y])
        # TODO: Need additional tests here especially for
        # clique nodes.

    def test_initialize_potentials(
            self, huang_darwiche_jt, huang_darwiche_dag):
        # Seems like there can be multiple assignments so
        # for this test we will set the assignments explicitely
        cliques = dict([(node.name, node) for node in
                        huang_darwiche_jt.clique_nodes])
        bbn_nodes = dict([(node.name, node) for node in
                          huang_darwiche_dag.nodes])
        assignments = {
            cliques['Clique_ACE']: [bbn_nodes['f_c'], bbn_nodes['f_e']],
            cliques['Clique_ABD']: [
                bbn_nodes['f_a'], bbn_nodes['f_b'],  bbn_nodes['f_d']]}
        huang_darwiche_jt.initialize_potentials(
            assignments, huang_darwiche_dag)
        for node in huang_darwiche_jt.sepset_nodes:
            for v in node.potential_tt.values():
                assert v == 1

        # Note that in H&D there are two places that show
        # initial potentials, one is for ABD and AD
        # and the second is for ACE and CE
        # We should test both here but we must enforce
        # the assignments above because alternate and
        # equally correct Junction Trees will give
        # different potentials.
        def r(x):
            return round(x, 3)

        tt = cliques['Clique_ACE'].potential_tt
        assert r(tt[('a', True), ('c', True), ('e', True)]) == 0.21
        assert r(tt[('a', True), ('c', True), ('e', False)]) == 0.49
        assert r(tt[('a', True), ('c', False), ('e', True)]) == 0.18
        assert r(tt[('a', True), ('c', False), ('e', False)]) == 0.12
        assert r(tt[('a', False), ('c', True), ('e', True)]) == 0.06
        assert r(tt[('a', False), ('c', True), ('e', False)]) == 0.14
        assert r(tt[('a', False), ('c', False), ('e', True)]) == 0.48
        assert r(tt[('a', False), ('c', False), ('e', False)]) == 0.32

        tt = cliques['Clique_ABD'].potential_tt
        assert r(tt[('a', True), ('b', True), ('d', True)]) == 0.225
        assert r(tt[('a', True), ('b', True), ('d', False)]) == 0.025
        assert r(tt[('a', True), ('b', False), ('d', True)]) == 0.125
        assert r(tt[('a', True), ('b', False), ('d', False)]) == 0.125
        assert r(tt[('a', False), ('b', True), ('d', True)]) == 0.180
        assert r(tt[('a', False), ('b', True), ('d', False)]) == 0.020
        assert r(tt[('a', False), ('b', False), ('d', True)]) == 0.150
        assert r(tt[('a', False), ('b', False), ('d', False)]) == 0.150

        # TODO: We should add all the other potentials here too.

    def test_jtclique_node_variable_names(self, huang_darwiche_jt):
        for node in huang_darwiche_jt.clique_nodes:
            if 'ADE' in node.name:
                assert set(node.variable_names) == set(['a', 'd', 'e'])

    def test_assign_clusters(self, huang_darwiche_jt, huang_darwiche_dag):

        # NOTE: This test will fail sometimes as assign_clusters
        # is currently non-deterministic, we should fix this.

        bbn_nodes = dict([(node.name, node) for node in
                          huang_darwiche_dag.nodes])
        assignments = huang_darwiche_jt.assign_clusters(huang_darwiche_dag)
        jt_cliques = dict([(node.name, node) for node
                           in huang_darwiche_jt.clique_nodes])
        # Note that these assignments are slightly different
        # to the ones in H&D. In their paper they never
        # give a full list of assignments so we will use
        # these default deterministic assignments for the
        # test. These are assumed to be a valid assignment
        # as all other tests pass.
        assert [] == assignments[jt_cliques['Clique_ADE']]
        assert [bbn_nodes['f_f']] == assignments[jt_cliques['Clique_DEF']]
        assert [bbn_nodes['f_h']] == assignments[jt_cliques['Clique_EGH']]
        assert [bbn_nodes['f_a'], bbn_nodes['f_c']] == \
            assignments[jt_cliques['Clique_ACE']]
        assert [bbn_nodes['f_b'], bbn_nodes['f_d']] == \
            assignments[jt_cliques['Clique_ABD']]
        assert [bbn_nodes['f_e'], bbn_nodes['f_g']] == \
            assignments[jt_cliques['Clique_CEG']]

        # Now we also need to ensure that every original
        # factor from the BBN has been assigned once
        # and only once to some cluster.
        assert set(
            [node for assignment in
             assignments.values() for node in assignment]) == \
            set(
                [node for node in huang_darwiche_dag.nodes])

    def test_propagate(self, huang_darwiche_jt, huang_darwiche_dag):
        jt_cliques = dict([(node.name, node) for node in
                           huang_darwiche_jt.clique_nodes])
        assignments = huang_darwiche_jt.assign_clusters(huang_darwiche_dag)
        huang_darwiche_jt.initialize_potentials(
            assignments, huang_darwiche_dag)

        huang_darwiche_jt.propagate(starting_clique=jt_cliques['Clique_ACE'])
        tt = jt_cliques['Clique_DEF'].potential_tt
        assert r5(tt[(('d', False), ('e', True), ('f', True))]) == 0.00150
        assert r5(tt[(('d', True), ('e', False), ('f', True))]) == 0.00365
        assert r5(tt[(('d', False), ('e', False), ('f', True))]) == 0.16800
        assert r5(tt[(('d', True), ('e', True), ('f', True))]) == 0.00315
        assert r5(tt[(('d', False), ('e', False), ('f', False))]) == 0.00170
        assert r5(tt[(('d', True), ('e', True), ('f', False))]) == 0.31155
        assert r5(tt[(('d', False), ('e', True), ('f', False))]) == 0.14880
        assert r5(tt[(('d', True), ('e', False), ('f', False))]) == 0.36165

        # TODO: Add more potential truth tables from other nodes.

    def test_marginal(self,  huang_darwiche_jt, huang_darwiche_dag):
        bbn_nodes = dict([(node.name, node) for node in
                          huang_darwiche_dag.nodes])
        assignments = huang_darwiche_jt.assign_clusters(huang_darwiche_dag)
        huang_darwiche_jt.initialize_potentials(
            assignments, huang_darwiche_dag)
        huang_darwiche_jt.propagate()

        # These test values come directly from
        # pg. 22 of H & D
        p_A = huang_darwiche_jt.marginal(bbn_nodes['f_a'])
        assert r3(p_A[(('a', True), )]) == 0.5
        assert r3(p_A[(('a', False), )]) == 0.5

        p_D = huang_darwiche_jt.marginal(bbn_nodes['f_d'])
        assert r3(p_D[(('d', True), )]) == 0.68
        assert r3(p_D[(('d', False), )]) == 0.32

        # The remaining marginals here come
        # from the module itself, however they
        # have been corrobarted by running
        # inference using the sampling inference
        # engine and the same results are
        # achieved.
        '''
        +------+-------+----------+
        | Node | Value | Marginal |
        +------+-------+----------+
        | a    | False | 0.500000 |
        | a    | True  | 0.500000 |
        | b    | False | 0.550000 |
        | b    | True  | 0.450000 |
        | c    | False | 0.550000 |
        | c    | True  | 0.450000 |
        | d    | False | 0.320000 |
        | d    | True  | 0.680000 |
        | e    | False | 0.535000 |
        | e    | True  | 0.465000 |
        | f    | False | 0.823694 |
        | f    | True  | 0.176306 |
        | g    | False | 0.585000 |
        | g    | True  | 0.415000 |
        | h    | False | 0.176900 |
        | h    | True  | 0.823100 |
        +------+-------+----------+
        '''
        p_B = huang_darwiche_jt.marginal(bbn_nodes['f_b'])
        assert r3(p_B[(('b', True), )]) == 0.45
        assert r3(p_B[(('b', False), )]) == 0.55

        p_C = huang_darwiche_jt.marginal(bbn_nodes['f_c'])
        assert r3(p_C[(('c', True), )]) == 0.45
        assert r3(p_C[(('c', False), )]) == 0.55

        p_E = huang_darwiche_jt.marginal(bbn_nodes['f_e'])
        assert r3(p_E[(('e', True), )]) == 0.465
        assert r3(p_E[(('e', False), )]) == 0.535

        p_F = huang_darwiche_jt.marginal(bbn_nodes['f_f'])
        assert r3(p_F[(('f', True), )]) == 0.176
        assert r3(p_F[(('f', False), )]) == 0.824

        p_G = huang_darwiche_jt.marginal(bbn_nodes['f_g'])
        assert r3(p_G[(('g', True), )]) == 0.415
        assert r3(p_G[(('g', False), )]) == 0.585

        p_H = huang_darwiche_jt.marginal(bbn_nodes['f_h'])
        assert r3(p_H[(('h', True), )]) == 0.823
        assert r3(p_H[(('h', False), )]) == 0.177


def test_make_node_func():
    UPDATE = {
        "prize_door": [
            # For nodes that have no parents
            # use the empty list to specify
            # the conditioned upon variables
            # ie conditioned on the empty set
            [[], {"A": 1/3, "B": 1/3, "C": 1/3}]],
        "guest_door": [
            [[], {"A": 1/3, "B": 1/3, "C": 1/3}]],
        "monty_door": [
            [[["prize_door", "A"], ["guest_door", "A"]], {"A": 0, "B": 0.5, "C": 0.5}],
            [[["prize_door", "A"], ["guest_door", "B"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "A"], ["guest_door", "C"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "B"], ["guest_door", "A"]], {"A": 0, "B": 0, "C": 1}],
            [[["prize_door", "B"], ["guest_door", "B"]], {"A": 0.5, "B": 0, "C": 0.5}],
            [[["prize_door", "B"], ["guest_door", "C"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "A"]], {"A": 0, "B": 1, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "B"]], {"A": 1, "B": 0, "C": 0}],
            [[["prize_door", "C"], ["guest_door", "C"]], {"A": 0.5, "B": 0.5, "C": 0}],
        ]
    }

    node_func = make_node_func("prize_door", UPDATE["prize_door"])
    assert get_args(node_func) == ['prize_door']
    assert node_func('A') == 1/3
    assert node_func('B') == 1/3
    assert node_func('C') == 1/3

    node_func = make_node_func("guest_door", UPDATE["guest_door"])
    assert get_args(node_func) == ['guest_door']
    assert node_func('A') == 1/3
    assert node_func('B') == 1/3
    assert node_func('C') == 1/3

    node_func = make_node_func("monty_door", UPDATE["monty_door"])
    assert get_args(node_func) == ['guest_door', 'prize_door', 'monty_door']
    assert node_func('A', 'A', 'A') == 0
    assert node_func('A', 'A', 'B') == 0.5
    assert node_func('A', 'A', 'C') == 0.5
    assert node_func('A', 'B', 'A') == 0
    assert node_func('A', 'B', 'B') == 0
    assert node_func('A', 'B', 'C') == 1
    assert node_func('A', 'C', 'A') == 0
    assert node_func('A', 'C', 'B') == 1
    assert node_func('A', 'C', 'C') == 0
    assert node_func('B', 'A', 'A') == 0
    assert node_func('B', 'A', 'B') == 0
    assert node_func('B', 'A', 'C') == 1
    assert node_func('B', 'B', 'A') == 0.5
    assert node_func('B', 'B', 'B') == 0
    assert node_func('B', 'B', 'C') == 0.5
    assert node_func('B', 'C', 'A') == 1
    assert node_func('B', 'C', 'B') == 0
    assert node_func('B', 'C', 'C') == 0
    assert node_func('C', 'A', 'A') == 0
    assert node_func('C', 'A', 'B') == 1
    assert node_func('C', 'A', 'C') == 0
    assert node_func('C', 'B', 'A') == 1
    assert node_func('C', 'B', 'B') == 0
    assert node_func('C', 'B', 'C') == 0
    assert node_func('C', 'C', 'A') == 0.5
    assert node_func('C', 'C', 'B') == 0.5
    assert node_func('C', 'C', 'C') == 0


def close_enough(x, y, r=3):
    return round(x, r) == round(y, r)


def test_build_bbn_from_conditionals():
    UPDATE = {
        "prize_door": [
            # For nodes that have no parents
            # use the empty list to specify
            # the conditioned upon variables
            # ie conditioned on the empty set
            [[], {"A": 1/3, "B": 1/3, "C": 1/3}]],
        "guest_door": [
            [[], {"A": 1/3, "B": 1/3, "C": 1/3}]],
        "monty_door": [
            [[["prize_door", "A"], ["guest_door", "A"]], {
                "A": 0,
                "B": 0.5,
                "C": 0.5
            }],
            [[["prize_door", "A"], ["guest_door", "B"]], {
                "A": 0,
                "B": 0,
                "C": 1}],
            [[["prize_door", "A"], ["guest_door", "C"]], {
                "A": 0,
                "B": 1,
                "C": 0}],
            [[["prize_door", "B"], ["guest_door", "A"]], {
                "A": 0,
                "B": 0,
                "C": 1}],
            [[["prize_door", "B"], ["guest_door", "B"]], {
                "A": 0.5,
                "B": 0,
                "C": 0.5}],
            [[["prize_door", "B"], ["guest_door", "C"]], {
                "A": 1,
                "B": 0,
                "C": 0}],
            [[["prize_door", "C"], ["guest_door", "A"]], {
                "A": 0,
                "B": 1,
                "C": 0}],
            [[["prize_door", "C"], ["guest_door", "B"]], {
                "A": 1,
                "B": 0,
                "C": 0}],
            [[["prize_door", "C"], ["guest_door", "C"]], {
                "A": 0.5,
                "B": 0.5,
                "C": 0}],
        ]
    }
    g = build_bbn_from_conditionals(UPDATE)
    result = g.query()
    assert close_enough(result[('guest_door', 'A')], 0.333)
    assert close_enough(result[('guest_door', 'B')], 0.333)
    assert close_enough(result[('guest_door', 'C')], 0.333)
    assert close_enough(result[('monty_door', 'A')], 0.333)
    assert close_enough(result[('monty_door', 'B')], 0.333)
    assert close_enough(result[('monty_door', 'C')], 0.333)
    assert close_enough(result[('prize_door', 'A')], 0.333)
    assert close_enough(result[('prize_door', 'B')], 0.333)
    assert close_enough(result[('prize_door', 'C')], 0.333)

    result = g.query(guest_door='A', monty_door='B')
    assert close_enough(result[('guest_door', 'A')], 1)
    assert close_enough(result[('guest_door', 'B')], 0)
    assert close_enough(result[('guest_door', 'C')], 0)
    assert close_enough(result[('monty_door', 'A')], 0)
    assert close_enough(result[('monty_door', 'B')], 1)
    assert close_enough(result[('monty_door', 'C')], 0)
    assert close_enough(result[('prize_door', 'A')], 0.333)
    assert close_enough(result[('prize_door', 'B')], 0)
    assert close_enough(result[('prize_door', 'C')], 0.667)


def valid_sample(samples, query_result):
    '''For a group of samples from
    a query result ensure that
    the sample is approximately equivalent
    to the query_result which is the
    true distribution.'''
    counts = Counter()
    for sample in samples:
        for var, val in sample.items():
            counts[(var, val)] += 1
    # Now lets normalize for each count...
    result = True
    for k, v in counts.items():
        counts[k] = v / len(samples)
        difference = abs(counts.get(k, 0) - query_result[k])
        if round(difference, 2) > 0.01:
            result = False
    return result


def test_draw_sample_monty(monty_bbn):
    '''Note this test is non-deterministic
    but should pass most of the time.'''
    query_result = monty_bbn.query()
    samples = monty_bbn.draw_samples(n=10000)
    assert valid_sample(samples, query_result)

    # Now test with some different queries...
    query = dict(guest_door='A')
    query_result = monty_bbn.query(**query)
    samples = monty_bbn.draw_samples(query, n=10000)
    assert valid_sample(samples, query_result)

    query = dict(guest_door='A', monty_door='B')
    query_result = monty_bbn.query(**query)
    samples = monty_bbn.draw_samples(query, n=10000)
    assert valid_sample(samples, query_result)


def test_draw_sample_sprinkler(sprinkler_bbn):

    query_result = sprinkler_bbn.query()
    samples = sprinkler_bbn.draw_samples({}, 10000)
    assert valid_sample(samples, query_result)
