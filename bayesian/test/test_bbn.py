from __future__ import division
import pytest

import os
from itertools import product as xproduct

from bayesian.bbn import *
from bayesian.graph import construct_priority_queue
from bayesian.utils import make_key
#from bayesian.examples.factor_graphs.earthquake import g as earthquake_fg
from bayesian.examples.bbns.monty_hall import g as monty_bbn
from bayesian.examples.bbns.earthquake import g as earthquake_bbn
from bayesian.examples.bbns.cancer import g as cancer_bbn
from bayesian.examples.bbns.happiness import g as happiness_bbn


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
    return monty_bbn


def pytest_funcarg__earthquake_bbn(request):
    return earthquake_bbn


def pytest_funcarg__cancer_bbn(request):
    return cancer_bbn


def pytest_funcarg__happiness_bbn(request):
    return happiness_bbn


def variable_domains_match_function_domains(fg):
    """
    Ensure that all factor node functions
    domains match the variable domains.
    """
    variable_domains = dict()
    for variable_node in fg.variable_nodes():
        variable_domains[variable_node.name] = variable_node.domain
    for factor_node in fg.factor_nodes():
        for k, v in factor_node.domains.items():
            assert v == variable_domains[k]


def all_configurations_equal(bbn, fg):
    '''We want to make sure that our
    conversion to factor graph is identical
    for all results to the BBN version.
    This is so that we can eventually
    switch out the old BBN inference engine
    that constructs truth tables as this
    takes too long.'''

    # Now we will test all combinations
    # of variable assignments including
    # some non-observations which
    # we will represent by a '-'
    vals = []

    # Firstly if the bbn does not have domains
    # we will build it here...
    if not bbn.domains:
        for node in bbn.nodes:
            if node.func.domains:
                bbn.domains.update(node.func.domains)
    for node in bbn.nodes:
        if node.variable_name not in bbn.domains:
            bbn.domains[node.variable_name] = [True, False]

    assert len(bbn.domains) == len(bbn.nodes)

    # TODO: We should ensure that the converted fg
    # has the correct .domains.
    fg.domains = bbn.domains

    for variable, domain in bbn.domains.items():
        vals.append(list(xproduct([variable], domain + ['-'])))
    permutations = list(xproduct(*vals))
    assert permutations
    # Now we have every possible combination
    # including unobserved variables.
    # We will construct the queries and
    # then compare the results...
    for permutation in permutations:
        assert permutation
        bbn_query = dict([p for p in permutation if p[1] != '-'])
        # Now construct the fg query with the
        # slightly different fg variable names...
        fg_query = dict([p for p in permutation if p[1] != '-'])
        # Now execute the two queries...
        bbn_result = bbn.query(**bbn_query)
        fg_result = fg.query(**fg_query)
        assert len(bbn_result) == len(fg_result)
        for (variable_name, value), v in bbn_result.items():
            print round(v, 6), round(fg_result[(variable_name, value)], 6)
            #assert round(v, 6) == (
            #    round(fg_result[(variable_name, value)], 6))
            assert abs(v - fg_result[(variable_name, value)]) < 0.0001
    return True


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

    def test_build_join_tree_cancer_bbn(self, cancer_bbn):
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

        jt = build_join_tree(cancer_bbn, priority_func_override)
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

    def test_monty_convert_to_factor_graph(self, monty_bbn):
        monty_converted = monty_bbn.convert_to_factor_graph()
        assert not monty_converted.has_cycles()

        # Now we will test all combinations
        # of the converted factor graph with
        # the original graph.
        assert all_configurations_equal(monty_bbn, monty_converted)

    def test_stat_535_lecture_8(self):
        '''Example From http://www.stat.washington.edu/courses/stat535/fall10/Handouts/l8-jt-variants.pdf'''

        def f_c(c):
            return 1

        def f_b(b):
            return 1

        def f_a(c, b, a):
            return 1

        def f_d(c, b, d):
            return 1

        def f_e(b, e):
            return 1

        bbn = build_bbn(
            f_a,
            f_b,
            f_c,
            f_d,
            f_e)

        fg = bbn.convert_to_factor_graph()
        # Note there are multiple ways the
        # junction tree assignments could have gone
        # TODO: Make the assignments deterministic.

    def test_earthquake_convert_to_factor_graph(
            self, earthquake_bbn):
        earthquake_converted = earthquake_bbn.convert_to_factor_graph()
        assert not earthquake_converted.has_cycles()
        # Now we will test all combinations
        # of the converted factor graph with
        # the original graph.
        assert all_configurations_equal(earthquake_bbn, earthquake_converted)

    def test_huang_darwiche_convert_to_factor_graph(
            self, huang_darwiche_dag):
        huang_darwiche_converted = huang_darwiche_dag.convert_to_factor_graph()

        # Check that the domain for variable nodes, if it exists
        # is always a list...
        for variable_node in huang_darwiche_converted.variable_nodes():
            if hasattr(variable_node, 'domain'):
                assert isinstance(variable_node.domain, list)
        import ipdb; ipdb.set_trace()
        huang_darwiche_converted.propagate()
        #assert variable_domains_match_function_domains(huang_darwiche_converted)
        assert not huang_darwiche_converted.has_cycles()

        # Now we will test all combinations
        # of the converted factor graph with
        # the original graph.
        assert all_configurations_equal(
            huang_darwiche_dag, huang_darwiche_converted)

    def test_cancer_convert_to_factor_graph(
            self, cancer_bbn):
        cancer_converted = cancer_bbn.convert_to_factor_graph()

        # Make sure that the domains have been
        # set on all
        # Now we will test all combinations
        # of the converted factor graph with
        # the original graph.
        assert all_configurations_equal(
            cancer_bbn, cancer_converted)


def test_expand_domains():
    original_variables = ['A', 'B', 'C']
    domains = dict(
        A = (True, False),
        B = (True, False),
        C = (True, False))
    expanded = expand_domains(
        original_variables,
        domains, 'ABC')
    assert 'ABC' in expanded
    assert len(expanded['ABC']) == 8
    for val in [
            [True, True, True],
            [True, True, False],
            [True, False, True],
            [True, False, False],
            [False, True, True],
            [False, True, False],
            [False, False, True],
            [False, False, False]]:
        assert val in expanded['ABC']
    assert len(expanded) == 1


def test_make_potential():
    def product_func(c, b, a):
        return c, b, a
    product_func.domains = dict(
        a = (True, False),
        b = (True, False),
        c = (True, False))
    domains = expand_domains(
        ('c', 'b', 'a'),
        product_func.domains,
        'a_b_c')
    potential_func = make_potential_func(
        ('a', 'b', 'c'),
        domains,
        product_func)
    for k, vals in domains.items():
        for v in vals:
            assert product_func(v[2], v[1], v[0]) == potential_func(v)


def test_clique_tree_sum_product_happiness(happiness_bbn):
    """For this test we will ensure that we get
    the same results from the sum product algorithm
    and the older more reliable (but slower)
    Huang Darwiche method ."""

    # Make a copy so that we can compare results...
    happiness_bbn_copy = copy.deepcopy(happiness_bbn)
    happiness_bbn.inference_method = 'clique_tree_sum_product'
    assert all_configurations_equal(happiness_bbn, happiness_bbn_copy)


def test_clique_tree_sum_product_monty(monty_bbn):
    """For this test we will ensure that we get
    the same results from the sum product algorithm
    and the older more reliable (but slower)
    Huang Darwiche method ."""

    # Make a copy so that we can compare results...
    monty_bbn_copy = copy.deepcopy(monty_bbn)
    monty_bbn.inference_method = 'clique_tree_sum_product'
    assert all_configurations_equal(monty_bbn, monty_bbn_copy)


def test_clique_tree_sum_product_cancer(cancer_bbn):
        cancer_copy = copy.deepcopy(cancer_bbn)
        cancer_copy.inference_method = 'clique_tree_sum_product'
        assert all_configurations_equal(
            cancer_bbn, cancer_copy)


def test_clique_tree_sum_product_with_evidence(happiness_bbn):
    happiness_bbn_copy = copy.deepcopy(happiness_bbn)
    happiness_bbn.inference_method = 'clique_tree_sum_product'
    # For the following evidence I was getting different
    # values for sum product and Huang Darwiche.
    result = happiness_bbn.query(g=True, i=True, h=True, j=True, l=True)
    assert result[('l', True)] == 1.0
    assert result[('h', False)] == 0.0
    assert result[('h', True)] == 1.0
    assert result[('s', False)] == 0.04000000000000001
    assert result[('i', True)] == 1.0
    assert result[('g', True)] == 1.0
    assert result[('i', False)] == 0.0
    assert result[('g', False)] == 0.0
    assert result[('c', True)] == 0.5510204081632653
    assert result[('j', False)] == 0.0
    assert result[('c', False)] == 0.44897959183673475
    assert result[('j', True)] == 1.0
    assert result[('s', True)] == 0.96
    assert result[('d', False)] == 0.5510204081632653
    assert result[('d', True)] == 0.44897959183673475
    assert result[('l', False)] == 0.0

def test_memoization_potential_funcs(happiness_bbn):
    """Test that each factor is called at most once for
    every argument combination."""
    happiness_bbn.inference_method = 'clique_tree_sum_product'
    happiness_bbn.q()
    for clique_node in happiness_bbn._jt.clique_nodes:
        assert len(clique_node.potential_func.call_count) <= (
            2 ** len(get_args(clique_node.potential_func)))
        print clique_node.potential_func.call_count


def test_memoization_on_messages(happiness_bbn):
    """Test that each factor is called at most once for
    every argument combination."""
    happiness_bbn.inference_method = 'clique_tree_sum_product'
    happiness_bbn.q()
    for clique_node in happiness_bbn._jt.clique_nodes:
        for source, message in clique_node.received_messages.items():
            assert len(message.call_count) <= (2 ** len(message.argspec))
