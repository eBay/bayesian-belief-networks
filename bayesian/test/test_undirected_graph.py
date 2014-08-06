import pytest
import os
from itertools import product as xproduct

from bayesian.examples.bbns.huang_darwiche import *
from bayesian.examples.undirected_graphs.monty_2 import *
from bayesian.undirected_graphical_model import *
from bayesian.linear_chain_crf import build_lccrf
from bayesian.examples.undirected_graphs.pos_tagging_as_crf import(
    indicator_func_8,
    indicator_func_9,
    indicator_func_10,
    indicator_func_11,
    indicator_func_12,
    indicator_func_13,
    indicator_func_14,
    indicator_func_15,
    indicator_func_16,)
from bayesian.examples.undirected_graphs.pos_tagging_as_crf import(
    build_um_from_sentence,
    attach_feature_functions)


def pytest_funcarg__huang_darwiche_factors(request):
    return [
        f_a, f_b, f_c, f_d,
        f_e, f_f, f_g, f_h]


def pytest_funcarg__undirected_monty_factors(request):
    return [
        f_u_prize_door, f_u_guest_door,
        f_u_monty_door]


def pytest_funcarg__undirected_monty_model(request):
    ug = get_graph()
    return ug


def pytest_funcarg__sprinkler_graph(request):
    '''The Sprinkler Example as a moralized undirected graph
    to be used in tests.
    '''
    cloudy = Node('Cloudy')
    sprinkler = Node('Sprinkler')
    rain = Node('Rain')
    wet_grass = Node('WetGrass')
    cloudy.neighbours = [
        sprinkler, rain]
    sprinkler.neighbours = [cloudy, wet_grass]
    rain.neighbours = [cloudy, wet_grass]
    wet_grass.neighbours = [
        sprinkler,
        rain]
    graph = UndirectedGraph([
        cloudy,
        sprinkler,
        rain,
        wet_grass])
    return graph


def pytest_funcarg__f_monty_door(request):
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
    return f_monty_door


def pytest_funcarg__pos_tag_weights(request):
    return (
        3.174603174603173,
        3.977272727272726,
        1.9607843137254881,
        5.194805194805184,
        6.779661016949148,
        1.9801980198019793,
        7.6923076922798295,
        1.9801980198019793,
        3.8461538461399147
    )


def pytest_funcarg__pos_tag_feature_functions(request):
    return (
        indicator_func_8,
        indicator_func_9,
        indicator_func_10,
        indicator_func_11,
        indicator_func_12,
        indicator_func_13,
        indicator_func_14,
        indicator_func_15,
        indicator_func_16,
    )


def pytest_funcarg__pos_tag_lccrf(request):
    output_alphabet = ['NAME', 'OTHER']
        # These are the weights from the lccrf training...
    weights = pytest_funcarg__pos_tag_weights(request)
    feature_functions = pytest_funcarg__pos_tag_feature_functions(request)
    lccrf = build_lccrf(
        output_alphabet, feature_functions)
    lccrf.weights = weights
    return lccrf


class TestUndirectedGraph():

    def test_get_graphviz_source(self, sprinkler_graph):
        gv_src = '''graph G {
  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];
  Cloudy [ shape="ellipse" color="blue"];
  Sprinkler [ shape="ellipse" color="blue"];
  Rain [ shape="ellipse" color="blue"];
  WetGrass [ shape="ellipse" color="blue"];
  Rain -- WetGrass;
  Sprinkler -- WetGrass;
  Cloudy -- Sprinkler;
  Cloudy -- Rain;
}
'''
        assert sprinkler_graph.get_graphviz_source() == gv_src


def test_cluster_sepset_consistancy(f_monty_door):
    '''See Equation (1) page 9 of H & D
    We need to ensure that for
    jt in order to perform correct
    inference the consistancy property
    holds for each cluster/sepset pair.
    We will use the Monty ug to
    test this here.
    In the Undirected version of
    Monty we have the cluster:
    Guest-Monty with its neighbouring
    Sepset Monty.
    The Potential func for
    Guest Monty is p(G) x p(M|G,P)
    i.e. it is a function of all
    three variables.
    The SepSet consists of {Monty}
    since this is the common node
    of the two clusters.
    Actually we only need to use each
    variable once in a cluster so
    for example we could create the
    potential funcs as possible:
    For cluster [GM]: p(G)
                [PM]: p(P)
         Sepset [M]:  p(M|G, P)
    In the above we have assigned
    var G to parent cluster GM,
    var P to parent cluster PM
    and variable M to the sepset cluster
    M
    Mmmm so unless I am doing this
    wrong, the above assignment
    DOES not work
    Lets try the following assignment:

    P -> PM
    G -> M
    M -> M

    so Potentials are:
    pot_PM = p(P)
    pot_M = p(g) * p(m|g, p)
    pot_GM = 1

    Ok this doesnt work either....
    Third possibility is to move
    the monty conditional into
    one of the PM or GM clusters like this:

    P -> PM
    G -> GM
    M -> PM
    ie
    pot_PM = p(p) * p(m|g, p)

    '''
    pot_GM = lambda p, g, m: 1.0
    pot_PM = lambda p, g, m: 1.0 / 3
    pot_M = lambda p, g, m: 1.0 / 3 * f_monty_door(p, g, m)

    # Now for consistancy we need
    # to ensure that pot_M = Sum_over_M(pot_GM)

    #total = 0
    #for guest_door in ['A', 'B', 'C']:
    #    total += pot_GM(guest_door)
    # Now for all vars in pot_M ie
    # all three of P, G, M for all
    # values we need to ensure that
    # pot_M(P, G, M) == pot_GM
    for prize_door, guest_door, monty_door in \
        xproduct(['A', 'B', 'C'], ['A', 'B', 'C'],
                 ['A', 'B', 'C']):
        total = 0
        for m_door in ['A', 'B', 'C']:
            total += pot_GM(prize_door, guest_door, m_door)
        assert pot_M(prize_door, guest_door, monty_door) == total


def test_build_graph(undirected_monty_factors):
    ugm = build_graph(
        undirected_monty_factors,
        domains=dict(
            prize_door=['A', 'B', 'C'],
            guest_door=['A', 'B', 'C'],
            monty_door=['A', 'B', 'C']))
    assert len(ugm.nodes) == 3
    for node in ugm.nodes:
        assert len(node.neighbours) == 2


def test_verify(undirected_monty_model):
    assert verify(undirected_monty_model)


def test_build_factor_graph(undirected_monty_model):
    fg = undirected_monty_model.build_factor_graph()
    fg.q()


def test_lccrf_results_same_as_crf_results(
        pos_tag_lccrf, pos_tag_feature_functions, pos_tag_weights):
    '''Since for the lccrf we are using
    the viterbi algorithm and for
    general crfs we are using the
    clique_tree_sum_product algorithm
    we want to verify that we get the
    same results when using identical weights.
    '''
    test_sentences = (
        'Claude Shannon',
        'the first president was George Washington',
        'George Washington was the first president',
    )
    from pprint import pprint
    for sentence in test_sentences:
        pos_tag_lccrf.q(sentence)
        X_seq = tuple(sentence.split())
        um = build_um_from_sentence(X_seq, ['NAME', 'OTHER'])
        um.log_domain = True
        attach_feature_functions(
            um, pos_tag_feature_functions,
            X_seq, pos_tag_weights)
        result = um.mpe_query()
        most_likely = dict()
        for var in um.variables:
            most_likely[var] = max(
                [x for x in result.items() if x[0][0]==var],
                key=lambda x:x[1])[0][1]

        pprint(most_likely)
        max_potential = max(result.values())
        # Which variables are at the max potential?
        # Now to normalize we take any of the
        # maxes and divide by the total of that
        # variables potential assignments
        last_var_name = 'y{}'.format(len(X_seq) - 1)
        total = sum([x[1] for x in result.items() if x[0][0] == last_var_name])
        mpe_prob = max_potential / total
        #import ipdb; ipdb.set_trace()
        # Seems like we need to know the *last*
        # variable that was computed in the
        # variable elimination since the
        # values assigned to other variables
        # than the mpe value may be higher...
        print mpe_prob
