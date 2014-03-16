import pytest
import os
from itertools import product as xproduct

from bayesian.bbn import *


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

    Ok this doesnt

    '''
    import ipdb; ipdb.set_trace()
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
