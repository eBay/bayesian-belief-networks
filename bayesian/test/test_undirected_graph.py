import pytest

import os

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
