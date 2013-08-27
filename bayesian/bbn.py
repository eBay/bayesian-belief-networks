from __future__ import division
'''Data Structures to represent a BBN as a DAG.'''

from StringIO import StringIO

class Node(object):

    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents[:]
        self.children = children[:]

    def __repr__(self):
        return '<Node %s>' % self.name


class BBNNode(Node):

    def __init__(self, factor):
        super(BBNNode, self).__init__()
        self.func = factor


class UndirectedNode(object):

    def __init__(self, name, neighbours=[]):
        self.name = name
        self.neighbours = neighbours[:]


    def __repr__(self):
        return '<Node %s>' % self.name


class Graph(object):

    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())



class UndirectedGraph(object):

    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name


    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edges.add(tuple(sorted(edge)))
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()



class BBN(Graph):
    '''A Directed Acyclic Graph'''

    def __init__(self, nodes, name=None):
        self.nodes = nodes

    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('digraph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for child in node.children:
                edge = (node.name, child.name)
                edges.add(edge)
        for source, target in edges:
            fh.write('  %s -> %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

    def moralize(self):
        '''Marry the unmarried parents'''
        pass
        #for node in self.nodes:
        #    unmarried = set


def connect(a, b):
    '''
    Make an edge between two nodes
    or between a source and several
    neighbours.
    '''
    if not isinstance(b, list):
        b = [b]
    for b_ in b:
        a.neighbours.append(b_)
        b_.neighbours.append(a)


def build_bbn(*args, **kwds):
    '''Builds a BBN Graph from
    a list of functions and domains'''
    variables = set()
    domains = kwds.get('domains', {})
    name = kwds.get('name')
    variable_nodes = dict()
    factor_nodes = dict()
    if isinstance(args[0], list):
        # Assume the functions were all
        # passed in a list in the first
        # argument. This makes it possible
        # to build very large graphs with
        # more than 255 functions.
        args = args[0]
    for factor in args:
        factor_args = get_args(factor)
        variables.update(factor_args)
        bbn_node = Node(factor.__name__, factor)
        factor_nodes[factor.__name__] = bbn_node
    # Now lets create the connections
    # To do this we need to find the
    # factor node representing the variables
    # in a child factors argument and connect
    # it to the child node
    for factor_name, factor_node in factor_nodes.items():
        for
    bbn = BBN(factor_nodes, name=name)
    return bbn
