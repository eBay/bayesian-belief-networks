'''Generic Graph Classes'''
from StringIO import StringIO

class Node(object):

    def __init__(self, name, parents=[], children=[]):
        self.name = name
        self.parents = parents[:]
        self.children = children[:]

    def __repr__(self):
        return '<Node %s>' % self.name


class UndirectedNode(object):

    def __init__(self, name, neighbours=[]):
        self.name = name
        self.neighbours = neighbours[:]

    def __repr__(self):
        return '<UndirectedNode %s>' % self.name


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

    def get_topological_sort(self):
        '''In order to make this sort
        deterministic we will use the
        variable name as a secondary sort'''
        l = []
        l_set = set() # For speed
        s = [n for n in self.nodes.values() if not n.parents]
        s.sort(reverse=True, key=lambda x:x.variable_name)
        while s:
            n = s.pop()
            l.append(n)
            l_set.add(n)
            # Now some of n's children may be
            # added to s if all their parents
            # are already accounted for.
            for m in n.children:
                if set(m.parents).issubset(l_set):
                    s.append(m)
                    s.sort(reverse=True, key=lambda x:x.variable_name)
        if len(l) == len(self.nodes):
            return l
        raise "Graph Has Cycles"


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

    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        fh.write(self.get_graphviz_source())


def connect(parent, child):
    '''
    Make an edge between a parent
    node and a child node.
    a - parent
    b - child
    '''
    parent.children.append(child)
    child.parents.append(parent)
