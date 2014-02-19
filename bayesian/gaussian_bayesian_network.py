'''Classes for pure Gaussian Bayesian Networks'''
import math
from functools import wraps
from numbers import Number
from collections import Counter
from itertools import product

from bayesian.graph import Graph, Node, connect
from bayesian.gaussian import make_gaussian_cdf
from bayesian.gaussian import marginalize_joint
from bayesian.gaussian import joint_to_conditional, conditional_to_joint
from bayesian.linear_algebra import zeros, Matrix
from bayesian.utils import get_args
from bayesian.utils import get_original_factors

#from bayesian.matfunc import Matrix, Square
#from numpy import matrix as Matrix
#import numpy as np

# Decorators for creation of Guassian Nodes.
def gaussian(mu, sigma):
    # This is the gaussian decorator
    # which is a decorator with parameters
    # This means it should return a
    # 'normal' decorated ie twice wrapped...

    def gaussianize(f):

        @wraps(f)
        def gaussianized(*args):
            x = args[0]
            return 1 / (sigma * (2 * math.pi) ** 0.5) * \
                math.exp((-(x - mu) ** 2) / (2 * sigma ** 2))

        gaussianized.mean = mu
        gaussianized.std_dev = sigma
        gaussianized.variance = sigma ** 2
        gaussianized.cdf = make_gaussian_cdf(mu, sigma)
        gaussianized.argspec = get_args(f)

        return gaussianized
    return gaussianize


def conditional_gaussian(mu, sigma, betas):

    def conditional_gaussianize(f):

        @wraps(f)
        def conditional_gaussianized(**args):
            '''Since this function will never
            be called directly we dont need anything here.
            '''
            pass

        # Actually the mean now becomes a
        # function of the dependent variable

        conditional_gaussianized.mean = mu
        conditional_gaussianized.std_dev = sigma
        conditional_gaussianized.variance = sigma ** 2
        conditional_gaussianized.raw_betas = betas
        conditional_gaussianized.argspec = get_args(f)

        return conditional_gaussianized

    return conditional_gaussianize


class GBNNode(Node):

    def __init__(self, factor):
        super(GBNNode, self).__init__(factor.__name__)
        self.func = factor
        self.argspec = get_args(factor)

    def __repr__(self):
        return '<GuassianNode %s (%s)>' % (
            self.name,
            self.argspec)

    @property
    def variance(self):
        return self.func.variance


class GaussianBayesianGraph(Graph):

    def __init__(self, nodes, name=None):
        self.nodes = nodes
        self.name = name
        # Assign integer indices
        # to the nodes to trace
        # matrix rows and cols
        # back to the nodes.
        # The indices must be in
        # topological order.
        ordered = self.get_topological_sort()
        for i, node in enumerate(ordered):
            node.index = i

    def get_joint_parameters(self):
        '''Return the vector of means
        and the covariance matrix
        for the full joint distribution.
        For now, by definition, all
        the variables in a Gaussian
        Bayesian Network are either
        univariate gaussian or
        conditional guassians.
        '''
        ordered = self.get_topological_sort()
        mu_x = Matrix([[ordered[0].func.mean]])
        sigma_x = Matrix([[ordered[0].func.variance]])
        # Iteratively build up the mu and sigma matrices
        for node in ordered[1:]:
            beta_0 = node.func.mean
            beta = zeros((node.index, 1))
            total = 0
            for parent in node.parents:
                beta_0 -= node.func.betas[parent.variable_name] * \
                          parent.func.mean
                beta[parent.index, 0] = node.func.betas[parent.variable_name]
            sigma_c = node.func.variance
            mu_x, sigma_x = conditional_to_joint(
                mu_x, sigma_x, beta_0, beta, sigma_c)
        return mu_x, sigma_x

    def joint_to_conditional(self, mu, sigma, graph_to_matrix):
        '''
        matrix_to_graph is a dict from matrix indices
        to node indices in the graph
        '''
        retval = dict()
        mu_conds = []
        sigma_conds = []

        # mu[graph_to_matrix[self.nodes['b'].index]]
        #mu_conds.append(mu[0])
        #sigma_conds.append(sigma[0][0])

        # First we need to determine which of
        # the conditionals we are interested in
        # and then 1-by-1 marginalize out
        # the rest
        #graph_to_matrix = dict([(v, k) for k, v in matrix_to_graph.items()])
        retval = dict()
        nodes_of_interest = [n for n in self.nodes.values()
                             if n.index in graph_to_matrix]
        for node in nodes_of_interest:

            joint_indices = []
            x_indices = []
            joint_indices.append(graph_to_matrix[node.index])
            y_index = graph_to_matrix[node.index]
            for parent in node.parents:
                if parent.index in graph_to_matrix:
                    joint_indices.append(graph_to_matrix[parent.index])
                    x_indices.append(graph_to_matrix[parent.index])
            print node.name, joint_indices, y_index, x_indices
            # Now if the parents are not in the matrix
            # ie x_indices is empty we just return the univariate guassian
            # for that variable
            if not x_indices:
                retval[node.name] = (mu[y_index], sigma[y_index][y_index])
            else:
                pass
                # This is where we need to construct the
                # parameters for joint to conditional...

                #r = joint_to_conditional(x_indices, y_index,
                #                         mu, sigma)
                #retval[node.name] = r

        # Now lets try the simple way...
        results = []
        sigma_xx = zeros((len(mu) - 1, len(mu) - 1))
        mu_x = zeros((len(mu) - 1, 1))
        mu_y = Matrix([[mu[len(mu) -1, 0]]])
        sigma_yy = Matrix([
            [sigma[len(sigma) -1, len(sigma) - 1]]])
        sigma_yx = Matrix([])


        for i in range(0, len(sigma_x)):
            mu_x[i, 0] = mu[i, 0]
            for j in range(0, len(sigma_x)):
                sigma_x[i, j] = sigma[i, j]
        #while len(results) < len(mu):



        return retval

    def query(self, **kwds):
        '''See equations 6 and 7'''
        z = zeros((len(kwds), 1))
        mu_Z = zeros((len(kwds), 1))
        mu_Y = zeros((len(self.nodes) - len(kwds), 1))
        mu_Z_map = {}
        mu_Y_map = {}
        old_mu, old_sigma = self.get_joint_parameters()
        c = Counter()
        for e in kwds:
            mu_Z_map[self.nodes[e].index] = c['z']
            mu_Z[c['z'], 0] = self.nodes[e].func.mean
            z[c['z'], 0] = kwds[e]
            c['z'] += 1
        for name, node in self.nodes.items():
            if name not in kwds:
                mu_Y_map[self.nodes[name].index] = c['y']
                mu_Y[c['y'], 0] = (
                    self.nodes[name].func.mean)
                c['y'] += 1

        # Construct block arrays
        yy_size = len(mu_Y)
        yz_size = len(mu_Y)
        zy_size = len(mu_Z)
        zz_size = len(mu_Z)
        sigma_YY = zeros((yy_size, yy_size))
        sigma_YZ = zeros((yz_size, zy_size))
        sigma_ZY = zeros((zy_size, yz_size))
        sigma_ZZ = zeros((zz_size, zz_size))

        evidence_indices = set([self.nodes[n].index for n in kwds.keys()])
        for a, b in product(range(len(old_sigma)), range(len(old_sigma))):
            v = old_sigma[a, b]
            if a in evidence_indices:
                if b in evidence_indices:
                    sigma_ZZ[mu_Z_map[a], mu_Z_map[b]] = old_sigma[a, b]
                else:
                    sigma_ZY[mu_Z_map[a], mu_Y_map[b]] = v
            else:
                if b in evidence_indices:
                    sigma_YZ[mu_Y_map[a], mu_Z_map[b]] = v
                else:
                    sigma_YY[mu_Y_map[a], mu_Y_map[b]] = v
        # Now we can apply equations 6 and 7 to
        # get the new joint parameters
        mu_Y_g_Z = mu_Y + sigma_YZ * (sigma_ZZ.I * (z - mu_Z))
        sigma_Y_g_Z = sigma_YY - sigma_YZ * sigma_ZZ.I * sigma_ZY
        # Note, we need to convert the matrices back
        # to the conditional form and also
        # de-map the entries back to the nodes.
        result = dict()
        # We will return a pair for each of the
        # variables being the mean and sd for
        # each variable.
        for k, v in kwds.items():
            result[k] = (v, 0)
        # Now for unseen variables ie those definied
        # by the joint mu_Y_g_Z and sigma_Y_g_Z
        # we want to return the conditionals for
        # each one.
        #mu_Y_map_inv = dict([(v, k) for k, v in mu_Y_map.items()])
        import ipdb; ipdb.set_trace()
        r = self.joint_to_conditional(mu_Y_g_Z, sigma_Y_g_Z, mu_Y_map)

        return mu_Y_g_Z, sigma_Y_g_Z


    def q(self, **kwds):
        '''Wrapper around query

        This method formats the query
        result in a nice human readable format
        for interactive use.
        '''
        result = self.query(**kwds)
        tab = PrettyTable(['Node', 'Value', 'Marginal'], sortby='Node')
        tab.align = 'l'
        tab.align['Marginal'] = 'r'
        tab.float_format = '%8.6f'
        for (node, value), prob in result.items():
            if kwds.get(node, '') == value:
                tab.add_row(['%s*' % node,
                             '%s%s*%s' % (GREEN, value, NORMAL),
                             '%8.6f' % prob])
            else:
                tab.add_row([node, value, '%8.6f' % prob])
        print tab

    def discover_sample_ordering(self):
        return discover_sample_ordering(self)

    def export(self, filename=None, format='graphviz'):
        '''Export the graph in GraphViz dot language.'''
        if filename:
            fh = open(filename, 'w')
        else:
            fh = sys.stdout
        if format != 'graphviz':
            raise 'Unsupported Export Format.'
        fh.write('graph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in self.nodes:
            if isinstance(node, FactorNode):
                fh.write('  %s [ shape="rectangle" color="red"];\n' % node.name)
            else:
                fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
        for node in self.nodes:
            for neighbour in node.neighbours:
                edge = [node.name, neighbour.name]
                edge = tuple(sorted(edge))
                edges.add(edge)
        for source, target in edges:
            fh.write('  %s -- %s;\n' % (source, target))
        fh.write('}\n')


def build_gbn(*args, **kwds):
    '''Builds a Gaussian Bayesian Graph from
    a list of functions'''
    variables = set()
    name = kwds.get('name')
    variable_nodes = dict()
    factor_nodes = dict()

    if isinstance(args[0], list):
        # Assume the functions were all
        # passed in a list in the first
        # argument. This makes it possible
        # to build very large graphs with
        # more than 255 functions, since
        # Python functions are limited to
        # 255 arguments.
        args = args[0]

    for factor in args:
        factor_args = get_args(factor)
        variables.update(factor_args)
        node = GBNNode(factor)
        factor_nodes[factor.__name__] = node

    # Now lets create the connections
    # To do this we need to find the
    # factor node representing the variables
    # in a child factors argument and connect
    # it to the child node.
    # Note that calling original_factors
    # here can break build_gbn if the
    # factors do not correctly represent
    # a valid network. This will be fixed
    # in next release
    original_factors = get_original_factors(factor_nodes.values())
    for var_name, factor in original_factors.items():
        factor.variable_name = var_name
    for factor_node in factor_nodes.values():
        factor_args = get_args(factor_node)
        parents = [original_factors[arg] for arg in
                   factor_args if original_factors[arg] != factor_node]
        for parent in parents:
            connect(parent, factor_node)
    # Now process the raw_betas to create a dict
    for factor_node in factor_nodes.values():
        # Now we want betas to always be a dict
        # but in the case that the node only
        # has one parent we will allow the user to specify
        # the single beta for that parent simply
        # as a number and not a dict.
        if hasattr(factor_node.func, 'raw_betas'):
            if isinstance(factor_node.func.raw_betas, Number):
                # Make sure that if they supply a number
                # there is only one parent
                assert len(get_args(factor_node)) == 2
                betas = dict()
                for arg in get_args(factor_node):
                    if arg != factor_node.variable_name:
                        betas[arg] = factor_node.func.raw_betas
                factor_node.func.betas = betas
            else:
                factor_node.func.betas = factor_node.func.raw_betas
    gbn = GaussianBayesianGraph(original_factors, name=name)
    return gbn


def build_graph(*args, **kwds):
    '''For compatibility, this is
    just a wrapper around build_gbn'''
    return build_gbn(*args, **kwds)
