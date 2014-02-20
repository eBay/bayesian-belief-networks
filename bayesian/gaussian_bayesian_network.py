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
from bayesian.gaussian import CovarianceMatrix, MeansVector
from bayesian.linear_algebra import zeros, Matrix
from bayesian.utils import get_args
from bayesian.utils import get_original_factors
from bayesian.exceptions import VariableNotInGraphError

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
        # Now set the names on the covariance matrix to
        # the graphs variabe names
        names = [n.variable_name for n in ordered]
        mu_x.set_names(names)
        sigma_x.set_names(names)
        return mu_x, sigma_x

    def query(self, **kwds):

        # Ensure the evidence variables are actually
        # present
        invalid_vars = [v for v in kwds.keys() if v not in self.nodes]
        if invalid_vars:
            raise VariableNotInGraphError(invalid_vars)

        mu, sigma = self.get_joint_parameters()

        # Iteratively apply the evidence...
        result = dict()
        result['evidence'] = kwds

        for k, v in kwds.items():
            x = MeansVector([[v]], names=[k])
            sigma_yy, sigma_yx, sigma_xy, sigma_xx = (
                sigma.split(k))
            mu_y, mu_x = mu.split(k)
            # See equations (6) and (7) of CK
            mu_y_given_x = MeansVector(
                (mu_y + sigma_yx * sigma_xx.I * (x - mu_x)).rows,
                names = mu_y.name_ordering)
            sigma_y_given_x = CovarianceMatrix(
                (sigma_yy - sigma_yx * sigma_xx.I * sigma_xy).rows,
                names=sigma_yy.name_ordering)
            sigma = sigma_y_given_x
            mu = mu_y_given_x

        result['joint'] = dict(mu=mu, sigma=sigma)
        return result

    def q(self, **kwds):
        '''Wrapper around query

        This method formats the query
        result in a nice human readable format
        for interactive use.
        '''
        result = self.query(**kwds)
        mu = result['joint']['mu']
        sigma = result['joint']['sigma']
        evidence = result['evidence']
        print 'Evidence: %s' % str(evidence)
        print 'Means:'
        print mu
        print 'Covariance Matrix:'
        print sigma


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
