'''Classes for pure Gaussian Bayesian Networks'''
import math
import types
from functools import wraps
from numbers import Number
from collections import Counter
from itertools import product as xproduct
from StringIO import StringIO

from bayesian.graph import Graph, Node, connect
from bayesian.gaussian import make_gaussian_cdf
from bayesian.gaussian import marginalize_joint
from bayesian.gaussian import joint_to_conditional, conditional_to_joint
from bayesian.gaussian import CovarianceMatrix, MeansVector
from bayesian.linear_algebra import zeros, Matrix
from bayesian.utils import get_args
from bayesian.utils import get_original_factors
from bayesian.exceptions import VariableNotInGraphError
from bayesian.linear_algebra import Matrix

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
        gaussianized.entropy = types.MethodType(
            lambda x: 0.5 * math.log(2 * math.pi * math.e * x.variance),
            gaussianized)

        return gaussianized
    return gaussianize


def conditional_gaussian(mu, sigma, betas):

    def conditional_gaussianize(f):

        @wraps(f)
        def conditional_gaussianized(*args, **kwds):
            '''Since this function will never
            be called directly we dont need anything here.
            '''
            # First we need to construct a vector
            # out of the args...
            x = zeros((len(args), 1))
            for i, a in enumerate(args):
                x[i, 0] = a
            sigma = conditional_gaussianized.covariance_matrix
            mu = conditional_gaussianized.joint_mu
            return 1 / (2 * math.pi * sigma.det()) ** 0.5 \
                * math.exp(-0.5 * ((x - mu).T * sigma.I * (x - mu))[0, 0])

        conditional_gaussianized.mean = mu
        conditional_gaussianized.std_dev = sigma
        conditional_gaussianized.variance = sigma ** 2
        conditional_gaussianized.raw_betas = betas
        conditional_gaussianized.argspec = get_args(f)
        conditional_gaussianized.entropy = types.MethodType(
            lambda x: len(x.joint_mu) / 2 * \
            (1 + math.log(2 * math.pi)) + \
            0.5 * math.log(x.covariance_matrix.det()), conditional_gaussianized)

        # NOTE: the joint parameters are
        # add to this function at the time of the
        # graph construction

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
                #beta_0 -= node.func.betas[parent.variable_name] * \
                #          parent.func.mean
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
        print 'Covariance Matrix:'
        print sigma
        print 'Means:'
        print mu


    def discover_sample_ordering(self):
        return discover_sample_ordering(self)


    def get_graphviz_source(self):
        fh = StringIO()
        fh.write('digraph G {\n')
        fh.write('  graph [ dpi = 300 bgcolor="transparent" rankdir="LR"];\n')
        edges = set()
        for node in sorted(self.nodes.values(), key=lambda x:x.name):
            fh.write('  %s [ shape="ellipse" color="blue"];\n' % node.name)
            for child in node.children:
                edge = (node.name, child.name)
                edges.add(edge)
        for source, target in sorted(edges, key=lambda x:(x[0], x[1])):
            fh.write('  %s -> %s;\n' % (source, target))
        fh.write('}\n')
        return fh.getvalue()

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
    # Now for any conditional gaussian nodes
    # we need to tell the node function what the
    # parent parameters are so that the pdf can
    # be computed.
    sorted = gbn.get_topological_sort()
    joint_mu, joint_sigma = gbn.get_joint_parameters()
    for node in sorted:
        if hasattr(node.func, 'betas'):
            # This means its multivariate gaussian
            names = [n.variable_name for n in node.parents] + [node.variable_name]
            node.func.joint_mu = MeansVector.zeros((len(names), 1), names=names)
            for name in names:
                node.func.joint_mu[name] = joint_mu[name][0, 0]
            node.func.covariance_matrix = CovarianceMatrix.zeros(
                (len(names), len(names)), names)
            for row, col in xproduct(names, names):
                node.func.covariance_matrix[row, col] = joint_sigma[row, col]
    return gbn


def build_graph(*args, **kwds):
    '''For compatibility, this is
    just a wrapper around build_gbn'''
    return build_gbn(*args, **kwds)
