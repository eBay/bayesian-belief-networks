from __future__ import division

import math
from collections import defaultdict
from itertools import combinations, product

from prettytable import PrettyTable

from bayesian.linear_algebra import Matrix, zeros


'''
Provides Gaussian Density functions and
approximation to Gaussian CDF.
see https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution
"Numerical approximations for the normal CDF"
For the multivariate case we use the statsmodels module.
'''

b0 = 0.2316419
b1 = 0.319381530
b2 = -0.356563782
b3 = 1.781477937
b4 = -1.821255978
b5 = 1.330274429


def std_gaussian_cdf(x):
    '''Zelen & Severo approximation'''
    g = make_gaussian(0, 1)
    t = 1 / (1 + b0 * x)
    return 1 - g(x) * (
        b1 * t +
        b2 * t ** 2 +
        b3 * t ** 3 +
        b4 * t ** 4 +
        b5 * t ** 5)


def make_gaussian(mean, std_dev):

    def gaussian(x):
        return 1 / (std_dev * (2 * math.pi) ** 0.5) * \
            math.exp((-(x - mean) ** 2) / (2 * std_dev ** 2))

    gaussian.mean = mean
    gaussian.std_dev = std_dev
    gaussian.cdf = make_gaussian_cdf(mean, std_dev)

    return gaussian


def make_gaussian_cdf(mean, std_dev):

    def gaussian_cdf(x):
        t = (x - mean) / std_dev
        if t > 0:
            return std_gaussian_cdf(t)
        elif t == 0:
            return 0.5
        else:
            return 1 - std_gaussian_cdf(abs(t))

    return gaussian_cdf


def make_log_normal(mean, std_dev, base=math.e):
    '''
    Example of approximate log normal distribution:
    In [13]: t = [5, 5, 5, 5, 6, 10, 10, 20, 50]

    In [14]: [math.log(x) for x in t]
    Out[14]:
    [1.6094379124341003,
    1.6094379124341003,
    1.6094379124341003,
    1.6094379124341003,
    1.791759469228055,
    2.302585092994046,
    2.302585092994046,
    2.995732273553991,
    3.912023005428146]

    When constructing the log-normal,
    keep in mind that the mean parameter is the
    mean of the log of the values.

    '''
    def log_normal(x):

        return 1 / (x * (2 * math.pi * std_dev * std_dev) ** 0.5) * \
            base ** (-((math.log(x, base) - mean) ** 2) / (2 * std_dev ** 2))

    log_normal.cdf = make_log_normal_cdf(mean, std_dev)

    return log_normal


def make_log_normal_cdf(mean, std_dev, base=math.e):

    def log_normal_cdf(x):
        gaussian_cdf = make_gaussian_cdf(0, 1)
        return gaussian_cdf((math.log(x, base) - mean) / std_dev)

    return log_normal_cdf


def discretize_gaussian(mu, stddev, buckets,
                        func_name='f_output_var', var_name='output_var'):
    '''Given gaussian distribution parameters
    generate python code that specifies
    a discretized function suitable for
    use in a bayesian belief network.
    buckets should be a list of values
    designating the endpoints of each
    discretized bin, for example if you
    have a variable in the domain [0; 1000]
    and you want 3 discrete intervals
    say [0-400], [400-600], [600-1000]
    then you supply n-1 values where n
    is the number of buckets as follows:
    buckets = [400, 600]
    The code that is generated will thus
    have three values and the prior for
    each value will be computed from
    the cdf function.

    In addition when the function is
    called with a numeric value it will
    automatically convert the numeric
    value into the correct discrete
    value.
    '''
    result = []

    cdf = make_gaussian_cdf(mu, stddev)
    cutoffs = [cdf(b) for b in buckets]
    probs = dict()

    # First the -infinity to the first cutoff....
    probs['%s_LT_%s' % (var_name, buckets[0])] = cutoffs[0]

    # Now the middle buckets
    for i, (b, c) in enumerate(zip(buckets, cutoffs)):
        if i == 0:
            continue
        probs['%s_GE_%s_LT_%s' % (
            var_name, buckets[i-1], b)] = c - cutoffs[i-1]

    # And the final bucket...
    probs['%s_GE_%s' % (
        var_name, buckets[-1])] = 1 - cutoffs[-1]

    # Check that the values = approx 1
    assert round(sum(probs.values()), 5) == 1

    # Now build the python fuction
    result.append('def %s(%s):' % (func_name, var_name))
    result.append('    probs = dict()')
    for k, v in probs.iteritems():
        result.append("    probs['%s'] = %s" % (k, v))
    result.append('    return probs[%s]' % var_name)

    # We will store the buckets as well as the arg_name
    # as attributes
    # of the function to make conversion to discrete
    # values easier.
    result.append('%s.buckets = %s' % (
        func_name, [buckets]))  # Since the argspec is a list of vars
                                # we will make the buckets a list of
                                # buckets one per arg for easy zip.


    return '\n'.join(result), probs.keys()


def discretize_multivariate_guassian(
        means, cov, buckets, parent_vars, cdf,
        func_name='f_output_var', var_name='output_var'):
    '''buckets should be an iterable of iterables
    where each element represnts the buckets into
    which the corresponding variable should be
    discretized.
    cov is the covariance matrix.
    cdf is a callable'''
    assert len(means) == len(stddevs)
    assert len(stddevs) == len(buckets)
    assert len(buckets) == len(parent_vars)

    inf = float("inf")
    result = []
    tt = dict()

    # First we will build the discrete value domains
    # for each of the parent variables.
    domains = defaultdict(list)
    for parent_var, bins in zip(parent_vars, buckets):
        for start, end in zip([-float("inf")] + bins, bins + [float("inf")]):
            if start == -inf:
                domains[parent_var].append(
                '%s_LT_%s' % (parent_var, end))
            elif end == inf:
                domains[parent_var].append(
                    '%s_GE_%s' % (parent_var, start))
            else:

                domains[parent_var].append(
                    '%s_GE_%s_LT_%s' % (
                        parent_var, start, end))

    # TODO Complete this possibly using statsmodels or scipy
    # to integrate over the pdfs.

    # We store the integrations in a dict with
    # n dimentional keys e.g.
    # probs[('A_LT_10', 'B_GT_10')] = 0.001 etc
    probs = dict()
    return domains


def marginalize_joint(x, mu, sigma):
    '''Given joint parameters we want to
    marginilize out the xth one.
    Assume that sigma is represented as a
    list of lists.'''
    new_mu = mu[:]
    del new_mu[x]
    new_sigma = []
    for i, row in enumerate(sigma):
        if i == x:
            continue
        new_row = row[:]
        del new_row[x]
        new_sigma.append(new_row)
    return new_mu, new_sigma


def joint_to_conditional(
        mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy):
    '''
    See Page 22 from MB08.
    p(X, Y) = N ([mu_x]; [sigma_xx sigma_xy])
                 [mu_y]; [sigma_yx sigma_yy]

    We will be returning the conditional
    distribution of p(Y|X)
    therefore we will always assume
    the shape of mu_y and sigma_yy to be (1, 1)
    Remember that the results of applying
    a single evidence variable in the
    iterative update procedure
    returns the *joint* distribution
    of the full graph given the evidence.
    However what we are actually interested in
    is reading off the individual factors
    of the graph given their dependancies.
    From a joint P(x1,x2...,y)
    return distribution of p(y|x1,x2...)
    mu and sigma should both be instances
    of some Matrix class which can
    invert matrices and do matrix
    arithemetic.
    size(sigma) = (len(mu), len(mu))
    len(mu) = len(x) + 1
    '''
    beta_0 = (mu_y - sigma_yx * sigma_xx.I * mu_x)[0, 0]
    beta = sigma_yx * sigma_xx.I
    sigma = sigma_yy - sigma_yx * sigma_xx.I * sigma_xy
    return beta_0, beta, sigma


def conditional_to_joint(
        mu_x, sigma_x, beta_0, beta, sigma_c):
    '''
    This is from page 19 of
    http://webdocs.cs.ualberta.ca/~greiner/C-651/SLIDES/MB08_GaussianNetworks.pdf
    We are given the parameters of a conditional
    gaussian p(Y|x) = N(beta_0 + beta'x; sigma)
    and also the unconditional means and sigma
    of the joint of the parents: mu_x and sigma_x
    Lets assume Y is always shape(1, 1)
    mu_x has shape(1, len(betas)) and
    sigma_x has shape(len(mu_x), len(mu_x))
    [[mu_X1],
     [mu_X2],
     .....
     [mu_Xn]]

    '''
    mu = MeansVector.zeros((len(beta.rows) + 1, 1))
    for i in range(len(mu_x.rows)):
        mu[i, 0] = mu_x[i, 0]
    assert (beta.T * mu_x).shape == (1, 1)
    mu_y = beta_0 + (beta.T * mu_x)[0, 0]
    mu[len(mu_x), 0] = mu_y
    sigma = CovarianceMatrix.zeros((mu.shape[0], mu.shape[0]))
    # Now the top left block
    # of the covariance matrix is
    # just a copy of the sigma_x matrix
    for i in range(0, len(sigma_x.rows)):
        for j in range(0, len(sigma_x.rows[0])):
            sigma[i, j] = sigma_x[i, j]
    # Now for the top-right and bottom-left corners
    for i in range(0, len(sigma_x)):
        total = 0
        for j in range(0, len(sigma_x)):
            total += beta[j, 0] * sigma_x[i, j]
        sigma[i, len(mu_x)] = total
        sigma[len(mu_x), i] = total
    # And finally for the bottom right corner
    sigma_y = sigma_c + (beta.T * sigma_x * beta)[0, 0]
    sigma[len(sigma_x), len(sigma_x)] = (
        sigma_y)
    return mu, sigma


class NamedMatrix(Matrix):
    '''Wrapper allowing referencing
    of columns and rows by variable
    name'''

    def __init__(self, rows=[], names=[]):
        super(NamedMatrix, self).__init__(rows)
        if not names:
            # Default to x1, x2....
            names = ['x%s' % x for x in range(1, len(rows) + 1)]
        self.set_names(names)

    @classmethod
    def zeros(cls, shape, names=[]):
        '''Alternate constructor that
        creates a zero based matrix'''
        rows, cols = shape
        matrix_rows = []
        for i in range(0, rows):
            matrix_rows.append([0] * cols)
        if not names:
            names = ['x%s' % x for x in range(1, rows + 1)]
        cov = cls(matrix_rows, names)
        return cov

    def set_name(self, col, name):
        self.names[name] = col
        self.index_to_name[col] = name
        self.name_ordering.append(name)

    def set_names(self, names):
        assert len(names) in self.shape
        self.names = dict(zip(names, range(len(names))))
        self.name_ordering = names
        self.index_to_name = dict([(v, k) for k, v in self.names.items()])

    def __getitem__(self, item):
        if isinstance(item, str):
            assert item in self.names
            item = self.names[item]
            return super(NamedMatrix, self).__getitem__(item)
        elif isinstance(item, tuple):
            row, col = item
            if isinstance(row, str):
                assert row in self.names
                row = self.names[row]
            if isinstance(col, str):
                assert col in self.names
                col = self.names[col]
            return super(NamedMatrix, self).__getitem__((row, col))
        else:
            return super(NamedMatrix, self).__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            row, col = item
            if isinstance(row, str):
                assert row in self.names
                row = self.names[row]
            if isinstance(col, str):
                assert col in self.names
                col = self.names[col]
            return super(NamedMatrix, self).__setitem__((row, col), value)
        else:
            return super(NamedMatrix, self).__setitem__(item, value)

    def col(self, j):
        if isinstance(j, str):
            assert j in self.names
            j = self.names[j]
        return [row[j] for row in self.rows]

    def __repr__(self):
        cols = self.name_ordering[:self.shape[1]]
        tab = PrettyTable([''] + cols)
        tab.align = 'r'
        for row in self.name_ordering:
            table_row = [row]
            for col in cols:
                table_row.append('%s' % self[row, col])
            tab.add_row(table_row)
        return tab.get_string()


class CovarianceMatrix(NamedMatrix):
    '''Wrapper allowing referencing
    of columns and rows by variable
    name'''

    def __init__(self, rows=[], names=[]):
        super(CovarianceMatrix, self).__init__(rows)
        if not names:
            # Default to x1, x2....
            names = ['x%s' % x for x in range(1, len(rows) + 1)]
        self.set_names(names)

    def split(self, name):
        '''Split into sigma_xx, sigma_yy etc...'''
        assert name in self.names
        x_names = [n for n in self.name_ordering if n != name]
        sigma_xx = CovarianceMatrix.zeros(
            (len(self) - 1, len(self) - 1),
            names=x_names)
        sigma_yy = CovarianceMatrix.zeros((1, 1), names=[name])
        #sigma_xy = zeros((len(sigma_xx), 1))
        sigma_xy = NamedMatrix.zeros((len(sigma_xx), 1), names=x_names)
        #sigma_yx = zeros((1, len(sigma_xx)))
        sigma_yx = NamedMatrix.zeros((1, len(sigma_xx)), names=x_names)

        for row, col in product(
                self.name_ordering,
                self.name_ordering):
            v = self[row, col]
            if row == name and col == name:
                sigma_yy[0, 0] = v
            elif row != name and col != name:
                sigma_xx[row, col] = v
            elif row == name:
                sigma_xy[col, 0] = v
            else:
                sigma_yx[0, row] = v
        return sigma_xx, sigma_xy, sigma_yx, sigma_yy


class MeansVector(NamedMatrix):
    '''Wrapper allowing referencing
    of rows by variable name.
    In this implementation we will
    always consider a vector of means
    to be a vertical matrix with
    a shape of n rows and 1 col.
    The rows will be named.
    '''


    def __init__(self, rows=[], names=[]):
        super(MeansVector, self).__init__(rows)
        if not names:
            # Default to x1, x2....
            names = ['x%s' % x for x in range(1, len(rows) + 1)]
        self.set_names(names)

    def __getitem__(self, item):
        if isinstance(item, str):
            assert item in self.names
            item = self.names[item]
            return super(MeansVector, self).__getitem__(item)
        elif isinstance(item, tuple):
            row, col = item
            if isinstance(row, str):
                assert row in self.names
                row = self.names[row]
            if isinstance(col, str):
                assert col in self.names
                col = self.names[col]
            return super(MeansVector, self).__getitem__((row, col))
        else:
            return super(MeansVector, self).__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, tuple):
            row, col = item
            assert col == 0 # means vector is always one col
            if isinstance(row, str):
                assert row in self.names
                row = self.names[row]
            return super(MeansVector, self).__setitem__((row, col), value)
        elif isinstance(item, str):
            # Since a MeansVector is always a n x 1
            # matrix we will allow setitem by row only
            # and infer col 0 always
            assert item in self.names
            row = self.names[item]
            return super(MeansVector, self).__setitem__((row, 0), value)
        else:
            return super(MeansVector, self).__setitem__(item, value)

    def __repr__(self):
        tab = PrettyTable(['', 'mu'])
        tab.align = 'r'
        rows = []
        for row in self.name_ordering:
            table_row = [row, '%s' % self[row, 0]]
            tab.add_row(table_row)
        return tab.get_string()

    def split(self, name):
        '''Split into mu_x and mu_y'''
        assert name in self.names
        x_names = [n for n in self.name_ordering if n != name]
        mu_x = MeansVector.zeros((len(self) - 1, 1),
                         names=x_names)
        mu_y = MeansVector.zeros((1, 1), names=[name])

        for row in self.name_ordering:
            v = self[row, 0]
            if row == name:
                mu_y[name, 0] = v
            else:
                mu_x[row, 0] = v
        return mu_x, mu_y
