from __future__ import division
import math
from collections import defaultdict
from itertools import combinations, product



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


def joint_to_conditional(x, y, mu, sigma):
    '''From a joint P(x1,x2...,y)
    return distribution of p(y|x1,x2...)
    In this case y is always a single
    variable. Both x and y are 0-based indices
    into mu and sigma.
    mu and sigma should both be instances
    of some Matrix class which can
    invert matrices and do matrix
    multiplication.
    size(sigma) = (len(mu), len(mu))
    len(mu) = len(x) + 1
    y is a scalar
    '''
    # Lets first work out beta_0 the
    # intercept.
    beta_0 = mu[y][0]
    for x_ in x:
        beta_0 -= sigma[y][x_] / sigma[x_][x_] * mu[x_][0]
    # Now for the rest of the betas....
    return beta_0


def conditional_to_joint(y, mu_y, sigma_y, x, mu_x, sigma_x, betas):
    ''' here we have p(y|x1,x2....)
    and we want to return the
    parameters for P(y, x1, x2...)
    y is a child node.
    x is a list of dependant nodes.
    mu_y is the mean of the conditioanl
    distribution. sigma_y is the
    std. dev. of y.
    mu_x: vector of prior means for x
    sigma_x: covariance matrix of joint x
    betas: weights of edges from x_ to y.
    '''
    #assert len(x) == len(mu_x)
    #assert len(mu_x) == len(betas)
    #assert len(sigma_x[0]) == len(x)
    mu = mu_y
    for m, b in zip(mu_x, betas):
        mu += m * b
    sigma = dict()
    all_vars = x[:] + [y]
    all_sigmas = sigma_x[:]
    for j in range(0, len(all_vars)):
        for i in range(0, j):
            total = 0
            if j == len(x):
                # We are on var y
                for pi, parent in enumerate(x):
                    total += sigma_x[i][pi] * \
                             betas[pi]
                sigma[(j, i)] = total
                sigma[(i, j)] = total
        total = 0
        if j == len(x):
            for pi, parent in enumerate(x):
                total += sigma[(j, pi)] * \
                         betas[pi]
        if j == len(x):
            sigma[(j, j)] = sigma_y + total
        else:
            sigma[(j, j)] = sigma_x[j][j]
    return mu, sigma

def conditional_to_joint_2(y, mu_y, sigma_y, x, mu_x, sigma_x, betas):
    '''
    This is from page 19 of
    http://webdocs.cs.ualberta.ca/~greiner/C-651/SLIDES/MB08_GaussianNetworks.pdf
    The notation finally makes sense now!
    '''
    sz = len(mu_x) + 1
    sigma = Square([[None] * sz] * sz)
    mu = mu_y
    for m, b in zip(mu_x, betas):
        mu += m * b
    # Now the top left block
    # of the covariance matrix is
    # just a copy of the sigma_x matrix
    for i in range(0, len(sigma_x)):
        for j in range(0, len(sigma_x)):
            sigma[i][j] = sigma_x[i][j]
    for i in range(0, len(sigma_x)):
        total = 0
        for j in range(0, len(sigma_x)):
            total += betas[j] * sigma_x[i][j]
        sigma[i][len(x)] = total
        sigma[len(x)][i] = total
    # And finally for the bottom right corner
    # Need to finish this....


def conditional_to_joint_using_my_linear_algebra(
        mu_1, mu_2, sigma_11, sigma_12, sigma_21, sigma_22):
    '''See Wikipedia article...'''
    pass

def joint_to_conditional_using_my_linear_algebra(
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
