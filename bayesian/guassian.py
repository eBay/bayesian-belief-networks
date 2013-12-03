from __future__ import division
import math

'''
Provides Guassian Density functions and
approximation to Guassian CDF.
see https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution
"Numerical approximations for the normal CDF"
'''

b0 = 0.2316419
b1 = 0.319381530
b2 = -0.356563782
b3 = 1.781477937
b4 = -1.821255978
b5 = 1.330274429


def std_guassian_cdf(x):
    '''Zelen & Severo approximation'''
    g = make_guassian(0, 1)
    t = 1 / (1 + b0 * x)
    return 1 - g(x) * (
        b1 * t +
        b2 * t ** 2 +
        b3 * t ** 3 +
        b4 * t ** 4 +
        b5 * t ** 5)


def make_guassian(mean, std_dev):

    def guassian(x):
        return 1 / (std_dev * (2 * math.pi) ** 0.5) * \
            math.exp((-(x - mean) ** 2) / (2 * std_dev ** 2))

    guassian.mean = mean
    guassian.std_dev = std_dev
    guassian.cdf = make_guassian_cdf(mean, std_dev)

    return guassian


def make_guassian_cdf(mean, std_dev):

    def guassian_cdf(x):
        t = (x - mean) / std_dev
        if t > 0:
            return std_guassian_cdf(t)
        elif t == 0:
            return 0.5
        else:
            return 1 - std_guassian_cdf(abs(t))

    return guassian_cdf


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
        guassian_cdf = make_guassian_cdf(0, 1)
        return guassian_cdf((math.log(x, base) - mean) / std_dev)

    return log_normal_cdf


def discretize_guassian(mu, stddev, buckets,
                        func_name='f_output_var', var_name='output_var'):
    '''Given guassian distribution parameters
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
    '''
    result = []

    cdf = make_guassian_cdf(mu, stddev)
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
    return '\n'.join(result)
