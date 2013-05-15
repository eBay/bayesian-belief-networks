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
