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


def guassian_cdf(x, mean, std_dev):
    t = (x - mean) / std_dev
    if t > 0:
        return std_guassian_cdf(t)
    elif t == 0:
        return 0.5
    else:
        return 1 - std_guassian_cdf(abs(t))


def make_guassian(mean, std_dev):

    def guassian(x):
        return 1 / (std_dev * (2 * math.pi) ** 0.5) * \
            math.exp((-(x - mean) ** 2) / 2 * std_dev ** 2)

    return guassian


if __name__ == '__main__':
    g = make_guassian(0, 1)
