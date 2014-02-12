import math
from itertools import product
from functools import wraps
import numpy as np

from bayesian.factor_graph import Node
from bayesian.gaussian import make_gaussian_cdf
from bayesian.utils import get_args



def conditional_covariance_matrix(sigma_11, sigma_12, sigma_22, sigma_21):
    return sigma_11 - sigma_12 * (sigma_22 ** -1) * sigma_21


def split(means, sigma):
    ''' Split the means and covariance matrix
    into 'parts' as in wikipedia article ie

    mu = | mu_1 |
         | mu_2 |

    sigma = | sigma_11 sigma_12 |
            | sigma_21 sigma_22 |

    We will assume that we always combine
    one variable at a time and thus we
    will always split by mu_2 ie mu_2 will
    always have dim(1,1) so that it can
    be subtracted from the scalar a
    Also we will make sim(sigma_22)
    always (1,1)


    '''
    mu_1 = means[0:-1]
    mu_2 = means[-1:]
    sigma_11 = sigma[0:len(means) -1, 0:len(means) -1]
    sigma_12 = sigma[:-1,-1:]
    sigma_21 = sigma_12.T
    sigma_22 = sigma[len(means) -1:, len(means) - 1:]
    return mu_1, mu_2, sigma_11, sigma_12, sigma_21, sigma_22


def conditional_mean(mu_1, mu_2, a, sigma_12, sigma_22):
    '''These arg names are from the Wikipedia article'''
    mean = mu_1 + sigma_12 * sigma_22 ** -1 * (a - mu_2)
    return mean


def build_sigma_from_std_devs(std_devs):
    retval = []
    for sd_i, sd_j in product(std_devs, std_devs):
        retval.append(sd_i * sd_j)
    return np.matrix(retval).reshape(len(std_devs), len(std_devs))


def get_parent_from_betas(betas, child):
    '''Return all betas ending at child'''
    return [k for k, v in betas.items() if k[0] == child]


def conditional_to_joint_sigma_2(s, C, variances, betas):
    '''
    This is derived from the psuedo code
    on page 538, Schachter and Kenley.
    http://www.stanford.edu/dept/MSandE/cgi-bin/people/faculty/shachter/pdfs/gaussid.pdf
    s is an ordering of the nodes in which no
    dependent variable occurs before its parents.
    To make this work we have to make sure we
    are using the same notation that they use.
    See beginning of chapter 2.
    For the example we have X1, X2 and X3
    Thus n = [1, 2, 3]
    s = [1, 2, 3] (This is the ordered sequence)
    for now C is a dict with the vals being
    the list of parents so
    C[1] = []
    C[2] = [1]
    C[3] = [2]
    betas is a dict of the betas keyed
    by a tuple representing the node
    indices.
    Ok I can verify that this one works!
    for the example in Koller and in the presentation page 19
    it gets correct results...
    Now to check for the river example.

    Woohoo, this works for the river example!!!!
    Now I will write an _3 version that
    uses more sensible arguments...

    '''

    sigma = np.zeros((len(s), len(s)))
    for j in s:
        for i in range(1, j-1+1):
            total = 0
            for k in C[j]:
                total += sigma[i - 1, k - 1] * betas[(j, k)]
            sigma[j-1, i-1] = total
            sigma[i-1, j-1] = total
        total = 0
        for k in C[j]:
            total += sigma[j-1, k-1] * betas[(j, k)]
        sigma[j-1, j-1] = variances[j-1] + total
    return sigma
