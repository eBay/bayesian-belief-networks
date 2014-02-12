import math
from itertools import product
import numpy as np

from bayesian.factor_graph import Node
from bayesian.gaussian import make_gaussian_cdf


def gaussian(mu, sigma):
    # This is the gaussian decorator
    # which is a decorator with parameters
    # This means it should return a
    # 'normal' decorated ie twice wrapped...

    def gaussianize(f):

        def gaussianized(x):
            return 1 / (sigma * (2 * math.pi) ** 0.5) * \
                math.exp((-(x - mu) ** 2) / (2 * sigma ** 2))

        gaussianized.mean = mu
        gaussianized.std_dev = sigma
        gaussianized.cdf = make_gaussian_cdf(mu, sigma)

        return gaussianized
    return gaussianize


def conditional_mean(mu_1, mu_2, a, sigma_12, sigma_22):
    '''These arg names are from the Wikipedia article'''
    mean = mu_1 + sigma_12 * sigma_22 ** -1 * (a - mu_2)
    return mean


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


def conditional_gaussian(mu, sigma, betas):

    def conditional_gaussianize(f):

        def conditional_gaussianized(**args):
            # We assume the last arg is the value
            # for the child node
            return 'something'

        # Actually the mean now becomes a
        # function of the dependent variable

        conditional_gaussianized.mean = mu
        conditional_gaussianized.std_dev = sigma
        conditional_gaussianized.betas = betas

        return conditional_gaussianized

    return conditional_gaussianize


def multiguassian(betas, parents, rho):
    # This is the decorator for creating
    # multivariate guassian nodes
    # from parents that are also gaussians.
    # See http://webdocs.cs.ualberta.ca/~greiner/C-651/SLIDES/MB08_GaussianNetworks.pdf
    # slide 18. The betas are just
    # some linear weights, one for
    # each parent and a bias weight.
    # betas should be list like.

    # ensure that betas and parents are
    # consistent
    assert len(betas) == (1 + len(parents))
    def multigaussianize(f):
        # The mean for this variable
        # is just the betas'*means of parents
        # ie b0 + b1mu1 + b2+mu2
        # Mmmm this seems to mean that actually
        # we cant define the mu and sigma
        # *until* we know which nodes its
        # connected to. So for now
        # we will simply record the betas....
        # ok if we change the decorator to
        # force the user to supply the
        # parent variables we can
        # compute the mu and sigma matrix
        # here
        mu = betas[0]
        parent_means = [f.mean for f in parents]
        for b, m in zip(betas[1:], parent_means):
            mu += b * m

        # Now for the covariance matrix...
        # seems like we also need the correlation
        # rho between the parents, not sure how
        # to interpret this but lets require the
        # user to define it...

        sigma = [['dummy'] * len(parents)] * len(parents)
        # This is where we will call conditional_to_joint_sigma_2...
        # For now lets just assume there are only 2 parents
        # even in the case of more than 2 we can combine
        # them pairwise.
        # look at http://en.wikipedia.org/wiki/Multivariate_normal_distribution
        # for the bivariate case...
        sigma[0][0] = parents[0].std_dev ** 2
        sigma[0][1] = rho * parents[0].std_dev * parents[1].std_dev
        sigma[1][0] = rho * parents[0].std_dev * parents[1].std_dev
        sigma[1][1] = parents[1].std_dev ** 2

        def multiguassianized(x):
            # Once we build the graph we
            # need to come back and
            # fill in the meat here...

            pass
        to_be_gaussianaed.betas = betas
        return to_be_gaussianaed
    return multiguassianize







class GaussianNode(object):

    def __init__(self, name, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.neighbours = []
        self.received_messages = {}
        self.value = None
