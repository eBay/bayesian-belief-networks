from __future__ import division
'''Simple Example Using Gaussian Variables'''
from bayesian.gaussian_bayesian_network import gaussian, conditional_gaussian
from bayesian.gaussian_bayesian_network import build_graph
from bayesian.utils import shrink_matrix
#from bayesian.gaussian import conditional_to_joint

'''
This example comes from page 3 of
http://people.cs.aau.dk/~uk/papers/castillo-kjaerulff-03.pdf

Note that to create a Guassian Node
we supply mean and standard deviation,
this differs from the example in the
above paper which uses variance (=std. dev.) ** 2

Note in the paper they specify variance,
wheres as this example we are  using std. dev.
instead hence for A the variance is 4 and std_dev is 2.
'''

@gaussian(3, 2)
def f_a(a):
    '''represents point A in the river system'''
    pass


@conditional_gaussian(4, 1, 1)
def f_b(a, b):
    '''Point b is a conditional Guassian
    with parent a.
    '''
    pass


@conditional_gaussian(9, 2, 2)
def f_c(a, c):
    '''Point c is a conditional Guassian
    with parent a'''
    pass


@conditional_gaussian(14, 1, betas=dict(b=1, c=1))
def f_d(b, c, d):
    pass


if __name__ == '__main__':

    g = build_graph(
        f_a,
        f_b,
        f_c,
        f_d)
    import ipdb; ipdb.set_trace()
    mu, sigma = g.query(a=7)
    print mu
    print sigma
    mu, sigma = g.query(a=7, c=17)
    print mu
    print sigma
    mu, sigma = g.query(a=7, c=17, b=8)
    print mu
    print sigma
    # Now to test conditional to joint....
    y = 1 # index of node in network
    mu_y = -5
    sigma_y = 4
    x = [0]
    mu_x = [1]
    sigma_x = [[4]]
    betas = [0.5]
    r = conditional_to_joint(y, mu_y, sigma_y, x, mu_x, sigma_x, betas)
    print r
    # great above worked now to try again with more than 1 x....
    y = 2
    mu_y = 4
    sigma_y = 3
    x = [0, 1]
    mu_x = [1, -4.5]
    sigma_x = [[4, 2], [2, 5]]
    betas = [0, -1]
    #    g.i()
    import ipdb; ipdb.set_trace()
    r = conditional_to_joint(y, mu_y, sigma_y, x, mu_x, sigma_x, betas)
    print r