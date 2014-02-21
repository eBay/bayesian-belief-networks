from __future__ import division
'''Simple Example Using Gaussian Variables'''
from bayesian.gaussian_bayesian_network import gaussian, conditional_gaussian
from bayesian.gaussian_bayesian_network import build_graph
from bayesian.utils import shrink_matrix

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


@conditional_gaussian(1, 1, 1)
def f_b(a, b):
    '''Point b is a conditional Guassian
    with parent a.
    '''
    pass


@conditional_gaussian(3, 2, 2)
def f_c(a, c):
    '''Point c is a conditional Guassian
    with parent a'''
    pass


@conditional_gaussian(1, 1, betas=dict(b=1, c=1))
def f_d(b, c, d):
    pass


if __name__ == '__main__':

    g = build_graph(
        f_a,
        f_b,
        f_c,
        f_d)
    g.q()
