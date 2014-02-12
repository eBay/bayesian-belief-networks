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
    sigma = g.get_joint_parameters()
    import ipdb; ipdb.set_trace()
    sigma_YY, sigma_YZ, sigma_ZY, sigma_ZZ = g.query(a=7)
    print sigma_YY
    print sigma_YZ
    print sigma_ZY
    print sigma_ZZ
    from bayesian.matfunc import Square

    m = Square([[4, 2, -2],[2, 5, -5],[-2, -5, 8]])

    print m.inverse()


    #    g.i()

import sys; sys.exit(0)

means = np.matrix([[f_a.mean], [f_b.mean], [f_c.mean], [f_d.mean]])

#std_devs = [f.std_dev for f in [f_a, f_b, f_c, f_d]]
std_devs = [f.std_dev for f in [f_a, f_b, f_c, f_d]]
sigma = build_sigma_from_std_devs(std_devs)
sigma = np.matrix([[4, 4, 8, 12],
                   [4, 5, 8, 13],
                   [8, 8, 20, 28],
                   [12, 13, 28, 42]])


N = 4
q = 3

#splits = split(means, sigma)
betas = {
    (1, 0): 1, # BA
    (2, 0): 2, # CA
    (3, 1): 1, # DB
    (3, 2): 1, # DC
}


variances = [s ** 2 for s in std_devs]


variances = [4, 4, 3]
betas = {
    (1, 0): 0.5,
    (2, 1): -1
}
C = {
    1: [],
    2: [1],
    3: [2]
}


betas = {
    (2, 1): 0.5,
    (3, 2): -1
}


sigma = conditional_to_joint_sigma_2([1, 2, 3], C, variances, betas)
print sigma


# Now for the river example first we have to modify the
# args called with...
C = {
    1: [],
    2: [1],
    3: [1],
    4: [2, 3]
}

betas = {
    (2, 1): 1, # BA
    (3, 1): 2, # CA
    (4, 2): 1, # DB
    (4, 3): 1, # DC
}


variances = [s ** 2 for s in std_devs]


sigma = conditional_to_joint_sigma_2([1,2,3,4], C, variances, betas)
print sigma


a = GuassianVariableNode('a')
b = GuassianVariableNode('b')

f_a_node = FactorNode('f_a', f_a)
f_b_node = FactorNode('f_b', f_b)

connect(f_a_node, a)
connect(f_b_node, [a, b])

graph = FactorGraph([
                    a,
                    b,
                    f_a,
                    f_b])



if __name__ == '__main__':
    graph.verify()
    graph.q()
