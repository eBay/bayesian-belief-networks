'''This example is from Koller & Friedman example 7.3 page 252'''
from bayesian.gaussian_bayesian_network import *

'''
This is a simple example with 3 variables:

         +----+    +----+    +----+
         | X1 |--->| X2 |--->| X3 |
         +----+    +----+    +----+

With parameters:

p(X1) ~ N(1; 4)
p(X2|X1) ~ N(0.5X1 - 3.5; 4)
p(X3|X2) ~ N(-X2 + 1; 3)

Remember that in our gaussian decorators
we are using Standard Deviation while the
Koller example uses Variance.

'''


@gaussian(1, 2)
def f_x1(x1):
    pass


@conditional_gaussian(-3.5, 2, 0.5)
def f_x2(x1, x2):
    pass


@conditional_gaussian(1, 3 ** 0.5, -1)
def f_x3(x2, x3):
    pass


if __name__ == '__main__':
    g = build_graph(f_x1, f_x2, f_x3)
    g.q()
