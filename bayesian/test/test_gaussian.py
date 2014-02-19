import pytest

from bayesian.gaussian import *
from bayesian.linear_algebra import Matrix, zeros

class TestGaussian():

    def test_joint_to_conditional(self):
        '''This is from the example
        on page 22
        '''
        mu_x = Matrix([[1]])
        mu_y = Matrix([[-4.5]])
        sigma_xx = Matrix([[4]])
        sigma_xy = Matrix([[2]])
        sigma_yx = Matrix([[2]])
        sigma_yy = Matrix([[5]])
        beta_0, beta, sigma = joint_to_conditional(
            mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy)
        assert beta_0 == -5 # checked
        assert beta == Matrix([[0.5]]) # checked
        assert sigma == Matrix([[4]])  # checked

        # Now do the same for P(X2|X3)
        mu_x = Matrix([[-4.5]])
        mu_y = Matrix([[8.5]])
        sigma_xx = Matrix([[5]])
        sigma_xy = Matrix([[-5]])
        sigma_yx = Matrix([[-5]])
        sigma_yy = Matrix([[8]])
        beta_0, beta, sigma = joint_to_conditional(
            mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy)
        assert beta_0 == 4 # checked
        assert beta == Matrix([[-1]]) #checked
        assert sigma == Matrix([[3]]) # checked

        # Now for the river example...
        # These values can be confirmed from page 4
        # First for p(B|A)
        mu_x = Matrix([[3]])
        mu_y = Matrix([[4]])
        sigma_xx = Matrix([[4]])
        sigma_xy = Matrix([[4]])
        sigma_yx = Matrix([[4]])
        sigma_yy = Matrix([[5]])
        beta_0, beta, sigma = joint_to_conditional(
            mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy)
        assert beta_0 == 1 # Checked see below comments
        # On page 3, the conditional for the factor f(b|a):
        # N(mu_B + beta_BA(a - mu_A), v_B)
        #
        #        mu_B + beta_BA(a - mu_A)
        #    ==  mu_B + (beta_BA * a) - beta_BA * mu_A
        #    ==     4 + 1 * a - 1 * 3
        #    ==     4 + a - 3
        #    ==     1 + a
        #    ==> beta_0 should be 1
        assert beta == Matrix([[1]]) # checked
        assert sigma == Matrix([[1]]) # checked

        # p(C|A)
        mu_x = Matrix([[3]])
        mu_y = Matrix([[9]])
        sigma_xx = Matrix([[4]])
        sigma_xy = Matrix([[8]])
        sigma_yx = Matrix([[8]])
        sigma_yy = Matrix([[20]])
        beta_0, beta, sigma = joint_to_conditional(
            mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy)
        # from page 3: f(c|a) ~ N(mu_C + beta_C(a - mu_A), v_C)
        #        mu_C + beta_CA(a - mu_A)
        #    ==  mu_C + beta_CA * a - beta_CA * mu_A
        #    ==     9 + 2 * a - 2 * 3
        #    ==     9 + 2a - 6
        #    ==     3 + 2a
        #    ==> beta_0 = 3 and beta_1 = 2
        assert beta_0 == 3 # checked
        assert beta == Matrix([[2]]) # checked
        assert sigma == Matrix([[4]]) # checked

        # Now the more complicated example
        # where we have multiple parent nodes
        # p(D|B, C)
        mu_x = Matrix([
            [4],
            [9]])
        mu_y = Matrix([[14]])
        sigma_xx = Matrix([
            [5, 8],
            [8, 20]])
        sigma_xy = Matrix([
            [13],
            [28]])
        sigma_yx = Matrix([
            [13, 28]])
        sigma_yy = Matrix([[42]])
        beta_0, beta, sigma = joint_to_conditional(
            mu_x, mu_y, sigma_xx, sigma_xy, sigma_yx, sigma_yy)
        # From page 3 :
        # f(d|b,c) ~ N(mu_D + beta_DB(b - mu_B) + beta_DC(c - mu_C), v_D)
        #              mu_D + beta_DB(b - mu_B) + beta_DC(c - mu_C
        #          ==  mu_D + beta_DB * b - beta_DB * mu_B + beta_DC * c - beta_DC * mu_C
        #          ==  14   + 1 * b - 1 * 4 + 1 * c - 1 * 9
        #          ==  14 + 1b - 4 + 1c -9
        #          ==  1 + 1b + 1c
        #          ==> beta_0 = 1, beta = (1  1)'
        assert beta_0 == 1 # checked
        assert beta == Matrix([[1, 1]]) # checked
        assert sigma == Matrix([[1]]) # checked

    def test_conditional_to_joint(self):
        # For the example in http://webdocs.cs.ualberta.ca/~greiner/C-651/SLIDES/MB08_GaussianNetworks.pdf
        # we will build up the joint parameters one by one to test...
        mu_x = Matrix([[1]])
        sigma_x = Matrix([[4]])
        beta_0 = -5
        beta = Matrix([[0.5]])
        sigma_c = 4
        mu, sigma = conditional_to_joint(
            mu_x, sigma_x, beta_0, beta, sigma_c)
        assert mu == Matrix([
            [1],
            [-4.5]])
        assert sigma == Matrix([
            [4, 2],
            [2, 5]])

        # Now we want to build up the second step of the process...
        mu_x = mu
        sigma_x = sigma
        beta_0 = 4
        beta = Matrix([
            [0], # Represents no edge from x1 to x3
            [-1]])
        sigma_c = 3
        mu, sigma = conditional_to_joint(
            mu_x, sigma_x, beta_0, beta, sigma_c)
        assert mu == Matrix([
            [1],
            [-4.5],
            [8.5]])
        assert sigma == Matrix([
            [4, 2, -2],
            [2, 5, -5],
            [-2, -5, 8]])

        # Now lets do the river example...
        mu_x = Matrix([
            [3]])
        sigma_x = Matrix([
            [4]])
        beta_0 = 1 # See above test for joint_to_conditional
        beta = Matrix([
            [1]])  # beta_BA
        sigma_c = 1
        mu, sigma = conditional_to_joint(
            mu_x, sigma_x, beta_0, beta, sigma_c)
        assert mu == Matrix([
            [3],
            [4]])
        assert sigma == Matrix([
            [4, 4],
            [4, 5]])

        # Test adding the d variable from the river example
        mu_x = Matrix([
            [3],
            [4],
            [9]])
        sigma_x = Matrix([
            [4, 4, 8],
            [4, 5, 8],
            [8, 8, 20]])
        beta_0 = 1 # See above test for joint_to_conditional
        beta = Matrix([
            [0], # No edge from a->c
            [1], # beta_DB Taken directly from page 4
            [1]]) # beta_DC Taken from page 4
        sigma_c = 1
        mu, sigma = conditional_to_joint(
            mu_x, sigma_x, beta_0, beta, sigma_c)
        assert mu == Matrix([
            [3],
            [4],
            [9],
            [14]])
        assert sigma == Matrix([
            [4, 4, 8, 12],
            [4, 5, 8, 13],
            [8, 8, 20, 28],
            [12, 13, 28, 42]])

        # Now we will test a graph which
        # has more than 1 parentless node.
        mu_x = Matrix([
            [3]])
        sigma_x = Matrix([
            [4]])
        beta_0 = 5
        beta = Matrix([[0]])
        sigma_c = 1
        mu, sigma = conditional_to_joint(
            mu_x, sigma_x, beta_0, beta, sigma_c)
        assert mu == Matrix([[3], [5]])
        assert sigma == Matrix([[4, 0], [0, 1]])
