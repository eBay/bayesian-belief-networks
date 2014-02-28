'''Tests for the small backup linera algebra module'''
import pytest

from bayesian.linear_algebra import *


def pytest_funcarg__matrix_a(request):
    m = Matrix([
        [1, 2, 3],
        [4, 5, 6]
        ])
    return m


def pytest_funcarg__matrix_b(request):
    m = Matrix([
        [1, 2],
        [3, 4],
        [5, 6]
        ])
    return m


def pytest_funcarg__matrix_c(request):
    m = Matrix([
        [4, 4, 8, 12],
        [4, 5, 8, 13],
        [8, 8, 20, 28],
        [12, 13, 28, 42]
        ])
    return m

def pytest_funcarg__matrix_e(request):
    m = Matrix([
        [4, 4, 8, 12],
        [4, 5, 8, 13],
        [8, 8, 20, 28],
        [12, 13, 28, 42]
        ])
    return m

def pytest_funcarg__matrix_f(request):
    '''differs in one cell to matrix_e'''
    m = Matrix([
        [4, 4, 8, 12],
        [4, 5, 88, 13],
        [8, 8, 20, 28],
        [12, 13, 28, 42]
        ])
    return m


def pytest_funcarg__matrix_d(request):
    m = Matrix([
        [0],
        [1],
        [2],
        [3],
        [4]
        ])
    return m


def pytest_funcarg__matrix_g(request):
    m = Matrix([[-2, 2, -3],
                [-1, 1, 3],
                [2, 0, -1]])
    return m


def close_enough(a, b):
    if abs(a - b) < 0.000001:
        return True
    return False


class TestLinearAlgebra():

    def test_zeros(self):
        m = zeros((4, 4))
        assert len(m.rows) == 4
        for i in range(4):
            assert len(m.rows[i]) == 4
        for i in range(4):
            for j in range(4):
                assert m[i, j] == 0

    def test_make_identity(self):
        m = make_identity(4)
        assert len(m.rows) == 4
        for i in range(4):
            assert len(m.rows[i]) == 4
        for i in range(4):
            for j in range(4):
                if i == j:
                    assert m[i, j] == 1
                else:
                    assert m[i, j] == 0

    def test_multiply(self, matrix_a, matrix_b):
        m = matrix_a * matrix_b
        assert len(m.rows) == 2
        for i in range(2):
            assert len(m.rows[i]) == 2
        assert m[0, 0] == 22
        assert m[0, 1] == 28
        assert m[1, 0] == 49
        assert m[1, 1] == 64

        sigma_YZ = Matrix([
            [8],
            [4],
            [12]])
        sigma_ZZ = Matrix([
            [4]])
        t = sigma_YZ * sigma_ZZ.I
        assert t == Matrix([
            [2],
            [1],
            [3]])

    def test_invert(self, matrix_c):
        c_inv = matrix_c.I
        assert c_inv[0, 0] == 2.25
        assert c_inv[0, 1] == -1.0
        assert c_inv[0, 2] == -0.5
        assert close_enough(c_inv[0, 3], -2.96059473e-16)

        assert c_inv[1, 0] == -1.0
        assert c_inv[1, 1] == 2
        assert c_inv[1, 2] == 1
        assert c_inv[1, 3] == -1

        assert c_inv[2, 0] == -0.5
        assert c_inv[2, 1] == 1
        assert c_inv[2, 2] == 1.25
        assert c_inv[2, 3] == -1

        assert close_enough(c_inv[3, 0], -2.66453526e-15)
        assert c_inv[3, 1] == -1
        assert c_inv[3, 2] == -1
        assert c_inv[3, 3] == 1

    def test_slicing(self, matrix_d):
        # Note slicing is NOT YET IMPLEMENTED
        pass

    def test_equality(self, matrix_c, matrix_d,
                      matrix_e, matrix_f):
        assert matrix_c == matrix_e
        assert matrix_c != matrix_d
        assert matrix_e != matrix_f

    def test_matrix_determinant(self, matrix_g):
        d = matrix_g.det()
        assert d == 18
