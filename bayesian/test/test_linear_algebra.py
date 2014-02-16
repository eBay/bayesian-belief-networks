'''Tests for the small backup linera algebra module'''
import pytest

from bayesian.linear_algebra import *

def pytest_funcarg__matrix_a(request):
    m = Matrix()
    m.rows.append([1, 2, 3])
    m.rows.append([4, 5, 6])
    return m


def pytest_funcarg__matrix_b(request):
    m = Matrix()
    m.rows.append([1, 2])
    m.rows.append([3, 4])
    m.rows.append([5, 6])
    return m


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
