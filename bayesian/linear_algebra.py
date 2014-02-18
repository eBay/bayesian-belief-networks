'''Very Basic backup Matrix ops for non-Numpy installs'''
from copy import deepcopy

class Matrix(object):

    def __init__(self, rows=[]):
        if not rows:
            self.rows = []
        else:
            self.rows = rows

    def append(self, row):
        '''Like list.append but must be a tuple.'''
        self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.rows[item][:]
        if isinstance(item, tuple):
            row, col = item
            return self.rows[row][col]

    def __setitem__(self, item, val):
        row, col = item
        assert row >= 0
        assert col >= 0
        assert row < len(self.rows)
        assert col < len(self.rows[0])
        self.rows[row][col] = val

    def __add__(self, other):
        assert self.shape == other.shape
        retval = zeros(self.shape)
        for i in range(len(self.rows)):
            for j in range(len(self.rows[0])):
                retval[i, j] = self[i, j] + other[i, j]
        return retval

    def __sub__(self, other):
        assert self.shape == other.shape
        retval = zeros(self.shape)
        for i in range(len(self.rows)):
            for j in range(len(self.rows[0])):
                retval[i, j] = self[i, j] - other[i, j]
        return retval

    def __mul__(self, other):
        assert len(self.rows[0]) == len(other.rows)
        assert len(self.rows[0]) >= len(self.rows)
        rows = len(self.rows)
        cols = len(other.rows[0])
        result = zeros((rows, cols))
        for new_i in range(0, len(self.rows)):
            for new_j in range(0, len(other.rows[0])):
                total = 0
                for k in range(0, len(self.rows[0])):
                    total += self[new_i, k] * other[k, new_j]
                result[new_i, new_j] = total
        return result

    def __div__(self, other):
        return self * other.I

    def __eq__(self, other):
        assert self.shape == other.shape
        for i in range(len(self.rows)):
            for j in range(len(self.rows[0])):
                if self[i, j] != other[i, j]:
                    return False
        return True

    def col(self, j):
        return [row[j] for row in self.rows]

    @property
    def T(self):
        '''Transpose'''
        result = zeros((len(self.rows[0]), len(self.rows)))
        for i in range(0, len(self.rows)):
            for j in range(0, len(self.rows[0])):
                result[j, i] = self[i, j]
        return result

    @property
    def I(self):
        '''Inverse named I tp emulate numpy API'''
        assert len(self.rows) == len(self.rows[0])
        if len(self.rows) == 1:
            return Matrix([[1.0 / self[0, 0]]])
        inverse = make_identity(len(self.rows))
        m = Matrix()
        m.rows = deepcopy(self.rows)
        for col in range(len(self.rows)):
            diag_row = col
            k = 1.0 / m[diag_row, col]
            for j in range(0, len(m.rows)):
                m[diag_row, j] *= k
            for j in range(0, len(inverse.rows)):
                inverse[diag_row, j] *= k
            source_row = diag_row
            for target_row in range(len(self.rows)):
                if source_row != target_row:
                    k = -m[target_row, col]
                    ones = make_identity(len(self.rows))
                    ones[source_row, target_row] = k
                    source_vals = ones.col(target_row)
                    for j in range(0, len(self.rows[0])):
                        target_vals = m.col(j)
                        m[target_row, j] = inner_product(
                            source_vals, target_vals)
                    for j in range(0, len(self.rows[0])):
                        target_vals = inverse.col(j)
                        inverse[target_row, j] = inner_product(
                            source_vals, target_vals)
        return inverse

    def __repr__(self):
        rows = []
        for i in range(0, len(self.rows)):
            row = ['%s' % j for j in self.rows[i]]
            rows.append(str('\t'.join(row)))
        return '\n'.join(rows)

def inner_product(x, y):
    assert len(x) == len(y)
    return sum(map(lambda (x, y): x * y, zip(x, y)))


def zeros(size):
    '''Emulate the Numpy np.zeros factory'''
    rows, cols = size
    m = Matrix()
    for i in range(0, rows):
        m.rows.append([0] * cols)
    return m


def make_identity(j):
    m = zeros((j, j))
    for i in range(0, j):
        m[i, i] = 1
    return m


if __name__ == '__main__':
    my_a = Matrix()
    my_b = Matrix()
    my_a.rows.append([0, 1, 2])
    my_a.rows.append([3, 4, 5])
    my_b.rows.append([0, 1])
    my_b.rows.append([2, 3])
    my_b.rows.append([4, 5])
    m = my_a * my_b
    print my_a * my_b
    import ipdb; ipdb.set_trace()
    mi = m.I
    print m
    print mi
