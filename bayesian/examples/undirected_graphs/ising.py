"""Simple 16 node Ising Model """

from bayesian.factor_graph import *

'''

     0    1    2   3

0    .    .    .   .

1    .    .    .   .

2    .    .    .   .

3    .    .    .   .

'''

# These constants come from Bishop page 390.
XY = 2.1
XX = 1.0
h = 0

# These are the 'x' variables
x00 = VariableNode("x00")
x01 = VariableNode("x01")
x02 = VariableNode("x02")
x03 = VariableNode("x03")
x10 = VariableNode("x10")
x11 = VariableNode("x11")
x12 = VariableNode("x12")
x13 = VariableNode("x13")
x20 = VariableNode("x20")
x21 = VariableNode("x21")
x22 = VariableNode("x22")
x23 = VariableNode("x23")
x30 = VariableNode("x30")
x31 = VariableNode("x31")
x32 = VariableNode("x32")
x33 = VariableNode("x33")



x_vars = [[x00, x01, x02, x03],
          [x10, x11, x12, x13],
          [x20, x21, x22, x23],
          [x30, x31, x32, x33]]


# 'y' variables
y00 = VariableNode("y00")
y01 = VariableNode("y01")
y02 = VariableNode("y02")
y03 = VariableNode("y03")
y10 = VariableNode("y10")
y11 = VariableNode("y11")
y12 = VariableNode("y12")
y13 = VariableNode("y13")
y20 = VariableNode("y20")
y21 = VariableNode("y21")
y22 = VariableNode("y22")
y23 = VariableNode("y23")
y30 = VariableNode("y30")
y31 = VariableNode("y31")
y32 = VariableNode("y32")
y33 = VariableNode("y33")


y_vars = [[y00, y01, y02, y03],
          [y10, y11, y12, y13],
          [y20, y21, y22, y23],
          [y30, y31, y32, y33]]


# Now for the factors...
def make_xy_factor(x_var, y_var):

    def xy_f(x, y):
        return -XY * x * y

    return xy_f


xy_factors = []
for x_row, y_row in zip(x_vars, y_vars):
    for x, y in zip(x_row, y_row):
        xy_factors.append(make_xy_factor(x, y))
        x.neighbours.append(y)
        y.neighbours.append(x)


def make_xx_factor(x1, x2):

    def xx_f(x1, x2):
        return -XX * x1 * x2

    return xx_f


xx_factors = []
for i, x_i in enumerate(x_vars):
    for j, x_ij in enumerate(x_i):
        for k, x_k in enumerate(x_vars):
            for l, x_kl in enumerate(x_k):
                if abs(i - k) + abs(j - l) == 1:
                    if k < i:
                        continue
                    if l < j:
                        continue
                    print x_ij, x_kl
                    xx_factors.append(make_xx_factor(x_ij, x_kl))
                    x_ij.neighbours.append(x_kl)
                    x_kl.neighbours.append(x_ij)


def local_energy(x, val):
    # The Energy function is given in Bishop 8.42
    x_sum = 0
    y_sum = 0
    for neighbour in x.neighbours:
        if neighbour.name.startswith('x'):
            x_sum += val * neighbour.value
        elif neighbour.name.startswith('y'):
            y_sum += val * neighbour.value
    return - XX*x_sum - XY*y_sum


def icm(x_vars):
    iteration = 0
    i_order = [0, 1, 2, 3]
    j_order = [0, 1, 2, 3]

    while iteration < 100:
        # In each iteration we will
        # update the x values in random order
        random.shuffle(i_order)
        random.shuffle(j_order)
        flip_count = 0
        for i in i_order:
            for j in j_order:
                # Now we need to calculate the
                # total Energy for both the
                # possible states...
                old_val = x_vars[i][j].value
                print local_energy(x_vars[i][j], -1), local_energy(x_vars[i][j], 1)
                if local_energy(x_vars[i][j], -1) < local_energy(x_vars[i][j], 1):
                    x_vars[i][j].value = -1
                else:
                    x_vars[i][j].value = 1
                if old_val != x_vars[i][j].value:
                    flip_count += 1
        # Now print out the current state...
        print 'Iteration: %s flips: %s' % (iteration, flip_count)
        for x_row in x_vars:
            print [x.value for x in x_row]
        # If nothing changed, we have reached stability....
        if not flip_count:
            return
        iteration += 1



if __name__ == '__main__':
    for xy_factor in xy_factors:
        print xy_factor
    print len(xx_factors)

    # Lets assign all y vars, the observation to +1
    for y_row in y_vars:
        for y in y_row:
            y.value = 1

    # Now we will randomly flip the
    # value of the y vars with a
    # chance of 10% to create noise
    for y_row in y_vars:
        for y in y_row:
            if random.random() < 0.1:
                y.value *= -1

    # Now lets print the y vals...
    for y_row in y_vars:
        print [y.value for y in y_row]

    # Now we initialize the x vals to
    # the observed y_vals...
    for x_row, y_row in zip(x_vars, y_vars):
        for x, y in zip(x_row, y_row):
            x.value = y.value

    # And print the x vals to confirm...
    for x_row in x_vars:
        print [x.value for x in x_row]

    # And now call the icm procedure...
    icm(x_vars)
    print 'Original Noisy Image: '
    for y_row in y_vars:
        print [y.value for y in y_row]

    print
    print 'Restore Image: '
    for x_row in x_vars:
        print [x.value for x in x_row]
