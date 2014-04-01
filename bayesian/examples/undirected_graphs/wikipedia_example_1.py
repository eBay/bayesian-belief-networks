'''This example is actually of a Hidden Markov Model'''
from __future__ import division
import random
from math import exp
from itertools import product as xproduct
from pprint import pprint

from bayesian.undirected_graphical_model import UndirectedNode
from bayesian.undirected_graphical_model import connect as connect_ug
from bayesian.undirected_graphical_model import UndirectedModel, UndirectedGraph
from bayesian.examples.undirected_graphs.monty import build_join_tree_from_ug
from bayesian.factor_graph import *
from bayesian.factor_graph import connect as fg_connect
from bayesian.graph import JoinTreeSepSetNode
from bayesian.utils import make_key
from bayesian.viterbi_crf import make_g_func, make_G_func
from bayesian.viterbi_crf import viterbi as my_viterbi
from bayesian.viterbi_crf import forward


build_fg = build_graph

START = '__START__'
STOP = '__STOP__'

'''
Although this is actually an example of an HMM
we will try and define it here as a CRF

states = ('Healthy', 'Fever')
end_state = 'E'

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
   }

emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }


Graphically:

   (Healthy)         (Fever)



   (normal)       (cold)    (dizzy)



'''

states = ('Healthy', 'Fever')
end_state = 'E'

observations = ('normal', 'cold', 'dizzy')

start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
   }

emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }


def crf_out(result):
    '''Convert a CRF query result into
    an output sequence...'''
    outputs = ['y0', 'y1', 'y2']
    best_score = defaultdict(float)
    best_label = dict()
    for k, v in result.items():
        if v >= best_score[k[0]]:
            best_score[k[0]] = v
            best_label[k[0]] = k[1]
    return [best_label[o] for o in outputs]



def viterbi(obs, states, start_p, trans_p, emit_p):
    '''From Wikipedia'''
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]

    # alternative Python 2.7+ initialization syntax
    # V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    # path = {y:[y] for y in states}

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]

        # Don't need to remember the old paths
        path = newpath

    #print_dptable(V)
    (prob, state) = max((V[t][y], y) for y in states)
    return (prob, path[state])


def fwd_bkw(x, states, a_0, a, e, end_st):
    '''From Wikipedia'''
    L = len(x)

    fwd = []
    f_prev = {}
    v = []
    v_prev = {}
    # forward part of the algorithm
    for i, x_i in enumerate(x):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = a_0[st]
            else:
                prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)

            f_curr[st] = e[st][x_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k]*a[k][end_st] for k in states)

    bkw = []
    b_prev = {}
    # backward part of the algorithm
    for i, x_i_plus in enumerate(reversed(x[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = a[st][end_st]
            else:
                b_curr[st] = sum(a[st][l]*e[l][x_i_plus]*b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(a_0[l] * e[l][x[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(L):
        posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in states})

    #assert p_fwd == p_bkw
    return fwd, bkw, posterior


def example():
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)

def forward_backward(x_seq):
    return fwd_bkw(x_seq,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)

'''
------------------------------------------
'''
# feature functions...
def f_0(yp, y, x_seq, j):
    if yp == START and y == 'Healthy':
        return 1
    return 0


def f_1(yp, y, x_seq, j):
    if yp == START and y == 'Fever':
        return 1
    return 0


def f_2(yp, y, x_seq, j):
    if yp == 'Healthy' and y == 'Fever':
        return 1
    return 0


def f_3(yp, y, x_seq, j):
    if yp == 'Fever' and y == 'Healthy':
        return 1
    return 0


def f_4(yp, y, x, j):
    if y == 'Fever' and x[j] == 'cold':
        return 1
    return 0


def f_5(yp, y, x, j):
    if y == 'Fever' and x[j] == 'dizzy':
        return 1
    return 0


def f_6(yp, y, x, j):
    if y == 'Fever' and x[j] == 'normal':
        return 1
    return 0


def f_6(yp, y, x, j):
    if y == 'Healthy' and x[j] == 'cold':
        return 1
    return 0


def f_7(yp, y, x, j):
    if y == 'Healthy' and x[j] == 'dizzy':
        return 1
    return 0


def f_8(yp, y, x, j):
    if y == 'Healthy' and x[j] == 'normal':
        return 1
    return 0


def f_9(yp, y, x, j):
    if yp == 'Healthy'and y == 'Healthy' and x[j] == 'normal':
        return 1
    return 0


def f_10(yp, y, x, j):
    if yp == 'Fever'and y == 'Fever' and x[j] == 'cold':
        return 1
    return 0

def f_11(yp, y, x, j):
    if yp == 'Fever'and y == 'Fever' and x[j] == 'cold' and x[j-1] == 'dizzy':
        return 1
    return 0


def f_12(yp, y, x, j):
    if x > 2 and yp == 'Fever'and y == 'Fever' and \
       x[j] == 'dizzy' and x[j-1] == 'cold' and \
       x[j-2] == 'dizzy':
        return 1
    return 0


def f_13(yp, y, x, j):
    if x < 3 and yp == 'Fever'and y == 'Fever' and \
       x[j] == 'cold' and x[j+1] == 'dizzy':
        return 1
    return 0


feature_functions = [
    f_0, f_1, f_2,
    f_3, f_4, f_5,
    f_6, f_7, f_8, f_9,
    f_10, f_11, f_12, f_13]


def lame_get_j(variable_names):
    indices = [int(name[1]) for name in variable_names if len(name)> 1]
    return min(indices)


def make_potential(x, j, L, feature_funcs):

    def potential(y_prev, y):
        total = 0
        for l, f in zip(L, feature_funcs):
            # Note closure over x and j
            res = f(y_prev, y, x, j)
            total += l * res
        return exp(total)

    return potential


# Now we will start off by creating the undirected
# graph (excluding the input node)
# Actually we should be modelling this as a LC-CRF!
# ie it has as many output nodes
# as input nodes....
#healthy_node = UndirectedNode('Healthy')
#fever_node = UndirectedNode('Fever')

#healthy_node.variable_name = 'Healthy'
#fever_node.variable_name = 'Fever'
#connect_ug(healthy_node, fever_node)

def make_big_F(little_f):
    '''Just a notational convention used
    by some authours. Note that the
    weight is NOT playing a role in
    this function, it is ONLY used
    for calculating the expected
    value of the empirical distribution.
    '''
    def big_F(x_seq, y_seq):
        total_i = 0
        # Dont forget the artificial begin and end!
        total_i += little_f(START, y_seq[0], x_seq, 0)
        for i in range(1, len(x_seq)):
            # Note I switched around the order of args for
            # the indicator funcs :(

            total_i += little_f(y_seq[i - 1], y_seq[i], x_seq, i)
        return total_i
    return big_F


def expected_model(x_seq, w, j, alphabet, F_j):
    total_y_dash = []
    for y_i, y_dash in enumerate(generate_all_y_seq(len(x_seq), alphabet)):
        total_y_dash.append(w[j] * F_j(x_seq, y_dash))
    return sum(total_y_dash) / len(total_y_dash)


def generate_all_y_seq(l, alphabet):
    '''Yield y_seq for all possible sequences
    l is the length of the input or training
    sequence'''
    for y_seq in xproduct(*([alphabet] * l)):
        yield y_seq


def build_potential_functions(X, fg, L, feature_funcs):
    '''Given an fg structure, together
    with the Lagrange Multipliers and the
    feature functions build the potential functions.
    X is the input sequence, since it is
    fixed for the LCCRF we are not including it
    for now as a variable.
    '''
    assert len(L) == len(feature_funcs)
    for factor_node in fg.factor_nodes():
        #print factor_node.variable_names
        # How to get the value of j.... mmmm
        # Maybe we should rather iterate over
        # variable nodes....
        j = lame_get_j(factor_node.variable_names)
        print j
        factor_node.func = make_potential(X, j, L, feature_funcs)
        factor_node.func.argspec = factor_node.variable_names[:]
        d = dict()
        for variable_name in factor_node.variable_names:
            d[variable_name] = fg_variable_nodes[variable_name].domain
        factor_node.domains = copy.copy(d)
        factor_node.func.domains = copy.copy(d)
    #import ipdb; ipdb.set_trace()


def total_expected_model(fg, L, indicator_func, training_examples, feature_funcs):
    '''Calculate the total expected value
    of the data according to the model.
    This is what Manning and Klein's
    presentation refers to as 'predicted count'.
    (http://www.cs.berkeley.edu/~klein/papers/maxent-tutorial-slides.pdf)
    It is just the models output probability
    using the current weights.
    '''
    total = 0
    for sequence, labels in training_examples:  # k
        build_potential_functions(sequence, fg, L, feature_funcs)
        F_j = make_big_F(indicator_func)
        print indicator_func
        print L
        print sequence
        result = fg.query()
        print result
        result = simple_normalize(result)
        out_seq = crf_out(result)
        p = 1
        for label, var in zip(
                out_seq, ['y0', 'y1', 'y2']):
            p *= result[(var, label)]
        total += p * F_j(sequence, out_seq)
    return total


def simple_normalize(r):
    ''' r is a result from fg.query() '''
    totals_by_variable = defaultdict(float)
    for k, v in r.items():
        totals_by_variable[k[0]] += v
    # Now we create a new result
    # normalized by the totals...
    d = dict()
    for k, v in r.items():
        d[k] = v / totals_by_variable[k[0]]
    return d


def train_by_max_product(fg, training_examples, feature_funcs,
                         learning_rate=0.1, max_iterations=1000):
    '''Ok now after reading Manning and Klein
    again its much more obvious.
    We have to train for one
    Li/Fi at a time...
    '''
    weights = defaultdict(float)
    iterations = 0
    alphabet = ['Healthy', 'Fever']
    L = []
    for fi in range(len(feature_funcs)):
        L.append(random.random())
    while iterations < max_iterations:
        correct = 0
        for j, feature_func in enumerate(feature_funcs):
            total_error = 0

            for sequence, label in training_examples:
                F_j = make_big_F(feature_func)
                exp_emp = F_j(sequence, label)
                # exp_mdl is the 'brute force' version.
                # We should use the actual model here...
                exp_mdl = expected_model(sequence, L, j, alphabet, F_j)
                import ipdb; ipdb.set_trace()
                build_potential_functions(sequence, fg, L, feature_funcs)
                exp_mdl_from_fg = total_expected_model(fg, L, feature_func,
                                                       training_examples, feature_funcs)

                # query the model by using forward..
                # Note we have to build the G funcs...
                g_ = {}
                G_ = {}
                for i in range(len(sequence)):
                    g_[i] = make_g_func(L, feature_funcs, sequence, i)
                    G_[i] = make_G_func(g_[i])

                exp_mdl_from_forward = forward(sequence, len(sequence) - 1,
                                G_, alphabet, {})
                import ipdb; ipdb.set_trace()
                delta = exp_emp - exp_mdl - L[j] / 10
                total_error += delta * delta

                L[j] = L[j] + delta * learning_rate
                build_potential_functions(sequence, fg, L, feature_funcs)
                print j, exp_emp, exp_mdl, delta
                #print L
                print sequence

                #print crf_out(fg.query())
                #build_potential_functions(sequence, fg, L, feature_funcs)
                #result = simple_normalize(fg.query())
                #p = 1
                #for l, var in zip(label,
                #                  ['y0', 'y1', 'y2', 'y3', 'y4', 'y5']):
                #    p *= result[(var, l)]
                #print 'P(correct_label)=%s' % p
                print '================'

                # Ok now lets use the viterbi_crf module
                # to verify these results...
                g_ = {}
                for i in range(len(sequence)):
                    g_[i] = make_g_func(L, feature_funcs, sequence, i)
                y_seq = my_viterbi(sequence, len(sequence) - 1,
                                g_, alphabet, {})
                print y_seq, label
                if y_seq == label:
                    correct += 1
                print iterations, '****', total_error, sum([l**2 for l in L]), float(correct) / len(feature_funcs)


            # Lets try to do this one feature at a time...
            #exp_emp = total_expected_empirical(
            #    feature_func, training_examples)
            #exp_mdl = total_expected_model(
            #    fg, L, feature_func, training_examples)
            #delta = exp_emp - exp_mdl
            #total_error += delta ** 2
            #L[j] = L[j] - delta * learning_rate #- L[j] / 10
            #print j, delta, L[j]
        print iterations, '****', total_error, sum([l**2 for l in L]), float(correct) / len(feature_funcs)
        #raw_input('yo!')
        #for sequence, labels in training_examples:
        #    build_potential_functions(sequence, fg, L, feature_funcs)
        #    print labels, crf_out(fg.query())
        #import ipdb; ipdb.set_trace()
        iterations += 1
    return weights



if __name__ == '__main__':
    for line in example():
        print(' '.join(map(str, line)))
    #while True:
    #    x_seq = raw_input('Enter observed sequence: ')
    #    print forward_backward(tuple(x_seq.split()))

    # Lets generate some samples from the hmm...
    # we will just go through all the possible
    # inputs of length 3 for now...
    input_alphabet = ['dizzy', 'normal', 'cold']
    training_examples = []
    for combo in xproduct(*([input_alphabet] * 3)):
        training_examples.append((combo, viterbi(combo,
                                                 states,
                                                 start_probability,
                                                 transition_probability,
                                                 emission_probability)[1]))




    # First we will specify an input...
    x_seq = ['dizzy', 'normal', 'cold']
    output_alphabet = ['Healthy', 'Fever']
    # Now lets create the output nodes...
    variable_nodes = []
    for i in range(0, len(x_seq)):
        node = UndirectedNode('y%s' % i)
        node.variable_name = 'y%s' % i
        variable_nodes.append(node)


    # Now we need to connect them...
    for y_a, y_b in zip(variable_nodes, variable_nodes[1:]):
        connect_ug(y_a, y_b)

    ug = UndirectedGraph(variable_nodes)
    print ug

    jt = ug.build_join_tree()

    fg_factor_nodes = {}

    for clique in jt.nodes:
        if isinstance(clique, JoinTreeSepSetNode):
            # Its unclear what the 'variable_names'
            # should be here, it could be the
            # common variables or it could
            # be the disjunction of the variable names
            fg_factor_node = FactorNode('SEPSET%s' % clique.name, lambda x: 1)
            #fg_factor_nodes[clique.name] = fg_factor_node
            #fg_factor_node.variable_names = clique.variable_names
            fg_factor_node.variable_names = clique.sepset.X.nodes.difference(clique.sepset.Y.nodes)

            continue
        # Note we will go and fix the function later for
        # now just set to unity.
        fg_factor_node = FactorNode(clique.name, lambda x: 1)
        fg_factor_nodes[clique.name] = fg_factor_node
        fg_factor_node.variable_names = clique.variable_names

    fg_variable_nodes = {}
    for node in ug.nodes:
        variable_node = VariableNode(node.variable_name)
        variable_node.domain = output_alphabet
        fg_variable_nodes[node.variable_name] = variable_node

    # Generate random initial weights...
    L = []
    for i in range(len(feature_functions)):
        L.append(random.random())

    for factor_node in fg_factor_nodes.values():
        print factor_node.variable_names
        # Mmmm we need a j???
        j = lame_get_j(factor_node.variable_names)
        factor_node.func = make_potential(x_seq, j, L, feature_functions)
        factor_node.func.argspec = factor_node.variable_names[:]
        d = dict()
        for variable_name in factor_node.variable_names:
            d[variable_name] = fg_variable_nodes[variable_name].domain
        factor_node.domains = copy.copy(d)
        factor_node.func.domains = copy.copy(d)
        print get_args(factor_node.func)


    # We have to actually connect the m
    for fg_factor_node in fg_factor_nodes.values():
        for variable_name in fg_factor_node.variable_names:
            fg_connect(fg_factor_node, fg_variable_nodes[variable_name])

    # Build the FactorGraph
    fg = FactorGraph(
        fg_variable_nodes.values() +
        fg_factor_nodes.values())


    weights = train_by_max_product(fg, training_examples, feature_functions)
    print weights
