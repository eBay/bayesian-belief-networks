'''Simple Part of Speech Tagging Example'''

from bayesian.linear_chain_crf import build_lccrf
from bayesian.linear_chain_crf import make_g_func, make_G_func
from bayesian.viterbi_crf import make_beta_func
from scipy import optimize as op

'''
In this example we build a very simple CRF to
tag sentences with just two tags.

'''

S1 = ('The', 'first', 'president', 'was', 'George', 'Washington')
S2 = ('George', 'Washington', 'was', 'the', 'first', 'president')
desired_output_1 = ['OTHER', 'OTHER', 'OTHER', 'OTHER', 'NAME', 'NAME']
desired_output_2 = ['NAME', 'NAME', 'OTHER', 'OTHER', 'OTHER', 'OTHER']

training_examples = [(S1, desired_output_1),
                     (S2, desired_output_2)]


def indicator_func_0(yp, _, x_bar, j):
    if yp == '__START__':
        return 1
    return 0


def indicator_func_1(yp, y, x_bar, j):
    if x_bar[j].lower() == 'the':
        return 1
    return 0


def indicator_func_2(yp, y, x_bar, j):
    if x_bar[j].lower() == 'first':
        return 1
    return 0


def indicator_func_3(yp, y, x_bar, j):
    if x_bar[j].lower() == 'president':
        return 1
    return 0


def indicator_func_4(yp, y, x_bar, j):
    if x_bar[j].lower() == 'was':
        return 1
    return 0


def indicator_func_5(yp, y, x_bar, j):
    if x_bar[j].lower() == 'george':
        return 1
    return 0


def indicator_func_6(yp, y, x_bar, j):
    if y == '__STOP__':
        return 1
    return 0


def indicator_func_7(yp, y, x_bar, j):
    if x_bar[j].lower() == 'washington':
        return 1
    return 0


def indicator_func_8(yp, y, x_bar, j):
    if x_bar[j].istitle() and y == 'NAME':
        return 1
    return 0


def indicator_func_9(yp, y, x_bar, j):
    if not x_bar[j].istitle() and y == 'OTHER':
        return 1
    return 0


def indicator_func_10(yp, y, x_bar, j):
    if j == 0 and y == 'OTHER':
        return 1
    return 0


def indicator_func_11(yp, y, x_bar, j):
    if yp == 'NAME' and y == 'NAME' and x_bar[j].istitle():
        # Names often come together
        return 1
    return 0


def indicator_func_12(yp, y, x_bar, j):
    if yp == 'OTHER' and y == 'OTHER' and not x_bar[j].istitle():
        # Names often come together
        return 1
    return 0


def indicator_func_13(yp, y, x_bar, j):
    # Known names
    if x_bar[j] == 'George':
        return 1
    return 0


def indicator_func_14(yp, y, x_bar, j):
    if j > 0 and yp == 'NAME' and y == 'NAME' and x_bar[j-1].istitle() \
       and x_bar[j].istitle():
        # Names often come together
        return 1
    return 0


def indicator_func_15(yp, y, x_bar, j):
    # Known names
    if x_bar[j] == 'Washington':
        return 1
    return 0


def indicator_func_16(yp, y, x_bar, j):
    if x_bar[0].lower() == 'the' and y == 'OTHER' and yp == '__START__':
        return 1
    return 0


# According to http://arxiv.org/PS_cache/arxiv/pdf/1011/1011.4088v1.pdf
# We should really have 'transition' and 'io' features...
def transition_func_0(yp, y, x_bar, j):
    if yp == 'OTHER' and y == 'OTHER':
        return 1
    return 0


def transition_func_1(yp, y, x_bar, j):
    if yp == 'OTHER' and y == 'NAME':
        return 1
    return 0


def transition_func_2(yp, y, x_bar, j):
    if yp == 'NAME' and y == 'OTHER':
        return 1
    return 0


def transition_func_3(yp, y, x_bar, j):
    if yp == 'NAME' and y == 'NAME':
        return 1
    return 0


# Now try using these funcs instead of f0 and f1...
feature_functions_ = [
    #transition_func_0,
    #transition_func_1,
    #transition_func_2,
    #transition_func_3,
    #indicator_func_0,
    #indicator_func_1,
    #indicator_func_2,
    #indicator_func_3,
    #indicator_func_4,
    #indicator_func_5,
    #indicator_func_6,
    #indicator_func_7,
    indicator_func_8,
    indicator_func_9,
    indicator_func_10,
    indicator_func_11,
    indicator_func_12,
    indicator_func_13,
    indicator_func_14,
    indicator_func_15,
    indicator_func_16,

]

def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


def make_bgfs_optimizable_func(feature_funcs, training_examples, output_alphabet):
    # Seems like the co-efficients ie weights
    # all go into a single variable as a list
    def optimizeable(X):
        # We will now simply return the full
        # joint of all the potentitals...
        total = 0
        for x_seq, y_seq in training_examples:
            # We need the log. probability of
            # the training data given
            # the weights
                    # First we need the 'G' functions...
            g_ = dict()
            for i in range(len(x_seq)):
                g_[i] = make_g_func(
                    X, feature_funcs,
                x_seq, i
            )
            bw = make_beta_func(g_, output_alphabet, len(x_seq))
            num_fw = reduce(
                lambda x, y: x * y,
                [bw(y_seq[i], i) for i in range(len(y_seq))])
            total += num_fw
        return -total
    return optimizeable


if __name__ == '__main__':
    optimal = make_bgfs_optimizable_func(
        feature_functions_, training_examples, ['NAME', 'OTHER'])
    from scipy import optimize as op
    w = [0.5] * len(feature_functions_)
    #optimal(w)
    #learned_weights = op.fmin_l_bfgs(optimal, w, approx_grad=1)
    lccrf = build_lccrf(
        ['NAME', 'OTHER'], feature_functions_)
    #lccrf.weights = op.fmin_l_bfgs_b(optimal, w, approx_grad=1)[0].tolist()

    print 'Training CRF...'
    weights = lccrf.batch_train(training_examples)
    #weights = lccrf.perceptron_train(
    #    training_examples, max_iterations=1000)
    print weights
    print lccrf.q('This is a test sentence with Claude Shannons name in it')
    lccrf.batch_query([x[0] for x in training_examples])
