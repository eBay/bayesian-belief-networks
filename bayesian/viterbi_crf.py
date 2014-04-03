import sys
from math import exp
from itertools import product as xproduct


START = '__START__'

'''
Note, this should really be called Forward-Backward
which is a more general algorithm than viterbi.
By changing between max and sum we can
either compute the most likely labelling
or the Z partition function.
'''

def make_g_func(w, feature_funcs, x_seq, t):

    def g_x(y_prev, y):
        total = 0
        for j, (w_, f_) in enumerate(zip(w, feature_funcs)):
            total += w_ * f_(y_prev, y, x_seq, t)
        return total

    return g_x


def make_G_func(g_func):

    def G_x(y_prev, y):
        return exp(g_func(y_prev, y))

    return G_x



    return g_x_3


def make_G_func_3(g_func_3):

    def G_x_3(y_prev, y, x_seq):
        return exp(g_func_3(y_prev, y, x_seq))

    return G_x_3


def propagate(x_seq, t, G_, output_alphabet, partials, aggregator):
    '''
    This is a generic propagation algorithm.
    By suppling different 'aggregator' functions
    one can use this for both viterbi and
    message passing algorithms.
    Note that in http://enterface10.science.uva.nl/pdf/lecture3_crfs.pdf
    there are both lowercase g_ and uppercase G_ functions
    defined the only difference being that
    G_ is uppercased.
    The only difference between finding the
    Z function and the most likely labeling
    is that for Z function we use 'sum'
    as the aggregator and for labelling
    we use max as the aggregator.
    For performace we do not need to use
    the exponentiated versions for
    Viterbi, however for Z function we
    do.

    '''


def sumlist(l, key):
    '''
    Works just like max over a list
    i.e. it applies the key which
    should be a function operating
    on
    '''
    return sum(map(key, l))


def viterbi(x_seq, t, g_, output_alphabet, partials):
    '''
    Use this to find the most likely output
    sequence given an input.
    '''
    vals = []
    if t == 0:
        for y_ in output_alphabet:
            vals.append(((y_, t), g_[t]('__START__', y_, x_seq)))
            partials[(y_, t)] = vals[-1][1]
    else:
        for y_prev in output_alphabet:
            if not (y_prev, t - 1) in partials:
                viterbi(x_seq, t - 1, g_, output_alphabet, partials)
        prevs = dict([(k, v) for k, v in partials.items() if k[1] == t - 1])
        currents = []
        for y in output_alphabet:
            key = (y, t)
            vals = []
            for (y_prev, _), v in prevs.items():
                vals.append((
                    key,
                    v + g_[t](y_prev, y, x_seq)))
            partials[key] = max([v[1] for v in vals])
            currents.append((key, partials[key]))
    # Now we need to return the most likely lable...
    y_seq = []
    p_seq = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        if i == t:
            denom = sum([x[1] for x in candidates])
            if denom == 0:
                print >> sys.stderr, '***WARNING: LOW PROBABILITIES ***'
                continue
            p_seq *= partials[(y_seq[-1], i)] / denom
    return y_seq, partials, p_seq


def old_forward(x_seq, t, G_, output_alphabet, partials):
    '''
    Use this to find the Z function for undirected
    graphs. To find the most likely labelling use
    viterbi which is just a special case of
    forward.
    '''
    vals = []
    if t == 0:
        for y_ in output_alphabet:
            vals.append(((y_, t), G_[t]('__START__', y_)))
            partials[(y_, t)] = vals[-1][1]
        return sumlist(vals, key=lambda x:x[1]), partials
    else:
        for y_prev in output_alphabet:
            if not (y_prev, t - 1) in partials:
                old_forward(x_seq, t - 1, G_, output_alphabet, partials)
        prevs = dict([(k, v) for k, v in partials.items() if k[1] == t - 1])
        currents = []
        for y in output_alphabet:
            key = (y, t)
            vals = []
            for (y_prev, _), v in prevs.items():
                vals.append((
                    key,
                    v + G_[t](y_prev, y)))
            partials[key] = sumlist(vals, key=lambda x:x[1])
            currents.append((key, partials[key]))
    # Now we need to return the sum up to position t
    y_seq = []
    p_seq = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        #if i == t:
        p_seq *= partials[(y_seq[-1], i)] / sum([x[1] for x in candidates])
    #candidates = [(k, v) for k, v in partials.items() if k[1] == t]
    return y_seq, partials, p_seq


def new_vit(x_seq, t, G_, partials, T, T_inv):
    '''
    My old vit was getting a slightly lower
    probability than expected.
    Also in order to reuse the G_ funcs
    it would be better to have them in
    a form that takes x_seq as a parameter.
    '''
    for y, y_prevs in T_inv[t].items():
        vals = []
        key = (y, t)
        for y_prev in y_prevs:
            if (y_prev, t - 1) not in partials:
                new_vit(x_seq, t - 1, G_, partials, T, T_inv)
            vals.append((key, partials[y_prev, t - 1] * G_[t](y_prev, y, x_seq)))
        partials[key] = max(vals, key=lambda x:x[1])[1]
    # Now extract the output tokens...
    y_seq = []
    p = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        p = max_[1]
    return y_seq, partials, p


def new_vit_G3(x_seq, t, G3_funcs, partials, T, T_inv):
    '''
    My old vit was getting a slightly lower
    probability than expected.
    Also in order to reuse the G_ funcs
    it would be better to have them in
    a form that takes x_seq as a parameter.
    '''
    for y, y_prevs in T_inv[t].items():
        vals = []
        key = (y, t)
        for y_prev in y_prevs:
            if (y_prev, t - 1) not in partials:
                new_vit(x_seq, t - 1, G_, partials, T, T_inv)
            vals.append((key, partials[y_prev, t - 1] * G_[t](y_prev, y, x_seq)))
        partials[key] = max(vals, key=lambda x:x[1])[1]
    # Now extract the output tokens...
    y_seq = []
    p = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        p = max_[1]
    return y_seq, partials, p


def forward(x_seq, t, G_, partials, T, T_inv):
    '''
    Use this to find the Z function for undirected
    graphs. To find the most likely labelling use
    viterbi which is just a special case of
    forward. This is a modified version of the
    usual forward algorithm.
    T and T_inv are the possible transitions in
    the training set. (i.e. T_inv is a mapping from
    a state y to all possible previous states
    in the training set.
    '''
    for y, y_prevs in T_inv[t].items():
        vals = []
        key = (y, t)
        for y_prev in y_prevs:
            if (y_prev, t - 1) not in partials:
                forward(x_seq, t - 1, G_, partials, T, T_inv)
            vals.append((key, partials[y_prev, t - 1] * G_[t](y_prev, y)))
        partials[key] = sumlist(vals, key=lambda x:x[1])
    return partials


def new_forward(x_seq, t, G_, partials, T, T_inv):
    '''
    My old vit was getting a slightly lower
    probability than expected.
    Also in order to reuse the G_ funcs
    it would be better to have them in
    a form that takes x_seq as a parameter.
    '''
    if t == 0:
        y_prevs = ['__START__']
    else:
        y_prevs = ['NAME', 'OTHER']
    for y in ['NAME', 'OTHER']:
        vals = []
        for y_prev in y_prevs:
            key = (y, t)
            if (y_prev, t - 1) not in partials:
                new_forward(x_seq, t - 1, G_, partials, T, T_inv)
            vals.append((key, partials[y_prev, t - 1] * G_[t](y_prev, y, x_seq)))
        partials[key] = sumlist(vals, key=lambda x:x[1])
    # Now extract the output tokens...
    y_seq = []
    p = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        p = max_[1]
    return y_seq, partials, p


def new_backward(x_seq, t, G_, partials, T, T_inv):
    '''
    My old vit was getting a slightly lower
    probability than expected.
    Also in order to reuse the G_ funcs
    it would be better to have them in
    a form that takes x_seq as a parameter.
    '''
    if t == len(x_seq):
        y_nexts = ['__STOP__']
    else:
        y_nexts = ['NAME', 'OTHER']
    for y in ['NAME', 'OTHER']:
        vals = []
        for y_next in y_nexts:
            key = (y, t)
            if (y_next, t + 1) not in partials:
                new_backward(x_seq, t + 1, G_, partials, T, T_inv)
            import ipdb; ipdb.set_trace()
            vals.append((key, partials[y_next, t + 1] * G_[t](y, y_next, x_seq)))
        partials[key] = sumlist(vals, key=lambda x:x[1])
    # Now extract the output tokens...
    y_seq = []
    p = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        p = max_[1]
    return y_seq, partials, p


def backward(x_seq, t, G_, output_alphabet, partials):
    '''
    Same as forward but in reverse direction...
    could we just reverse x_seq???
    Use this to find the Z function for undirected
    graphs. To find the most likely labelling use
    viterbi which is just a special case of
    forward.
    '''
    vals = []
    if t == 0:
        for y_ in output_alphabet:
            vals.append(((y_, t), G_[t]('__START__', y_)))
            partials[(y_, t)] = vals[-1][1]
        return sumlist(vals, key=lambda x:x[1]), partials
    else:
        for y_prev in output_alphabet:
            if not (y_prev, t - 1) in partials:
                forward(x_seq, t - 1, G_, output_alphabet, partials)
        prevs = dict([(k, v) for k, v in partials.items() if k[1] == t - 1])
        currents = []
        for y in output_alphabet:
            key = (y, t)
            vals = []
            for (y_prev, _), v in prevs.items():
                vals.append((
                    key,
                    v + G_[t](y_prev, y)))
            partials[key] = sumlist(vals, key=lambda x:x[1])
            currents.append((key, partials[key]))
    # Now we need to return the sum up to position t
    y_seq = []
    p_seq = 1
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        #if i == t:
        p_seq *= partials[(y_seq[-1], i)] / sum([x[1] for x in candidates])
    #candidates = [(k, v) for k, v in partials.items() if k[1] == t]
    return y_seq, partials, p_seq

def backward(x_seq, t, G_, partials, T, T_inv):
    '''
    Use this to find the Z function for undirected
    graphs. To find the most likely labelling use
    viterbi which is just a special case of
    forward. This is a modified version of the
    usual forward algorithm.
    T and T_inv are the possible transitions in
    the training set. (i.e. T_inv is a mapping from
    a state y to all possible previous states
    in the training set.
    '''
    for y, y_nexts in T[t].items():
        vals = []
        key = (y, t)
        for y_next in y_nexts:
            if (y_next, t + 1) not in partials:
                backward(x_seq, t + 1, G_, partials, T, T_inv)
            vals.append((key, partials[y_next, t + 1] * G_[t](y, y_next)))
            partials[key] = sumlist(vals, key=lambda x:x[1])
    return partials


def make_beta_func(G2_func, alphabet, l):

    betas = {
        ('__STOP__', l): 1
        }

    def beta_func(y, t):
        if t == l:
            return 1
        if (y, t) in betas:
            return betas[y, t]
        y_nexts = alphabet
        if t == l - 1:
            y_nexts = ['__STOP__']
        vals = []
        for y_ in y_nexts:
            vals.append(beta_func(y_, t + 1))
        if t == -1:
            betas[y, t] = sum(vals)
        else:
            betas[y, t] = sum(vals) * G2_func[t](y_, y)
        return betas[y, t]

    return beta_func

def make_beta3_func(G3_func, alphabet, l):

    betas = {
        ('__STOP__', l): 1
        }

    def beta_func(y, t):
        if t == l:
            return 1
        if (y, t) in betas:
            return betas[y, t]
        y_nexts = alphabet
        if t == l - 1:
            y_nexts = ['__STOP__']
        vals = []
        for y_ in y_nexts:
            vals.append(beta_func(y_, t + 1))
        if t == -1:
            betas[y, t] = sum(vals)
        else:
            betas[y, t] = sum(vals) * G2_func[t](y_, y)
        return betas[y, t]

    return beta_func



def make_alpha_func(G2_func, alphabet, l):

    alphas = {
        ('__START__', -1): 1
        }

    def alpha_func(y, t):
        if t == -1:
            return 1
        if (y, t) in alphas:
            return alphas[y, t]
        y_prevs = alphabet
        if t == 0:
            y_prevs = ['__START__']
        vals = []
        for y_ in y_prevs:
            vals.append(alpha_func(y_, t - 1))
        if t == l:
            alphas[y, t] = sum(vals)
        else:
            alphas[y, t] = sum(vals) * G2_func[t](y_, y)
        return alphas[y, t]

    return alpha_func


def make_viterbi_func(G2_func, alphabet, l):

    alphas = {
        ('__START__', -1): 1
        }

    def viterbi_func(y, t):
        if t == -1:
            return 1
        if (y, t) in alphas:
            return alphas[y, t]
        y_prevs = alphabet
        if t == 0:
            y_prevs = ['__START__']
        vals = []
        for y_ in y_prevs:
            vals.append(viterbi_func(y_, t - 1))
        if t == l:
            alphas[y, t] = max(vals)
        else:
            alphas[y, t] = max(vals) * G2_func[t](y_, y)
        return alphas[y, t]

    return viterbi_func


def viterbi_decoder(x_seq, t, G_funcs, output_alphabet,
                    partials={(START, -1):1}):
    if t == -1:
        return 1
    y_prevs = output_alphabet
    if t == 0:
        y_prevs = [START]
    for y_prev in y_prevs:
        for y in output_alphabet:

            vals = []
            key = (y, t)
            if (y_prev, t - 1) not in partials:
                viterbi_decoder(x_seq, t - 1, G_funcs, output_alphabet, partials)
            vals.append(partials[y_prev, t - 1] * G_funcs[t](y_prev, y, x_seq))
    import ipdb; ipdb.set_trace()
    partials[key] = max(vals)
    # Now extract the output tokens...
    y_seq = []
    p = 1

    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        p = max_[1]
    return y_seq, partials, p


def PropagationEngine(x_seq, feature_funcs, output_alphabet):

    # Firstly we define the 'g_t' functions
    # These are functions from 1 to T where
    # in the above T is the length of the
    # sequence. each g_t function takes
    # paramaters y-1 and y ie:
    # g_2(y0, y1)
    g_ = dict()
    for t in range(0, len(x_seq)):
        g_[t] = make_g_func(w, feature_funcs, x_seq, t)

    # Now define the uppercase G funcs...x_seq
    G_ = dict()
    for k, v in g_.items():
        G_[k] = make_G_func(v)

    # Now we need to record the partial forward
    # and partial backward paths...
    #partial_forward = dict()
    #partial_backward = dict()

    # To start out we need to initialize (We could get rid of this)
    #partial_forward([], '__START__') = 0

    # Now for the rest of the sequence...
    #for t in range(0, len(x_seq)):
    #    for y in output_alphabet:
    #        partial_forward
