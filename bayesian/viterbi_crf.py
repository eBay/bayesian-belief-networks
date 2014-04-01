'''From http://enterface10.science.uva.nl/pdf/lecture3_crfs.pdf'''
from math import exp

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
            vals.append(((y_, t), g_[t]('__START__', y_)))
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
                    v + g_[t](y_prev, y)))
            partials[key] = max([v[1] for v in vals])
            currents.append((key, partials[key]))
    # Now we need to return the most likely lable...
    y_seq = []
    p_seq = 1
    #import ipdb; ipdb.set_trace()
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        max_ = max(candidates, key=lambda x: x[1])
        y_seq.append(max_[0][0])
        if i == t:
            p_seq *= partials[(y_seq[-1], i)] / sum([x[1] for x in candidates])
    return y_seq, partials, p_seq


def forward(x_seq, t, G_, output_alphabet, partials):
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


def forward(x_seq, t, G_, output_alphabet, partials, T_inv, T):
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
    vals = []
    if t == 0:
        for y in list(T[-1]['__START__']):
            vals.append(G_[t]('__START__', y))
            partials[(y, t)] = G_[t]('__START__', y)
    else:
        for y, y_prevs in T_inv[t].items():
            vals = []
            key = (y, t)
            for y_prev in y_prevs:
                if (y_prev, t - 1) not in partials:
                    forward(x_seq, t - 1, G_, output_alphabet, partials, T_inv, T)
                vals.append((key, partials[y_prev, t - 1] * G_[t](y_prev, y)))
            partials[key] = sumlist(vals, key=lambda x:x[1])
    return partials


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
