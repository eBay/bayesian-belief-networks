'''From http://enterface10.science.uva.nl/pdf/lecture3_crfs.pdf'''
from math import exp


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


def V(x_seq, t, g_, output_alphabet, partials):
    vals = []
    if t == 0:
        for y_ in output_alphabet:
            vals.append(((y_, t), g_[t]('__START__', y_)))
            partials[(y_, t)] = vals[-1][1]
        max_ = max(vals, key=lambda x:x[1])
        return [max_[0][0]]
    else:
        for y_prev in output_alphabet:
            if not (y_prev, t - 1) in partials:
                V(x_seq, t - 1, g_, output_alphabet, partials)
        prevs = dict([(k, v) for k, v in partials.items() if k[1] == t - 1])
        currents = []
        for y in output_alphabet:
            key = (y, t)
            vals = []
            for (y_prev, _), v in prevs.items():
                vals.append((
                    key,
                    v + g_[t](y_prev, y)))
            partials[key] = sum([v[1] for v in vals])
            currents.append((key, partials[key]))
        #max_ = max(currents, key=lambda x:x[1])
        #sum_ = sum([c[1] for c in currents])
    # Now we need to return the most likely lable...
    y_seq = []
    for i in range(t + 1):
        candidates = [(k, v) for k, v in partials.items() if k[1] == i]
        y_seq.append(max(candidates, key=lambda x: x[1])[0][0])
    return y_seq


def partial_forward(G_, output_alphabet, x_seq, path, partial_forward, t):
    if len(path) == 0:
        max_val = -1000
        for y in output_alphabet:
            partial_forward[(y, 0)] = G_[0]('__START__', y)
            if partial_forward[(y, 0)] > max_val:
                max_val = partial_forward[(y, 0)]
                max_arg = y
        # Now take the max so far and
        # record it in the path...
        path.append((max_arg, max_val))
        return path, partial_forward
    # Now for the recursion...
    max_val = -1000
    #for y in alphabet:










def viterbi(x_seq, feature_funcs, output_alphabet):

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
