'''Simple Part of Speech Tagging Example'''
from collections import defaultdict
from pprint import pprint

from bayesian.graph import UndirectedNode, UndirectedGraph
from bayesian.undirected_graphical_model import UndirectedModelNode
from bayesian.bbn import JoinTreeSepSetNode, clique_tree_sum_product, build_bbn
from bayesian.bbn import make_undirected_copy, make_moralized_copy
from bayesian.undirected_graphical_model import (
    UndirectedModel, connect as ug_connect)
from bayesian.linear_chain_crf import build_lccrf
from bayesian.linear_chain_crf import make_g_func, make_G_func
from bayesian.viterbi_crf import make_beta_func
#from bayesian.factor_graph import make_product_func, make_unity
from bayesian.factor_graph import make_unity
from bayesian.utils import get_args
from scipy import optimize as op

'''
We will try to construct the same
graph as in the LCCRF and we should
be able to get the same results
using the max-product algorithm...

We now get approximately the same result
using the clique_tree_sum_product on
a directly created undirected graph.

The numbers are higher possibly because we
are using the sum rather than the max.

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



def build_um_from_sentence(S, y_domain):
    # Okay for each token in the sequence we
    # need a y node and initially just assign
    # the unity function, the actual feature
    # functions will be applied later...

    y_nodes = []
    for i, token in enumerate(S):
        node_name = 'y%s' % i
        y_node = UndirectedModelNode(node_name, make_unity([node_name], True))
        y_node.func.domains = {
            node_name: y_domain }
        y_node.variable_name = node_name
        y_nodes.append(y_node)

    # Now for each pair of adjacent nodes
    # we want to connect them...
    for left_node, right_node in zip(y_nodes, y_nodes[1:]):
        ug_connect(left_node, right_node)

    # And now we need the X node for the
    # input sentence...
    X_node = UndirectedModelNode('X', make_unity(['X'], True))
    X_node.func.domains = dict(X = [S])
    X_node.variable_name = 'X'

    # And connect the X node to all the
    # y nodes building the domains as we go...
    domains = dict(X=[S])
    for y_node in y_nodes:
        domains[y_node.variable_name] = y_domain
        ug_connect(y_node, X_node)

    # And now build the graph...
    node_dict = dict([(node.name, node) for
                      node in y_nodes + [X_node]])
    um = UndirectedModel(node_dict, domains=domains)
    # Now we just want to modify the argspecs
    # so that the feature functions get attached
    # correctly. (Until we find a better way to do this)
    for node in um.nodes:
        if node.name == 'y0':
            node.argspec = ['y0', 'X']
        elif node.name.startswith('y'):
            i = int(node.name[1:])
            node.argspec = ['y%s' % (i - 1), 'y%s' % i, 'X']
        else:
            # For the X node we basicly leave it at unity...
            pass
    return um






def make_starter_function(name, feature_function, S):
    def starter_function(y, X):
        return feature_function('__START__', y, X, 0)
    starter_function.argspec = ('y0', 'X')
    starter_function.domains = dict(
        y0 = ['NAME', 'OTHER'],
        # Note here we are clamping X to only
        # allow the input sentence for this single
        # example. This is because in a tempate
        # based model we generally 'unroll' the
        # templates for every training example
        # and we are generally not interested in
        # prediction the value of X. Indeed there
        # would be two many possible values of
        # X ie input sequences to even make this
        # feasible. Also make sure that S is
        # a tuple so that it can be cached.
        X = [S])
    starter_function.original_func = feature_function
    if hasattr(feature_function, 'weight'):
        starter_function.weight = feature_function.weight
    return starter_function


def expand_feature_functions(jt, feature_functions, S):
    # For now lets assume all the feature functions
    # are of the same 'template' and therefore apply
    # to all clusters....
    t = Template("chain", ['y_prev', 'y', 'X', 'i'])
    for cluster in jt.clique_nodes:

        print cluster
        combined_vars = dict()
        for sepset_node in cluster.neighbours:
            for variable_name in sepset_node.variable_names:
                combined_vars[variable_name] = sorted(sepset_node.variable_names)
        variable_names = cluster.variable_names
        merged_variables = resolve_merged_variables(variable_names, combined_vars)
        new_feature_functions = []
        #print cluster.clique.nodes
        for feature_func in feature_functions_:
            # Lets assume that we have a template
            # that we know is associated with
            # the function...
            y_nodes = [n for n in cluster.clique.nodes if n.name.startswith('y')]
            y_node_names = [n.name for n in y_nodes]
            if 'y0' in y_node_names and 'y1' in y_node_names:
                new_feature_functions.append(
                    make_starter_function('y0', feature_func, S))
            y_nodes.sort(key=lambda x:int(x.name[1]))
            new_feature_functions.append(
                expand_feature_function(feature_func, y_nodes, jt.sepset_nodes))
            # We also need the artificial start node....
        # We should attach the expanded functions to
        # each clique so we have them later...
        cluster.expanded_feature_functions = new_feature_functions



#---------------------------------------------------
# Functions for the Directed Version
#
# These I created purely to view the factor graph
# that gets created when starting off from the
# directed version of a sequence tagging model.
# --------------------------------------------------


def f_X(X):
    return 1


def f_y0(X, y0):
    return 1


def f_y1(X, y0, y1):
    return 1


def f_y2(X, y1, y2):
    return 1


def f_y3(X, y2, y3):
    return 1


def f_y4(X, y3, y4):
    return 1


def get_bbn(S):
    '''S is the custom sentence for this dag'''
    g = build_bbn(
        f_X, f_y0, f_y1,
        f_y2, f_y3, f_y4,
        domains = dict(
            #X = (('George', 'Washington'),),
            X = (S,),
            y0 = ('NAME', 'OTHER'),
            y1 = ('NAME', 'OTHER'),
            y2 = ('NAME', 'OTHER'),
            y3 = ('NAME', 'OTHER'),
            y4 = ('NAME', 'OTHER')))
    return g


def make_sum_func(factors):
    '''
    Note that this func is different to
    the make_sum_func in factor_graph in that
    for the initial functions assigned to the
    um nodes we include the weights that are
    multiplied by each feature function.
    The summed functions used within the
    sum product algorithm do not need
    to remultiply by these weights....
    '''
    args_map = {}
    all_args = []
    domains = {}
    for factor in factors:
        args_map[factor] = get_args(factor)
        all_args += args_map[factor]
        if hasattr(factor, 'domains'):
            domains.update(factor.domains)
    args = list(set(all_args))

    def sum_func(*sum_func_args):
        arg_dict = dict(zip(args, sum_func_args))
        result = 0
        for factor in factors:
            # We need to build the correct argument
            # list to call this factor with.
            factor_args = []
            for arg in get_args(factor):
                if arg in arg_dict:
                    factor_args.append(arg_dict[arg])
            #if not factor_args:
            #    # Since we always require
            #    # at least one argument we
            #    # insert a dummy argument
            #    # so that the unity function works.
            #    factor_args.append('dummy')
            try:
                res = factor.weight * factor(*factor_args)
            except:
                # This is just for debugging purposes
                # remove before final
                import ipdb; ipdb.set_trace()
                res = factor.weight * factor(*factor_args)
            if res < 0:
                import ipdb; ipdb.set_trace()
                print 'negative result from product func...'
                res = factor.weight * factor(*factor_args)
            result += res
        return result

    sum_func.argspec = args
    sum_func.factors = factors
    sum_func.domains = domains
    # For now I will not memoize
    #return memoize(sum_func)
    return sum_func


def customize_feature_function(node, function, S):
    y_nodes = [a for a in node.argspec if a.startswith('y')]
    y_nodes.sort()
    j = int(y_nodes[-1][1:])
    def customized_function(*args):
        return function(args[0], args[1], args[2], j)
        #return function(yp, y, X, j)
    customized_function.argspec = y_nodes + ['X']
    customized_function.weight = function.weight
    domains = dict(
        X = [S])
    for node in y_nodes:
        domains[node] = ['NAME', 'OTHER']
    customized_function.domains = domains
    return customized_function


def attach_feature_functions(g, feature_functions, S, weights):
    '''From either a directed or undirected
    graph g we want to attach the feature functions
    to one of the nodes. We will start with
    the bbn from get_bbn so we can see what the
    jt assignments look like'''
    for function, weight in zip(feature_functions, weights):
        function.weight = weight
    for node in g.nodes:
        if len(node.argspec) == 2:
            # This is the first token
            # so we need a starter function...
            starter_functions = []
            for func in feature_functions:
                starter_functions.append(
                    make_starter_function(node.name, func, S))
            # Now that we have all the starter functions
            # we replace the nodes function with
            # the sum func of these starter functions...
            node.func = make_sum_func(starter_functions)
            node.argspec = node.func.argspec
        elif len(node.argspec) == 1:
            # This is the bbn node with just the
            # X variable... I think this one
            # can just remain as unity since
            # X can only take one value...
            node.func.argspec = node.argspec
        elif len(node.argspec) == 3:
            functions = []
            for func in feature_functions:
                functions.append(
                    customize_feature_function(node, func, S))
            print [f.argspec for f in functions]
            node.func = make_sum_func(functions)
            node.argspec = node.func.argspec


if __name__ == '__main__':
    output_alphabet = ['NAME', 'OTHER']
    # These are the weights from the lccrf training...
    weights = [3.174603174603173, 3.977272727272726, 1.9607843137254881, 5.194805194805184, 6.779661016949148, 1.9801980198019793, 7.6923076922798295, 1.9801980198019793, 3.8461538461399147]
    S = ('Claude', 'Shannon', 'was', 'a', 'genius')
    # First lets look at what the convert_to_bbn does
    # with a sequential like directed model

    # LEAVE THIS IN FOR NOW AS SANITY CHECK
    dag = get_bbn(S)
    attach_feature_functions(dag, feature_functions_, S, weights)
    dag.inference_method = 'clique_tree_sum_product'
    dag.q()

    # For the um we should always get the same result as the bbn
    um_direct = build_um_from_sentence(S, output_alphabet)
    attach_feature_functions(um_direct, feature_functions_, S, weights)
    um_direct.q()


    w = [0.5] * len(feature_functions_)
    lccrf = build_lccrf(
        ['NAME', 'OTHER'], feature_functions_)
    lccrf.weights = weights

    # So lets do this in a repeated loop...
    while True:
        X = raw_input('Input Sentence -> ')
        X_seq = tuple(X.split())
        um = build_um_from_sentence(X_seq, ['NAME', 'OTHER'])
        # Now we need to move the argspec stuff
        # into the build_um....
        attach_feature_functions(um, feature_functions_, X_seq, weights)
        um_result = um.q()
        print um_result
        y_nodes = sorted([node.name for node
                          in um.nodes if node.name.startswith('y')])
        labels = []
        for y_node in y_nodes:
            y_probs = [p for p in um_result.items() if y_node in p[0]]
            max = 0
            max_label = None
            for prob in y_probs:
                if prob[1] > max:
                    max = prob[1]
                    max_label = prob[0][1]
            labels.append(max_label)
        print labels
        # Now we will also call the lccrf to compare...
        lccrf_result = lccrf.q(X)
        pprint(lccrf_result)

    # We now get the same results for both so
    # I will start looking at training...
    # for this we would need the normalizing
    # Z/partition function.

    #print 'Training CRF...'
    #weights = lccrf.batch_train(training_examples)

    lccrf.weights = weights
    print weights
    # So these are the weights that we get with around 7 or 8 feature functions...
    # [3.174603174603173, 3.977272727272726, 1.9607843137254881, 5.194805194805184, 6.779661016949148, 1.9801980198019793, 7.6923076922798295, 1.9801980198019793, 3.8461538461399147]
    print lccrf.q('This is a test sentence with Claude Shannons name in it')
    lccrf.batch_query([x[0] for x in training_examples])

    # the funcs are in a g_ dict
    # example call: g_[t]('__START__', y_, x_seq)
    # returns 3.174603174603173

    # "Shannon" the result is
    # p 3.174603174603173 / (3.174603174603173 + 1.9607843137254881)
    # 0.6181818181818184
    #import ipdb; ipdb.set_trace()
    #print lccrf.q('Shannon')
    # Okay that worked so now lets see if we can get the same result
    # for the factor graph.... remember that first we have to
    # rebuild the factor graph with just one y node....
