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
        y_node = UndirectedModelNode(node_name, make_unity([node_name]))
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
    X_node = UndirectedModelNode('X', make_unity('X'))
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


def make_sum_func(factors, weights):
    '''
    This is a copy of the factor_graph.make_product_func
    except that it sums since we are working in the
    log domain. I also need to add the weights in here...
    And also it needs the whole arg mapping
    thing....
    Need a generic way to do these...
    Mmmm the problem is these things can get
    wrapped over many times and we dont know
    which expect a value to be unwrapped as
    a tuple and a raw value..
    may need to either create an Args type
    or have some better method....

    '''
    #assert len(factors) == len(weights)
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
            node.func = make_sum_func(starter_functions, weights)
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
            node.func = make_sum_func(functions, weights)
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
        um.q()
        # Now we will also call the lccrf to compare...
        pprint(lccrf.q(X))


    # The undirected copy is essentially
    # the same as the dag for this model
    # just with the arrows removed.
    # Now lets look at the moralized version
    #mg = make_moralized_copy(ug, dag)
    #mg.export('mg_moralized.gv')

    # Since the ug is already fully
    # moralized this is once again just the
    # same.
    # So now we need to just correctly
    # construct the initial potentials
    # just as we tried earlier below

    # Now lets also look at the jt that is built
    # with build_join_tree()
    #jt = dag.build_join_tree()
    #jt.export('jt_sequential.gv')
    # Okay so actually the two join trees
    # are the same so thats good!

    # Now I probably need to create a
    # UndirectedGraph.convert_to_factor_graph
    # as well and then see if the fg looks
    # the same...



    #optimal = make_bgfs_optimizable_func(
    #    feature_functions_, training_examples, ['NAME', 'OTHER'])
    #from scipy import optimize as op
    w = [0.5] * len(feature_functions_)
    #optimal(w)
    #learned_weights = op.fmin_l_bfgs(optimal, w, approx_grad=1)
    lccrf = build_lccrf(
        ['NAME', 'OTHER'], feature_functions_)
    #lccrf.weights = op.fmin_l_bfgs_b(optimal, w, approx_grad=1)[0].tolist()

    # Lets try to build the graph for S1...
    #s1_ug = build_ug(training_examples[0][0])
    # Now lets look at the graph...
    #s1_ug.export('s1_ug.gv')
    # Ok! that graph looks good, now the only difference
    # is that the cancer graph has the potential functions
    # already assigned and "attached" so we have to figure
    # out where those go...

    # Ok so now we will try to build the join tree from
    # that ug...

    #s1_jt = build_join_tree_from_ug(s1_ug)
    #s1_jt.export('s1_jt.gv')
    #jt_from_ug = build_join_tree_from_ug(ug)
    #jt_from_ug.export('jt_from_ug.gv')

    # Okay so now I have a convert_to_factor_graph
    # method in undirected graph class so lets
    # see if we get the same factor graph now...
    # Actually there is alread a 'build_factor_graph'
    # method in a class called UndirectedModel
    # but not sure if its the same as convert_to_factor_graph
    # so I will move the convert_to_factor_graph from
    # the UndirectedGraph class to the UndirectedModel class
    #s1_um = build_um(training_examples[0][0])
    # Lets first take a look at it...
    #s1_um.export('s1_um.gv')

    # Ok that looks fine now to see if the
    # convert_to_factor_graph will work....
    #s1_fg = s1_um.convert_to_factor_graph()

    # Ok, so this is good! (With exception
    # that I put a dummy unity func on
    # all the undirected nodes, but thats ok
    # as now I can go and swap in the
    # real potential functions...

    # For now lets just add the unity func to
    # each node and see what transpires....
    # I am beginning to think though that
    # it may be better to reverse a factor
    # graph (which may have cycles) to
    # an undirected graph and then run
    # the convert_to_factor_graph on the
    # undirected graph. Note that convert_to_factor_graph
    # as it is operating now should really be
    # convert_to_acyclic_factor_graph as it
    # also constructs a jt before conversion
    # to a factor graph.

    # the equivalent functions would look like...
    # In the case of the linear chain CRF
    # all the functions operate on (y_prev, y, X, i)
    # so they simply all apply to each clique
    # mmmmm also we dont have the Y nodes....
    # do we need the Y nodes as well?
    # need to look at literature again....
    # Ok so it seems the potential function
    # for each clique is just the product
    # of all functions taking the
    # variables within that clique
    # as arguments. However the
    # indicator functions are written
    # as 'general' functions ie they
    # use i and y_prev etc so how do
    # we find the ones that apply just to
    # certain nodes?????
    #s1_jt.assign_clusters(s1_ug) <--- This wont work because the nodes are not functions as in BBNs

    # Ok so thinking about it more...
    # Basically for the linear chain CRF
    # since each function has the
    # parameters (y_prev, y, X, i) basically
    # one way to proceed is to create
    # multiple copies of each function
    # like this:
    # def f_start_y0(y0, X, 0)
    # def f_y0_y1(y0, y1, X, 1)
    # def f_y1_y2(y1, y2, X, 2)
    # etc etc....
    # Then we can include each one of
    # these functions in the potential for
    # each cluster...
    #expanded_feature_functions = expand_feature_functions(s1_jt, feature_functions_, S1)
    # The above should have also attached the
    # expanded feature functions to each clique
    # so lets just print them out here to verify...
    #for cluster in s1_jt.nodes:
    #    if isinstance(cluster, JoinTreeSepSetNode):
    #        # For now we have not assigned any
    #        # functions to the SepSet nodes
    #        # which may be okay for linear
    #        # chains but not in arb. graphs...
    #        continue
    #        print cluster.name
    #    else:
    #        print cluster.clique.nodes
    #    for f in cluster.expanded_feature_functions:
    #        print f.__name__, get_args(f), f.original_func.__name__
    #    print '----------------------------------------'

    # So now we have the factor graph built with
    # dummy unity functions and we also
    # have the "expanded" feature functions
    # (the "unrolled" functions) so now
    # we need to look at swapping in the
    # expanded functions instead of the unity functions

    # Lets try to assign the feature functions
    # to the factor nodes now...
    # We could also write a version of expand_feature_functions
    # that operates on the factor graph instead of the jt...

    #expanded_feature_functions_fg = expand_feature_functions_fg(s1_fg, feature_functions_, S1, weights)

    #for factor_node in s1_fg.factor_nodes():
    #    print factor_node
    #    for f in factor_node.expanded_feature_functions:
    #        print f.__name__, get_args(f), f.original_func.__name__
    #    print '----------------------------------------'

    # Okay so the above also built the potential function
    # and assigned it to the factor nodes....
    # Now we need to see if we get the same result
    # as Viterbi!!!!!
    # So first we will let the training run....
    # And then we need to dump the parameter weights...

    #print 'Training CRF...'
    #weights = lccrf.batch_train(training_examples)

    lccrf.weights = weights
    print weights
    # So these are the weights that we get with around 7 or 8 feature functions...
    # [3.174603174603173, 3.977272727272726, 1.9607843137254881, 5.194805194805184, 6.779661016949148, 1.9801980198019793, 7.6923076922798295, 1.9801980198019793, 3.8461538461399147]
    print lccrf.q('This is a test sentence with Claude Shannons name in it')
    lccrf.batch_query([x[0] for x in training_examples])

    # So now how to proceed... we need to incorporate these weights
    # into the potential functions somehow...

    # Lets try a really simple sentence with just one word...
    # For the single word we will step through it and
    # we want the lccrf answer and fg answer to be the same...

    # First thing the .query does is build the "g" funcs
    # which closes a sum over the products of feature funcs
    # with the weights for each i being the position
    # of each token ie it creates a separate closure
    # for each position where the position is closed over.
    # Then it calls viterbi with those funcs...
    # the funcs are in a g_ dict
    # example call: g_[t]('__START__', y_, x_seq)
    # returns 3.174603174603173
    # basically it calls the g_ func for each
    # possibility in the output alphabet
    # and then takes the maximum result
    # it then normalizes by the sum of the
    # g_ values above from the *last* token
    # in the sequence to get the actual probs
    # With above weights and input sequence
    # "Shannon" the result is
    # p 3.174603174603173 / (3.174603174603173 + 1.9607843137254881)
    # 0.6181818181818184
    #import ipdb; ipdb.set_trace()
    #print lccrf.q('Shannon')
    # Okay that worked so now lets see if we can get the same result
    # for the factor graph.... remember that first we have to
    # rebuild the factor graph with just one y node....

    # Okay so we have to figure out how to incorporate the
    # weights into the factor graph...
    # It seems that we could maybe put them in
    # the expand_feature_functions_fg....
    import ipdb; ipdb.set_trace()
    generic_um = build_um_from_feature_functions(
        feature_functions_, ('Claude', 'Shannon', 'was', 'a', 'dude'),
        weights, output_alphabet)
    # Just as a test I will manually replace the node
    # functions with dummy functions as they
    # would be for an unrolled lccrf...

    for node in generic_um.nodes:
        if node.name == 'y0':
            node.func = f_y0
        elif node.name == 'y1':
            node.func = f_y1
        elif node.name == 'y2':
            node.func = f_y2
        elif node.name == 'X':
            node.func = f_X
        elif node.name == 'y3':
            node.func = f_y3
        elif node.name == 'y4':
            node.func = f_y4
    import ipdb; ipdb.set_trace()
    generic_fg = generic_um.convert_to_factor_graph()

    jt = generic_um.build_join_tree()

    clique_tree_sum_product(jt, generic_um)

    assignments = jt.assign_clusters(generic_um)

    um_small = build_small_um('Shannon')
    print um_small
    #jt = um_small.build_join_tree()
    fg_small = um_small.convert_to_factor_graph()
    #fg_small.export('fg_small.gv')
    # Ok so the weight is only on the original func it
    # needs to also be on the func
    expanded_feature_functions_fg_small = (
        expand_feature_functions_fg(
            fg_small, feature_functions_,
            ('Shannon',), weights))
    for factor_node in fg_small.factor_nodes():
        print factor_node
        for f in factor_node.expanded_feature_functions:
            print f.__name__, get_args(f), f.original_func.__name__
        print '----------------------------------------'

    # Remember that X should always be a tuple or list
    # NOT just a string....
    f = fg_small.factor_nodes()[0].func
    print f('NAME', ('Shannon',))
    print f('OTHER', ('Shannon',))
    # Ok!!! Woohoo these are now returning
    # the same as the g_ func above
    # ie the non-normalized results...
    # so now lets see what the .q does...
    fg_small.q()
    # Yeah! it works correctly only thing
    # is its still not normalized but that
    # should be easy....
    # Before I normalize though I am now going to
    # see what I get for a two token sentence
    # Actually before that lets see if
    # I can call build the one token version
    # using generic functions....
    generic_um = build_um_from_feature_functions(
        feature_functions_, ('Shannon',), weights, output_alphabet)
    #generic_um.export('generic_um.gv')
    # Okay so the generic um looks correct....
    # Now to look at the converted um to factor graph...
    generic_fg = generic_um.convert_to_factor_graph()
    #generic_fg.export('generic_fg.gv')
    # generic_fg also looks good....
    # Now to attach the feature functions...
    expand_feature_functions_fg(
        generic_fg, feature_functions_, ('Shannon',), weights)

    # Now when we query it we should get the same result...
    import ipdb; ipdb.set_trace()
    generic_fg.q()



    generic_um = build_um_from_feature_functions(
        feature_functions_, ('Claude', 'Shannon', 'was'),
        weights, output_alphabet)
    jt = generic_um.build_join_tree()
    assignments = jt.assign_clusters(generic_um)
    #generic_um.export('generic_um.gv')
    # Okay so the generic um looks correct....
    # Now to look at the converted um to factor graph...
    generic_fg = generic_um.convert_to_factor_graph()
    #generic_fg.export('generic_fg.gv')
    # generic_fg also looks good....
    # Now to attach the feature functions...
    expand_feature_functions_fg(
        generic_fg, feature_functions_, ('Claude', 'Shannon', 'was'), weights)

    # Now when we query it we should get the same result...
    generic_fg.q()


    # Ok this looks like its missing the X domain...
    # and also the y_domains... mmmm
    # Ok sow now I get the same result as the
    # manually built fg with the weird small
    # exception that the X variable in the
    # generic one shows correctly as a tuple
    # while the manually built one does not
    # for some bizarre reason...
    # Anyway this is enough progress for
    # a push...

    # Okay so at this point its probably worthwhile having
    # a repl loop to play through several examples....
    while True:
        X = raw_input('Input Sentence -> ')
        X_seq = tuple(X.split())
        if len(X_seq) > 2:
            import ipdb; ipdb.set_trace()
            print X_seq
        um = build_um_from_feature_functions(
            feature_functions_, X_seq, weights, output_alphabet)
        fg = um.convert_to_factor_graph()
        expand_feature_functions_fg(fg, feature_functions_, X_seq, weights)
        fg.q()
        # Okay this loop works for sequences up to two
        # tokens but not 3 or more, so need to step
        # through, when we have 3 or more there are
        # more than 1 clique so I guess its something to
        # do with the clique interactions...
        # Okay so actually the problem looks like
        # the factor argspec when it combines a plane
        # variable node to a clique variable node
        # the argspec is wrong, interstingly it
        # has the correct call args somehow...
        # So I need to look at the propagation and see
        # where that argspec is messing up....
        # I should how for example the earthquake and/or cancer
        # graphs get it right because they also have
        # combined clique variable nodes...
        # Ok so cancer converted doesnt have any
        # combined variable clique nodes all the sepset
        # intersections just have one variable...
        # Lets try the Huange Darwiche graph....
        # Okay so the Huang Darwiche one has a similar issue...
        # Lets start off by looking at the domains for
        # the combined nodes...
        # Okay so the combined node has a domain
        # consisting of tuples which may be ok...
        # So the argspecs for the factor node
        # neighbours is just the original vars
        # so the argspec doesnt match the domain,
        # so now lets look at how the clique tree
        # sum product does it for the huang_darwiche
        # graph....
        # Okay so the test_clique_tree_huang_darwiche_sum_product
        # works so lets look at the difference....
        # Also we should ensure that the test is actually
        # calling the clique_tree_sum_product routine...
        # okay it is calling the right function...
        # Looks like from initialize_factors in bbn
        # we just have to build the sum_func using
        # the *original* nodes
        # We could also try placing the functions
        # on the original um.... This should
        # be easier???
        # The jt created by the um.build_join_tree
        # actually calls build_join_tree_from_cliques
        # for some reason which is slightly different.
        # It means that calling clique_tree_sum_product
        # doesnt work on the jt, maybe it would work
        # on the clique_tree????
