'''Unit tests for the examples in the examples dir.'''
from bayesian.factor_graph import build_graph
from bayesian.factor_graph import make_product_func, make_not_sum_func
from bayesian.examples.factor_graphs.cancer import g as cancer_fg
from bayesian.examples.factor_graphs.cancer import (
    fP, fS, fC, fX, fD)
from bayesian.examples.bbns.happiness import g as happiness_bbn
from bayesian.examples.bbns.happiness import (
    f_coherence, f_difficulty, f_intelligence,
    f_grade, f_sat, f_letter, f_job, f_happy)
from bayesian.bbn import clique_tree_sum_product

from pprint import pprint

'''
Since one of the goals of this package
are to have many working examples its
very important that the examples work
correctly "out of the box".
Please add unit tests for all examples
and give references to their sources.

Note that the identical graph also
appears in test_graph where many more
lower level tests are run. These tests
however import the code directly from
the examples directory.
'''


def pytest_funcarg__cancer_graph(request):
    return cancer_fg


def pytest_funcarg__happiness_bbn(request):
    return happiness_bbn


class TestCancerGraph():

    '''
    See table 2.2 of BAI_Chapter2.pdf
    For verification of results.
    (Note typo in some values)
    '''

    def test_no_evidence(self, cancer_graph):
        '''Column 2 of upper half of table'''
        result = cancer_graph.query()
        assert round(result[('P', 'high')], 3) == 0.1
        assert round(result[('P', 'low')], 3) == 0.9
        assert round(result[('S', True)], 3) == 0.3
        assert round(result[('S', False)], 3) == 0.7
        assert round(result[('C', True)], 3) == 0.012
        assert round(result[('C', False)], 3) == 0.988
        assert round(result[('X', True)], 3) == 0.208
        assert round(result[('X', False)], 3) == 0.792
        assert round(result[('D', True)], 3) == 0.304
        assert round(result[('D', False)], 3) == 0.696

    def test_D_True(self, cancer_graph):
        '''Column 3 of upper half of table'''
        result = cancer_graph.query(D=True)
        assert round(result[('P', 'high')], 3) == 0.102
        assert round(result[('P', 'low')], 3) == 0.898
        assert round(result[('S', True)], 3) == 0.307
        assert round(result[('S', False)], 3) == 0.693
        assert round(result[('C', True)], 3) == 0.025
        assert round(result[('C', False)], 3) == 0.975
        assert round(result[('X', True)], 3) == 0.217
        assert round(result[('X', False)], 3) == 0.783
        assert round(result[('D', True)], 3) == 1
        assert round(result[('D', False)], 3) == 0

    def test_S_True(self, cancer_graph):
        '''Column 4 of upper half of table'''
        result = cancer_graph.query(S=True)
        assert round(result[('P', 'high')], 3) == 0.1
        assert round(result[('P', 'low')], 3) == 0.9
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 0.032
        assert round(result[('C', False)], 3) == 0.968
        assert round(result[('X', True)], 3) == 0.222
        assert round(result[('X', False)], 3) == 0.778
        assert round(result[('D', True)], 3) == 0.311
        assert round(result[('D', False)], 3) == 0.689

    def test_C_True(self, cancer_graph):
        '''Column 5 of upper half of table'''
        result = cancer_graph.query(C=True)
        assert round(result[('P', 'high')], 3) == 0.249
        assert round(result[('P', 'low')], 3) == 0.751
        assert round(result[('S', True)], 3) == 0.825
        assert round(result[('S', False)], 3) == 0.175
        assert round(result[('C', True)], 3) == 1
        assert round(result[('C', False)], 3) == 0
        assert round(result[('X', True)], 3) == 0.9
        assert round(result[('X', False)], 3) == 0.1
        assert round(result[('D', True)], 3) == 0.650
        assert round(result[('D', False)], 3) == 0.350

    def test_C_True_S_True(self, cancer_graph):
        '''Column 6 of upper half of table'''
        result = cancer_graph.query(C=True, S=True)
        assert round(result[('P', 'high')], 3) == 0.156
        assert round(result[('P', 'low')], 3) == 0.844
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 1
        assert round(result[('C', False)], 3) == 0
        assert round(result[('X', True)], 3) == 0.9
        assert round(result[('X', False)], 3) == 0.1
        assert round(result[('D', True)], 3) == 0.650
        assert round(result[('D', False)], 3) == 0.350

    def test_D_True_S_True(self, cancer_graph):
        '''Column 7 of upper half of table'''
        result = cancer_graph.query(D=True, S=True)
        assert round(result[('P', 'high')], 3) == 0.102
        assert round(result[('P', 'low')], 3) == 0.898
        assert round(result[('S', True)], 3) == 1
        assert round(result[('S', False)], 3) == 0
        assert round(result[('C', True)], 3) == 0.067
        assert round(result[('C', False)], 3) == 0.933
        assert round(result[('X', True)], 3) == 0.247
        assert round(result[('X', False)], 3) == 0.753
        assert round(result[('D', True)], 3) == 1
        assert round(result[('D', False)], 3) == 0


class TestHappinessGraph(object):

    def test_get_join_tree(self, happiness_bbn):
        jt = happiness_bbn.build_join_tree()
        # Great! The default jt seems
        # to be the same one that is on page 21
        # of http://webdocs.cs.ualberta.ca/~greiner/
        # C-651/SLIDES/MB03_CliqueTrees.pdf

        # Now if we want to compute the
        # marginal for only one variable,
        # in the example this is J (job)
        # then we choose as root node, a
        # clique that contains J. In the
        # example they choose GJSL as the
        # 'root' clique.

        # Since there are multiple valid
        # assignments we will manually
        # set the assignments here to
        # correspond to the assignments
        # in the example on page 23.
        # Actually we have to assign the
        # BBN nodes, not the functions!!!
        clique_nodes = sorted(jt.clique_nodes, key=lambda x: sorted(x.clique.nodes, key=lambda x:x.name))
        cd_clique = [clique for clique in clique_nodes if clique.name == 'f_difficulty_f_coherence'][0]
        gid_clique = [clique for clique in clique_nodes if clique.name == 'f_grade_f_difficulty_f_intelligence'][0]
        hgj_clique = [clique for clique in clique_nodes if clique.name == 'f_grade_f_job_f_happy'][0]
        gsi_clique = [clique for clique in clique_nodes if clique.name == 'f_grade_f_intelligence_f_sat'][0]
        gjsl_clique = [clique for clique in clique_nodes if clique.name == 'f_grade_f_sat_f_job_f_letter'][0]



        fg = happiness_bbn.convert_to_factor_graph()
        #import ipdb; ipdb.set_trace()
        # Check that the eliminated
        # var from cd to gid is c...
        elim_vars = cd_clique.get_eliminate_vars(gid_clique)
        assignments = jt.assign_clusters(happiness_bbn)
        cd_clique.initialize_factors(assignments[cd_clique])
        import ipdb; ipdb.set_trace()
        # Now we need to construct a message
        # to the target...
        # Step 1 slide 25
        target = cd_clique.get_target()
        message = cd_clique.make_message(target)
        target.received_messages[cd_clique.name] = message

        # Step 2
        target = gid_clique.get_target()
        gid_clique.initialize_factors(assignments[gid_clique])
        message = gid_clique.make_message(target)
        target.received_messages[gid_clique.name] = message

        # Step 3
        target = gsi_clique.get_target()
        gsi_clique.initialize_factors(assignments[gsi_clique])
        message = gsi_clique.make_message(target)
        target.received_messages[gsi_clique.name] = message

        # Step 4
        target = hgj_clique.get_target()
        hgj_clique.initialize_factors(assignments[hgj_clique])
        message = hgj_clique.make_message(target)
        target.received_messages[hgj_clique.name] = message

        # Step 5
        gjsl_clique.initialize_factors(assignments[gjsl_clique])
        final_func = make_product_func(
            [gjsl_clique.potential_func] +
            gjsl_clique.received_messages.values())
        import ipdb; ipdb.set_trace()
        final_func = make_not_sum_func(final_func, 'j')
        #### YESSSSSS!!!!!! ######
        # Now to complete the full propagation....

def test_clique_tree_sum_product(happiness_bbn):
    clique_tree = happiness_bbn.build_join_tree()
    import ipdb; ipdb.set_trace()
    #result = clique_tree_sum_product(clique_tree, happiness_bbn)
    #print result
    happiness_bbn.inference_method = 'clique_tree_sum_product'
    happiness_bbn.q()
