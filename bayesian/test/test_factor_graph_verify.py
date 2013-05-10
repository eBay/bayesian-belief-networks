import pytest
from bayesian.factor_graph import *


def pytest_funcarg__x1(request):
    x1 = VariableNode('x1')
    return x1


def pytest_funcarg__x2(request):
    x2 = VariableNode('x2')
    return x2


def pytest_funcarg__fA_node(request):
    
    def fA(x1):
        return 0.5

    fA_node = FactorNode('fA', fA)
    return fA_node


def pytest_funcarg__simple_valid_graph(request):

    def fA(x1):
        return 0.5
    
    fA_node = FactorNode('fA', fA)
    x1 = VariableNode('x1')
    connect(fA_node, x1)
    graph = FactorGraph([fA_node, x1])
    return graph


def pytest_funcarg__graph_with_function_as_node(request):
    '''
    A common error is to instantiate the
    graph with the function instead of
    the function node wrapper.
    '''
    def fA(x1):
        return 0.5
    
    fA_node = FactorNode('fA', fA)
    x1 = VariableNode('x1')

    connect(fA_node, x1)
    graph = FactorGraph([fA, x1])
    return graph
    

class TestVerify():

    def test_verify_variable_node_neighbour_type(self, x1, fA_node):
        connect(fA_node, x1)
        assert fA_node.verify_neighbour_types() is True
        assert x1.verify_neighbour_types() is True

    def test_verify_variable_node_neighbour_type_symmetry(self, x1, fA_node):
        connect(x1, fA_node)
        assert fA_node.verify_neighbour_types() is True
        assert x1.verify_neighbour_types() is True

    def test_verify_variable_node_wrong_neighbour_type(self, x1, x2):
        connect(x1, x2)
        assert x1.verify_neighbour_types() is False
        assert x2.verify_neighbour_types() is False

    def test_nodes_of_correct_type(self, simple_valid_graph):
        assert simple_valid_graph.verify() is True
        
    def test_broken_graph_bad_factor_node(self, graph_with_function_as_node):
        '''
        Make sure exception is raised for 
        broken graph.
        '''
        with pytest.raises(InvalidGraphException):
            graph_with_function_as_node.verify()
