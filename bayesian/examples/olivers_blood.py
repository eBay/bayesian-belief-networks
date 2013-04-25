from bayesian.factor_graph import *

'''
This example is take from page 43 of ThinkBayes.pdf

The prolem is stated as follows:

    Two people have left traces of their own blood
    at the scene of a crime. A suspect, Oliver, is
    tested and found to have type 'O' blood. The
    blood groups of the two traces are found to be
    of type 'O' (a common type in the local
    population, having frequency 60%) and of
    type 'AB' (a rare type, with frequency 1%).
    Do these data [the traces found at the scene]
    give evidence in favor of the proposition that
    Oliver was one of the people [who left blood at the scene]?

Implementing this as a factor graph is trivial
as it has only 2 nodes.
                                         
'''

def f_oliver(oliver):
    table=dict()
    if oliver.value is True:
        return 0.01
    else:
        return 2 * 0.6 * 0.01
    
f_oliver_node = FactorNode('f_oliver', f_oliver)
oliver = VariableNode('oliver')

connect(f_oliver_node, oliver)

graph = FactorGraph([
        oliver,
        f_oliver_node])

graph.verify()

if __name__ == '__main__':
    graph.propagate()
    graph.status(omit=[])
        


