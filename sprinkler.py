'''Example of Sprinkler, Rain and Wet Grass from Wikipedia'''
import sys

from bayesian import RandomVariable, log
from bayesian import Prob, Joint, Conditional, make_prob
from bayesian import joint_to_conditional, conditional_to_joint
from bayesian import Node, who_can_answer, Division, Mult


def rain_func(x):
    assert type(x) == RandomVariable
    if x.val == True:
        return 0.2
    elif x.val == False:
        return 0.8
    raise 'pdf cannot resolve for %s' % x


def sprinkler_func(r, s):
    if r.val == False and s.val == True:
        return 0.4
    if r.val == False and s.val == False:
        return 0.6
    if r.val == True and s.val == True:
        return 0.01
    if r.val == True and s.val == False:
        return 0.99


def grass_wet_func(s, r, g):
    ''' 
    This needs to be a joint probability distribution
    over the inputs and the node itself
    '''
    table = dict()
    table['fft'] = 0.0
    table['fff'] = 1.0
    table['ftt'] = 0.8
    table['ftf'] = 0.2
    table['tft'] = 0.9
    table['tff'] = 0.1
    table['ttt'] = 0.99
    table['ttf'] = 0.01
    key = ''
    key = key + 't' if s else key + 'f'
    key = key + 't' if r else key + 'f'
    key = key + 't' if g else key + 'f'
    return table[key]

def resolvable(question, answerable):
    for answer in answerable:
        if question == answerable:
            return True
    return False


@log
def resolve(question, answerable):
    if question == []:
        return []
    if type(question) == float:
        return question
    if type(question) == Conditional:
        if question in answerable:
            return 'Answered!'
        question = conditional_to_joint(question)
        return resolve(question, answerable)
    if type(question) == Prob:
        if question in answerable:
            return answerable[question].func(question.x)
        else:
            return 'I give up'
    if type(question) == Division:
        if type(question.numerator) == float and type(question.denominator) == float:
            return question.numerator / question.denominator
        return Division(resolve(question.numerator, answerable), resolve(question.denominator, answerable))
    if type(question) == Mult:
        if type(question.x) == float and type(question.y) == float:
            return question.x * question.y
        return Mult(resolve(question.x, answerable), resolve(question.y, answerable))
    if type(question) == Joint:
        question = joint_to_conditional(question)
        return resolve(question, answerable)
    raise "Help!"



if __name__ == '__main__':
    a = RandomVariable('A')
    b = RandomVariable('B')
    c = RandomVariable('C')
    d = Joint([a, b, c])

    rain_var = RandomVariable('R')
    rain_node = Node(rain_var, rain_func, parents=[])

    sprinkler_var = RandomVariable('S')
    sprinkler_node = Node(sprinkler_var, sprinkler_func, parents = [rain_node])

    grass_wet_var = RandomVariable('G')
    grass_wet_node = Node(grass_wet_var, grass_wet_func, parents = [sprinkler_node, rain_node])

    network = [rain_node, sprinkler_node, grass_wet_node]

    # Try to answer the question from Wikipedia:
    # "What is the probability that it is raining, given the grass is wet?"

    # The above question can be asked like this in my framework:

    answerable = {}
    for node in network:
        node_answers = node.can_answer()
        print node, node_answers
        for node_answer in node_answers:
            answerable[node_answer] = node


    r = RandomVariable('R', True)
    g = RandomVariable('G')
    s = RandomVariable('S')
    question = Prob(r)
    print 'Question: ', question
    print 'Answer: ', resolve(question, answerable)


    r = RandomVariable('R', False)
    g = RandomVariable('G')
    s = RandomVariable('S')
    question = Prob(r)
    print 'Question: ', question
    print 'Answer: ', resolve(question, answerable)


    r = RandomVariable('R', True)
    g = RandomVariable('G')
    s = RandomVariable('S', True)
    question = Conditional(s, [r])
    print 'Question: ', question
    import ipdb; ipdb.set_trace()
    print 'Answer: ', resolve(question, answerable)

