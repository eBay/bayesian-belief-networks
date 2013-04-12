'''Example of Sprinkler, Rain and Wet Grass from Wikipedia'''
import sys

from bayesian import RandomVariable, log
from bayesian import Prob, Joint, Conditional, make_prob, expand_arguments
from bayesian import joint_to_conditional, conditional_to_joint, marginalize
from bayesian import Node, who_can_answer, Division, Mult, get_all_variables


def rain_func(g, s, x):
    if x == True:
        return 0.2
    elif x == False:
        return 0.8
    raise 'pdf cannot resolve for %s' % x


def sprinkler_func(g, s, r):
    if r == False and s == True:
        return 0.4
    if r == False and s == False:
        return 0.6
    if r == True and s == True:
        return 0.01
    if r == True and s == False:
        return 0.99


def grass_wet_func(g, s, r):
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


#@log
def resolve(question, answerable, backref):
    if question == []:
        return []
    if type(question) == float:
        return question
    if type(question) == Conditional:
        if question in answerable:
            return question, answerable[question].func
        question = conditional_to_joint(question)
        return resolve(question, answerable, backref)
    if type(question) == Prob:
        if question in answerable:
            return answerable[question].func(question.x)
        else:
            return 'I give up'
    if type(question) == Division:
        import ipdb; ipdb.set_trace()
        if type(question.numerator) == float and type(question.denominator) == float:
            return question.numerator / question.denominator
        return Division(resolve(question.numerator, answerable, backref), resolve(question.denominator, answerable, backref))
    if type(question) == Mult:
        if type(question.x) == float and type(question.y) == float:
            return question.x * question.y
        return Mult(resolve(question.x, answerable, backref), resolve(question.y, answerable, backref))
    if type(question) == Joint:
        # When we have a joint there are TWO equivalent conditionals
        # We need to make sure that the one we generate is solvable
        # ie if we have a jP(R,G) there is no point getting
        # the expansion: cP(R|G) / R
        #import ipdb; ipdb.set_trace()
        if not backref[question.vars[0].name].parents:
            reverse_joint = Joint(question.vars[::-1])
            question = joint_to_conditional(reverse_joint)
        else:
            question = joint_to_conditional(question)
        return resolve(question, answerable, backref)
    raise "Help!"



if __name__ == '__main__':
    a = RandomVariable('A')
    b = RandomVariable('B')
    c = RandomVariable('C')
    d = Joint([a, b, c])

    r = RandomVariable('R')
    rain_node = Node(r, rain_func, parents=[])

    s = RandomVariable('S')
    sprinkler_node = Node(s, sprinkler_func, parents = [rain_node])
    

    g = RandomVariable('G')
    grass_wet_node = Node(g, grass_wet_func, parents = [sprinkler_node, rain_node])

    network = [rain_node, sprinkler_node, grass_wet_node]

    backref = dict(
       R = rain_node,
       G = grass_wet_node,
       S = sprinkler_node)
    #print get_all_variables([rain_node, sprinkler_node, grass_wet_node])
    #sys.exit(0)

    # Try to answer the question from Wikipedia:
    # "What is the probability that it is raining, given the grass is wet?"

    # The above question can be asked like this in my framework:

    #answerable = {}
    #for node in network:
    #    node_answers = node.can_answer()
    #    print node, node_answers
    #    for node_answer in node_answers:
    #        answerable[node_answer] = node


    #r.val = True
    #g.val = None
    #s.val = None
    #question = Prob(r)
    #print 'Question: ', question
    #print 'Answer: ', resolve(question, answerable, backref)


    #r.val = False
    #g.val = None
    #s.val = None
    #question = Prob(r)
    #print 'Question: ', question
    ##print 'Answer: ', resolve(question, answerable, backref)


    #r.val = True
    #g.val = None
    #s.val = True
    #question = Conditional(s, [r])
    #print 'Question: ', question
    #print 'Answer: ', resolve(question, answerable, backref)


    r.val = False
    g.val = False
    s.val = False
    question = Conditional(r, [g])
    print 'Question: ', question
    question = conditional_to_joint(question)
    print question

    
    numerator = question.numerator
    print 'Numerator before transposing: ', numerator
    # Now whenever we have a joint we want to re-order it in the best way
    # for the network, in our case its ['G','S', 'R']
    numerator_args = dict(G=None, S=None, R = None)
    for var in numerator.vars:
        numerator_args[var.name] = var.val
        
    

    numerator_args = [numerator_args[x] for x in ['G', 'S', 'R']]
    print 'Numerator args:', numerator_args


    denominator = question.denominator

    print 'Denominator before transposing: ', denominator

    denominator_args = dict(G=None, S=None, R = None)

    if type(denominator) == Joint:
        for var in denominator.vars:
            denominator_args[var.name] = var.val
    else:
        denominator_args[denominator.x.name] = denominator.x.val

    denominator_args = [denominator_args[x] for x in ['G', 'S', 'R']]
    print 'Denominator args:', denominator_args

    # Now for each of the numerator and denominator we
    # need to marginalize the joint of ALL variables that are not 
    # constrained.

    #numerator_combos = expand_arguments(numerator_args)
    #print numerator_combos

    

    num = marginalize(lambda g, s, r: grass_wet_func(g,s,r) * rain_func(g,s,r) * sprinkler_func(g,s,r), numerator_args)
    den = marginalize(lambda g, s, r: rain_func(g,s,r) * grass_wet_func(g,s,r) * sprinkler_func(g,s,r), denominator_args)

    print num / den

    #denominator_combos = expand_arguments(denominator_args)
    #print denominator_combos

    #template = joint_to_conditional(Joint([g, s, r]))
    #print template

    #print 'Answer: ', resolve(question, answerable, backref)
    
