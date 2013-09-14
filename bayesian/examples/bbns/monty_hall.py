'''The Monty Hall Problem Modelled as a Bayesian Belief Network'''
from bayesian.bbn import *

'''
As BBN:


         GuestDoor     ActualDoor
               \         /
                MontyDoor
                p(M|G,A)


As Factor Graph:

        fGuestDoor               fActualDoor
             |                        |
         GuestDoor                ActualDoor
             |                        |
             +------fMontyDoor--------+
                         |
                     MontyDoor

Now Query: Given Guest chooses door A
and Monty chooses door B, should guest
switch to C or stay with A?

'''


def f_prize_door(prize_door):
    return 1.0 / 3


def f_guest_door(guest_door):
    return 1.0 / 3


def f_monty_door(prize_door, guest_door, monty_door):
    if prize_door == guest_door:
        if prize_door == monty_door:
            return 0
        else:
            return 0.5
    elif prize_door == monty_door:
        return 0
    elif guest_door == monty_door:
        return 0
    return 1


if __name__ == '__main__':

    g = build_bbn(
        f_prize_door,
        f_guest_door,
        f_monty_door,
        domains=dict(
            prize_door=['A', 'B', 'C'],
            guest_door=['A', 'B', 'C'],
            monty_door=['A', 'B', 'C']))
    # Initial Marginals without any knowledge.
    # Observe that the likelihood for
    # all three doors is 1/3.
    print 'Initial Marginal Probabilities:'
    g.q()
    # Now suppose the guest chooses
    # door A and Monty chooses door B.
    # Should we switch our choice from
    # A to C or not?
    # To answer this we "query" the
    # graph with instantiation of the
    # observed variables as "evidence".
    # The likelihood for door C has
    # indeed increased to 2/3 therefore
    # we should switch to door C.
    #print 'Marginals after knowing Guest chose A and Monty chose B.'
    #g.q(guest_door='A', monty_door='B')
