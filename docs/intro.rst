============
Introduction
============

Welcome to the Bayesian package.
This Bayesian package allows one to create
Bayesian Belief Networks using pure Python functions.
As there are many good introductions to
Bayesian Belief Networks, this document will
not cover the theory of them, but rather will
assume some familiarity by the user.
Rather this documentation will seek to
illustrate how Bayesian Belief Networks
can be constructed using pure Python functions.

=============
Factorization
=============
Throughout this document we will use *factor* to
refer simultaneously to a Python function (or callable)
that represents a factor in a Bayseian Belief Network.
If you have read one or more of the excellent introductory
texts you will by now have realized that the key to
BBNs is to *factorize* the full Joint Probability distribution
function of the model variables, into the relevant factors.
Essentially a BBN is a way to express this factorization.
By providing a framework which takes care of the construction
of the graph, and the inference on the graph, this package
will allow the developer of the model to concentrate
on the details of the individual factors themselves.
This is best illustrated with a simple example,
probably the most famous example in Bayseian probability texts.

======================
The Monty Hall Problem
======================
Depending on what you believe, the story goes like this:

	  A certain Monty Hall, hosted a television game
	  show, in which a guest is asked to pick one
	  of three closed doors, behind one of which
	  is a valuable prize. The host, Monty, knows
	  beforehand which of the three doors obscures
	  the prize. After the guest has made her choice,
	  Monty then opens one of the two remaining doors.
	  (Monty will never however open *the* door that
	  has the prize behind it)
	  After opening the door, Monty will then offer
	  the guest the opportunity to *change* her choice
	  of door, or to stick with the original one she
	  chose. Assuming that the guest would very much
	  like to go home with the prize, should she
	  change her choice of door after observing
	  that whichever door Monty opened clearly does
	  *not* reveal the prize?

Lets begin modelling this problem. Firstly lets
think about the events occuring during this game.
(While it is not strictly necessary to think of
the events in chronological order, it helps to
illustrate this example.)
Firstly, before the guest arrives presumable, the
prize is hidden behind one of the doors. Lets
model this with a random variable we will call
'PrizeDoor'. Clearly this variable can take
on one of three values, 'A', 'B', or 'C'.
Now the guest picks a door, lets introduce
a second variable and call it 'GuestDoor'.
Once again 'GuestDoor' can also take on
one of the three values 'A', 'B', or 'C'.
Now Monty selects one of the doors, and again,
this could be 'A' or 'B' or 'C'.
When constructing a model, we need to
think about each *factor* as introducing a
new variable into the graph.
Thus lets start with the first chronological
event, the hiding of the prize.
Assuming that the prize is hidden behind
any of the three doors with equal likelyhood, then
the chance of the show organizers having selected
the value of 'A', 'B' or 'C' would be equal at
1/3. We will now introduce our first Python
function which will be used to model the game:

::
    def f_prize_door(prize_door):
        return 1.0 / 3


    def f_guest_door(guest_door):
        return 1.0 / 3


    def f_monty_door(prize_door, guest_door, monty_door):
        if prize_door.value == guest_door.value:
            if prize_door.value == monty_door.value:
                return 0
            else:
                return 0.5
        elif prize_door.value == monty_door.value:
            return 0
        elif guest_door.value == monty_door.value:
            return 0
        return 1

There are a few things to note here:
The function names and arguments will be introspected by
the module to construc the network.
As of now, because Python does not support dynamic creation
of variables with specific names, variables passed to
factor functions need to support a "value" attribute
which contains the value the variable is bound too.

There are further examples in the examples subdirectory.
