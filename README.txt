Small Bayesian Belief Propagation Framework using
Sum-Product Algorithm on Factor Graphs.

Todo:

1) Change requirement for PMFs to use .value
2) Make the storage and retrieval of pre-generated
   samples from SQLite files transparent
3) Rename VariableNode to DiscreteVariableNode
4) Add GuassianVariableNode for continuous variables
5) Deprecate "status" method in favour of q
6) Allow build_graph to take a single parameter
   being a list of functions so as to overcome
   the 255 argument limit in Python

Unit Tests:

In order to run the unit tests you need the pytest framwork.
This can be installed in a virtuanlenv with:

$ pip install pytest

To run the tests in a development environment:

$ PYTHONPATH=. py.test bayesian/test


Resources
=========

http://www.fil.ion.ucl.ac.uk/spm/course/slides10-vancouver/08_Bayes.pdf
http://www.ee.columbia.edu/~vittorio/Lecture12.pdf
http://www.csse.monash.edu.au/bai/book/BAI_Chapter2.pdf
http://www.comm.utoronto.ca/frank/papers/KFL01.pdf
http://www.snn.ru.nl/~bertk/ (Many real-world examples listed)
http://www.cs.ubc.ca/~murphyk/Bayes/Charniak_91.pdf
http://www.sciencedirect.com/science/article/pii/S0888613X96000692

Junction Tree Algorithm:
http://www.inf.ed.ac.uk/teaching/courses/pmr/docs/jta_ex.pdf
http://ttic.uchicago.edu/~altun/Teaching/CS359/junc_tree.pdf
http://eniac.cs.qc.cuny.edu/andrew/gcml/lecture10.pdf
http://leo.ugr.es/pgm2012/proceedings/eproceedings/evers_a_framework.pdf
