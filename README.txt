Small Bayesian Belief Propagation Framework using
Sum-Product Algorithm on Factor Graphs.

Todo:

1) Change requirement for PMFs to use .value
2) Make the storage and retrieval of pre-generated 
   samples from SQLite files transparent
3) Rename VariableNode to DiscreteVariableNode
4) Add GuassianVariableNode for continuous variables
5) Deprecate "status" method in favour of q

Unit Tests:

In order to run the unit tests you need the pytest framwork.
This can be installed in a virtuanlenv with:

$ pip install pytest

To run the tests in a development environment:

$ PYTHONPATH=. py.test


Resources:
http://www.fil.ion.ucl.ac.uk/spm/course/slides10-vancouver/08_Bayes.pdf
http://www.ee.columbia.edu/~vittorio/Lecture12.pdf
http://www.csse.monash.edu.au/bai/book/BAI_Chapter2.pdf
