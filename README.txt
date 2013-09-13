Pythonic Bayesian Belief Network Framework

Allows creation of BBNs with pure Python
functions. Currently three different inference
methods are supported with more to come:

- Message Passing and the Junction Tree Algorithm
- The Sum Product Algorithm
- MCMC Sampling for approximate inference


Other Features

- Automated conversion to Junction Trees
- Inference of Graph Structure from Mass Functions
- Automatic conversion to Factor Graphs
- Seemless storage of samples for future use
- Exact inference on cyclic graphs
- Export of graphs to GraphViz (dot language) format

Please see the short tutorial in the docs/tutorial directory
for a short introduction on how to build a BBN.
There are also many examples in the examples directory.


Installation

$ python setup.py install
$ pip install -r requirements.txt

Building The Tutorial

$ pip install sphinx
$ cd docs/tutorial
$ make clean
$ make html

Unit Tests:

In order to run the unit tests you need the pytest framwork.
This can be installed in a virtuanlenv with:

$ pip install pytest

To run the tests in a development environment:

$ PYTHONPATH=. py.test bayesian/test


Todo:

1) Change requirement for PMFs to use .value
2) Rename VariableNode to DiscreteVariableNode
3) Add GuassianVariableNode for continuous variables


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
