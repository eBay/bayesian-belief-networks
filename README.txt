Pythonic Bayesian Belief Network Framework
------------------------------------------

Allows creation of Bayesian Belief Networks
and other Graphical Models with pure Python
functions. Where tractable exact inference
is used. Currently four different inference
methods are supported with more to come.

Graphical Models Supported
--------------------------

- Bayesian Belief Networks with discrete variables
- Gaussian Bayesian Networks with continous variables having gaussian distributions


Inference Engines
-----------------

- Message Passing and the Junction Tree Algorithm
- The Sum Product Algorithm
- MCMC Sampling for approximate inference
- Exact Propagation in Gaussian Bayesian Networks


Other Features
--------------

- Automated conversion to Junction Trees
- Inference of Graph Structure from Mass Functions
- Automatic conversion to Factor Graphs
- Seemless storage of samples for future use
- Exact inference on cyclic graphs
- Export of graphs to GraphViz (dot language) format
- Discrete and Continuous Variables (with some limitations)
- Minimal dependancies on non-standard library modules.

Please see the short tutorial in the docs/tutorial directory
for a short introduction on how to build a Bayesian Belief Network.
There are also many examples in the examples directory.


Installation
------------

$ python setup.py install
$ pip install -r requirements.txt

Building The Tutorial

$ pip install sphinx
$ cd docs/tutorial
$ make clean
$ make html

Unit Tests:

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
http://arxiv.org/pdf/1301.7394v1.pdf

Junction Tree Algorithm:
http://www.inf.ed.ac.uk/teaching/courses/pmr/docs/jta_ex.pdf
http://ttic.uchicago.edu/~altun/Teaching/CS359/junc_tree.pdf
http://eniac.cs.qc.cuny.edu/andrew/gcml/lecture10.pdf
http://leo.ugr.es/pgm2012/proceedings/eproceedings/evers_a_framework.pdf

Guassian Bayesian Networks:
http://www.cs.ubc.ca/~murphyk/Teaching/CS532c_Fall04/Lectures/lec17x4.pdf
http://webdocs.cs.ualberta.ca/~greiner/C-651/SLIDES/MB08_GaussianNetworks.pdf
http://people.cs.aau.dk/~uk/papers/castillo-kjaerulff-03.pdf
