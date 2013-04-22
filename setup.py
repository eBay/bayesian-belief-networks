from distutils.core import setup

setup(
    name='Bayesian',
    version='0.1.0',
    author='Neville Newey',
    author_email='nn@theneweys.org',
    packages=['bayesian', 'bayesian.test', 'bayesian.examples'],
    license='LICENSE.txt',
    description='Small Bayesian Inference Engine using Factor Graphs.',
    long_description=open('README.txt').read(),
    install_requires=[],
)
