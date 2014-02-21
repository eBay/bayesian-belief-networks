import bif_parser
from time import time
from prettytable import *

# Perform exact and/or persistent sampling
# inference on a given .bif file,
# showing the time taken and the convergence
# of probability in the case of increasing samples

if __name__ == '__main__':

    # Name of .bif file
    name = 'insurance'

    # (Variable, Value) pair in marginals table to focus on
    key = ('RuggedAuto', 'Football')

    start = time()
    module_name = bif_parser.parse(name)
    print str(time()-start) + "s to parse .bif file into python module"
    start = time()
    module = __import__(module_name)
    print str(time()-start) + "s to import the module"
    start = time()
    fg = module.create_graph()
    print str(time()-start) + "s to create factor graph"
    start = time()
    bg = module.create_bbn()
    print str(time()-start) + "s to create bayesian network"

    # Methods of inference to demonstrate
    exact = True
    sampling = True

    if exact:
        start = time()
        if not sampling:

            # Set exact=True, sampling=False to
            # just show the exact marginals table
            # and select a key of interest
            bg.q()
        else:
            print 'Exact probability:', bg.query()[key]
        print 'Time taken for exact query:', time()-start

    if sampling:
        fg.inference_method = 'sample_db'

        table = PrettyTable(["Number of samples",
                             "Time to generate samples",
                             "Time to query", "Probability",
                             "Difference from previous"])

        for power in range(10):
            n = 2**power
            fg.n_samples = n
            start = time()
            fg.generate_samples(n)
            generate_time = time() - start
            start = time()
            q = fg.query()
            query_time = time() - start
            p = q[key]
            diff = "" if power == 0 else abs(p-prev_p)
            prev_p = p
            table.add_row([n, generate_time, query_time, p, diff])

        print table
