import re


def parse(filename):
    """Parses the .bif file with the
    given name (exclude the extension from the argument)
    and produces a python file with create_graph() and create_bbn() functions
    to return the network. The name of the module is returned.
    The bbn/factor_graph objects will have the filename as their model name."""

    # Setting up I/O
    module_name = filename+'_bn'
    outfile = open(module_name + '.py', 'w')

    def write(s):
        outfile.write(s+"\n")
    infile = open(filename+'.bif')
    infile.readline()
    infile.readline()

    # Import statements in the produced module
    write("""from bayesian.factor_graph import *
from bayesian.bbn import *
""")

    # Regex patterns for parsing
    variable_pattern = re.compile(r"  type discrete \[ \d+ \] \{ (.+) \};\s*")
    prior_probability_pattern_1 = re.compile(
        r"probability \( ([^|]+) \) \{\s*")
    prior_probability_pattern_2 = re.compile(r"  table (.+);\s*")
    conditional_probability_pattern_1 = (
        re.compile(r"probability \( (.+) \| (.+) \) \{\s*"))
    conditional_probability_pattern_2 = re.compile(r"  \((.+)\) (.+);\s*")

    variables = {}  # domains
    functions = []  # function names (nodes/variables)

    # For every line in the file
    while True:
        line = infile.readline()

        # End of file
        if not line:
            break

        # Variable declaration
        if line.startswith("variable"):
            match = variable_pattern.match(infile.readline())

            # Extract domain and place into dictionary
            if match:
                variables[line[9:-3]] = match.group(1).split(", ")
            else:
                raise Exception("Unrecognised variable declaration:\n" + line)
            infile.readline()

        # Probability distribution
        elif line.startswith("probability"):

            match = prior_probability_pattern_1.match(line)
            if match:

                # Prior probabilities
                variable = match.group(1)
                function_name = "f_" + variable
                functions.append(function_name)
                line = infile.readline()
                match = prior_probability_pattern_2.match(line)
                write("""dictionary_%(var)s = %(dict)s

def %(function)s(%(var)s):
    return dictionary_%(var)s[%(var)s]
"""
                      % {
                          'function': function_name,
                          'var': variable,
                          'dict': str(dict(
                              zip(variables[variable],
                                  map(float, match.group(1).split(", ")))))
                      }
                )
                infile.readline()  # }

            else:
                match = conditional_probability_pattern_1.match(line)
                if match:

                    # Conditional probabilities
                    variable = match.group(1)
                    function_name = "f_" + variable
                    functions.append(function_name)
                    given = match.group(2)
                    dictionary = {}

                    # Iterate through the conditional probability table
                    while True:
                        line = infile.readline()  # line of the CPT
                        if line == '}\n':
                            break
                        match = conditional_probability_pattern_2.match(line)
                        given_values = match.group(1).split(", ")
                        for value, prob in zip(
                                variables[variable],
                                map(float, match.group(2).split(", "))):
                            dictionary[tuple(given_values + [value])] = prob
                    write("""dictionary_%(var)s = %(dict)s
def %(function)s(%(given)s, %(var)s):
    return dictionary_%(var)s[(%(given)s, %(var)s)]
"""
                          % {'function': function_name,
                             'given': given,
                             'var': variable,
                             'dict': str(dictionary)})
                else:
                    raise Exception(
                        "Unrecognised probability declaration:\n" + line)

    write("""functions = %(funcs)s
domains_dict = %(vars)s

def create_graph():
    g = build_graph(
        *functions,
        domains = domains_dict)
    g.name = '%(name)s'
    return g

def create_bbn():
    g = build_bbn(
        *functions,
        domains = domains_dict)
    g.name = '%(name)s'
    return g
"""
          % {
              'funcs': ''.join(c for c in str(functions) if c not in "'\""),
              'vars': str(variables), 'name': filename})
    outfile.close()
    return module_name
