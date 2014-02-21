'''Example from Wikipedia: http://en.wikipedia.org/wiki/Bayesian_network'''

from bayesian.bbn import *
from bayesian.utils import make_key


def f_rain(rain):
    if rain is True:
        return 0.2
    return 0.8


def f_sprinkler(rain, sprinkler):
    if rain is False and sprinkler is True:
        return 0.4
    if rain is False and sprinkler is False:
        return 0.6
    if rain is True and sprinkler is True:
        return 0.01
    if rain is True and sprinkler is False:
        return 0.99


def f_grass_wet(sprinkler, rain, grass_wet):
    table = dict()
    table['fft'] = 0.0
    table['fff'] = 1.0
    table['ftt'] = 0.8
    table['ftf'] = 0.2
    table['tft'] = 0.9
    table['tff'] = 0.1
    table['ttt'] = 0.99
    table['ttf'] = 0.01
    return table[make_key(sprinkler, rain, grass_wet)]


if __name__ == '__main__':
    g = build_bbn(
        f_rain,
        f_sprinkler,
        f_grass_wet)
