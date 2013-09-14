'''This is the earthquake example from 2.5.1 in BAI'''
from bayesian.factor_graph import *


def f_burglary(burglary):
    if burglary is True:
        return 0.01
    return 0.99


def f_earthquake(earthquake):
    if earthquake is True:
        return 0.02
    return 0.98


def f_alarm(burglary, earthquake, alarm):
    table = dict()
    table['ttt'] = 0.95
    table['ttf'] = 0.05
    table['tft'] = 0.94
    table['tff'] = 0.06
    table['ftt'] = 0.29
    table['ftf'] = 0.71
    table['fft'] = 0.001
    table['fff'] = 0.999
    key = ''
    key = key + 't' if burglary else key + 'f'
    key = key + 't' if earthquake else key + 'f'
    key = key + 't' if alarm else key + 'f'
    return table[key]


def f_johncalls(alarm, johncalls):
    table = dict()
    table['tt'] = 0.9
    table['tf'] = 0.1
    table['ft'] = 0.05
    table['ff'] = 0.95
    key = ''
    key = key + 't' if alarm else key + 'f'
    key = key + 't' if johncalls else key + 'f'
    return table[key]


def f_marycalls(alarm, marycalls):
    table = dict()
    table['tt'] = 0.7
    table['tf'] = 0.3
    table['ft'] = 0.01
    table['ff'] = 0.99
    key = ''
    key = key + 't' if alarm else key + 'f'
    key = key + 't' if marycalls else key + 'f'
    return table[key]


if __name__ == '__main__':
    g = build_graph(
        f_burglary,
        f_earthquake,
        f_alarm,
        f_johncalls,
        f_marycalls)
    g.q()
