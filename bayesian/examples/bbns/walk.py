'''Simple Example Containing A Cycle'''
from __future__ import division

from bayesian.bbn import *
from bayesian.utils import make_key

'''

                          Rain Forecast
                              |
                 +------------+
                 |            |
               Rain           |
                 |            |
                 +---------+  |
                           |  |
                           Walk


Our decision to go for a walk is based on two factors,
the forecast for rain and on actual rain observed.

'''


def f_forecast(forecast):
    if forecast is True:
        return 0.6
    return 0.4


def f_rain(forecast, rain):
    table = dict()
    table['tt'] = 0.95
    table['tf'] = 0.05
    table['ft'] = 0.1
    table['ff'] = 0.9
    return table[make_key(forecast, rain)]


def f_walk(forecast, rain, walk):
    table = dict()
    table['fff'] = 0.01
    table['fft'] = 0.99
    table['ftf'] = 0.99
    table['ftt'] = 0.01
    table['tff'] = 0.8
    table['tft'] = 0.2
    table['ttf'] = 0.999
    table['ttt'] = 0.001
    return table[make_key(forecast, rain, walk)]


if __name__ == '__main__':
    g = build_bbn(
        f_forecast,
        f_rain,
        f_walk)
    g.q()
