from __future__ import division
'''Simple Example Containing A Cycle'''

from bayesian.factor_graph import *


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


def little_bool(var):
    return str(var).lower()[0]


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
    key = ''
    key = key + little_bool(forecast)
    key = key + little_bool(rain)
    return table[key]


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
    key = ''
    key = key + little_bool(forecast)
    key = key + little_bool(rain)
    key = key + little_bool(walk)
    return table[key]


forecast = VariableNode('forecast')
rain = VariableNode('rain')
walk = VariableNode('walk')

f_forecast_node = FactorNode('f_forecast', f_forecast)
f_rain_node = FactorNode('f_rain', f_rain)
f_walk_node = FactorNode('f_walk', f_walk)


connect(f_forecast_node, forecast)
connect(f_rain_node, [forecast, rain])
connect(f_walk_node, [forecast, rain, walk])

graph = FactorGraph([
                    forecast,
                    rain,
                    walk,
                    f_forecast_node,
                    f_rain_node,
                    f_walk_node])


def tabulate(counts, normalizer):
    table = PrettyTable(['Variable', 'Value', 'p'])
    table.align = 'l'
    deco = [(k, v) for k, v in counts.items()]
    deco.sort()
    for k, v in deco:
        if k[1] is not False:
            table.add_row(list(k) + [v / normalizer])
    print table


if __name__ == '__main__':
    graph.verify()
    print graph.get_sample()

    n = 10000
    counts = defaultdict(int)
    for i in range(0, n):
        table = PrettyTable(['Variable', 'Value'])
        table.align = 'l'
        sample = graph.get_sample()
        for var in sample:
            key = (var.name, var)
            counts[key] += 1

    from pprint import pprint
    pprint(counts)
    print 'Sampled:'
    tabulate(counts, n)
