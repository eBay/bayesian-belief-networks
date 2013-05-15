from __future__ import division
'''Basic Stats functionality'''

import math
from collections import defaultdict

from prettytable import PrettyTable


class Vector(object):

    def __init__(self, l):
        self.l = l

    @property
    def mean(self):
        return sum(self.l) / len(self.l)

    @property
    def median(self):
        l = self.l[:]
        l.sort()
        mid = int(float(len(l)) / 2)
        if len(l) % 2 == 1:
            return l[mid]
        else:
            v = Vector(l[mid - 1:mid])
            return v.mean

    @property
    def mode(self):
        '''
        NB: For now we are always
        returning only one mode
        so if the sample is multimodal
        this is not reliable
        '''
        l = self.l[:]
        counts = defaultdict(int)
        for x in l:
            counts[x] += 1
        deco = [(k, v) for k, v in counts.items()]
        deco.sort(reverse=True, key=lambda x: x[1])
        return deco[0][0]

    @property
    def population_std_dev(self):
        return math.sqrt(self.population_variance)

    @property
    def std_dev(self):
        '''Corrected sample standard deviation.'''
        return math.sqrt(self.variance)

    @property
    def population_variance(self):
        mu = self.mean
        sumsq = sum([math.pow(x - mu, 2) for x in self.l])
        return sumsq / len(self.l)

    @property
    def variance(self):
        '''Corrected (unbiased) sample variance'''
        mu = self.mean
        sumsq = sum([math.pow(x - mu, 2) for x in self.l])
        return sumsq / (len(self.l) - 1)

    @property
    def mean_absolute_deviation(self):
        '''Mean of absolute differences to mean'''
        mu = self.mean
        return sum([abs(x - mu) for x in self.l]) / len(self.l)

    @property
    def median_absolute_deviation(self):
        '''Mean of absolute differences to median'''
        mu = self.median
        return sum([abs(x - mu) for x in self.l]) / len(self.l)

    @property
    def mode_absolute_deviation(self):
        '''Mean of absolute differences to a mode*'''
        mu = self.mode
        return sum([abs(x - mu) for x in self.l]) / len(self.l)

    def describe(self):
        tab = PrettyTable(['Property', 'value'])
        tab.align['Property'] = 'l'
        tab.align['value'] = 'r'
        tab.add_row(['Total Numbers', len(self.l)])
        tab.add_row(['Mean', self.mean])
        tab.add_row(['Median', self.median])
        tab.add_row(['Mode*', self.mode])
        tab.add_row(['Sample Standard Deviation', self.std_dev])
        tab.add_row(['Sample Variance', self.variance])
        tab.add_row(['Populatoin Standard Deviation',
                     self.population_std_dev])
        tab.add_row(['Population Variance',
                     self.population_variance])
        tab.add_row(['Mean Absolute Deviation',
                     self.mean_absolute_deviation])
        tab.add_row(['Median Absolute Deviation',
                     self.median_absolute_deviation])
        tab.add_row(['Mode Absolute Deviation',
                     self.mode_absolute_deviation])
        print tab
