'''Simple Model of Pleasanton Weather'''
from __future__ import division

from bayesian.bbn import build_bbn

'''
This is an extremely simple five
variable model of the weather in
Pleasanton, California.
Any real weather model would have
orders of magnitude more parameters.
However the weather in this area
is remarkably consistent.
All probabilities are simple guesses
based on having lived in the area
for several years.

Here I am rather loosely defining
temp approximately as:

'hot' - above 90' F
'medium' - 55 to 90' F
'cold' - below 55' F

Rain is a binary value, so even
though in Autumn/Fall, if it does
rain, it may be only a few drops.
The value for rain would in this
case still be True.
'''

# Temp today conditioned on yesterday

spring_temp = {
    ('hot', 'hot'): 0.6,
    ('hot', 'medium'): 0.3,
    ('hot', 'cold'): 0.1,
    ('medium', 'hot'): 0.2,
    ('medium', 'medium'): 0.6,
    ('medium', 'cold'): 0.2,
    ('cold', 'hot'): 0.05,
    ('cold', 'medium'): 0.4,
    ('cold', 'cold'): 0.55}


summer_temp = {
    ('hot', 'hot'): 0.9,
    ('hot', 'medium'): 0.099,
    ('hot', 'cold'): 0.001,
    ('medium', 'hot'): 0.4,
    ('medium', 'medium'): 0.59,
    ('medium', 'cold'): 0.01,
    ('cold', 'hot'): 0.1,
    ('cold', 'medium'): 0.89,
    ('cold', 'cold'): 0.01}


autumn_temp = {
    ('hot', 'hot'): 0.6,
    ('hot', 'medium'): 0.3,
    ('hot', 'cold'): 0.1,
    ('medium', 'hot'): 0.2,
    ('medium', 'medium'): 0.6,
    ('medium', 'cold'): 0.2,
    ('cold', 'hot'): 0.05,
    ('cold', 'medium'): 0.4,
    ('cold', 'cold'): 0.55}


winter_temp = {
    ('hot', 'hot'): 0.2,
    ('hot', 'medium'): 0.6,
    ('hot', 'cold'): 0.2,
    ('medium', 'hot'): 0.05,
    ('medium', 'medium'): 0.5,
    ('medium', 'cold'): 0.45,
    ('cold', 'hot'): 0.01,
    ('cold', 'medium'): 0.19,
    ('cold', 'cold'): 0.80}


season_temp = dict(
    spring=spring_temp,
    summer=summer_temp,
    autumn=autumn_temp,
    winter=winter_temp)


# Rain today conditioned on yesterday

spring_rain = {
    (False, False): 0.35,
    (False, True): 0.65,
    (True, False): 0.35,
    (True, True): 0.65}


summer_rain = {
    (False, False): 0.95,
    (False, True): 0.05,
    (True, False): 0.8,
    (True, True): 0.2}


autumn_rain = {
    (False, False): 0.8,
    (False, True): 0.2,
    (True, False): 0.7,
    (True, True): 0.3}


winter_rain = {
    (False, False): 0.2,
    (False, True): 0.8,
    (True, False): 0.2,
    (True, True): 0.8}


season_rain = dict(
    spring=spring_rain,
    summer=summer_rain,
    autumn=autumn_rain,
    winter=winter_rain)


def f_season(season):
    return 0.25


def f_temp_yesterday(temp_yesterday):
    if temp_yesterday == 'hot':
        return 0.5
    elif temp_yesterday == 'medium':
        return 0.25
    elif temp_yesterday == 'cold':
        return 0.25


def f_rain_yesterday(rain_yesterday):
    return 0.5


def f_temp(season, temp_yesterday, temp):
    return season_temp[season][(temp_yesterday, temp)]


def f_rain(season, rain_yesterday, rain):
    return season_rain[season][(rain_yesterday, rain)]


if __name__ == '__main__':
    g = build_bbn(
        f_temp_yesterday,
        f_rain_yesterday,
        f_season,
        f_temp,
        f_rain,
        domains=dict(
            temp_yesterday=('hot', 'medium', 'cold'),
            temp=('hot', 'medium', 'cold'),
            season=('spring', 'summer', 'autumn', 'winter')))
    g.q()
