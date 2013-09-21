import pytest
from bayesian.factor_graph import *
from bayesian.persistance import *

def f_prize_door(prize_door):
    return 1.0 / 3


def f_guest_door(guest_door):
    return 1.0 / 3


def f_monty_door(prize_door, guest_door, monty_door):
    if prize_door == guest_door:
        if prize_door == monty_door:
            return 0
        else:
            return 0.5
    elif prize_door == monty_door:
        return 0
    elif guest_door == monty_door:
        return 0
    return 1


def pytest_funcarg__monty_graph(request):
    g = build_graph(
        f_prize_door,
        f_guest_door,
        f_monty_door,
        domains=dict(
            prize_door=['A', 'B', 'C'],
            guest_door=['A', 'B', 'C'],
            monty_door=['A', 'B', 'C']))
    return g


class TestPersistance():

    def test_create_sqlite_db_when_inference_method_changed(self, monty_graph):
        assert monty_graph.inference_method == 'sumproduct'
        # Now switch the inference_method to sample_db...
        monty_graph.inference_method = 'sample_db'
        assert monty_graph.inference_method == 'sample_db'

    def test_key_to_int(self):
        assert key_to_int((('a', True),)) == 1
        assert key_to_int((('a', False),
                           ('b', False),
                           ('c', False))) == 0

        assert key_to_int((('a', False),
                           ('b', True),
                           ('c', False))) == 2

        assert key_to_int((('a', True),
                           ('b', False),
                           ('c', True))) == 5

    def test_diskdict(self):
        d = DiskDict()
        d[(('a', False),)] = 0.4
        assert d[(('a', False),)] == 0.4

        d = DiskDict()
        key = (('a', True),
               ('b', False),
               ('c', True))
        d[key] = 1
        assert d[key] == 1

        key = (('a', False),
               ('b', True),
               ('c', True))

        d[key] = 3
        assert d[key] == 3

        # Test default like dict behaviour...
        d = DiskDict(float)
        assert d[key] == 0

        # Test copying
        e = d.copy()
        for k, v in d.iteritems():
            assert e[k] == v
        assert len(e) == len(d)
        # Now read the raw files and ensure they are the same

        data_d = open(d.name).read()
        data_e = open(e.name).read()
        assert data_d == data_e

        # Test items and iteritems...
        for k, v in d.items():
            assert e[k] == d[k]

        # Test all the other attributes...
        ignore_attribs = ('name', '_db')
        for k, v in d.__dict__.iteritems():
            if k in ignore_attribs:
                continue
            print k, v
            assert e.__dict__[k] == v
