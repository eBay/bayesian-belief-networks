import pytest
from bayesian.factor_graph import *


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
