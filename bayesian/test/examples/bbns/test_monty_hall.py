'''Test the Monty Hall example as a BBN.'''
from bayesian.bbn import build_bbn
from bayesian.examples.bbns.monty_hall import (
    f_guest_door, f_prize_door, f_monty_door)


def pytest_funcarg__monty_hall_graph(request):
    g = build_bbn(
        f_guest_door, f_prize_door, f_monty_door,
        domains={
            'guest_door': ['A', 'B', 'C'],
            'monty_door': ['A', 'B', 'C'],
            'prize_door': ['A', 'B', 'C']})
    return g


def close_enough(x, y, r=3):
    return round(x, r) == round(y, r)


class TestMontyGraph():

    def test_no_evidence(self, monty_hall_graph):
        result = monty_hall_graph.query()
        assert close_enough(result[('guest_door', 'A')], 0.333)
        assert close_enough(result[('guest_door', 'B')], 0.333)
        assert close_enough(result[('guest_door', 'C')], 0.333)
        assert close_enough(result[('monty_door', 'A')], 0.333)
        assert close_enough(result[('monty_door', 'B')], 0.333)
        assert close_enough(result[('monty_door', 'C')], 0.333)
        assert close_enough(result[('prize_door', 'A')], 0.333)
        assert close_enough(result[('prize_door', 'B')], 0.333)
        assert close_enough(result[('prize_door', 'C')], 0.333)

    def test_guest_A_monty_B(self, monty_hall_graph):
        result = monty_hall_graph.query(guest_door='A', monty_door='B')
        assert close_enough(result[('guest_door', 'A')], 1)
        assert close_enough(result[('guest_door', 'B')], 0)
        assert close_enough(result[('guest_door', 'C')], 0)
        assert close_enough(result[('monty_door', 'A')], 0)
        assert close_enough(result[('monty_door', 'B')], 1)
        assert close_enough(result[('monty_door', 'C')], 0)
        assert close_enough(result[('prize_door', 'A')], 0.333)
        assert close_enough(result[('prize_door', 'B')], 0)
        assert close_enough(result[('prize_door', 'C')], 0.667)
