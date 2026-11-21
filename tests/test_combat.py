"""Test 3: the combat rules, on small hand-made cases.

sim.combat(garrison_owner, garrison_ships, arriving) -> (owner, ships), where
`arriving` maps owner -> ships landing this turn. -1 is neutral.
"""

from sim import combat


def test_single_attacker_captures():
    assert combat(-1, 10, {0: 15}) == (0, 5)


def test_single_attacker_too_weak():
    assert combat(-1, 10, {0: 7}) == (-1, 3)


def test_two_player_tie_leaves_no_survivor():
    assert combat(-1, 5, {0: 10, 1: 10}) == (-1, 5)
    assert combat(1, 5, {0: 10, 1: 10}) == (1, 5)   # even the owner's own fleet is cancelled out


def test_three_attackers_top_minus_second():
    # 20 - 15 = 5 of player 0 survive (player 2's 5 make no difference), then beat 3 neutrals.
    assert combat(-1, 3, {0: 20, 1: 15, 2: 5}) == (0, 2)
    # A tie between the top two cancels everything, even with a third player present.
    assert combat(-1, 3, {0: 15, 1: 15, 2: 5}) == (-1, 3)


def test_defender_reinforces_while_enemy_arrives():
    # The owner's 8 and the enemy's 12 fight first: 4 enemy ships go on to hit the garrison.
    assert combat(0, 10, {0: 8, 1: 12}) == (0, 6)
    # The other way round: 4 of the owner's ships survive and join the garrison.
    assert combat(0, 10, {0: 12, 1: 8}) == (0, 14)


def test_garrison_left_at_exactly_zero_keeps_owner():
    assert combat(1, 10, {0: 10}) == (1, 0)
    assert combat(1, 10, {0: 11}) == (0, 1)


def test_reinforcing_own_planet():
    assert combat(2, 7, {2: 5}) == (2, 12)
