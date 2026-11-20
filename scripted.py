"""Simple deterministic scripted agents.

Each agent is a plain Kaggle agent function `agent(obs, config=None) -> moves`
that depends only on the observation, so the same state always produces the
same moves. They are deliberately simple, and each one leans on a different
part of the rules so that games between them show a model the whole game:

  passive         never moves                    -> pure production growth
  nearest-sniper  Kaggle's example agent         -> basic capture arithmetic; its
                  (aims at where the target is)     un-led shots miss moving planets
  lead-sniper     same, but aims ahead           -> orbits, comet paths, travel time,
                                                    production during flight, the sun
  weakest-first   richest planet hits the        -> long flights, fleet size vs speed
                  weakest planet anywhere
  economist       buys the fastest payback       -> production values
  trickler        streams of tiny fleets         -> slow small fleets, attrition
                                                    without capture
  doomstack       funnels everything into one    -> reinforcing own planets, huge
                  planet, then one huge fleet       fleets at top speed
  turtle          sits on big garrisons and      -> defensive combat, reinforcement
                  reinforces planets under attack
  comet-chaser    grabs comets, evacuates them   -> comet capture, comet expiry
  hunter          only attacks other players     -> player-vs-player combat, elimination
  gang-up         every planet hits one target   -> several fleets fighting at one planet
  sweeper         fires a fixed cycle of fleet   -> fleet speed curve, sun and
                  sizes in a rotating direction     out-of-bounds losses, stray hits

`SCRIPTED` maps those names to the functions. `noisy` wraps any agent with
seeded random mistakes (see generate_replays.py).

Only depends on features.py / actions.py, so these can also be shipped in a
Kaggle submission.
"""

import math
import random

from actions import fleet_speed
from features import (
    CENTER,
    F_OWNER,
    F_SHIPS,
    P_ID,
    P_OWNER,
    P_PROD,
    P_RADIUS,
    P_SHIPS,
    P_X,
    P_Y,
    _fleet_target,
    _get,
    is_orbiting,
)

SUN_RADIUS = 10.0
LAUNCH_GAP = 0.1  # fleets spawn this far outside the source planet's surface
TWO_PI = 2.0 * math.pi


def dist(a, b):
    return math.hypot(a[P_X] - b[P_X], a[P_Y] - b[P_Y])


def angle_to(src, target):
    return math.atan2(target[P_Y] - src[P_Y], target[P_X] - src[P_X])


def order(src, angle, ships):
    return [int(src[P_ID]), float(angle), int(ships)]


def _segment_distance(p, a, b):
    """Distance from point p to the segment a-b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    l2 = abx * abx + aby * aby
    if l2 == 0.0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / l2))
    return math.hypot(p[0] - a[0] - t * abx, p[1] - a[1] - t * aby)


class World:
    """One observation, parsed, plus the geometry helpers the agents share."""

    def __init__(self, obs):
        self.me = _get(obs, "player", 0)
        self.step = _get(obs, "step", 0) or 0
        self.planets = list(_get(obs, "planets", []) or [])
        self.fleets = list(_get(obs, "fleets", []) or [])
        self.angular_velocity = _get(obs, "angular_velocity", 0.0) or 0.0
        self.comet_paths = {}  # comet planet id -> (path, index of its current position)
        for group in _get(obs, "comets", []) or []:
            for pid, path in zip(group["planet_ids"], group["paths"]):
                self.comet_paths[pid] = (path, group["path_index"])
        self.mine = [p for p in self.planets if p[P_OWNER] == self.me]
        self.targets = [p for p in self.planets if p[P_OWNER] != self.me]
        self._incoming = None

    def position(self, planet, turns):
        """Where `planet` will be `turns` ticks from now, or None if it is a
        comet that will have left the board by then."""
        pid = planet[P_ID]
        if pid in self.comet_paths:
            path, index = self.comet_paths[pid]
            i = index + int(round(turns))
            return (path[i][0], path[i][1]) if 0 <= i < len(path) else None
        if not is_orbiting(planet):
            return planet[P_X], planet[P_Y]
        if self.step == 0:
            turns = max(turns - 1, 0)  # planets do not rotate on the very first tick
        dx, dy = planet[P_X] - CENTER, planet[P_Y] - CENTER
        r = math.hypot(dx, dy)
        a = math.atan2(dy, dx) + self.angular_velocity * turns
        return CENTER + r * math.cos(a), CENTER + r * math.sin(a)

    def intercept(self, src, target, ships):
        """Aim a fleet of `ships` from `src` at where `target` will be when it
        gets there. Returns (angle, turns, (x, y)), or None if the target is a
        comet that leaves first."""
        speed = fleet_speed(ships)
        pos = (target[P_X], target[P_Y])
        turns = 1
        for _ in range(4):
            gap = math.hypot(pos[0] - src[P_X], pos[1] - src[P_Y]) - src[P_RADIUS] - LAUNCH_GAP - target[P_RADIUS]
            turns = max(1, math.ceil(gap / speed))
            pos = self.position(target, turns)
            if pos is None:
                return None
        return math.atan2(pos[1] - src[P_Y], pos[0] - src[P_X]), turns, pos

    def path_clear(self, src, pos, target):
        """True if the straight line from `src` to `pos` misses the sun and
        every other planet (checked at their current positions)."""
        a = (src[P_X], src[P_Y])
        if _segment_distance((CENTER, CENTER), a, pos) < SUN_RADIUS + 1.0:
            return False
        for p in self.planets:
            if p[P_ID] == src[P_ID] or p[P_ID] == target[P_ID]:
                continue
            if _segment_distance((p[P_X], p[P_Y]), a, pos) < p[P_RADIUS] + 0.5:
                return False
        return True

    def need(self, target, turns):
        """Ships needed to capture `target` with a fleet landing in `turns`
        ticks: its garrison, plus what it produces meanwhile if owned, plus 1."""
        growth = target[P_PROD] * turns if target[P_OWNER] != -1 else 0
        return target[P_SHIPS] + growth + 1

    def plan(self, src, target, extra=0, check_path=True):
        """Smallest capturing shot from `src` at `target`: (angle, ships, turns),
        or None if it is blocked or the target cannot be reached."""
        ships = target[P_SHIPS] + 1 + extra
        hit = None
        for _ in range(2):  # fleet size sets speed, speed sets travel time, time sets size
            hit = self.intercept(src, target, ships)
            if hit is None:
                return None
            ships = self.need(target, hit[1]) + extra
        if check_path and not self.path_clear(src, hit[2], target):
            return None
        return hit[0], ships, hit[1]

    def incoming(self):
        """{planet_id: [my_ships, other_ships]} for fleets in flight, found by
        extending each fleet's heading to the first planet it would hit."""
        if self._incoming is None:
            self._incoming = {}
            for f in self.fleets:
                pid = _fleet_target(f, self.planets)
                if pid is not None:
                    entry = self._incoming.setdefault(pid, [0, 0])
                    entry[0 if f[F_OWNER] == self.me else 1] += f[F_SHIPS]
        return self._incoming

    def covered(self, target):
        """True if my fleets already heading for `target` are enough to take it."""
        return self.incoming().get(target[P_ID], (0, 0))[0] > target[P_SHIPS]

    def claim(self, target, ships):
        """Count a fleet launched this turn, so two planets don't double up."""
        self.incoming().setdefault(target[P_ID], [0, 0])[0] += ships


def _snipe(w, sources):
    """Each source takes its nearest target that is not already covered and
    has a clear shot; it fires when it can afford the capture, else it waits."""
    moves = []
    for src in sources:
        for target in sorted(w.targets, key=lambda t: dist(src, t)):
            if w.covered(target):
                continue
            shot = w.plan(src, target)
            if shot is None:
                continue
            angle, ships, _ = shot
            if ships <= src[P_SHIPS]:
                moves.append(order(src, angle, ships))
                w.claim(target, ships)
            break
    return moves


def passive(obs, config=None):
    """Never moves."""
    return []


def nearest_sniper(obs, config=None):
    """The example agent from the Kaggle page: every planet sends garrison + 1
    ships at its nearest planet it doesn't own, as soon as it can afford to.
    Aims at the target's current position, so it misses fast-moving planets."""
    w = World(obs)
    moves = []
    for src in w.mine:
        if not w.targets:
            break
        target = min(w.targets, key=lambda t: dist(src, t))
        ships = target[P_SHIPS] + 1
        if src[P_SHIPS] >= ships:
            moves.append(order(src, angle_to(src, target), ships))
    return moves


def lead_sniper(obs, config=None):
    """Nearest-planet sniper that aims ahead of moving targets, allows for
    enemy production during the flight, and won't shoot through the sun."""
    w = World(obs)
    return _snipe(w, w.mine)


def weakest_first(obs, config=None):
    """Works through targets from the smallest garrison up, wherever they are,
    hitting each from one of its three richest planets."""
    w = World(obs)
    spare = {p[P_ID]: p[P_SHIPS] for p in w.mine}
    moves = []
    for target in sorted(w.targets, key=lambda t: (t[P_SHIPS], t[P_ID])):
        richest = sorted(w.mine, key=lambda s: (-spare[s[P_ID]], s[P_ID]))[:3]
        if not richest or spare[richest[0][P_ID]] <= target[P_SHIPS]:
            break  # nothing left that can afford this or any bigger target
        if w.covered(target):
            continue
        for src in richest:
            shot = w.plan(src, target)
            if shot is not None and shot[1] <= spare[src[P_ID]]:
                moves.append(order(src, shot[0], shot[1]))
                spare[src[P_ID]] -= shot[1]
                w.claim(target, shot[1])
                break
    return moves


def economist(obs, config=None):
    """Each planet buys the affordable target that pays back fastest:
    (ships spent / target production) + travel time. Keeps 5 ships at home."""
    w = World(obs)
    moves = []
    for src in w.mine:
        budget = src[P_SHIPS] - 5
        options = []
        for target in w.targets:
            if target[P_SHIPS] >= budget or w.covered(target):
                continue
            shot = w.plan(src, target, check_path=False)
            if shot is not None and shot[1] <= budget:
                options.append((shot[1] / target[P_PROD] + shot[2], target[P_ID], target, shot))
        for _, _, target, (angle, ships, turns) in sorted(options, key=lambda o: o[:2]):
            pos = w.position(target, turns)
            if pos is not None and w.path_clear(src, pos, target):
                moves.append(order(src, angle, ships))
                w.claim(target, ships)
                break
    return moves


def trickler(obs, config=None):
    """Every third turn each planet sends a tiny fleet (2-7 ships) at its
    nearest target. Small fleets crawl, and they wear a garrison down without
    capturing until enough of them have landed."""
    w = World(obs)
    moves = []
    for src in w.mine:
        k = w.step + src[P_ID]
        if k % 3 or src[P_SHIPS] < 4 or not w.targets:
            continue
        target = min(w.targets, key=lambda t: dist(src, t))
        ships = min(src[P_SHIPS] - 1, 2 + (k // 3) % 6)
        hit = w.intercept(src, target, ships)
        if hit is not None:
            moves.append(order(src, hit[0], ships))
    return moves


def doomstack(obs, config=None):
    """Every planet ships its garrison to the capital (its highest-production
    planet). Once the capital holds 40+, it throws 90% of the stack at the
    most productive planet that stack can take."""
    w = World(obs)
    if not w.mine:
        return []
    capital = max(w.mine, key=lambda p: (p[P_PROD], -p[P_ID]))
    moves = []
    for src in w.mine:
        if src is capital or src[P_SHIPS] < 12:
            continue
        ships = src[P_SHIPS] - 1
        hit = w.intercept(src, capital, ships)
        if hit is not None and w.path_clear(src, hit[2], capital):
            moves.append(order(src, hit[0], ships))
    if capital[P_SHIPS] >= 40:
        stack = int(capital[P_SHIPS] * 0.9)
        for target in sorted(w.targets, key=lambda t: (-t[P_PROD], dist(capital, t))):
            hit = w.intercept(capital, target, stack)
            if hit is None or w.need(target, hit[1]) > stack or not w.path_clear(capital, hit[2], target):
                continue
            moves.append(order(capital, hit[0], stack))
            break
    return moves


def turtle(obs, config=None):
    """Keeps 10 ships on every planet. Planets about to fall are reinforced
    from the nearest neighbours; it only expands from a planet holding more
    than twice what the capture costs."""
    w = World(obs)
    incoming = w.incoming()
    spare = {p[P_ID]: p[P_SHIPS] - 10 for p in w.mine}
    moves = []
    threatened = {p[P_ID] for p in w.mine if incoming.get(p[P_ID], (0, 0))[1]}
    for home in w.mine:
        friendly, hostile = incoming.get(home[P_ID], (0, 0))
        deficit = hostile - home[P_SHIPS] - friendly + 1
        if deficit <= 0:
            continue
        for helper in sorted(w.mine, key=lambda p: dist(p, home)):
            give = min(spare[helper[P_ID]], deficit)
            if helper is home or helper[P_ID] in threatened or give <= 0:
                continue
            hit = w.intercept(helper, home, give)
            if hit is None or not w.path_clear(helper, hit[2], home):
                continue
            moves.append(order(helper, hit[0], give))
            spare[helper[P_ID]] -= give
            deficit -= give
            if deficit <= 0:
                break
    for src in w.mine:
        if src[P_ID] in threatened:
            continue
        for target in sorted(w.targets, key=lambda t: dist(src, t)):
            if w.covered(target):
                continue
            shot = w.plan(src, target)
            if shot is None:
                continue
            if 2 * shot[1] <= spare[src[P_ID]]:
                moves.append(order(src, shot[0], shot[1]))
                w.claim(target, shot[1])
            break
    return moves


def comet_chaser(obs, config=None):
    """Goes for comets first (intercepting them along their published paths),
    pulls its ships off a comet just before it leaves the board, and otherwise
    plays like lead-sniper."""
    w = World(obs)
    moves = []
    busy = set()
    for src in w.mine:  # evacuate comets on their last two positions
        if src[P_ID] not in w.comet_paths:
            continue
        busy.add(src[P_ID])
        path, index = w.comet_paths[src[P_ID]]
        homes = [p for p in w.mine if p[P_ID] not in w.comet_paths]
        if len(path) - 1 - index <= 1 and src[P_SHIPS] > 0 and homes:
            home = min(homes, key=lambda p: dist(src, p))
            hit = w.intercept(src, home, src[P_SHIPS])
            if hit is not None:
                moves.append(order(src, hit[0], src[P_SHIPS]))
    for comet in w.targets:
        if comet[P_ID] not in w.comet_paths or w.covered(comet):
            continue
        for src in sorted(w.mine, key=lambda p: dist(p, comet))[:3]:
            if src[P_ID] in busy:
                continue
            shot = w.plan(src, comet)
            if shot is not None and shot[1] <= src[P_SHIPS]:
                moves.append(order(src, shot[0], shot[1]))
                w.claim(comet, shot[1])
                busy.add(src[P_ID])
                break
    return moves + _snipe(w, [p for p in w.mine if p[P_ID] not in busy])


def hunter(obs, config=None):
    """Expands like lead-sniper until it owns three planets, then ignores
    neutrals: every planet with 10+ ships attacks its nearest enemy planet."""
    w = World(obs)
    enemies = [p for p in w.targets if p[P_OWNER] != -1]
    if len(w.mine) < 3 or not enemies:
        return _snipe(w, w.mine)
    moves = []
    for src in w.mine:
        if src[P_SHIPS] < 10:
            continue
        for victim in sorted(enemies, key=lambda t: dist(src, t))[:4]:
            shot = w.plan(src, victim, extra=2)
            if shot is None:
                continue
            if shot[1] <= src[P_SHIPS]:
                moves.append(order(src, shot[0], shot[1]))
            break
    return moves


def gang_up(obs, config=None):
    """Picks one target (cheap for its production, and close to where its ships
    are). When half of every garrison together is 1.3x what the capture costs,
    all planets fire at it in the same turn."""
    w = World(obs)
    total = sum(p[P_SHIPS] for p in w.mine)
    if total <= 0 or not w.targets:
        return []
    centre = [0, 0, sum(p[P_X] * p[P_SHIPS] for p in w.mine) / total, sum(p[P_Y] * p[P_SHIPS] for p in w.mine) / total]
    open_targets = [t for t in w.targets if not w.covered(t)]
    if not open_targets:
        return []
    target = min(open_targets, key=lambda t: ((t[P_SHIPS] + 1) / t[P_PROD] + 0.5 * dist(centre, t), t[P_ID]))
    shots = []
    for src in w.mine:
        ships = src[P_SHIPS] // 2
        if ships < 3:
            continue
        hit = w.intercept(src, target, ships)
        if hit is not None and w.path_clear(src, hit[2], target):
            shots.append((src, hit, ships))
    if not shots:
        return []
    cost = w.need(target, max(hit[1] for _, hit, _ in shots))
    if sum(ships for _, _, ships in shots) < 1.3 * cost:
        return []
    return [order(src, hit[0], ships) for src, hit, ships in shots]


GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
SWEEP_SIZES = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144)


def sweeper(obs, config=None):
    """Not trying to win: every other turn each planet fires the next fleet
    size from a fixed cycle (1, 2, 3, 5, ... 144, capped at half its garrison)
    in a direction that steps round by the golden angle. Shows fleets of every
    size flying in every direction: into the sun, off the board, into planets."""
    w = World(obs)
    moves = []
    for src in w.mine:
        k = w.step + src[P_ID]
        if k % 2 or src[P_SHIPS] < 2:
            continue
        ships = min(SWEEP_SIZES[(k // 2) % len(SWEEP_SIZES)], src[P_SHIPS] // 2)
        moves.append(order(src, (k * GOLDEN_ANGLE) % TWO_PI, ships))
    return moves


SCRIPTED = {
    "passive": passive,
    "nearest-sniper": nearest_sniper,
    "lead-sniper": lead_sniper,
    "weakest-first": weakest_first,
    "economist": economist,
    "trickler": trickler,
    "doomstack": doomstack,
    "turtle": turtle,
    "comet-chaser": comet_chaser,
    "hunter": hunter,
    "gang-up": gang_up,
    "sweeper": sweeper,
}


def noisy(agent, eps, seed):
    """Wrap `agent` so it makes seeded random mistakes at rate `eps`.

    Each of its moves is, with probability eps, knocked off course, resized or
    dropped. With probability eps per turn it also adds a random launch, and
    with probability eps/5 an illegal order (someone else's planet, or more
    ships than the planet holds), which the game ignores. This widens the
    states and actions a model sees beyond what tidy scripts produce, while
    staying reproducible for a given seed.
    """
    rng = random.Random(seed)

    def agent_with_noise(obs, config=None):
        moves = []
        for pid, angle, ships in agent(obs, config):
            if rng.random() < eps:
                kind = rng.randrange(3)
                if kind == 0:
                    angle += rng.gauss(0.0, 0.3)
                elif kind == 1:
                    ships = max(1, int(ships * rng.uniform(0.2, 1.0)))
                else:
                    continue
            moves.append([pid, angle, ships])
        me = _get(obs, "player", 0)
        planets = _get(obs, "planets", []) or []
        mine = [p for p in planets if p[P_OWNER] == me and p[P_SHIPS] > 0]
        if mine and rng.random() < eps:
            src = rng.choice(mine)
            moves.append(order(src, rng.uniform(0.0, TWO_PI), rng.randint(1, src[P_SHIPS])))
        if planets and rng.random() < eps / 5:
            src = rng.choice(planets)
            moves.append(order(src, rng.uniform(0.0, TWO_PI), src[P_SHIPS] + rng.randint(1, 20)))
        return moves

    return agent_with_noise
