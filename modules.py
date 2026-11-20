"""Custom PyTorch modules.

    GameEncoder    the whole game state  ->  one vector of size n
    ActionChooser  a vector of inputs    ->  a choice for each of several decisions

Shapes use B for the batch size.

Example:

    obs = game.observation(t, player=0)            # from a Replay, FastGame or Kaggle
    batch = stack_observations([obs])              # dict of padded tensors, B = 1

    encoder = GameEncoder(out_size=128)
    chooser = ActionChooser(in_size=128, num_decisions=48, num_options=10)

    state_vector = encoder(batch)                  # (1, 128)
    choices = chooser.choose(state_vector)         # (1, 48), one option number per decision
"""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------------------------
# Turning an observation into tensors
# ---------------------------------------------------------------------------------------------

# What each row of the encoder's input holds. Positions are divided by the board size (100) and
# ship counts are log-scaled, so every number is roughly between 0 and 1.
PLANET_FEATURES = ["planet_id", "owner", "x", "y", "radius", "ships", "production", "is_comet"]
FLEET_FEATURES = ["fleet_id", "owner", "x", "y", "cos_angle", "sin_angle", "from_planet" "ships"]
GLOBAL_FEATURES = ["step", "angular_velocity"]


def _scale_ships(ships):
    return math.log1p(max(ships, 0)) / math.log1p(1000)


def observation_to_tensors(obs):
    """One Kaggle-style observation -> a dict of three unbatched tensors.

    planets  (num_planets, 8)  one row per planet, in the order of PLANET_FEATURES
    fleets   (num_fleets, 8)   one row per fleet, in the order of FLEET_FEATURES
    globals  (2,)              in the order of GLOBAL_FEATURES
    """
    me = obs["player"]
    comets = set(obs.get("comet_planet_ids") or [])

    planets = []
    for planet_id, owner, x, y, radius, ships, production in obs["planets"]:
        planets.append([
            planet_id / 60,
            owner,
            x / 100,
            y / 100,
            radius / 3,
            _scale_ships(ships),
            production / 5,
            planet_id in comets,
        ])

    fleets = []
    for fleet_id, owner, x, y, angle, from_planet, ships in obs["fleets"]:
        fleets.append([
            fleet_id / 1000,
            owner,
            x / 100,
            y / 100,
            math.cos(angle),
            math.sin(angle),
            from_planet / 60,
            _scale_ships(ships),
        ])

    return {
        "planets": torch.tensor(planets, dtype=torch.float32).reshape(-1, len(PLANET_FEATURES)),
        "fleets": torch.tensor(fleets, dtype=torch.float32).reshape(-1, len(FLEET_FEATURES)),
        "globals": torch.tensor([obs.get("step", 0) / 500, obs.get("angular_velocity", 0) / 0.05],
                                dtype=torch.float32),
    }


def stack_observations(observations):
    """A list of observations -> one batch, padded to the largest number of planets and fleets.

    planets        (B, max_planets, 8)
    planet_mask    (B, max_planets)    True where the row is a real planet, False for padding
    fleets         (B, max_fleets, 8)
    fleet_mask     (B, max_fleets)
    globals        (B, 2)
    """
    items = [observation_to_tensors(obs) for obs in observations]
    batch = {}
    for kind in ("planets", "fleets"):
        rows = [item[kind] for item in items]
        batch[kind] = pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(r) for r in rows])
        batch[kind[:-1] + "_mask"] = torch.arange(batch[kind].shape[1]) < lengths[:, None]
    batch["globals"] = torch.stack([item["globals"] for item in items])
    return batch


# ---------------------------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------------------------

def mlp(in_size, hidden_size, out_size):
    """Two linear layers with a ReLU in between."""
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )


def masked_mean(rows, mask):
    """Average of the real rows, ignoring padding.

    rows (B, N, D), mask (B, N) -> (B, D). A game with no rows at all (no fleets) gives zeros.
    """
    mask = mask.unsqueeze(-1).float()                 # (B, N, 1)
    total = (rows * mask).sum(dim=1)                  # (B, D)
    count = mask.sum(dim=1).clamp(min=1)              # (B, 1), at least 1 to avoid dividing by 0
    return total / count


# ---------------------------------------------------------------------------------------------
# GameEncoder
# ---------------------------------------------------------------------------------------------

class GameEncoder(nn.Module):
    """Everything in one game state -> one vector of size `out_size`.

    The number of planets and fleets changes from turn to turn, so the encoder works in three steps:

      1. Each planet row goes through the same small network (`planet_net`), and so does each fleet
         row (`fleet_net`). Every row becomes a vector of size `hidden_size`.
      2. The planet vectors are averaged into one vector, and so are the fleet vectors. The average
         does not depend on how many rows there are or what order they come in.
      3. The two averages and the global numbers are joined end to end, and `output_net` turns them
         into the final vector of size `out_size`.
    """

    def __init__(self, out_size, single_encoding_size=4, hidden_size=128):
        super().__init__()
        self.planet_net = mlp(len(PLANET_FEATURES), hidden_size, single_encoding_size)
        self.fleet_net = mlp(len(FLEET_FEATURES), hidden_size, single_encoding_size)
        self.output_net = mlp(2 * single_encoding_size + len(GLOBAL_FEATURES), hidden_size, out_size)

    def forward(self, batch):
        """`batch` is the dict from `stack_observations`. Returns (B, out_size)."""
        # Step 1: one vector per planet and one per fleet.
        planet_vectors = self.planet_net(batch["planets"])        # (B, num_planets, single_encoding_size)
        fleet_vectors = self.fleet_net(batch["fleets"])           # (B, num_fleets, single_encoding_size)

        # Step 3: join everything and map it to the output size.
        joined = torch.cat([planet_vectors, fleet_vectors, batch["globals"]], dim=-1)
        return self.output_net(joined)                            # (B, out_size)


# ---------------------------------------------------------------------------------------------
# ActionChooser
# ---------------------------------------------------------------------------------------------

class ActionChooser(nn.Module):
    """A vector of inputs -> a choice for each of `num_decisions` decisions.

    Each decision picks one of `num_options` options. For example, with one decision per planet
    slot, the options could be "hold" or "attack target 1..9". What the decisions and options
    mean is up to the code that turns choices into game orders; this module only scores them.

    `forward` gives a score (logit) for every option of every decision. `choose` turns the scores
    into choices: the highest-scoring option, or a random one drawn in proportion to the scores
    (for exploration during training).
    """

    def __init__(self, in_size, num_decisions, num_options, hidden_size=128):
        super().__init__()
        self.num_decisions = num_decisions
        self.num_options = num_options
        self.net = mlp(in_size, hidden_size, num_decisions * num_options)

    def forward(self, inputs):
        """inputs (B, in_size) -> scores (B, num_decisions, num_options)."""
        scores = self.net(inputs)
        return scores.view(-1, self.num_decisions, self.num_options)

    def choose(self, inputs, sample=False):
        """inputs (B, in_size) -> choices (B, num_decisions), each a number from 0 to num_options - 1.

        sample=False picks the best option for each decision; sample=True draws one at random.
        """
        scores = self.forward(inputs)
        if sample:
            return torch.distributions.Categorical(logits=scores).sample()
        return scores.argmax(dim=-1)
