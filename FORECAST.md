# Per-planet incoming-ships forecast: notes

Implements `PLANET_FORECAST_SPEC.md`. Files:

| File | What it does |
|---|---|
| `sim.py` | `simulate_turn(state, moves)`: one engine turn on NumPy arrays; `make_state(game, t)` builds a state from a replay |
| `forecast.py` | `forecast(state, n_players)`: runs the fleets in flight forward 50 turns with no new launches |
| `precompute.py` | runs the forecast for every state of every game, saves `replay_memory/forecast/NNN/gNNNNNNN.npz` |
| `features.py` | (new section at the bottom) `planet_tokens`, `global_token`, `start_turns`, `owner_class`, `forecast_table` |
| `dataset.py` | `ForecastDataset`: shuffled stream of `(game, t, player)` samples, batched with `np.stack` |
| `encoder.py` | `StateEncoder`: planet MLP + global MLP + self-attention over 45 tokens, pooled to `z`, two heads |
| `tests/` | Tests 1–4 plus a model smoke test: `python -m pytest -s` |

## The four questions

1. **`observation(t, player)`** returns `player`, `step` (= t), `angular_velocity`, `planets`, `fleets`,
   `comets` (groups with `planet_ids`, `paths`, `path_index`) and `comet_planet_ids`. It does not include
   `initial_planets` (those are the positions in `planets(0)`) or `next_fleet_id` (`make_state` counts the
   valid orders before turn t instead).
2. **Angular velocity** is `game.meta["angular_velocity"]`.
3. **Comets:** at state t, comet k of group g is at `comet_paths[g, k, t - comet_first_step[g]]`, so the
   engine's `path_index` is `t - comet_first_step[g]`. The group is on the board for `comet_path_len[g]`
   states.
4. **Orbit offset c = -1.** An orbiting planet at state s ≥ 1 sits at `initial_angle + w * (s - 1)`. The
   turn from state s to s + 1 runs with `step = s`. State 0 holds the initial positions, and state 1 has the
   same positions, because the first turn runs with step 0. The largest error over a game is 7e-6
   (float32 recordings).

## Test results

| Test | Result |
|---|---|
| 1: simulator vs recordings | 200 games (25 from each mix of 2p/4p, standard/scrambled, clean/noisy), 44,655 turns, **0 mismatches**. 451 turns where a comet group spawns were skipped. Checks planet ids, owners and ships, fleet ids, owners, ships and source planets, positions to 1e-3, `action_valid` and `action_fleet`. As a sanity check, breaking the multi-owner combat rule on purpose gave 664 mismatches in 40 games. |
| 2: forecast vs what happened | 40 games, every state, 235,590 (state, fleet) pairs. **0 mismatches** apart from 1,706 pairs (0.72%) where the recording shows the fleet hitting a comet spawned after t, which the forecast cannot know about. All those cases end at t = 49, 149, 249, … (just before a spawn). The test also checks that every fleet's ships end up in exactly one of incoming / lost / unresolved. |
| 3: combat rules | 7 hand-made cases pass: lone attacker (capture and fail), two-player tie, three attackers (also a top-two tie with a third present), defender reinforcing while an enemy arrives (both ways), garrison left at exactly 0. |
| 4: features | Shapes `[44, 146]` and `[20]`, no NaN, padding all zero, planets sorted by id. In a standard 2-player game the two views have the same set of positions, and each player's "me" incoming block equals the other's "enemy opposite" block at every planet. The example table below checks out by hand: production 4 a turn plus arrivals. |

```
planet 4 at state 340, seen by player 0: enemy+3, 44 ships      (games/000/g0000000.npz)
  turns |    me  en+1  en+2  en+3 |    owner  ships
      1 |     0     0     0   390 |  enemy+3    438
      2 |     0     0     0   354 |  enemy+3    796
      3 |     0     0     0    95 |  enemy+3    895
      4 |     0     0     0     0 |  enemy+3    899
    ...
  16-25 |     0     0     0    33 |  enemy+3   1051
  26-50 |     0     0     0     0 |  enemy+3   1151
```

One false alarm along the way: Test 2 first allowed 0.01 of extra reach when reading collisions off
the recordings, and flagged a fleet that passed 0.001–0.01 from a planet's edge on its way off the board.
That planet's garrison only grew by its production, so the forecast ("edge") was right. The margin is now
1e-3, well above the float32 rounding error.

## Precompute time

20 games (4,683 states) in 17.4 s, which is **3.7 ms per state**, with numba on the fleet loop. The full
replay memory is 3,493,376 states:

- one process: about **3.6 hours**;
- `python precompute.py --workers 12`: roughly **20–25 minutes** (16 cores). The per-game function is the
  same either way.

The full run has **not** been started. Only the first 20 games have forecast files so far.

## Decisions I made where the spec or the code left a gap

- **numba.** `swept_pair_hit`, `point_to_segment_distance`, `fleet_speed` and `move_fleets` in `sim.py`
  have `@njit`. They are still plain loops copied from the engine. Set `NUMBA_DISABLE_JIT=1` to run them
  as ordinary Python, for example in a debugger. Test 1 gives 0 mismatches with and without numba.
- **Fleets that hit a comet on the turn it leaves** collide with it (the engine checks the collision), but
  the comet is gone before combat, so their ships vanish. `events` lists them as `"planet"` with the comet's
  id, and the forecast counts them in `lost`, as agreed.
- **Code the spec calls "existing" did not exist** anywhere: `start_turns`, `owner_class`, the 16 current
  features, the 12 globals, the fleet cross-attention model, the dataset, and the heads and losses. I wrote
  them as follows:
  - `start_turns`: the starting planets are the group of four symmetric planets (ids 4g..4g+3) in which
    each player owns its seat's member (2p: members 0 and 3; 4p: members 0–3, which is how the engine hands
    them out). The rotation is the number of anticlockwise quarter turns that brings that planet into
    x > 50, y > 50. This also works for scrambled starts, where many planets are owned at state 0.
  - `owner_class`: enemy classes 2 / 3 / 4 are one quarter turn **anticlockwise** (the way planets orbit) /
    opposite / clockwise from me, after rotating to my view. In 2-player games the enemy is always 3.
  - Current features (16): positions `(x - 50) / 50` after rotation; velocity = the planet's move during the
    next turn, `/ 4`; `orbit_r / 50`; radius `/ 3`; production `/ 5`; `log1p(ships) / 8`;
    comet turns left `/ 40`; owner one-hot.
  - Globals (12): step / 500, angular velocity / 0.05, is 4-player, `log1p(total ships) / 8` per class (4),
    total production / 50 per class (4), turns to the next comet spawn / 100. The forecast adds
    `log1p(lost) / 8` and `log1p(unresolved) / 8` per class. I divided by 8 for the same scale as the other
    ship features; the spec only said `log1p`.
  - Targets and heads: `win` (did this player win; binary cross-entropy on a logit) and `future_share`
    (this player's share of all ships 50 turns later, or at the end; MSE on a sigmoid output).
  - Pooling: `z = Linear([global token, mean of real planet tokens])`.
- **Where the new code lives.** The old `features.encode` is still used by `scripted.py` (which
  `replay.py` imports) and by `PolicyNet`, so the token code is a new section at the bottom of `features.py`
  rather than a replacement. The model is a new `encoder.py`, so `model.py`'s `PolicyNet` keeps working.
- **Replay location.** The data is in `..\Machine_Learning\orbit_wars\replay_memory`. The tests read
  `ORBIT_REPLAYS` (default `./replay_memory`), and `precompute.py` takes `--root`. Forecast files go in a
  `forecast/` folder next to `games/` inside that root.

## Things I'm not sure about

- Test 2 reads "how a fleet disappeared" off the recording by geometry. It agrees with the forecast
  everywhere outside new-comet hits, but it is a reading of float32 positions, not ground truth from the
  engine.
- The forecast ignores comets that spawn later (as specified). About 0.7% of (state, fleet) pairs are
  affected, all within 50 turns before a spawn at states 50, 150, 250, 350 or 450. If that matters, the
  spawn times are fixed and known, so the model could at least be told "a spawn is coming" (global feature
  11 does this).
