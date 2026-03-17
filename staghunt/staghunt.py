"""Spatial Stag Hunt environment for LFA agents.

9x9 grid. Players pick up resources (stag or hare) by walking over them,
then choose to interact when adjacent. The payoff matrix resolves when
EITHER player chooses the INTERACT action while adjacent
and both carrying a resource:
  Stag-Stag: 4,4 | Stag-Hare: 0,2 | Hare-Stag: 2,0 | Hare-Hare: 2,2

Players can swap their carried resource by walking over a different type.
Multiple rounds per episode. Full information — both players visible.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple

# --- Map (9x9) ---
# W=wall, S=stag, H=hare, n=spawn, ' '=empty
# 4S in corners (~3 steps from spawn), 4H beside spawns (~1 step)
# Equal count but hare closer = safe default; stag farther = coordination
MAP_STR = [
    "WWWWWWWWW",
    "W S   S W",
    "W       W",
    "W H n H W",
    "W  n n  W",
    "W H n H W",
    "W       W",
    "W S   S W",
    "WWWWWWWWW",
]

# Cell types
EMPTY = 0
WALL = 1
RESOURCE_STAG = 2
RESOURCE_HARE = 3
SPAWN_POINT = 4

MAP_H = len(MAP_STR)
MAP_W = len(MAP_STR[0])
NUM_PLAYERS = 2
NUM_ACTIONS = 6

# Actions
NOOP = 0
UP = 1
DOWN = 2
RIGHT = 3
LEFT = 4
INTERACT = 5

# Payoff matrix
STAG_REWARD = 4.0
HARE_REWARD = 2.0
STEP_COST = 0.0

# Timing
MAX_FRAMES = 75


def _parse_map():
    grid = jnp.zeros((MAP_H, MAP_W), dtype=jnp.int32)
    spawn_list = []
    stag_list = []
    hare_list = []

    for r, row_str in enumerate(MAP_STR):
        for c, ch in enumerate(row_str):
            if ch == "W":
                grid = grid.at[r, c].set(WALL)
            elif ch == "n":
                grid = grid.at[r, c].set(SPAWN_POINT)
                spawn_list.append((r, c))
            elif ch == "S":
                grid = grid.at[r, c].set(RESOURCE_STAG)
                stag_list.append((r, c))
            elif ch == "H":
                grid = grid.at[r, c].set(RESOURCE_HARE)
                hare_list.append((r, c))
            else:
                grid = grid.at[r, c].set(EMPTY)

    return (
        grid,
        jnp.array(spawn_list, dtype=jnp.int32),
        jnp.array(stag_list, dtype=jnp.int32),
        jnp.array(hare_list, dtype=jnp.int32),
    )


BASE_GRID, SPAWN_POINTS, STAG_POSITIONS, HARE_POSITIONS = _parse_map()
NUM_SPAWNS = SPAWN_POINTS.shape[0]
NUM_STAG_RESOURCES = STAG_POSITIONS.shape[0]
NUM_HARE_RESOURCES = HARE_POSITIONS.shape[0]


class State(NamedTuple):
    grid: jnp.ndarray                     # [H, W]
    player_pos: jnp.ndarray               # [2, 2]
    player_inv: jnp.ndarray               # [2] -1=empty, 0=stag, 1=hare
    round_step: jnp.ndarray               # scalar
    step_count: jnp.ndarray               # scalar
    round_count: jnp.ndarray              # scalar
    interaction_happened: jnp.ndarray      # bool
    last_rewards: jnp.ndarray             # [2]
    last_interaction_strategy: jnp.ndarray # [2]
    done: jnp.ndarray                     # bool
    key: jnp.ndarray


def _random_spawn(key: jnp.ndarray) -> jnp.ndarray:
    """Pick 2 distinct spawn points at random."""
    n = SPAWN_POINTS.shape[0]
    k1, k2 = jax.random.split(key)
    idx0 = jax.random.randint(k1, (), 0, n)
    idx1 = jax.random.randint(k2, (), 0, n - 1)
    idx1 = jnp.where(idx1 >= idx0, idx1 + 1, idx1)
    return jnp.stack([SPAWN_POINTS[idx0], SPAWN_POINTS[idx1]])


def reset(key: jnp.ndarray) -> State:
    key, spawn_key = jax.random.split(key)
    player_pos = _random_spawn(spawn_key)

    return State(
        grid=BASE_GRID,
        player_pos=player_pos,
        player_inv=jnp.full((2,), -1, dtype=jnp.int32),
        round_step=jnp.int32(0),
        step_count=jnp.int32(0),
        round_count=jnp.int32(0),
        interaction_happened=jnp.bool_(False),
        last_rewards=jnp.zeros(2, dtype=jnp.float32),
        last_interaction_strategy=jnp.full((2,), -1, dtype=jnp.int32),
        done=jnp.bool_(False),
        key=key,
    )


def _move_player(pos: jnp.ndarray, action: jnp.ndarray, grid: jnp.ndarray) -> jnp.ndarray:
    # INTERACT and NOOP both produce zero movement
    delta = jnp.where(
        action == UP,
        jnp.array([-1, 0], dtype=jnp.int32),
        jnp.where(
            action == DOWN,
            jnp.array([1, 0], dtype=jnp.int32),
            jnp.where(
                action == RIGHT,
                jnp.array([0, 1], dtype=jnp.int32),
                jnp.where(
                    action == LEFT,
                    jnp.array([0, -1], dtype=jnp.int32),
                    jnp.zeros(2, dtype=jnp.int32),  # NOOP or INTERACT
                ),
            ),
        ),
    )

    new_pos = pos + delta
    new_pos = jnp.clip(new_pos, jnp.array([0, 0]), jnp.array([MAP_H - 1, MAP_W - 1]))
    blocked = grid[new_pos[0], new_pos[1]] == WALL
    return jnp.where(blocked, pos, new_pos)


def _nearest_dist(pos: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Manhattan distance to nearest position in array."""
    dists = jnp.sum(jnp.abs(positions - pos[None, :]), axis=1)
    return jnp.min(dists)


def _on_resource(pos: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Check if pos is on any of the given resource positions."""
    dists = jnp.sum(jnp.abs(positions - pos[None, :]), axis=1)
    return jnp.any(dists == 0)


@jax.jit
def step(state: State, actions: jnp.ndarray) -> State:
    key, spawn_key = jax.random.split(state.key)
    interaction_happened = jnp.bool_(False)
    last_is = jnp.full((2,), -1, dtype=jnp.int32)

    new_pos = state.player_pos
    inv = state.player_inv

    # Movement (INTERACT action = stay in place, same as NOOP for movement)
    for i in range(NUM_PLAYERS):
        moved_pos = _move_player(new_pos[i], actions[i], state.grid)
        new_pos = new_pos.at[i].set(moved_pos)

    # Collision resolution: revert both if they end up on same cell
    same_cell = jnp.all(new_pos[0] == new_pos[1])
    new_pos = jnp.where(same_cell, state.player_pos, new_pos)

    # Pickup: walking over a resource picks it up (or swaps if carrying other type)
    for i in range(NUM_PLAYERS):
        on_stag = _on_resource(new_pos[i], STAG_POSITIONS)
        on_hare = _on_resource(new_pos[i], HARE_POSITIONS)
        new_inv = jnp.where(on_stag, jnp.int32(0),
                            jnp.where(on_hare, jnp.int32(1), inv[i]))
        inv = inv.at[i].set(new_inv)

    # Interaction: either player chooses INTERACT, adjacent, and both carrying
    either_interact = (actions[0] == INTERACT) | (actions[1] == INTERACT)
    dist = jnp.sum(jnp.abs(new_pos[0] - new_pos[1]))
    adjacent = dist <= 1  # adjacent or same cell (same cell shouldn't happen due to collision)
    both_carrying = (inv[0] >= 0) & (inv[1] >= 0)
    interact = either_interact & adjacent & both_carrying

    # Compute rewards from payoff matrix
    s0 = inv[0]
    s1 = inv[1]
    r0 = jnp.where(
        s0 == 1,
        HARE_REWARD,
        jnp.where((s0 == 0) & (s1 == 0), STAG_REWARD, 0.0),
    )
    r1 = jnp.where(
        s1 == 1,
        HARE_REWARD,
        jnp.where((s1 == 0) & (s0 == 0), STAG_REWARD, 0.0),
    )
    base_rewards = jnp.array([-STEP_COST, -STEP_COST], dtype=jnp.float32)
    interaction_rewards = jnp.array([r0, r1], dtype=jnp.float32)
    rewards = jnp.where(interact, base_rewards + interaction_rewards, base_rewards)
    interaction_happened = interact
    last_is = jnp.where(interact, jnp.array([s0, s1], dtype=jnp.int32), last_is)

    # Round reset on interaction only
    round_resolved = interact
    respawn_pos = _random_spawn(spawn_key)
    next_pos = jnp.where(round_resolved, respawn_pos, new_pos)
    next_inv = jnp.where(round_resolved, jnp.full((2,), -1, dtype=jnp.int32), inv)
    next_round_step = jnp.where(round_resolved, jnp.int32(0), state.round_step + 1)
    next_round_count = jnp.where(round_resolved, state.round_count + 1, state.round_count)

    new_step = state.step_count + 1
    done = new_step >= MAX_FRAMES

    return State(
        grid=BASE_GRID,
        player_pos=next_pos,
        player_inv=next_inv,
        round_step=next_round_step,
        step_count=new_step,
        round_count=next_round_count,
        interaction_happened=interaction_happened,
        last_rewards=rewards,
        last_interaction_strategy=last_is,
        done=done,
        key=key,
    )
