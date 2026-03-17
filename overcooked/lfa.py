"""LFA observation and joint-action feature utilities for Overcooked."""

from typing import Tuple

import numpy as np

from overcooked_jaxmarl.overcooked import (
    State, Overcooked, POT_READY_STATUS, POT_EMPTY_STATUS, POT_FULL_STATUS, MAX_ONIONS_IN_POT,
)
from overcooked_jaxmarl.common import OBJECT_TO_INDEX

# Raw maze map pot status encoding (differs from the env's observation-layer constants).
# The maze cell's channel-2 value uses: 8=empty, 7=1 onion, 6=2 onions,
# 5=full (starts cooking), 4..1=cooking countdown, 0=ready.
RAW_POT_EMPTY = POT_EMPTY_STATUS
RAW_POT_FULL = POT_FULL_STATUS

NUM_ACTIONS = 6


def extract_obs(state: State, env: Overcooked) -> np.ndarray:
    """Extract a 49-dim observation vector from state (single-pot layout).

    obs breakdown (cramped_room, 1 pot):
      a0_x, a0_y                          (2)
      a0_inv (onion/plate/dish one-hot)   (3)
      a0_dir (N/S/E/W one-hot)            (4)
      a1_x, a1_y                          (2)
      a1_inv                              (3)
      a1_dir                              (4)
      dist_agents                         (1)
      pot_feats (n_onions, cooking,
                 progress, ready)          (4)
      a0_dists (pot, onion, plate, goal)  (4)
      a1_dists                            (4)
      time_frac                           (1)
      bias                                (1)
      goal_potentials                     (16)
      ─────────────────────────────────────
      Total                                49

    phi(obs, a0, a1) expands to D = 49+6+6+294+294+36 = 685.
    """
    w = env.width
    h = env.height
    max_dist = float(w + h)
    padding = (state.maze_map.shape[0] - h) // 2

    agent_pos = np.array(state.agent_pos, dtype=np.float32)
    agent_inv = np.array(state.agent_inv, dtype=np.int32)

    # --- Agent positions (4) ---
    a0_x = float(agent_pos[0, 0]) / max(w - 1, 1)
    a0_y = float(agent_pos[0, 1]) / max(h - 1, 1)
    a1_x = float(agent_pos[1, 0]) / max(w - 1, 1)
    a1_y = float(agent_pos[1, 1]) / max(h - 1, 1)

    # --- Agent inventory one-hots (3 each, 6 total) ---
    def _inv_features(inv_val):
        return np.array([
            float(inv_val == OBJECT_TO_INDEX["onion"]),
            float(inv_val == OBJECT_TO_INDEX["plate"]),
            float(inv_val == OBJECT_TO_INDEX["dish"]),
        ], dtype=np.float32)

    a0_inv = _inv_features(int(agent_inv[0]))
    a1_inv = _inv_features(int(agent_inv[1]))

    # --- Agent orientations (4 each, 8 total) ---
    agent_dir_idx = np.array(state.agent_dir_idx, dtype=np.int32)

    def _dir_features(dir_idx):
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[int(np.clip(dir_idx, 0, 3))] = 1.0
        return one_hot

    a0_dir = _dir_features(agent_dir_idx[0])
    a1_dir = _dir_features(agent_dir_idx[1])

    # --- Inter-agent distance (1) ---
    dist_agents = (abs(agent_pos[0, 0] - agent_pos[1, 0])
                   + abs(agent_pos[0, 1] - agent_pos[1, 1])) / max_dist

    # --- Pot features (4, single pot) ---
    pot_pos = np.array(state.pot_pos, dtype=np.float32)
    maze = np.array(state.maze_map)

    px, py = int(pot_pos[0, 0]), int(pot_pos[0, 1])
    pot_status = int(maze[padding + py, padding + px, 2])

    if pot_status >= RAW_POT_EMPTY:
        n_onions, is_cooking, cook_progress, is_ready = 0, 0.0, 0.0, 0.0
    elif pot_status > RAW_POT_FULL:
        n_onions = RAW_POT_EMPTY - pot_status
        is_cooking, cook_progress, is_ready = 0.0, 0.0, 0.0
    elif pot_status == POT_READY_STATUS:
        n_onions, is_cooking, cook_progress, is_ready = 3, 0.0, 1.0, 1.0
    else:
        n_onions, is_cooking = 3, 1.0
        cook_progress = 1.0 - pot_status / RAW_POT_FULL
        is_ready = 0.0

    onion_norm = float(n_onions) / MAX_ONIONS_IN_POT
    pot_feats = np.array([onion_norm, is_cooking, cook_progress, is_ready],
                         dtype=np.float32)

    any_pot_needs_onions = float(onion_norm < 1.0 and is_cooking == 0.0 and is_ready == 0.0)
    any_pot_cooking_or_ready = float(is_cooking > 0.0 or is_ready > 0.0)
    any_pot_ready = float(is_ready > 0.0)

    # --- Proximity helper ---
    def _prox(ax, ay, positions):
        if len(positions) == 0:
            return 0.0
        dists = [abs(ax - float(p[0])) + abs(ay - float(p[1])) for p in positions]
        return 1.0 - (min(dists) / max_dist)

    onion_pile_pos = np.array(
        [env.layout["onion_pile_idx"] % w, env.layout["onion_pile_idx"] // w],
        dtype=np.float32,
    ).T
    plate_pile_pos = np.array(
        [env.layout["plate_pile_idx"] % w, env.layout["plate_pile_idx"] // w],
        dtype=np.float32,
    ).T
    goal_pos = np.array(state.goal_pos, dtype=np.float32)

    a0_px, a0_py = float(agent_pos[0, 0]), float(agent_pos[0, 1])
    a1_px, a1_py = float(agent_pos[1, 0]), float(agent_pos[1, 1])

    # --- Per-agent distances (4 each, 8 total) ---
    a0_dists = np.array([
        _prox(a0_px, a0_py, pot_pos),
        _prox(a0_px, a0_py, onion_pile_pos),
        _prox(a0_px, a0_py, plate_pile_pos),
        _prox(a0_px, a0_py, goal_pos),
    ], dtype=np.float32)
    a1_dists = np.array([
        _prox(a1_px, a1_py, pot_pos),
        _prox(a1_px, a1_py, onion_pile_pos),
        _prox(a1_px, a1_py, plate_pile_pos),
        _prox(a1_px, a1_py, goal_pos),
    ], dtype=np.float32)

    # --- Time and bias (2) ---
    time_frac = float(state.time) / env.max_steps

    # --- Goal potentials (16) ---
    # Hand-crafted interaction terms a linear model cannot construct.
    a0_has_onion, a0_has_plate, a0_has_dish = a0_inv[0], a0_inv[1], a0_inv[2]
    a1_has_onion, a1_has_plate, a1_has_dish = a1_inv[0], a1_inv[1], a1_inv[2]
    a0_empty = 1.0 - a0_has_onion - a0_has_plate - a0_has_dish
    a1_empty = 1.0 - a1_has_onion - a1_has_plate - a1_has_dish

    # "deliver onion to pot" potential
    a0_deliver_onion = a0_has_onion * a0_dists[0] * any_pot_needs_onions
    a1_deliver_onion = a1_has_onion * a1_dists[0] * any_pot_needs_onions

    # "pick up soup from pot" potential
    a0_pickup_soup = a0_has_plate * a0_dists[0] * any_pot_ready
    a1_pickup_soup = a1_has_plate * a1_dists[0] * any_pot_ready

    # "deliver dish to goal" potential
    a0_deliver_dish = a0_has_dish * a0_dists[3]
    a1_deliver_dish = a1_has_dish * a1_dists[3]

    # "go get onion" potential
    a0_go_get_onion = a0_empty * a0_dists[1] * any_pot_needs_onions
    a1_go_get_onion = a1_empty * a1_dists[1] * any_pot_needs_onions

    # "go get plate" potential
    a0_go_get_plate = a0_empty * a0_dists[2] * any_pot_cooking_or_ready
    a1_go_get_plate = a1_empty * a1_dists[2] * any_pot_cooking_or_ready

    # adjacency to pot
    adj_thresh = 1.0 / max_dist + 1e-6
    a0_adjacent_pot = float(a0_dists[0] >= (1.0 - adj_thresh))
    a1_adjacent_pot = float(a1_dists[0] >= (1.0 - adj_thresh))

    # collision risk
    collision_risk = float(float(dist_agents) <= adj_thresh)

    goal_potentials = np.array([
        a0_deliver_onion, a1_deliver_onion,
        a0_pickup_soup, a1_pickup_soup,
        a0_deliver_dish, a1_deliver_dish,
        a0_go_get_onion, a1_go_get_onion,
        a0_go_get_plate, a1_go_get_plate,
        a0_adjacent_pot, a1_adjacent_pot,
        collision_risk,
        any_pot_needs_onions, any_pot_cooking_or_ready, any_pot_ready,
    ], dtype=np.float32)

    # --- Concatenate (49 dims) ---
    obs = np.concatenate([
        np.array([a0_x, a0_y], dtype=np.float32),
        a0_inv,
        a0_dir,
        np.array([a1_x, a1_y], dtype=np.float32),
        a1_inv,
        a1_dir,
        np.array([float(dist_agents)], dtype=np.float32),
        pot_feats,
        a0_dists,
        a1_dists,
        np.array([time_frac, 1.0], dtype=np.float32),
        goal_potentials,
    ])
    return obs


def potential(state: State, env: Overcooked) -> Tuple[float, float]:
    """Compute potential-based shaping value for each agent.

    Returns (Φ_0, Φ_1).  Use  γ·Φ(s') − Φ(s)  as an additive reward.
    The potential encodes pipeline progress + proximity to next sub-goal,
    giving dense per-step signal without altering the optimal policy.
    """
    obs = extract_obs(state, env)
    # Goal potentials start at index 33
    # (after 2+3+4+2+3+4+1+4+4+4+1+1=33)
    # a0_deliver_onion=33, a1_deliver_onion=34
    # a0_pickup_soup=35, a1_pickup_soup=36
    # a0_deliver_dish=37, a1_deliver_dish=38
    # a0_go_get_onion=39, a1_go_get_onion=40
    # a0_go_get_plate=41, a1_go_get_plate=42
    a0_inv = obs[2:5]
    a1_inv = obs[11:14]
    a0_has_onion, a0_has_plate, a0_has_dish = a0_inv[0], a0_inv[1], a0_inv[2]
    a1_has_onion, a1_has_plate, a1_has_dish = a1_inv[0], a1_inv[1], a1_inv[2]

    any_pot_ready = obs[48]  # last element of goal_potentials

    def _agent_potential(has_onion, has_plate, has_dish, deliver_onion,
                         pickup_soup, deliver_dish, go_get_onion, go_get_plate):
        stage = (has_onion * 1.0
                 + has_plate * any_pot_ready * 2.0
                 + has_dish * 3.0)
        prox = max(deliver_onion, pickup_soup, deliver_dish,
                   go_get_onion, go_get_plate)
        return stage + 0.5 * prox

    phi0 = _agent_potential(a0_has_onion, a0_has_plate, a0_has_dish,
                            obs[33], obs[35], obs[37], obs[39], obs[41])
    phi1 = _agent_potential(a1_has_onion, a1_has_plate, a1_has_dish,
                            obs[34], obs[36], obs[38], obs[40], obs[42])
    return float(phi0), float(phi1)


def get_obs_dim(env: Overcooked) -> int:
    """Return observation dimension for the given layout (single-pot).

    2 (a0 pos) + 3 (a0 inv) + 4 (a0 dir)
    + 2 (a1 pos) + 3 (a1 inv) + 4 (a1 dir) + 1 (dist_agents)
    + 4 (pot feats) + 4 (a0 dists) + 4 (a1 dists) + 1 (time_frac) + 1 (bias)
    + 16 (goal_potentials) = 49
    """
    return 49


def get_feature_dim(env: Overcooked) -> int:
    """Return total feature dimension for phi(obs, a0, a1)."""
    obs_dim = get_obs_dim(env)
    A = NUM_ACTIONS
    return obs_dim + A + A + obs_dim * A + obs_dim * A + A * A


def phi_single(obs: np.ndarray, a0: int, a1: int) -> np.ndarray:
    """Compute phi(obs, a0, a1) for one joint action."""
    A = NUM_ACTIONS
    a0_oh = np.zeros(A, dtype=np.float32)
    a0_oh[a0] = 1.0
    a1_oh = np.zeros(A, dtype=np.float32)
    a1_oh[a1] = 1.0

    obs_a0 = np.outer(obs, a0_oh).ravel()
    obs_a1 = np.outer(obs, a1_oh).ravel()
    a0a1 = np.outer(a0_oh, a1_oh).ravel()

    return np.concatenate([obs, a0_oh, a1_oh, obs_a0, obs_a1, a0a1])


def phi_all_actions(obs: np.ndarray) -> np.ndarray:
    """Compute phi for all joint actions at the given observation."""
    A = NUM_ACTIONS
    D_obs = obs.shape[0]
    D = D_obs + A + A + D_obs * A + D_obs * A + A * A

    a0_idx = np.repeat(np.arange(A, dtype=np.int32), A)
    a1_idx = np.tile(np.arange(A, dtype=np.int32), A)
    a0_oh = np.eye(A, dtype=np.float32)[a0_idx]  # [A*A, A]
    a1_oh = np.eye(A, dtype=np.float32)[a1_idx]  # [A*A, A]

    obs_rep = np.broadcast_to(obs.astype(np.float32), (A * A, D_obs))
    obs_a0 = np.einsum("bi,bj->bij", obs_rep, a0_oh, optimize=True).reshape(A * A, D_obs * A)
    obs_a1 = np.einsum("bi,bj->bij", obs_rep, a1_oh, optimize=True).reshape(A * A, D_obs * A)
    a0a1 = np.einsum("bi,bj->bij", a0_oh, a1_oh, optimize=True).reshape(A * A, A * A)

    phi_flat = np.concatenate([obs_rep, a0_oh, a1_oh, obs_a0, obs_a1, a0a1], axis=1)
    return phi_flat.astype(np.float32, copy=False).reshape(A, A, D)


def phi_all_actions_batch(obs_batch: np.ndarray) -> np.ndarray:
    """Compute phi for all joint actions for a batch of observations.

    Args:
        obs_batch: [B, D_obs]
    Returns:
        [B, A, A, D]
    """
    obs_batch = np.asarray(obs_batch, dtype=np.float32)
    B, D_obs = obs_batch.shape
    A = NUM_ACTIONS
    D = D_obs + A + A + D_obs * A + D_obs * A + A * A

    a0_idx = np.repeat(np.arange(A, dtype=np.int32), A)
    a1_idx = np.tile(np.arange(A, dtype=np.int32), A)
    a0_oh = np.eye(A, dtype=np.float32)[a0_idx]  # [A*A, A]
    a1_oh = np.eye(A, dtype=np.float32)[a1_idx]  # [A*A, A]

    BA = B * A * A
    obs_rep = np.repeat(obs_batch, A * A, axis=0)  # [B*A*A, D_obs]
    a0_rep = np.tile(a0_oh, (B, 1))
    a1_rep = np.tile(a1_oh, (B, 1))

    obs_a0 = np.einsum("bi,bj->bij", obs_rep, a0_rep, optimize=True).reshape(BA, D_obs * A)
    obs_a1 = np.einsum("bi,bj->bij", obs_rep, a1_rep, optimize=True).reshape(BA, D_obs * A)
    a0a1 = np.einsum("bi,bj->bij", a0_rep, a1_rep, optimize=True).reshape(BA, A * A)

    phi_flat = np.concatenate([obs_rep, a0_rep, a1_rep, obs_a0, obs_a1, a0a1], axis=1)
    return phi_flat.astype(np.float32, copy=False).reshape(B, A, A, D)
