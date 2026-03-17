"""LFA feature extraction for Spatial Stag Hunt."""

import jax
import jax.numpy as jnp

from staghunt.staghunt import (
    State,
    NUM_ACTIONS,
    MAP_H,
    MAP_W,
    MAX_FRAMES,
    STAG_POSITIONS,
    HARE_POSITIONS,
)

A = NUM_ACTIONS
MAX_DIST = float(MAP_H + MAP_W)

OBS_DIM = 15

FEATURE_DIM = OBS_DIM + A + A + OBS_DIM * A + OBS_DIM * A + A * A


def _nearest_dist(pos: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Manhattan distance to nearest position."""
    dists = jnp.sum(jnp.abs(positions - pos[None, :]), axis=1)
    return jnp.min(dists).astype(jnp.float32)


@jax.jit
def extract_features(state: State) -> jnp.ndarray:
    """Extract 15-dim symmetric observation (full info, both players)."""
    pos0 = state.player_pos[0]
    pos1 = state.player_pos[1]

    p0_row = pos0[0].astype(jnp.float32) / float(MAP_H - 1)
    p0_col = pos0[1].astype(jnp.float32) / float(MAP_W - 1)
    p1_row = pos1[0].astype(jnp.float32) / float(MAP_H - 1)
    p1_col = pos1[1].astype(jnp.float32) / float(MAP_W - 1)

    p0_ds = _nearest_dist(pos0, STAG_POSITIONS) / MAX_DIST
    p0_dh = _nearest_dist(pos0, HARE_POSITIONS) / MAX_DIST
    p1_ds = _nearest_dist(pos1, STAG_POSITIONS) / MAX_DIST
    p1_dh = _nearest_dist(pos1, HARE_POSITIONS) / MAX_DIST

    p0_has_stag = (state.player_inv[0] == 0).astype(jnp.float32)
    p0_has_hare = (state.player_inv[0] == 1).astype(jnp.float32)
    p1_has_stag = (state.player_inv[1] == 0).astype(jnp.float32)
    p1_has_hare = (state.player_inv[1] == 1).astype(jnp.float32)

    inter_dist = jnp.sum(jnp.abs(pos0 - pos1)).astype(jnp.float32) / MAX_DIST

    round_frac = state.round_step.astype(jnp.float32) / max(float(MAX_FRAMES), 1.0)

    return jnp.array(
        [
            p0_row, p0_col,
            p1_row, p1_col,
            p0_ds, p0_dh,
            p1_ds, p1_dh,
            p0_has_stag, p0_has_hare,
            p1_has_stag, p1_has_hare,
            inter_dist,
            round_frac,
            1.0,
        ],
        dtype=jnp.float32,
    )


def get_feature_dim() -> int:
    return int(FEATURE_DIM)


@jax.jit
def phi_single(state: State, a0: jnp.ndarray, a1: jnp.ndarray) -> jnp.ndarray:
    state_feats = extract_features(state)[None, :]
    a0_b = jnp.asarray(a0, dtype=jnp.int32).reshape(1)
    a1_b = jnp.asarray(a1, dtype=jnp.int32).reshape(1)
    return phi_batch_jax(state_feats, a0_b, a1_b)[0]


@jax.jit
def phi_all_actions(state: State) -> jnp.ndarray:
    state_feats = extract_features(state)
    a0_all = jnp.repeat(jnp.arange(A, dtype=jnp.int32), A)
    a1_all = jnp.tile(jnp.arange(A, dtype=jnp.int32), A)
    sf_batch = jnp.broadcast_to(state_feats[None, :], (A * A, OBS_DIM))
    phi_flat = phi_batch_jax(sf_batch, a0_all, a1_all)
    return phi_flat.reshape(A, A, FEATURE_DIM)


@jax.jit
def phi_batch_jax(
    state_feat_batch: jnp.ndarray,
    a0_batch: jnp.ndarray,
    a1_batch: jnp.ndarray,
) -> jnp.ndarray:
    """phi = [obs, a0_oh, a1_oh, obs x a0, obs x a1, a0 x a1]"""
    sf = jnp.asarray(state_feat_batch, dtype=jnp.float32)
    a0b = jnp.asarray(a0_batch, dtype=jnp.int32)
    a1b = jnp.asarray(a1_batch, dtype=jnp.int32)
    B = sf.shape[0]

    a0_oh = jax.nn.one_hot(a0b, A, dtype=jnp.float32)
    a1_oh = jax.nn.one_hot(a1b, A, dtype=jnp.float32)

    a0a1 = jnp.einsum("bi,bj->bij", a0_oh, a1_oh).reshape(B, A * A)

    obs_x_a0 = jnp.einsum("bi,bj->bij", sf, a0_oh).reshape(B, OBS_DIM * A)
    obs_x_a1 = jnp.einsum("bi,bj->bij", sf, a1_oh).reshape(B, OBS_DIM * A)

    return jnp.concatenate([sf, a0_oh, a1_oh, obs_x_a0, obs_x_a1, a0a1], axis=1)
