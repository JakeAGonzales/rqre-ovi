import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, RegularPolygon

THEME = {
    "fig_bg":       "#F5F3F7",
    "floor_1":      "#F5F3F7",
    "floor_2":      "#EDEAF2",
    "wall":         "#3D3555",
    "wall_edge":    "#2B2440",
    "wall_hi":      "#524A6B",
    "stag":         "#7C6FE0",
    "stag_glow":    "#A89EF0",
    "hare":         "#E06F8B",
    "hare_glow":    "#F0A0B4",
    "spawn":        "#C8C0DD",
    "agent_0":      "#5B8DEE",
    "agent_1":      "#9B59B6",
    "skin":         "#F0DAC4",
    "text":         "#2B2440",
    "hud_bg":       "#FFFFFF",
    "chose_glow":   "#58D68D",
    "grid_line":    "#DDD8E4",
}

EMPTY = 0
WALL = 1
RESOURCE_STAG = 2
RESOURCE_HARE = 3
SPAWN_POINT = 4


def _draw_wall(ax, x, y):
    base = FancyBboxPatch(
        (x + 0.02, y + 0.02), 0.96, 0.96,
        boxstyle="round,pad=0,rounding_size=0.08",
        fc=THEME["wall"], ec=THEME["wall_edge"], lw=0.6, zorder=1,
    )
    ax.add_patch(base)
    hi = FancyBboxPatch(
        (x + 0.15, y + 0.55), 0.7, 0.3,
        boxstyle="round,pad=0,rounding_size=0.06",
        fc=THEME["wall_hi"], ec="none", alpha=0.35, zorder=2,
    )
    ax.add_patch(hi)


def _draw_floor(ax, x, y, parity):
    color = THEME["floor_1"] if parity else THEME["floor_2"]
    ax.add_patch(Rectangle((x, y), 1, 1, fc=color, ec="none", zorder=0))


def _draw_spawn(ax, x, y):
    _draw_floor(ax, x, y, True)
    c = Circle((x + 0.5, y + 0.5), 0.22, fc=THEME["spawn"],
               ec=THEME["spawn"], lw=0, alpha=0.5, zorder=1)
    ax.add_patch(c)
    c2 = Circle((x + 0.5, y + 0.5), 0.1, fc=THEME["spawn"],
                ec="none", alpha=0.3, zorder=1)
    ax.add_patch(c2)


def _draw_resource(ax, x, y, rtype):
    _draw_floor(ax, x, y, True)
    if rtype == "stag":
        color, glow = THEME["stag"], THEME["stag_glow"]
        label = "S"
    else:
        color, glow = THEME["hare"], THEME["hare_glow"]
        label = "H"

    glow_c = Circle((x + 0.5, y + 0.5), 0.30, fc=glow, alpha=0.25, zorder=2)
    ax.add_patch(glow_c)

    diamond = RegularPolygon((x + 0.5, y + 0.5), 4, radius=0.22,
                             fc=color, ec="white", lw=0.8, zorder=3)
    ax.add_patch(diamond)

    ax.text(x + 0.5, y + 0.5, label, ha="center", va="center",
            fontsize=6, fontweight="bold", color="white", zorder=4)


def _draw_agent(ax, cx, cy, agent_idx, inv_val):
    color = THEME[f"agent_{agent_idx}"]

    # Glow if carrying something
    if inv_val >= 0:
        glow_color = THEME["stag"] if inv_val == 0 else THEME["hare"]
        glow = Circle((cx, cy), 0.42, fc=glow_color,
                      alpha=0.25, zorder=19)
        ax.add_patch(glow)

    # Body
    body = Circle((cx, cy - 0.04), 0.32, fc=color, ec="white", lw=0.8, zorder=20)
    ax.add_patch(body)

    # Head
    head = Circle((cx, cy + 0.14), 0.18, fc=THEME["skin"], ec=color, lw=0.6, zorder=22)
    ax.add_patch(head)

    # Eyes
    for sign in [-1, 1]:
        ex = cx + 0.07 * sign
        ey = cy + 0.14
        ax.add_patch(Circle((ex, ey), 0.03, fc=THEME["text"], zorder=23))

    # Inventory label (S/H) or agent number
    if inv_val == 0:
        label = "S"
        label_color = THEME["stag_glow"]
    elif inv_val == 1:
        label = "H"
        label_color = THEME["hare_glow"]
    else:
        label = str(agent_idx)
        label_color = "white"
    ax.text(cx, cy - 0.08, label, ha="center", va="center",
            fontsize=6, fontweight="bold", color=label_color, zorder=24)

def render_state(state, ax=None, step_num=0, total_rewards=None, title_extra=""):
    grid = np.array(state.grid)
    H, W = grid.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(W * 0.6, 8), max(H * 0.6, 5)))
        fig.patch.set_facecolor(THEME["fig_bg"])
    else:
        fig = ax.figure

    ax.clear()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    player_pos = np.array(state.player_pos)
    player_inv = np.array(state.player_inv)

    agent_cells = {}
    for i in range(2):
        r, c = int(player_pos[i, 0]), int(player_pos[i, 1])
        agent_cells[(r, c)] = i

    for row in range(H):
        for col in range(W):
            dx = col
            dy = H - 1 - row

            cell = int(grid[row, col])

            if (row, col) in agent_cells:
                _draw_floor(ax, dx, dy, (row + col) % 2 == 0)
                idx = agent_cells[(row, col)]
                _draw_agent(ax, dx + 0.5, dy + 0.5, idx, int(player_inv[idx]))
            elif cell == WALL:
                _draw_wall(ax, dx, dy)
            elif cell == RESOURCE_STAG:
                _draw_resource(ax, dx, dy, "stag")
            elif cell == RESOURCE_HARE:
                _draw_resource(ax, dx, dy, "hare")
            elif cell == SPAWN_POINT:
                _draw_spawn(ax, dx, dy)
            else:
                _draw_floor(ax, dx, dy, (row + col) % 2 == 0)

    # HUD
    r_str = ""
    if total_rewards is not None:
        r_str = f"  |  P0: {total_rewards[0]:.0f}  P1: {total_rewards[1]:.0f}"
    title = f"Step {step_num}{r_str}"
    if title_extra:
        title += f"  |  {title_extra}"
    ax.set_title(title, fontsize=11, fontweight="bold", color=THEME["text"],
                pad=8, fontfamily="monospace")

    # Legend
    legend_items = [
        mpatches.Patch(fc=THEME["stag"], label="Stag"),
        mpatches.Patch(fc=THEME["hare"], label="Hare"),
        mpatches.Patch(fc=THEME["agent_0"], label="Agent 0"),
        mpatches.Patch(fc=THEME["agent_1"], label="Agent 1"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4,
             fontsize=7, framealpha=0.8, edgecolor=THEME["grid_line"],
             bbox_to_anchor=(0.5, -0.06))

    return fig, ax


def animate_rollout(state_seq, filename="staghunt.gif", fps=5):
    s0 = state_seq[0] if not isinstance(state_seq[0], tuple) else state_seq[0][0]
    H, W = np.array(s0.grid).shape
    fig, ax = plt.subplots(figsize=(max(W * 0.6, 8), max(H * 0.6, 5)))
    fig.patch.set_facecolor(THEME["fig_bg"])

    cum_rewards = np.zeros(2)

    def _update(frame_idx):
        nonlocal cum_rewards
        item = state_seq[frame_idx]
        if isinstance(item, tuple):
            state, rewards = item
            cum_rewards += np.array(rewards)
        else:
            state = item
        render_state(state, ax=ax, step_num=frame_idx, total_rewards=cum_rewards)

    anim = animation.FuncAnimation(fig, _update, frames=len(state_seq),
                                   interval=1000 // fps)
    anim.save(filename, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    import jax
    from staghunt.staghunt import reset, step, NUM_ACTIONS

    key = jax.random.PRNGKey(42)
    state = reset(key)

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(THEME["fig_bg"])
    render_state(state, ax=ax)
    plt.tight_layout()
    plt.savefig("staghunt_frame.png", dpi=150, bbox_inches="tight")
    plt.show()
