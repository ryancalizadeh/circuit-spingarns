"""
Figures for report_nested_dissection.md, written to report_assets/nested_dissection/. Everything is computed from the
ordering in algorithms/ordering.py and the crossbar's graph: the nonzero pattern of the nodal matrix depends only on the
array's shape, so no figure depends on circuit values, seeds or the machine. Prints the numbers the report quotes.

    python nested_dissection_figures.py
"""
import ctypes
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from cvxopt import cholmod
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

from Config import GROUND, BranchKind, Config
from algorithms.ordering import LEAF_SIZE, _count, _expand, _leaf_blocks, _split, nested_dissection
from algorithms.spingarns import _CholmodFactor, _capsule_pointer
from plotting import GRID, INK, INK_2, MUTED, SERIES_COLORS, SURFACE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report_assets" / "nested_dissection"
# Separators by recursion depth: one blue ramp, darkest for the first cut (validated as an ordinal ramp on SURFACE).
DEPTH = ["#0d366b", "#256abf", "#5598e7", "#86b6ef"]
FILL = SERIES_COLORS[1]  # fill entries and fill edges
FIRST, SECOND = SERIES_COLORS[2], SERIES_COLORS[3]  # the two halves of a cut, in figure 3
LEAF = "#c9c8c2"  # nodes not in any separator
FADED = "#e6e5e0"
WHITE = "#ffffff"
OFFSET = 0.2  # word node up-left of its cross-point, bit node down-right


# ---------------------------------------------------------------------------------------------------------- the model

def nodal_matrix(cfg):
    """K = A A^T: the nodal matrix with every port resistance 1 ohm. Its pattern, the only thing used here, is that of
    A Gamma^-1 A^T for any Gamma."""
    A = cfg.incidence()
    return (A @ A.T).tocsr()


def name(cfg, n):
    """'w01' for the word node w[0, 1], 'b21' for the bit node b[2, 1]."""
    pq, q = cfg.p * cfg.q, cfg.q
    return f"w{n // q}{n % q}" if n < pq else f"b{(n - pq) // q}{(n - pq) % q}"


def dissection(cfg, leaf_size=LEAF_SIZE, max_depth=None):
    """
    nested_dissection's recursion, also stopped at max_depth, keeping each numbered group in elimination order as
    (kind, depth, nodes, region) with kind "leaf" or "separator". The groups' nodes, concatenated, are the ordering;
    without max_depth it is exactly nested_dissection(cfg, leaf_size).
    """
    groups = []

    def dissect(region, depth):
        if _count(region) <= leaf_size or depth == max_depth:
            groups.append(("leaf", depth, _expand(cfg.q, _leaf_blocks(cfg, region)), region))
            return
        separator, first, second = _split(cfg, region)
        dissect(first, depth + 1)
        dissect(second, depth + 1)
        groups.append(("separator", depth, _expand(cfg.q, [separator]), region))

    dissect((0, cfg.p, 0, cfg.q, 0, 0), 0)
    return groups


def ordering(groups):
    return np.concatenate([nodes for _, _, nodes, _ in groups])


def eliminate(K, perm, snapshots=()):
    """
    The elimination game on K's graph in the order perm, on dense booleans (small matrices only). Returns the pattern of
    K[perm][:, perm], the lower-triangular pattern of its Cholesky factor L, the number of fill entries each step
    created, and the graph (symmetric pattern, in positions) after each step count in snapshots.
    """
    S = (K[perm][:, perm] != 0).toarray()
    original, created, saved = S.copy(), [], {}
    for k in range(len(perm)):
        if k in snapshots:
            saved[k] = S.copy()
        later = np.flatnonzero(S[k + 1:, k]) + k + 1
        clique = np.ix_(later, later)
        created.append(int(np.tril(~S[clique]).sum()))
        S[clique] = True
    if len(perm) in snapshots:
        saved[len(perm)] = S.copy()
    return original, np.tril(S), created, saved


def column_counts(K, perm=None):
    """Nonzeros in each column of L, diagonal included, from CHOLMOD's symbolic analysis alone (simplicial, so nothing
    the size of L is allocated): in the order perm, or in CHOLMOD's own (AMD here) for perm=None."""
    n = K.shape[0]
    lower = sp.tril(K, format="coo")
    Kc = cvxopt.spmatrix(cvxopt.matrix(lower.data), cvxopt.matrix(lower.row.astype(np.int64)),
                         cvxopt.matrix(lower.col.astype(np.int64)), (n, n))
    saved = dict(cholmod.options)
    cholmod.options.update(nmethods=0 if perm is None else 1, supernodal=0)
    try:
        F = cholmod.symbolic(Kc, uplo="L", **({} if perm is None else {"p": cvxopt.matrix(np.asarray(perm, np.int64))}))
    finally:
        cholmod.options.clear()
        cholmod.options.update(saved)
    f = _CholmodFactor.from_address(_capsule_pointer(F, b"CHOLMOD FACTOR D L"))
    return np.ctypeslib.as_array((ctypes.c_int64 * n).from_address(f.ColCount)).copy()


# ------------------------------------------------------------------------------------------------------------ drawing

def save(fig, name):
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def text_on(color):
    """Ink or white, whichever reads on a fill of this color."""
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else WHITE


def positions(cfg):
    r, c = np.indices((cfg.p, cfg.q))
    r, c = r.ravel(), c.ravel()
    return np.concatenate([c - OFFSET, c + OFFSET]), np.concatenate([r - OFFSET, r + OFFSET])


def graph_edges(cfg):
    """(tail, head) of every branch between two nodes, by kind: wires and devices (sources and loads go to ground)."""
    edges = {}
    for kind in (BranchKind.WORD_WIRE, BranchKind.BIT_WIRE, BranchKind.DEVICE):
        s = cfg.slices[kind]
        edges[kind] = np.column_stack([cfg.tail[s], cfg.head[s]])
    return edges


def draw_crossbar(ax, cfg, colors, size, rings=None, labels=None, inner=None, edge_width=1.0, stubs=False, alpha=None):
    """
    The crossbar's graph: word nodes as circles, bit nodes as squares, word and bit wires solid, devices thin. colors
    and size are per node (or one value); rings marks nodes drawn with a heavy outline; labels writes each node's name
    beside it and inner a number inside it. stubs draws the sources and loads that run to ground.
    """
    x, y = positions(cfg)
    edges = graph_edges(cfg)
    for kind, width, color in ((BranchKind.WORD_WIRE, 1.6 * edge_width, INK_2),
                               (BranchKind.BIT_WIRE, 1.6 * edge_width, INK_2),
                               (BranchKind.DEVICE, 0.9 * edge_width, MUTED)):
        e = edges[kind]
        segments = np.stack([np.column_stack([x[e[:, 0]], y[e[:, 0]]]), np.column_stack([x[e[:, 1]], y[e[:, 1]]])], 1)
        ax.add_collection(LineCollection(segments, colors=color, linewidths=width, zorder=1))
    if stubs:
        for r in range(cfg.p):
            w = cfg.word_node(r, 0)
            ax.plot([x[w] - 0.55, x[w]], [y[w], y[w]], color=MUTED, lw=1.2, ls=(0, (2, 1.5)), zorder=1)
            ax.text(x[w] - 0.6, y[w], f"E{r}", ha="right", va="center", fontsize=8, color=INK_2)
        for c in range(cfg.q):
            b = cfg.bit_node(cfg.p - 1, c)
            ax.plot([x[b], x[b]], [y[b], y[b] + 0.5], color=MUTED, lw=1.2, ls=(0, (2, 1.5)), zorder=1)
            ax.text(x[b], y[b] + 0.56, "load", ha="center", va="top", fontsize=7.5, color=INK_2)
    n, pq = cfg.num_nodes, cfg.p * cfg.q
    colors = np.broadcast_to(np.asarray(colors, dtype=object), (n,))
    size = np.broadcast_to(np.asarray(size, dtype=float), (n,))
    ring = np.zeros(n, bool) if rings is None else np.isin(np.arange(n), rings)
    for marker, nodes in (("o", np.arange(pq)), ("s", np.arange(pq, n))):
        ax.scatter(x[nodes], y[nodes], s=size[nodes], c=list(colors[nodes]), marker=marker, zorder=3, alpha=alpha,
                   edgecolors=[INK if ring[k] else SURFACE for k in nodes],
                   linewidths=[2.2 if ring[k] else 0.8 for k in nodes])
    if labels:
        for k in range(n):
            dx, dy, ha, va = (-0.17, -0.17, "right", "bottom") if k < pq else (0.17, 0.17, "left", "top")
            ax.text(x[k] + dx, y[k] + dy, name(cfg, k), ha=ha, va=va, fontsize=7.5, color=INK_2, zorder=4)
    if inner is not None:
        for k in range(n):
            ax.text(x[k], y[k], str(inner[k]), ha="center", va="center", fontsize=7, color=text_on(colors[k]),
                    zorder=4)
    ax.set_xlim(-0.75 if not stubs else -1.25, cfg.q - 0.25)
    ax.set_ylim(cfg.p - 0.25 + (0.45 if stubs else 0), -0.75)
    ax.set_aspect("equal")
    ax.axis("off")


def style_matrix(ax, n, title, labels=None):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    ax.xaxis.tick_top()
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=7.5, length=2)
    if labels is not None:
        ax.set_xticks(range(n), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(n), labels, fontsize=7)


def draw_entries(ax, M, color, s):
    i, j = np.nonzero(M)
    ax.scatter(j, i, s=s, c=color, marker="s", linewidths=0, zorder=2)


def draw_pattern(ax, original, fill=None, s=20.0):
    """Original nonzeros in ink, fill (entries of L that K lacks) in orange."""
    draw_entries(ax, original, INK, s)
    if fill is not None:
        draw_entries(ax, fill, FILL, s)


def draw_groups(ax, groups, labels=None):
    """Shade each separator's diagonal block in its depth's blue and rule a line at every group boundary; labels
    optionally names each group above its block."""
    start = 0
    for k, (kind, depth, nodes, _) in enumerate(groups):
        size = len(nodes)
        if size == 0:
            continue
        if kind == "separator":
            ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size, facecolor=DEPTH[min(depth, 3)], alpha=0.22,
                                   linewidth=0, zorder=0))
        if start > 0:
            ax.axhline(start - 0.5, color=MUTED, lw=0.6, zorder=1)
            ax.axvline(start - 0.5, color=MUTED, lw=0.6, zorder=1)
        if labels is not None:
            ax.text(start + size / 2 - 0.5, -0.9, labels[k], ha="center", va="bottom", fontsize=8.5, color=INK,
                    transform=ax.transData)
        start += size


def group_colors(cfg, groups, leaf=LEAF):
    colors = np.empty(cfg.num_nodes, dtype=object)
    for kind, depth, nodes, _ in groups:
        colors[nodes] = DEPTH[min(depth, 3)] if kind == "separator" else leaf
    return colors


def legend_handles(entries):
    """Marker-only legend entries: (label, marker, face color, edge color, edge width)."""
    return [Line2D([], [], linestyle="none", marker=m, markersize=8, markerfacecolor=f, markeredgecolor=e,
                   markeredgewidth=w, label=label) for label, m, f, e, w in entries]


def legend(ax, handles, **kwargs):
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, **kwargs)


# ------------------------------------------------------------------------------------------------------------ figures

def figure_star():
    """Figure 1: a star eliminated hub first (every leaf joins every other) and hub last (no fill)."""
    n = 7
    angles = np.pi / 2 - 2 * np.pi * np.arange(6) / 6
    hub_xy, leaf_xy = np.zeros((1, 2)), np.column_stack([np.cos(angles), np.sin(angles)])
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    for row, (hub, title) in enumerate(((0, "hub eliminated first"), (6, "hub eliminated last"))):
        leaves = [k for k in range(n) if k != hub]
        xy = np.empty((n, 2))
        xy[hub], xy[leaves] = hub_xy, leaf_xy
        K = np.eye(n, dtype=bool)
        K[hub, :] = K[:, hub] = True
        L = np.tril(K.copy())
        for k in range(n):  # the elimination game
            later = [j for j in range(k + 1, n) if L[j, k] or L[k, j]]
            for a in later:
                for b in later:
                    if a > b:
                        L[a, b] = True
        fill = L & ~np.tril(K)
        ax = axes[row, 0]
        if hub == 0:
            for a in range(1, n):
                for b in range(a + 1, n):
                    ax.plot(*xy[[a, b]].T, color=FILL, lw=1.1, ls=(0, (3, 2)), zorder=1)
        for leaf in leaves:
            ax.plot(*xy[[hub, leaf]].T, color=INK_2, lw=1.6, zorder=2)
        ax.scatter(*xy.T, s=420, c=[DEPTH[0] if k == hub else LEAF for k in range(n)], edgecolors=SURFACE,
                   linewidths=1.2, zorder=3)
        for k in range(n):
            ax.text(*xy[k], str(k), ha="center", va="center", fontsize=9.5,
                    color=text_on(DEPTH[0] if k == hub else LEAF), zorder=4)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        caption = ("eliminating node 0 joins all six\nleaves into a clique: 15 fill edges"
                   if hub == 0 else "each leaf has one neighbour left\nwhen it goes: no fill")
        ax.set_title(f"({'ad'[row]}) {title}\n{caption}", loc="left", color=INK, fontsize=10)
        style_matrix(axes[row, 1], n, f"({'be'[row]}) K in this order: {int(K.sum())} nonzeros")
        draw_pattern(axes[row, 1], K, s=230)
        style_matrix(axes[row, 2], n, f"({'cf'[row]}) L: {int(L.sum())} nonzeros, {int(fill.sum())} fill")
        draw_pattern(axes[row, 2], np.tril(K), fill, s=230)
    handles = legend_handles([("nonzero of K", "s", INK, INK, 0), ("fill: nonzero of L, not of K", "s", FILL, FILL, 0)])
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "star.png")


def figure_crossbar(cfg):
    """Figure 2: the 3x3 crossbar's graph in its natural numbering, and the pattern of its nodal matrix."""
    K = nodal_matrix(cfg)
    n, pq = cfg.num_nodes, cfg.p * cfg.q
    labels = [name(cfg, k) for k in range(n)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axes[0]
    colors = [SERIES_COLORS[0]] * pq + [SERIES_COLORS[2]] * pq
    draw_crossbar(ax, cfg, colors, 330, labels=True, inner=np.arange(n), edge_width=1.3, stubs=True)
    ax.set_title("(a) A 3×3 crossbar: node k at index k (natural order)", loc="left", color=INK, fontsize=10.5)
    handles = legend_handles([("word node w[r, c]  (index r·q + c)", "o", SERIES_COLORS[0], SURFACE, 0.8),
                              ("bit node b[r, c]  (index p·q + r·q + c)", "s", SERIES_COLORS[2], SURFACE, 0.8)])
    handles += [Line2D([], [], color=INK_2, lw=2, label="word wire (along a row) / bit wire (down a column)"),
                Line2D([], [], color=MUTED, lw=1.2, label="device, w[r, c] to b[r, c]"),
                Line2D([], [], color=MUTED, lw=1.2, ls=(0, (2, 1.5)), label="source / load to ground (diagonal only)")]
    legend(ax, handles, loc="upper center", bbox_to_anchor=(0.5, -0.0), ncol=2)
    ax = axes[1]
    M = (K != 0).toarray()
    style_matrix(ax, n, f"(b) Pattern of K = A Γ⁻¹ Aᵀ: {int(M.sum())} nonzeros, {n} of them diagonal", labels)
    for start, size, color in ((0, pq, SERIES_COLORS[0]), (pq, pq, SERIES_COLORS[2])):
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), size, size, facecolor=color, alpha=0.10, lw=0, zorder=0))
    ax.axhline(pq - 0.5, color=MUTED, lw=0.6)
    ax.axvline(pq - 0.5, color=MUTED, lw=0.6)
    draw_pattern(ax, M, s=150)
    notes = [((4.6, -0.3), "word wires:\nw[r, c] – w[r, c+1]"), ((14.6, 8.75), "bit wires:\nb[r, c] –\nb[r+1, c]"),
             ((12.6, 1.8), "devices: w[r, c] – b[r, c],\na diagonal p·q = 9 off"), ((3.6, 10.8), "devices\n(symmetric)")]
    for (x0, y0), text in notes:
        ax.text(x0, y0, text, ha="left", va="top", fontsize=8, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.3"))
    fig.tight_layout()
    save(fig, "crossbar_graph.png")


def figure_cuts():
    """Figure 3: the first cut of an 8x8 crossbar, and the cut of its first half."""
    cfg = Config(8, 8, seed=0)
    n = cfg.num_nodes
    root = (0, 8, 0, 8, 0, 0)
    s0, first, second = _split(cfg, root)
    s1, first_top, first_bottom = _split(cfg, first)
    nodes = lambda region: _expand(cfg.q, _leaf_blocks(cfg, region))
    s0, s1 = _expand(cfg.q, [s0]), _expand(cfg.q, [s1])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.4), facecolor=SURFACE)

    colors = np.empty(n, dtype=object)
    colors[nodes(first)], colors[nodes(second)], colors[s0] = FIRST, SECOND, DEPTH[0]
    stranded = np.array([cfg.bit_node(r, 4) for r in range(8)])
    draw_crossbar(axes[0], cfg, colors, 95, rings=stranded)
    axes[0].set_title("(a) First cut, across the wider side: separator = word nodes of column 4\n"
                      f"region {root}  →  first {first}, second {second}", loc="left", color=INK, fontsize=10)
    axes[0].annotate("bit nodes of column 4: their only sideways\nlinks run through the separator, so they are\n"
                     "stranded and join the first half", (4.2, 7.2), xytext=(1.0, 8.55), fontsize=8.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8), va="top")

    colors = np.full(n, FADED, dtype=object)
    colors[nodes(first_top)], colors[nodes(first_bottom)] = FIRST, SECOND
    colors[s0], colors[s1] = DEPTH[0], DEPTH[1]
    stranded = np.array([cfg.word_node(4, c) for c in range(4)])
    draw_crossbar(axes[1], cfg, colors, 95, rings=stranded)
    axes[1].set_title("(b) Second cut, of the first half (taller than wide): separator = bit nodes of row 4\n"
                      f"region {first}  →  first {first_top}, second {first_bottom}", loc="left", color=INK,
                      fontsize=10)
    axes[1].annotate("word nodes of row 4 are stranded\nthe same way and join the top", (1.8, 3.8),
                     xytext=(-0.6, 8.55), fontsize=8.5, color=INK_2, va="top",
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].annotate("the separator also takes b[4, 4],\nfrom the stranded column", (4.2, 4.2), xytext=(4.6, 8.55),
                     fontsize=8.5, color=INK_2, va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    axes[1].text(6.1, 3.5, "second half of (a):\ndissected on its own", ha="center", va="center", fontsize=8.5,
                 color=INK_2, zorder=5, bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.4"))
    for ax in axes:
        ax.set_ylim(9.0, -0.75)
    handles = legend_handles([("word node", "o", LEAF, SURFACE, 0.8), ("bit node", "s", LEAF, SURFACE, 0.8),
                              ("separator, first cut", "o", DEPTH[0], SURFACE, 0.8),
                              ("separator, second cut", "s", DEPTH[1], SURFACE, 0.8),
                              ("first half (numbered first)", "o", FIRST, SURFACE, 0.8),
                              ("second half", "o", SECOND, SURFACE, 0.8),
                              ("stranded line", "o", FIRST, INK, 2.2)])
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "cuts.png")


HAND = dict(names=["L1", "L2", "S2", "L3", "L4", "S3", "S1"])


def figure_hand_roles(cfg, groups):
    """Figure 4: the hand example's groups on the graph, with each node's elimination position, and the dissection
    tree."""
    perm = ordering(groups)
    position = np.empty(cfg.num_nodes, int)
    position[perm] = np.arange(cfg.num_nodes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1, 1.15]))
    ax = axes[0]
    draw_crossbar(ax, cfg, group_colors(cfg, groups), 420, labels=True, inner=position, edge_width=1.3)
    x, y = positions(cfg)
    spots = {"L1": (-0.62, 0.55), "L2": (-0.62, 2.05), "L3": (2.62, 0.55), "L4": (2.62, 2.05), "S1": (1.0, -0.62),
             "S2": (-0.62, 1.2), "S3": (2.62, 1.2)}
    for group, label in zip(groups, HAND["names"]):
        tx, ty = spots[label]
        ax.text(tx, ty, label, ha="center", va="center", fontsize=10, fontweight="bold",
                color=DEPTH[group[1]] if group[0] == "separator" else INK_2)
    ax.set_xlim(-0.95, 2.95)
    ax.set_title("(a) The ordering for leaf_size = 4: the number in each node is its\nposition in perm (0 = "
                 "eliminated first)", loc="left", color=INK, fontsize=10.5)
    handles = legend_handles([("leaf node", "o", LEAF, SURFACE, 0.8), ("separator, depth 0", "o", DEPTH[0], SURFACE, 0.8),
                              ("separator, depth 1", "o", DEPTH[1], SURFACE, 0.8)])
    legend(ax, handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) The dissection tree, numbered in post-order: left subtree,\nright subtree, then the separator",
                 loc="left", color=INK, fontsize=10.5)
    spots = {"S1": (0.5, 0.86), "S2": (0.26, 0.54), "S3": (0.74, 0.54), "L1": (0.13, 0.2), "L2": (0.38, 0.2),
             "L3": (0.62, 0.2), "L4": (0.87, 0.2)}
    start = 0
    info = {}
    for (kind, depth, nodes, region), label in zip(groups, HAND["names"]):
        info[label] = (kind, depth, nodes, region, start)
        start += len(nodes)
    for parent, children in (("S1", ("S2", "S3")), ("S2", ("L1", "L2")), ("S3", ("L3", "L4"))):
        for child in children:
            ax.plot(*zip(spots[parent], spots[child]), color=MUTED, lw=1.2, zorder=1)
    for label, (kind, depth, nodes, region, begin) in info.items():
        color = DEPTH[depth] if kind == "separator" else LEAF
        span = f"{begin}" if len(nodes) == 1 else f"{begin}–{begin + len(nodes) - 1}"
        what = "separator" if kind == "separator" else "leaf"
        text = (f"{label}: {what}, position{'s' * (len(nodes) > 1)} {span}\n" + " ".join(name(cfg, k) for k in nodes) +
                f"\nregion {region}")
        ax.text(*spots[label], text, ha="center", va="center", fontsize=8, color=text_on(color), zorder=3,
                linespacing=1.4, bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor=SURFACE))
    ax.text(0.5, 0.0, "Leaf region tuples hold the nodes: word rows [r0, r1+ew) × columns [c0, c1),\n"
            "bit rows [r0, r1) × columns [c0, c1+eb). Separator tuples are the region that was cut.",
            ha="center", va="bottom", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    save(fig, "hand_roles.png")


def passes_through(x, y, u, v, m, within=0.08):
    """Whether node m lies on the segment from node u to node v."""
    d = np.array([x[v] - x[u], y[v] - y[u]])
    t = np.dot([x[m] - x[u], y[m] - y[u]], d) / np.dot(d, d)
    return 0 < t < 1 and np.hypot(x[u] + t * d[0] - x[m], y[u] + t * d[1] - y[m]) < within


def figure_hand_elimination(cfg, groups):
    """Figure 5: the elimination graph of the hand example after 0, 7, 9 and 15 eliminations."""
    K = nodal_matrix(cfg)
    perm = ordering(groups)
    steps = (0, 7, 9, 15)
    original, _, created, saved = eliminate(K, perm, steps)
    colors = group_colors(cfg, groups)
    x, y = positions(cfg)
    titles = ["(a) Start: the graph of K", "(b) After L1 and L2 (7 eliminated)", "(c) After S2: left half gone",
              "(d) After L3, L4, S3: only S1 left"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.9), facecolor=SURFACE)
    for ax, k, title in zip(axes, steps, titles):
        S = saved[k]
        alive = perm[k:]
        for a in range(k, len(perm)):
            for b in range(k, a):
                if S[a, b]:
                    u, v = perm[a], perm[b]
                    is_fill = not original[a, b]
                    # an edge whose straight line would run through another node is drawn as an arc
                    blocked = any(passes_through(x, y, u, v, m) for m in range(cfg.num_nodes) if m not in (u, v))
                    ax.add_patch(FancyArrowPatch((x[u], y[u]), (x[v], y[v]), arrowstyle="-", shrinkA=0, shrinkB=0,
                                                 connectionstyle=f"arc3,rad={0.3 if blocked else 0}",
                                                 color=FILL if is_fill else INK_2, zorder=1 + is_fill,
                                                 lw=1.6 if is_fill else 1.2, ls=(0, (3, 2)) if is_fill else "-"))
        gone = perm[:k]
        for marker, sel in (("o", lambda m: m < cfg.p * cfg.q), ("s", lambda m: m >= cfg.p * cfg.q)):
            g = gone[sel(gone)]
            ax.scatter(x[g], y[g], s=110, marker=marker, facecolors="none", edgecolors=GRID, linewidths=1.2, zorder=2)
            a = alive[sel(alive)]
            ax.scatter(x[a], y[a], s=320, marker=marker, c=list(colors[a]), edgecolors=SURFACE, linewidths=0.8,
                       zorder=3)
        for node in alive:
            ax.text(x[node], y[node], name(cfg, node), ha="center", va="center", fontsize=6,
                    color=text_on(colors[node]), zorder=4)
        made = sum(created[:k])
        live_fill = int(np.tril(S[k:, k:] & ~original[k:, k:]).sum())
        ax.set_title(f"{title}\nfill created so far: {made}; dashed: {live_fill} between nodes still left",
                     loc="left", color=INK, fontsize=9.5)
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(2.6, -0.6)
        ax.set_aspect("equal")
        ax.axis("off")
    handles = [Line2D([], [], color=INK_2, lw=1.4, label="edge of K"),
               Line2D([], [], color=FILL, lw=1.6, ls=(0, (3, 2)), label="fill edge"),
               *legend_handles([("leaf node", "o", LEAF, SURFACE, 0.8),
                                ("separator, depth 0 / 1", "o", DEPTH[0], SURFACE, 0.8),
                                ("eliminated", "o", "none", GRID, 1.2)])]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "hand_elimination.png")
    return created


def figure_hand_matrices(cfg, groups):
    """Figure 6: K and L of the hand example, natural order against nested dissection."""
    K = nodal_matrix(cfg)
    n = cfg.num_nodes
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.8), facecolor=SURFACE)
    stats = {}
    for row, (label, perm, grouping) in enumerate((("natural order", np.arange(n), None),
                                                   ("nested dissection, leaf_size = 4", ordering(groups), groups))):
        original, L, _, _ = eliminate(K, perm)
        fill = L & ~np.tril(original)
        names = [name(cfg, k) for k in perm]
        stats[label] = (int(original.sum()), int(L.sum()), int(fill.sum()))
        ax = axes[row, 0]
        style_matrix(ax, n, f"({'ac'[row]}) K, {label}: {int(original.sum())} nonzeros\n\n", names)
        if grouping:
            draw_groups(ax, grouping)
        draw_pattern(ax, original, s=150)
        ax = axes[row, 1]
        style_matrix(ax, n, f"({'bd'[row]}) L, {label}: {int(L.sum())} nonzeros, of which {int(fill.sum())} fill\n\n",
                     names)
        if grouping:
            draw_groups(ax, grouping)
        draw_pattern(ax, np.tril(original), fill, s=150)
    for ax in axes[1]:
        start = 0
        for (_, _, nodes, _), label in zip(groups, HAND["names"]):
            ax.text(start + len(nodes) / 2 - 0.5, n - 0.3, label, ha="center", va="top", fontsize=8.5, color=INK)
            start += len(nodes)
    handles = legend_handles([("nonzero of K", "s", INK, INK, 0), ("fill", "s", FILL, FILL, 0),
                              ("separator block, depth 0", "s", DEPTH[0] + "38", DEPTH[0] + "38", 0),
                              ("separator block, depth 1", "s", DEPTH[1] + "38", DEPTH[1] + "38", 0)])
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "hand_matrices.png")
    return stats


def figure_depths(cfg):
    """Figure 7: a 16x16 crossbar dissected to depth 0, 1, ... until the leaves fit LEAF_SIZE: the regions on the grid,
    K in the resulting order and its factor's pattern."""
    K = nodal_matrix(cfg)
    n = cfg.num_nodes
    final = dissection(cfg)
    deepest = max(depth for _, depth, _, _ in final)
    assert np.array_equal(ordering(final), nested_dissection(cfg))
    fig, axes = plt.subplots(3, deepest + 1, figsize=(3.55 * (deepest + 1), 11.6), facecolor=SURFACE,
                             gridspec_kw=dict(height_ratios=[1, 1, 1]))
    rows = []
    for depth in range(deepest + 1):
        groups = dissection(cfg, max_depth=depth)
        perm = ordering(groups)
        original, L, _, _ = eliminate(K, perm)
        fill = L & ~np.tril(original)
        leaves = [len(nodes) for kind, _, nodes, _ in groups if kind == "leaf" and len(nodes)]
        rows.append((depth, len(leaves), max(leaves), int(L.sum()), int(fill.sum())))
        ax = axes[0, depth]
        draw_region_map(ax, cfg, groups)
        heading = "natural order" if depth == 0 else f"{depth} level{'s' * (depth > 1)} of cuts"
        if depth == deepest:
            heading += " (final)"
        ax.set_title(f"depth {depth}: {heading}\n{len(leaves)} leaf region{'s' * (len(leaves) > 1)}, "
                     f"largest {max(leaves)} nodes", loc="left", color=INK, fontsize=9.5)
        ax = axes[1, depth]
        style_matrix(ax, n, f"K in this order")
        draw_groups(ax, groups)
        draw_pattern(ax, original, s=0.35)
        ax = axes[2, depth]
        style_matrix(ax, n, f"L: {int(L.sum()):,} nonzeros\n({int(fill.sum()):,} fill)")
        draw_groups(ax, groups)
        draw_pattern(ax, np.tril(original), fill, s=0.35)
        if depth == 1:
            sizes = [len(nodes) for _, _, nodes, _ in groups]
            centers = np.cumsum(sizes) - np.array(sizes) / 2
            for ax in axes[1:, 1]:
                for c, text in zip(centers, ("A", "B")):
                    ax.text(c, c, text, ha="center", va="center", fontsize=12, fontweight="bold", color=INK_2,
                            bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.75))
                ax.text(n + 12, centers[2], "S", ha="left", va="center", fontsize=12, fontweight="bold",
                        color=DEPTH[0], clip_on=False)
                ax.text(centers[0], centers[1], "0", ha="center", va="center", fontsize=12, color=MUTED)
            axes[1, 1].text(centers[1], centers[0], "0", ha="center", va="center", fontsize=12, color=MUTED)
    handles = legend_handles([("leaf regions (alternating grays)", "s", LEAF, SURFACE, 0.8)] +
                             [(f"separator, depth {d}", "s", DEPTH[d], SURFACE, 0.8) for d in range(min(deepest, 4))] +
                             [("nonzero of K", "s", INK, INK, 0), ("fill", "s", FILL, FILL, 0)])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    save(fig, "depths.png")
    return rows


LEAF_DARK = "#a3a29c"


def draw_region_map(ax, cfg, groups):
    """
    The groups on the grid as an image: each cross-point is a 2x2 cell holding its word node top-left and its bit node
    bottom-right. Separators take their depth's blue; leaf regions alternate between two grays in elimination order,
    so that neighbouring regions differ.
    """
    rgb = lambda color: [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    image = np.full((2 * cfg.p, 2 * cfg.q, 3), rgb(SURFACE))
    pq, leaves = cfg.p * cfg.q, 0
    for kind, depth, nodes, _ in groups:
        if kind == "separator":
            color = DEPTH[min(depth, 3)]
        elif len(nodes):
            color, leaves = (LEAF, LEAF_DARK)[leaves % 2], leaves + 1
        else:
            continue
        word, bit = nodes[nodes < pq], nodes[nodes >= pq] - pq
        image[2 * (word // cfg.q), 2 * (word % cfg.q)] = rgb(color)
        image[2 * (bit // cfg.q) + 1, 2 * (bit % cfg.q) + 1] = rgb(color)
    ax.imshow(image, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def figure_scaling(sizes):
    """Figure 8: nonzeros of L and Cholesky flops against the number of nodes, in three orderings."""
    data = {key: ([], []) for key in ("natural", "AMD", "GNDO")}
    N = []
    for n in sizes:
        cfg = Config(n, n, seed=0)
        K = nodal_matrix(cfg)
        N.append(cfg.num_nodes)
        for key, perm in (("natural", np.arange(cfg.num_nodes)), ("AMD", None), ("GNDO", nested_dissection(cfg))):
            counts = column_counts(K, perm).astype(float)
            data[key][0].append(counts.sum())
            data[key][1].append((counts ** 2).sum())
    N = np.array(N, float)
    styles = {"natural": (MUTED, "s", "natural order"), "AMD": (SERIES_COLORS[4], "^", "CHOLMOD's AMD"),
              "GNDO": (SERIES_COLORS[0], "o", "GNDO (this ordering)")}
    big = N >= 2 * 128 ** 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    for ax, which, title, unit in ((axes[0], 0, "(a) Nonzeros of L (memory, and the cost of each solve)", "nonzeros"),
                                   (axes[1], 1, "(b) Flops to factorize, Σ (column count)²", "flops")):
        ax.set_facecolor(SURFACE)
        for key, (color, marker, label) in styles.items():
            y = np.array(data[key][which])
            ax.loglog(N, y, color=color, marker=marker, lw=2, markersize=7, markeredgecolor=SURFACE,
                      markeredgewidth=1.5, label=f"{label}: slope {slope(N[big], y[big]):.2f}")
            ax.annotate(label.split(" (")[0], (N[-1], y[-1]), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)
        g = np.array(data["GNDO"][which])
        ref = N * np.log2(N) if which == 0 else N ** 1.5
        ref_label = "∝ N log N" if which == 0 else "∝ N¹·⁵"
        ax.loglog(N, ref * g[-1] / ref[-1] * 0.45, color=MUTED, lw=1, ls=(0, (4, 3)), label=f"{ref_label} (reference)")
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("nodes N = 2n² of an n×n crossbar  (n = 8 … 1024)", color=INK_2)
        ax.set_ylabel(unit, color=INK_2)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK_2)
        ax.set_xlim(N[0] / 1.6, N[-1] * 6)
        legend(ax, ax.get_legend_handles_labels()[0], loc="upper left", title="slopes over n = 128 … 1024",
               title_fontsize=8)
    fig.tight_layout()
    save(fig, "scaling.png")
    return N, data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figure_star()

    hand = Config(3, 3, seed=0)
    figure_crossbar(hand)
    figure_cuts()
    groups = dissection(hand, leaf_size=4)
    assert np.array_equal(ordering(groups), nested_dissection(hand, 4))
    figure_hand_roles(hand, groups)
    created = figure_hand_elimination(hand, groups)
    perm = ordering(groups)
    print("hand example perm:", list(perm), [name(hand, k) for k in perm])
    print("fill created by each elimination:", dict(zip([name(hand, k) for k in perm], created)))
    print("hand example (K, L, fill):", figure_hand_matrices(hand, groups))

    for depth, leaves, largest, lnz, fill in figure_depths(Config(16, 16, seed=0)):
        print(f"16x16 depth {depth}: {leaves} leaves, largest {largest}, nnz(L) {lnz}, fill {fill}")

    sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    N, data = figure_scaling(sizes)
    for k, n in enumerate(sizes):
        print(f"{n}x{n} N={int(N[k])}: " + "  ".join(f"{key} lnz {v[0][k]:.4g} flops {v[1][k]:.4g}"
                                                      for key, v in data.items()))


if __name__ == "__main__":
    main()
