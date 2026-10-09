"""
Geometric nested-dissection ordering of the crossbar's nodes, for factorizing the nodal matrix A Gamma^-1 A^T.

Eliminating a node in Cholesky joins its not-yet-eliminated neighbours into a clique, and those new edges are the fill.
Nested dissection keeps the fill small: it finds a separator, a small set of nodes whose removal splits the graph in two,
numbers both halves before it, and recurses into the halves. No fill ever joins the two halves, only the separator's own
block fills in. On a 2-D grid this is optimal up to a constant factor: O(N log N) nonzeros in the factor and O(N^1.5)
flops for N nodes.

The crossbar's graph is a p x q grid of cross-points, each holding a word node w[r, c] and a bit node b[r, c] joined by a
device. Only word wires join neighbouring columns and only bit wires join neighbouring rows (sources and loads go to
ground, which is not a node), so the separators can be read off the grid instead of being searched for with a graph
partitioner: a vertical cut through column c needs only the word nodes w[:, c], and a horizontal cut through row r only
the bit nodes b[r, :]. Each separator is as small as on a grid with one node per cross-point.

A cut strands the other kind of node on its line: the bit nodes of a cut column touch neither half, since their only link
sideways ran through w[:, c]. They stay with the left half, where later horizontal cuts split them, and the word nodes of
a cut row likewise stay with the top half. A region is therefore the tuple (r0, r1, c0, c1, ew, eb): word nodes in rows
[r0, r1 + ew) x columns [c0, c1) and bit nodes in rows [r0, r1) x columns [c0, c1 + eb), where ew and eb (0 or 1) count
a stranded word row below and a stranded bit column to the right.
"""
import numpy as np

# Regions of at most this many nodes are numbered in natural order rather than dissected further. Measured with CHOLMOD on
# the default crossbar at 1024x1024 (2026-10-08): leaves of 16 to 128 nodes factor equally fast, larger leaves make the
# ordering itself cheaper (0.36 s at 16, 0.13 s at 64), and 64 stores the fewest factor entries (95.6M, vs 99.3M at 16
# and 103.8M at 128).
LEAF_SIZE = 64


def _count(region):
    r0, r1, c0, c1, ew, eb = region
    return (r1 + ew - r0) * (c1 - c0) + (r1 - r0) * (c1 + eb - c0)


def _split(cfg, region):
    """
    Cut `region` across its longer side, at the middle. Returns (separator, first, second): the separator as a block
    (base, row start, row stop, column start, column stop) of node indices base + r q + c, and the two halves as
    regions. Only called on regions of at least two nodes, which always admit a cut.
    """
    r0, r1, c0, c1, ew, eb = region
    pq, q = cfg.p * cfg.q, cfg.q
    can_cut_columns = c1 > c0 and r1 + ew > r0  # some word node to cut through
    can_cut_rows = r1 > r0 and c1 + eb > c0  # some bit node to cut through
    if can_cut_columns and (c1 + eb - c0 >= r1 + ew - r0 or not can_cut_rows):
        cm = (c0 + c1) // 2
        separator = (0, r0, r1 + ew, cm, cm + 1)  # the word nodes w[:, cm]
        return separator, (r0, r1, c0, cm, ew, 1), (r0, r1, cm + 1, c1, ew, eb)
    rm = (r0 + r1) // 2
    separator = (pq, rm, rm + 1, c0, c1 + eb)  # the bit nodes b[rm, :]
    return separator, (r0, rm, c0, c1, 1, eb), (rm + 1, r1, c0, c1, ew, eb)


def _leaf_blocks(cfg, region):
    r0, r1, c0, c1, ew, eb = region
    return [(0, r0, r1 + ew, c0, c1), (cfg.p * cfg.q, r0, r1, c0, c1 + eb)]


def _expand(q, blocks):
    """Node indices of `blocks`, in order: block by block, each row-major. Vectorized over all blocks at once."""
    base, rs, re, cs, ce = (np.array(col, dtype=np.int64) for col in zip(*blocks))
    width = np.maximum(ce - cs, 0)
    sizes = np.maximum(re - rs, 0) * width
    keep = sizes > 0
    base, rs, cs, width, sizes = base[keep], rs[keep], cs[keep], width[keep], sizes[keep]
    block = np.repeat(np.arange(len(sizes)), sizes)
    offset = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    w = width[block]
    return base[block] + (rs[block] + offset // w) * q + cs[block] + offset % w


def nested_dissection(cfg, leaf_size=LEAF_SIZE):
    """
    Geometric nested-dissection ordering of cfg's 2pq nodes (see the module docstring): perm[k] is the k-th node to
    eliminate, so the matrix to factorize is K[perm][:, perm]. Regions of at most leaf_size nodes are numbered in natural
    order.
    """
    if leaf_size < 1:
        raise ValueError("leaf_size must be at least 1")
    blocks = []

    def dissect(region):
        if _count(region) <= leaf_size:
            blocks.extend(_leaf_blocks(cfg, region))
            return
        separator, first, second = _split(cfg, region)
        dissect(first)
        dissect(second)
        blocks.append(separator)

    dissect((0, cfg.p, 0, cfg.q, 0, 0))
    perm = _expand(cfg.q, blocks)
    assert perm.size == cfg.num_nodes
    return perm
