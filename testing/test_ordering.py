import sys
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from Config import Config
from algorithms.ordering import LEAF_SIZE, _count, _expand, _leaf_blocks, _split, nested_dissection

SHAPES = [(1, 1), (1, 9), (9, 1), (3, 4), (13, 7), (5, 31), (64, 64)]


def region_nodes(cfg, region):
    return _expand(cfg.q, _leaf_blocks(cfg, region))


def test_is_a_permutation():
    """Every node is eliminated exactly once, for any shape and leaf size, down to one node per leaf."""
    for p, q in SHAPES:
        cfg = Config(p, q, seed=0)
        for leaf_size in (1, 2, LEAF_SIZE, 10 ** 9):
            perm = nested_dissection(cfg, leaf_size)
            assert np.array_equal(np.sort(perm), np.arange(cfg.num_nodes)), (p, q, leaf_size)
    assert np.array_equal(nested_dissection(Config(3, 4, seed=0), 10 ** 9), np.arange(24))  # one leaf: natural order


def test_every_split_separates_its_halves():
    """At every level, down to single nodes: the separator and the two halves partition the region, no branch joins
    the two halves (so no fill can join them), and both halves are smaller than the region."""
    for p, q in SHAPES[:-1] + [(16, 16)]:
        cfg = Config(p, q, seed=0)
        A = cfg.incidence()
        adjacency = (abs(A) @ abs(A).T).tocsr()
        regions = [(0, p, 0, q, 0, 0)]
        while regions:
            region = regions.pop()
            if _count(region) <= 1:
                continue
            separator, first, second = _split(cfg, region)
            s, f, g = _expand(q, [separator]), region_nodes(cfg, first), region_nodes(cfg, second)
            assert np.array_equal(np.sort(np.concatenate([s, f, g])), np.sort(region_nodes(cfg, region))), region
            assert adjacency[f][:, g].nnz == 0, region
            assert len(s) > 0 and _count(first) < _count(region) and _count(second) < _count(region), region
            regions += [first, second]


def test_separators_are_one_line_of_one_kind_of_node():
    """The first cut crosses the longer side through its middle and takes only the nodes that join the two halves: the
    p word nodes of the middle column of a wide array, the q bit nodes of the middle row of a tall one."""
    wide, tall = Config(16, 32, seed=0), Config(32, 16, seed=0)
    separator, _, _ = _split(wide, (0, 16, 0, 32, 0, 0))
    assert sorted(_expand(32, [separator])) == sorted(wide.word_node(np.arange(16), 16))
    separator, _, _ = _split(tall, (0, 32, 0, 16, 0, 0))
    assert sorted(_expand(16, [separator])) == sorted(tall.bit_node(16, np.arange(16)))


def test_reduces_fill():
    """Factorized in the nested-dissection ordering, the nodal matrix fills in far less than in the natural one."""
    cfg = Config(48, 48, seed=0)
    A = cfg.incidence()
    K = (A @ A.T).tocsc()
    perm = nested_dissection(cfg)
    options = dict(permc_spec="NATURAL", diag_pivot_thresh=0, options=dict(SymmetricMode=True))
    natural, dissected = spla.splu(K, **options).nnz, spla.splu(K[perm][:, perm].tocsc(), **options).nnz
    assert dissected < 0.5 * natural, (dissected, natural)


def test_rejects_empty_leaves():
    try:
        nested_dissection(Config(2, 2, seed=0), 0)
    except ValueError:
        return
    assert False, "accepted leaf_size=0"


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
