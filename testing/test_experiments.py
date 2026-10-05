import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from algorithms.common import Solver, SolveOutput
from plotting import make_plots
from run_experiments import parse_size, run_sweep


class FakeConverging(Solver):
    """Stand-in solver: zero solution, iteration count growing with size, runtime growing with size."""
    name = "fake_ok"

    def solve(self, cfg):
        time.sleep(1e-6 * cfg.p * cfg.q)
        z = np.zeros(cfg.num_branches)
        return SolveOutput(z, z, True, 10 + cfg.p + cfg.seed, {"residual": [1.0, 0.1, 0.01]})


class FakeStalling(Solver):
    """Stand-in solver that converges only on the smallest arrays."""
    name = "fake_stall"

    def solve(self, cfg):
        z = np.zeros(cfg.num_branches)
        return SolveOutput(z, z, cfg.p * cfg.q <= 16, self.max_iterations)


class FakeCrashing(Solver):
    name = "fake_crash"

    def solve(self, cfg):
        raise RuntimeError("singular matrix")


def test_parse_size():
    assert parse_size("64") == (64, 64)
    assert parse_size("16x32") == (16, 32)
    assert parse_size("16X32") == (16, 32)


def sweep(out):
    sizes, runs = [(4, 4), (8, 8), (8, 16)], 3
    solvers = [FakeConverging(), FakeStalling(max_iterations=50), FakeCrashing()]
    summaries = run_sweep(sizes, runs, solvers, out, base_seed=10)
    rows = make_plots(out, [s.name for s in solvers])
    return sizes, runs, solvers, summaries, rows


def test_sweep_records_saves_and_plots():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        sizes, runs, solvers, summaries, rows = sweep(out)
        n = len(sizes) * runs * len(solvers)

        records = [json.loads(line) for line in open(out / "runs.jsonl")]
        assert records == json.loads(json.dumps(summaries)) and len(records) == n
        assert len(list((out / "pickles").glob("*.pkl"))) == n
        for f in ["experiment.json", "summary.json", "runtime.png", "iterations.png", "convergence.png"]:
            assert (out / f).stat().st_size > 0, f

        by_alg = {s.name: [r for r in records if r["algorithm"] == s.name] for s in solvers}
        assert all(r["status"] == "converged" and r["iterations"] == 10 + r["p"] + r["seed"]
                   for r in by_alg["fake_ok"])
        stall = {r["p"] * r["q"]: r["status"] for r in by_alg["fake_stall"]}
        assert stall == {16: "converged", 64: "not_converged", 128: "not_converged"}
        assert all(r["status"] == "error" and "singular matrix" in r["error"] for r in by_alg["fake_crash"])
        # non-converged runs report no iteration count
        assert all(r["iterations"] is None for r in records if r["status"] != "converged")
        # every algorithm saw the same seeds, i.e. the same circuits
        assert {(r["p"], r["seed"]) for r in by_alg["fake_ok"]} == {(r["p"], r["seed"]) for r in by_alg["fake_crash"]}

        ok = [r for r in rows if r["algorithm"] == "fake_ok" and r["p"] == 8 and r["q"] == 8][0]
        assert ok["runs"] == ok["converged"] == runs
        assert ok["iterations_mean"] == np.mean([10 + 8 + s for s in (10, 11, 12)])
        crash = [r for r in rows if r["algorithm"] == "fake_crash"]
        assert all(r["errors"] == runs and r["runtime_mean"] is None for r in crash)

        with open(out / "pickles" / "fake_ok_8x16_seed11.pkl", "rb") as f:
            res = pickle.load(f)
        assert res.config.p == 8 and res.config.q == 16 and res.config.seed == 11
        assert res.v.shape == (3 * 8 * 16,) and res.history["residual"] == [1.0, 0.1, 0.01]


if __name__ == "__main__":
    if len(sys.argv) > 1:  # keep the output for inspection: python testing/test_experiments.py OUT_DIR
        sweep(Path(sys.argv[1]))
        sys.exit()
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    for t in tests:
        t()
        print(f"{t.__name__}: ok")
    print(f"{len(tests)} tests passed")
