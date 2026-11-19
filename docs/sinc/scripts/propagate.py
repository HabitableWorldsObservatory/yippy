"""Re-run the delivery's physicaloptix propagator without touching its directory.

The delivery directory holds ``generate.py`` and ``common.py``, which define the
``Designer`` that made the stored nodes. This module imports that class read-only,
redirects the JAX and matplotlib caches to a scratch folder, and exposes it so
the evidence scripts propagate extra positions with the delivery's own settings.
"""

import os
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402


def designer(delivery, level=0, scratch=None):
    """Return the delivery's ``Designer`` at refinement ``level`` (0 = delivery)."""
    scratch = Path(scratch or os.environ.get("TMPDIR", "/tmp")) / "yippy-sinc-evidence"
    scratch.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(delivery))
    sys.dont_write_bytecode = True
    import common  # noqa: F401  (sets caches inside the delivery; undone below)
    from generate import Designer

    os.environ["MPLCONFIGDIR"] = str(scratch / "mpl-cache")
    jax.config.update("jax_compilation_cache_dir", str(scratch / "jax-cache"))
    return Designer(level)


def images(designer_, positions, log_every=50):
    """Propagate ``positions`` [Q, 2] in lambda0/D and return float64 [Q, 128, 128]."""
    out = np.empty((len(positions), 128, 128))
    for i, xy in enumerate(positions):
        out[i] = designer_.image(xy)
        if log_every and i % log_every == 0:
            print("propagated", i, len(positions), flush=True)
    return out


SEED = 20261106
CELL = np.array([[3.8, 5.32], [4.18, 5.70]])  # served cell, lambda0/D, lower closed
IWA = 6.72  # design inner working angle from the public design name, lambda0/D
NEAR_EDGE = 6.92  # upper edge of the "near the inner working angle" bin
LINE_STEP = 0.38 / 4
LINE_HALF = 421  # fine samples each side of zero: 421 * 0.095 = 40 lambda0/D
WORST = (3.9425, 5.32038)  # worst interior check of the delivery (check-001)


def qualification_positions(seed=SEED):
    """Seeded held-out positions in the served cell.

    Returns a dict of name to [N, 2] arrays: 40 behind the inner working angle,
    40 within 0.2 lambda0/D above it, and 200 uniform over the whole cell.
    """
    rng = np.random.default_rng(seed)

    def draw(n, lo=0.0, hi=np.inf):
        out = []
        while len(out) < n:
            point = CELL[0] + rng.random(2) * (CELL[1] - CELL[0])
            if lo <= np.hypot(*point) < hi:
                out.append(point)
        return np.array(out)

    return {
        "behind": draw(40, hi=IWA),
        "near": draw(40, lo=IWA, hi=NEAR_EDGE),
        "random": draw(200),
    }


def line_positions(axis, fixed):
    """Source positions along one axis at h/4 spacing, other coordinate fixed."""
    s = np.arange(-LINE_HALF, LINE_HALF + 1) * LINE_STEP
    if axis == "x":
        return np.stack([s, np.full_like(s, fixed)], axis=1)
    return np.stack([np.full_like(s, fixed), s], axis=1)


REFINE_NODES = [(0, 0), (5, 5), (10, 14), (11, 14), (10, 15), (11, 15)]


LEVEL1_CHECKS = [
    1,
    0,
    3,
    5,
    12,
    15,
]  # check-001 (worst) and the delivery's refinement set


def level1(delivery, cache):
    """Re-propagate every node and a few checks at twice the sampling (level 1)."""
    import common

    cache = Path(cache)
    d1 = designer(delivery, 1)
    if not (cache / "nodes_level1.npy").exists():
        np.save(
            cache / "nodes_level1.npy",
            images(d1, common.coords("starter_training.csv"), 100),
        )
    checks = common.coords("starter_checks.csv")[LEVEL1_CHECKS]
    np.save(cache / "checks_level1.npy", images(d1, checks, 0))


def main(delivery, cache):
    """Propagate every set the evidence scripts need and cache it as .npy files."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    d0 = designer(delivery)
    import common

    training = common.coords("starter_training.csv")
    if not (cache / "nodes64.npy").exists():
        np.save(cache / "nodes64.npy", images(d0, training))
    for name, pos in qualification_positions().items():
        if not (cache / f"truth_{name}.npy").exists():
            np.save(cache / f"positions_{name}.npy", pos)
            np.save(cache / f"truth_{name}.npy", images(d0, pos))
    for axis, fixed in (("x", WORST[1]), ("y", WORST[0])):
        if not (cache / f"line_{axis}.npy").exists():
            np.save(cache / f"line_{axis}.npy", images(d0, line_positions(axis, fixed)))
    if not (cache / "refine_level2.npy").exists():
        pts = np.array(
            [WORST] + [(ix * 0.38, iy * 0.38) for ix, iy in REFINE_NODES], dtype=float
        )
        np.save(cache / "refine_positions.npy", pts)
        np.save(cache / "refine_level0.npy", images(d0, pts, 0))
        for level in (1, 2):
            dl = designer(delivery, level)
            np.save(cache / f"refine_level{level}.npy", images(dl, pts, 0))
            del dl


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "level1":
        level1(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2])
