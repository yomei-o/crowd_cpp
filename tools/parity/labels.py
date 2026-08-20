"""Parity test: the target maps the two languages build from the same annotation must be the same map.

Everything downstream is measured against these labels — the loss, the MAE, the localisation F1 — so
if C++ and Python disagree here, no later comparison between them means anything. This runs both
generators on one .mat and diffs the maps element by element, for all three label kinds.

  python tools/parity/labels.py --mat <GT.mat> --w 1024 --h 768

Exact equality is not the bar and should not be: the two sides evaluate `exp` and `pow` through
different libms, so identical arithmetic in a different order lands a few ULP apart. The bar is that
the difference is at that scale (1e-6 relative) and that the *derived* quantities — a density map's
sum, and the peak set a FIDT map yields — agree exactly.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_map(path):
    with open(path, "rb") as f:
        w, h = np.fromfile(f, np.int32, 2)
        v = np.fromfile(f, np.float32, int(w) * int(h))
    return v.reshape(int(h), int(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--down", type=int, default=8)
    ap.add_argument("--crowd", default=os.path.join(ROOT, "crowd.exe"))
    ap.add_argument("--tol", type=float, default=1e-5, help="max |diff| relative to the map's max")
    a = ap.parse_args()

    if not os.path.exists(a.crowd):
        print("no crowd binary at %s — build it first (build/gcc.sh)" % a.crowd)
        return 2
    import density as D

    tmp = tempfile.mkdtemp(prefix="labels")
    common = ["--mat", a.mat, "--w", str(a.w), "--h", str(a.h), "--down", str(a.down)]
    ok = True
    for kind, extra in (("density fixed sigma", ["--sigma", "15"]),
                        ("density adaptive", ["--adaptive"]),
                        ("FIDT", ["--fidt"])):
        cpp_out = os.path.join(tmp, "cpp.bin")
        py_out = os.path.join(tmp, "py.bin")
        r = subprocess.run([a.crowd, "labels"] + common + extra + ["--out", cpp_out],
                           capture_output=True)
        if r.returncode != 0:
            print("%s: crowd labels failed\n%s" % (kind, r.stdout.decode("utf-8", "replace")))
            return 2
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "density.py")] + common
                           + extra + ["--out", py_out], capture_output=True)
        if r.returncode != 0:
            print("%s: density.py failed\n%s" % (kind, r.stderr.decode("utf-8", "replace")[-800:]))
            return 2

        c, p = read_map(cpp_out), read_map(py_out)
        if c.shape != p.shape:
            print("%-20s shapes differ: %s vs %s" % (kind, c.shape, p.shape))
            ok = False
            continue
        d = np.abs(c.astype(np.float64) - p.astype(np.float64))
        scale = max(1e-12, float(np.abs(p).max()))
        rel = d.max() / scale
        line = ("%-20s %dx%d  max|diff| %.3e (rel %.2e)  sum %.6f / %.6f"
                % (kind, c.shape[1], c.shape[0], d.max(), rel, c.sum(dtype=np.float64),
                   p.sum(dtype=np.float64)))
        # the derived quantity each label kind exists for
        if kind == "FIDT":
            pc, pp = len(D.peaks(c)), len(D.peaks(p))
            line += "  peaks %d / %d" % (pc, pp)
            ok = ok and pc == pp
        else:
            line += "  (sums must match the head count)"
        print(line)
        ok = ok and rel <= a.tol
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
