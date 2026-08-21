"""Eval parity: `crowd eval` and `tools/eval.py` must report the same numbers.

  python tools/parity/eval.py                 # density, FIDT and label modes
  python tools/parity/eval.py --keep          # keep the temp fixture to look at

Why this is a test and not a one-off diff: the two evaluators are what M6 and M7 are judged on. If
they drift, one of the two headline numbers in RESUME becomes fiction and nothing else would notice.

What is required to match, and what is not:

  * `--labels` (scoring the target maps themselves): **exactly**. Both sides build the labels with
    their own generator and never run a network, so there is nothing to differ.
  * anything that runs the network (density MAE, FIDT F1 at any threshold): **within a tolerance**.
    The two runtimes agree to about 4e-7 per cell, and counting local maxima is a discrete decision:
    a peak that close to the threshold is kept by one side and dropped by the other. Measured — with
    per-pixel noise of 4e-7, the peak count of a random-initialised model on the fixture moves from
    224 to 235, because such a map is full of equal-valued plateaus; a *trained* model over 6 real
    images differed by 0.004 of F1; the same fixture differs by 0.0002.

    This was written the other way round first ("the absolute sweep must match exactly") because the
    sweep table prints 3 decimals and rounded the difference away. The 4-decimal headline showed it.
    A tolerance that is stated is worth more than an equality that only holds at the print width.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "parity"))
import resume as R      # noqa: E402  - the synthetic ShanghaiTech builder lives there

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The tolerance model-based comparisons are held to. 0.02 is ten times the worst difference
# measured on the fixture (0.0002) and five times the worst on real images (0.004), so it
# catches a real divergence without failing on float noise.
MODEL_TOL = 0.02
LMDS_TOL = MODEL_TOL


def crowd_exe():
    for name in ("crowd.exe", "crowd"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            return p
    return None


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout.splitlines()


def numbers(lines):
    """Every float on every line, so two outputs can be compared value by value."""
    out = []
    for ln in lines:
        for tok in ln.replace("|", " ").replace(":", " ").replace(",", " ").split():
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def compare(name, cpp, py, tol):
    """tol == 0 means the text itself must match."""
    if tol == 0:
        same = cpp == py
        if not same:
            for a, b in zip(cpp, py):
                if a != b:
                    print("    cpp: %s" % a)
                    print("    py : %s" % b)
                    break
        print("  [%s] %s (exact)" % ("OK" if same else "FAIL", name))
        return same
    a, b = numbers(cpp), numbers(py)
    if len(a) != len(b):
        print("  [FAIL] %s: %d numbers vs %d" % (name, len(a), len(b)))
        return False
    worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    ok = worst <= tol
    print("  [%s] %s (worst difference %.4f, tolerance %.2f)"
          % ("OK" if ok else "FAIL", name, worst, tol))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    exe = crowd_exe()
    if exe is None:
        print("[SKIP] no crowd binary — build it with build/cc.sh or build/gcc.sh")
        return 0

    tmp = tempfile.mkdtemp(prefix="crowd_evalparity_", dir=os.path.join(ROOT, "scratch"))
    print("fixture: %s" % tmp)
    part = R.make_fake_dataset(tmp, n_train=2, n_test=2)
    dens = os.path.join(tmp, "dens.onnx")
    fidt = os.path.join(tmp, "fidt.onnx")
    ok = True
    try:
        for out, dec in ((dens, "0"), (fidt, "4")):
            rc, lines = run([exe, "init-csrnet", "--out", out, "--decoder", dec, "--seed", "99"])
            if rc != 0:
                print("\n".join(lines[-4:]))
                return 1

        cases = [
            ("labels 1/2, 8px", ["eval", "--data", part, "--fidt", "--down", "2", "--labels"], 0),
            ("labels 1/2, 4px", ["eval", "--data", part, "--fidt", "--down", "2", "--labels",
                                 "--loc-thr", "4"], 0),
            ("density MAE/RMSE", ["eval", "--data", part, "--model", dens], 0.01),
            ("FIDT sweep", ["eval", "--data", part, "--model", fidt, "--fidt", "--down", "2",
                            "--sweep"], LMDS_TOL),
            ("FIDT absolute 0.30", ["eval", "--data", part, "--model", fidt, "--fidt", "--down", "2",
                                    "--peak-thr", "0.30"], LMDS_TOL),
        ]
        for name, args, tol in cases:
            rc1, cpp = run([exe] + args)
            rc2, py = run([sys.executable, os.path.join("tools", "eval.py")] + args[1:])
            if rc1 != 0 or rc2 != 0:
                print("  [FAIL] %s: exit %d / %d" % (name, rc1, rc2))
                print("\n".join((cpp + py)[-6:]))
                ok = False
                continue
            ok = compare(name, cpp, py, tol) and ok
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print("eval parity: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
