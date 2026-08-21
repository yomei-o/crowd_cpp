"""Resume parity: a run that was interrupted and continued must equal the run that never stopped.

  python tools/parity/resume.py                 # both implementations
  python tools/parity/resume.py --impl cpp      # one of them
  python tools/parity/resume.py --steps 6 --stop 3 --keep    # keep the temp dir to look at

Why this exists: a Kaggle session died after 1h50m-2h45m twice in one day and took two 20,000-step
runs with it (RESUME.md has the timeline). `--resume` is what makes a long run survivable, and a
resume that is only approximately right is a silent lie — the losses drift a little, the curve looks
plausible, and nobody notices. So this compares the printed losses of steps stop+1..N, digit for
digit, between

  (a) an uninterrupted N-step run, and
  (b) an N-step run **actually killed** after the checkpoint for step `stop` is on disk, then
      continued with `--resume`.

Killing the process rather than asking it to stop is the point: that is the failure mode this has to
survive. The dataset is a synthetic ShanghaiTech (4 train + 2 test 128x96 images with .mat point
lists), so this needs neither the 349MB download nor a GPU — it runs on CPU in about a minute.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_fake_dataset(root, n_train=4, n_test=2, w=128, h=96):
    """A ShanghaiTech-shaped part directory: <split>_data/images/IMG_n.jpg plus
    ground-truth/GT_IMG_n.mat, with the annotation wrapped the way the real files wrap it
    (cell -> struct -> location), and v7 zlib compression, because that is what the readers meet."""
    from PIL import Image
    from scipy.io import savemat

    part = os.path.join(root, "part_B")
    for split, n in (("train", n_train), ("test", n_test)):
        img_dir = os.path.join(part, split + "_data", "images")
        gt_dir = os.path.join(part, split + "_data", "ground-truth")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        for i in range(1, n + 1):
            rng = np.random.default_rng(1000 * (1 if split == "train" else 2) + i)
            px = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
            pts = np.stack([rng.uniform(4, w - 4, 12), rng.uniform(4, h - 4, 12)], 1)
            for x, y in pts:                      # a bright blob per head, so the images are not pure noise
                px[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4] = 240
            Image.fromarray(px).save(os.path.join(img_dir, "IMG_%d.jpg" % i), quality=95)
            st = np.zeros((1, 1), dtype=[("location", "O"), ("number", "O")])
            st[0, 0]["location"] = pts.astype(np.float64)
            st[0, 0]["number"] = np.array([[len(pts)]], np.float64)
            cell = np.empty((1, 1), dtype=object)
            cell[0, 0] = st
            savemat(os.path.join(gt_dir, "GT_IMG_%d.mat" % i), {"image_info": cell},
                    do_compression=True)
    return part


def losses_of(lines):
    """`step N loss X` -> {N: "X"}, keeping the text so the comparison is on the printed digits."""
    out = {}
    for ln in lines:
        p = ln.split()
        if len(p) >= 4 and p[0] == "step" and p[2] == "loss":
            out[int(p[1])] = p[3]
    return out


def run(cmd, cwd=ROOT):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout.splitlines()


def run_until_ckpt(cmd, ckpt, stop_step, cwd=ROOT, timeout=1800):
    """Start `cmd`, wait until the checkpoint written after step `stop_step` exists, then kill it."""
    if os.path.exists(ckpt):
        os.remove(ckpt)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace")
    lines, seen, t0 = [], False, time.time()
    while time.time() - t0 < timeout:
        ln = p.stdout.readline()
        if not ln:
            break
        lines.append(ln.rstrip())
        if ("step %d loss" % stop_step) in ln:
            seen = True
        if seen:
            for _ in range(300):                  # the write happens just after the line is printed
                if os.path.exists(ckpt):
                    break
                time.sleep(0.1)
            break
    p.kill()
    p.wait()
    return lines


def check(impl, tmp, part, steps, stop, verbose):
    ckpt = os.path.join(tmp, "resume_%s.ck" % impl)
    log_a = os.path.join(tmp, "a_%s.csv" % impl)
    log_b = os.path.join(tmp, "b_%s.csv" % impl)
    common = ["--data", part, "--steps", str(steps), "--crop", "0", "--eval-every", "0",
              "--dump-loss", "--seed", "7", "--lr", "1e-5", "--lr-final", "0.02"]
    if impl == "py":
        base = [sys.executable, os.path.join("tools", "train_csrnet.py")]
    else:
        exe = os.path.join(ROOT, "crowd.exe")
        if not os.path.exists(exe):
            exe = os.path.join(ROOT, "crowd")
        if not os.path.exists(exe):
            print("[SKIP] no crowd binary — build it with build/cc.sh or build/gcc.sh")
            return None
        init = os.path.join(tmp, "tiny.onnx")
        rc, out = run([exe, "init-csrnet", "--out", init, "--width", "0.25", "--seed", "7"])
        if rc != 0:
            print("\n".join(out[-5:]))
            return False
        base = [exe, "train", "--init", init]
        common += ["--batch", "1"]

    print("[%s] (a) uninterrupted %d steps" % (impl, steps), flush=True)
    rc, a_lines = run(base + common + ["--log", log_a])
    if rc != 0:
        print("\n".join(a_lines[-10:]))
        return False
    a = losses_of(a_lines)

    print("[%s] (b) same run, killed after the checkpoint for step %d" % (impl, stop), flush=True)
    b_lines = run_until_ckpt(base + common + ["--ckpt", ckpt, "--ckpt-every", str(stop),
                                              "--log", log_b], ckpt, stop)
    if not os.path.exists(ckpt):
        print("no checkpoint was written")
        print("\n".join(b_lines[-10:]))
        return False
    b = losses_of(b_lines)

    print("[%s]     resuming from the checkpoint" % impl, flush=True)
    rc, c_lines = run(base + common + ["--resume", ckpt, "--log", log_b])
    if rc != 0:
        print("\n".join(c_lines[-10:]))
        return False
    c = losses_of(c_lines)

    ok = True
    first = min(c) if c else 0
    if first != stop + 1:
        print("  FAIL: the resumed run starts at step %d, expected %d" % (first, stop + 1))
        ok = False
    for k in range(stop + 1, steps + 1):
        if k not in a or k not in c:
            print("  FAIL: step %d missing (uninterrupted %s, resumed %s)"
                  % (k, k in a, k in c))
            ok = False
            continue
        same = a[k] == c[k]
        ok = ok and same
        if verbose or not same:
            print("  step %-4d uninterrupted %s   resumed %s   %s"
                  % (k, a[k], c[k], "same" if same else "DIFFERENT"))
    # the killed prefix must also match, or the two runs were never the same run
    for k in range(1, stop + 1):
        if k in b and k in a and a[k] != b[k]:
            print("  FAIL: step %d differs before the kill (%s vs %s)" % (k, a[k], b[k]))
            ok = False
    if os.path.exists(log_b):
        with open(log_b, encoding="utf-8") as f:
            rows = [r for r in f.read().splitlines() if r and not r.startswith("step,")]
        got = [int(r.split(",")[0]) for r in rows]
        if got != list(range(1, len(got) + 1)):
            print("  FAIL: the resumed CSV is not one continuous curve: %s" % got)
            ok = False
        elif verbose:
            print("  csv rows after the resume: steps %s (appended, not truncated)" % got)
    print("[%s] %s" % (impl, "resume == uninterrupted, to the printed digit" if ok else "MISMATCH"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default="both", choices=["py", "cpp", "both"])
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--stop", type=int, default=3, help="kill the run after this step's checkpoint")
    ap.add_argument("--keep", action="store_true", help="do not delete the temp dataset")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="crowd_resume_")
    print("fixture: %s" % tmp)
    part = make_fake_dataset(tmp)
    results = {}
    try:
        for impl in (["py", "cpp"] if a.impl == "both" else [a.impl]):
            results[impl] = check(impl, tmp, part, a.steps, a.stop, a.verbose)
    finally:
        if not a.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    bad = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    print("%d passed, %d failed%s"
          % (sum(1 for v in results.values() if v), len(bad),
             (", %d skipped" % len(skipped)) if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
