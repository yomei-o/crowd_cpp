"""Pull artefacts off the Kaggle box as they are written, instead of at the end of the run.

  python scratch/pull.py --out scratch/pulled --fast run_cos.csv \
      --slow models/csrnet_B_cos.onnx ck/cos.ck

Why: on 2026-08-20 a session died 8,000 steps into two 20,000-step runs and took the exported best
models and the CSV logs with it. They had been written to /kaggle/working the moment each eval
improved — the loss was purely that nobody had copied them down. So this polls `stat` through kbridge
and downloads a file whenever its size or mtime changes.

Two speeds, because the sizes differ by two orders of magnitude: the CSV is a few KB (poll it often),
while an exported CSRNet is 62MB and a checkpoint with Adam's moments is ~190MB (poll those rarely,
but do poll them — a checkpoint that only exists on a box that can vanish is not a backup).

It also happens to be the interaction that keeps the session alive, but that is kbridge's
`--keepalive` job; do not rely on this script for it.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def post(base, path, body, timeout=180):
    req = urllib.request.Request(base + path, method="POST", data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def stat_all(base, paths):
    """One round trip for every file: `stat` prints `path size mtime` or nothing if it is missing."""
    cmd = "cd /kaggle/working && stat -c '%n %s %Y' " + " ".join("'" + p + "'" for p in paths) \
          + " 2>/dev/null"
    out = post(base, "/sh", {"cmd": cmd}).get("stdout", "")
    got = {}
    for ln in out.splitlines():
        f = ln.split()
        if len(f) == 3:
            got[f[0]] = (int(f[1]), int(f[2]))
    return got


def download(base, remote, local, timeout=1800):
    os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
    url = base + "/download?path=" + urllib.parse.quote(remote) + "&raw=1"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(local + ".part", "wb") as f:
        n = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            n += len(chunk)
    os.replace(local + ".part", local)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--out", default="scratch/pulled")
    ap.add_argument("--fast", nargs="*", default=[], help="small files, polled every --every")
    ap.add_argument("--slow", nargs="*", default=[], help="big files, polled every --slow-every")
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--slow-every", dest="slow_every", type=int, default=1200)
    ap.add_argument("--hours", type=float, default=12.0)
    a = ap.parse_args()

    seen = {}
    last_slow = 0.0
    t0 = time.time()
    print("pulling into %s every %ds (%s) / %ds (%s)"
          % (a.out, a.every, ", ".join(a.fast) or "-", a.slow_every, ", ".join(a.slow) or "-"),
          flush=True)
    while time.time() - t0 < a.hours * 3600:
        due = list(a.fast)
        if time.time() - last_slow >= a.slow_every:
            due += list(a.slow)
            last_slow = time.time()
        try:
            st = stat_all(a.base, due)
        except Exception as e:                            # noqa: BLE001 - the session may be gone
            print(time.strftime("%H:%M:%S ") + "stat failed: %s" % e, flush=True)
            time.sleep(a.every)
            continue
        for p in due:
            if p not in st or seen.get(p) == st[p]:
                continue
            local = os.path.join(a.out, p.replace("/", "_"))
            try:
                n = download(a.base, p, local)
            except Exception as e:                        # noqa: BLE001
                print(time.strftime("%H:%M:%S ") + "%s: %s" % (p, e), flush=True)
                continue
            seen[p] = st[p]
            print(time.strftime("%H:%M:%S ") + "pulled %s -> %s (%.1f MB)"
                  % (p, local, n / 1e6), flush=True)
        time.sleep(a.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
