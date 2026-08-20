"""Keep a Kaggle session from being reaped, and log whether it worked.

Today's evidence for why this exists (2026-08-20): a session died at some point inside a 56-minute
window in which nothing was sent through kbridge — while two training jobs were running on the box the
whole time. So the reaper does not appear to care that the kernel is busy; it appears to care about
*interaction* through the proxy. An earlier session the same day died the same way. That is a
hypothesis, not a documented fact, which is why this script logs its own outcome: run it alongside a
long job and the log says whether the session outlived the 2-3 hours we kept losing.

  python scratch/keepalive.py --every 240 --hours 9 --log scratch/keepalive.log

Belt and braces: this is *not* a substitute for `--resume` in the trainer. A heartbeat cannot help if
the browser tab that owns the session is closed, the laptop sleeps, or the 9-hour cap is reached. It
only removes the failure mode we actually observed.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def ping(base, cmd):
    req = urllib.request.Request(base + "/sh", method="POST",
                                 data=json.dumps({"cmd": cmd}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r).get("stdout", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--every", type=int, default=240, help="seconds between pings")
    ap.add_argument("--hours", type=float, default=9.0, help="give up after this long")
    ap.add_argument("--log", default="")
    # Cheap on purpose: `date` plus the GPU's utilisation, so the log doubles as a record of whether
    # the job was still computing when the session died.
    ap.add_argument("--cmd", default="date +%H:%M:%S; nvidia-smi --query-gpu=utilization.gpu "
                                     "--format=csv,noheader | head -1")
    a = ap.parse_args()

    log = open(a.log, "a", buffering=1) if a.log else None

    def say(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        if log:
            log.write(line + "\n")

    say("keepalive: every %ds for up to %.1fh" % (a.every, a.hours))
    t0 = time.time()
    n = 0
    while time.time() - t0 < a.hours * 3600:
        try:
            out = ping(a.base, a.cmd).replace("\n", " | ")
            n += 1
            say("ping %d ok (%.0f min in): %s" % (n, (time.time() - t0) / 60, out))
        except urllib.error.HTTPError as e:
            say("ping failed after %.0f min: HTTP %s -> the session is gone" % ((time.time() - t0) / 60, e.code))
            return 1
        except Exception as e:                                  # noqa: BLE001 - any transport failure
            say("ping failed after %.0f min: %s" % ((time.time() - t0) / 60, e))
            return 1
        time.sleep(a.every)
    say("done: session survived %.1f hours of pinging" % ((time.time() - t0) / 3600))
    return 0


if __name__ == "__main__":
    sys.exit(main())
