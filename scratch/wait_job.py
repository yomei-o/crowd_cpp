"""Block until a kbridge job leaves the 'running' state, then print its state and the log tail.

if hasattr(sys.stdout, "reconfigure"):   # Ultralytics logs box-drawing chars; cp932 console chokes
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

  python scratch/wait_job.py 20260819-061157-det2 [poll_seconds]

Polls the job list rather than a file's existence: an export from a *previous* run is already on disk,
so file-existence polling returns instantly with a stale model (learned that the hard way).
"""
import json, sys, time, urllib.request

jid = sys.argv[1]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 60


def get(url):
    return json.load(urllib.request.urlopen(url, timeout=60))


for _ in range(120):
    jobs = get("http://127.0.0.1:8787/job").get("jobs", [])
    j = next((x for x in jobs if x.get("id") == jid), None)
    if j is None:
        print("no such job: %s" % jid)
        sys.exit(2)
    if j.get("state") != "running":
        print("%s -> %s (exit %s)" % (jid, j.get("state"), j.get("exit_code")))
        log = get("http://127.0.0.1:8787/job/%s/log?offset=0" % jid).get("data", "")
        print("\n".join(log.splitlines()[-25:]))
        sys.exit(0)
    print("%s still running (%.1f min)" % (jid, (time.time() - j.get("started", 0)) / 60), flush=True)
    time.sleep(every)
print("timed out waiting for %s" % jid)
sys.exit(1)
