"""Run a shell command on the Kaggle box via kbridge. Command comes from argv or stdin,
so no quoting survives-the-shell games: `python scratch/kb.py 'ls -la'` or heredoc into stdin."""
import json, sys, urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cmd = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
req = urllib.request.Request("http://127.0.0.1:8787/sh", method="POST",
                            data=json.dumps({"cmd": cmd}).encode(),
                            headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=180))
sys.stdout.write(d.get("stdout", ""))
if d.get("stderr"): sys.stderr.write("[stderr] " + d["stderr"][:1000])
