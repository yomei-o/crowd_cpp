"""Submit a long-running job to the Kaggle box via kbridge (command from argv/stdin), print its id."""
import json, sys, urllib.request
cmd = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
req = urllib.request.Request("http://127.0.0.1:8787/job", method="POST",
                            data=json.dumps({"cmd": cmd}).encode(),
                            headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=60)))
