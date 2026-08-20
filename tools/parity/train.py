"""Parity test: the C++ and PyTorch trainers compute the same loss and the same gradients.

`crowd train --dump-fixture` writes step 1's *exact* batch (the normalised crops and the target maps),
the loss it computed and every parameter gradient. This replays those same numbers through
tools/csrnet.py and compares. Feeding both sides the same batch is the point: comparing two trainers
on their own batches only ever measures the samplers.

  crowd train --data <part> --init models/csrnet.onnx --steps 1 --batch 2 --crop 128 \
      --dump-fixture scratch/fix.bin
  python tools/parity/train.py --fixture scratch/fix.bin --onnx models/csrnet.onnx

Passing means the loss agrees to 1e-5 relative and every gradient tensor to 1e-4 relative. Both sides
are float32 and the sums run over ~16M parameters worth of convolution, so bit equality is not the
bar — a systematic difference (a wrong dilation in the backward pass, a mis-scaled loss) is what this
catches, and it shows up thousands of times above that threshold.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_fixture(path):
    with open(path, "rb") as f:
        if f.read(8) != b"CSRFIX01":
            raise SystemExit("%s is not a csrnet fixture" % path)
        n, c, crop, mh, npar = np.fromfile(f, np.int32, 5)
        x = np.fromfile(f, np.float32, int(n) * int(c) * int(crop) * int(crop))
        x = x.reshape(int(n), int(c), int(crop), int(crop))
        y = np.fromfile(f, np.float32, int(n) * int(mh) * int(mh)).reshape(int(n), 1, int(mh), int(mh))
        loss = float(np.fromfile(f, np.float32, 1)[0])
        grads = {}
        for _ in range(int(npar)):
            nl = int(np.fromfile(f, np.int32, 1)[0])
            name = f.read(nl).decode("utf-8")
            ne = int(np.fromfile(f, np.int32, 1)[0])
            grads[name] = np.fromfile(f, np.float32, ne)
    return dict(x=x, y=y, loss=loss, grads=grads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "csrnet.onnx"))
    ap.add_argument("--loss-tol", dest="loss_tol", type=float, default=1e-5)
    ap.add_argument("--grad-tol", dest="grad_tol", type=float, default=1e-4)
    a = ap.parse_args()

    import torch
    import csrnet as C

    fx = read_fixture(a.fixture)
    model = C.CSRNet()
    C.load_onnx(model, a.onnx, verbose=False)
    model.train()

    x = torch.from_numpy(fx["x"])
    y = torch.from_numpy(fx["y"])
    p = model(x)
    if tuple(p.shape) != tuple(y.shape):
        print("output %s but the fixture's target is %s" % (tuple(p.shape), tuple(y.shape)))
        return 1
    loss = ((p - y) ** 2).sum() / x.shape[0]
    model.zero_grad(set_to_none=True)
    loss.backward()

    lv = float(loss)
    rel = abs(lv - fx["loss"]) / max(1e-9, abs(lv))
    print("batch %s, map %s" % (tuple(x.shape), tuple(y.shape)))
    print("  loss  C++ %.6f   python %.6f   rel %.2e" % (fx["loss"], lv, rel))
    ok = rel <= a.loss_tol

    worst, worst_name, compared = 0.0, "", 0
    for name, p_ in model.named_onnx().items():
        g_cpp = fx["grads"].get(name)
        if g_cpp is None:
            print("  gradient missing from the fixture: %s" % name)
            ok = False
            continue
        g_py = p_.grad.detach().cpu().numpy().reshape(-1)
        if g_py.shape != g_cpp.shape:
            print("  %s: shapes %s vs %s" % (name, g_py.shape, g_cpp.shape))
            ok = False
            continue
        scale = max(1e-12, float(np.abs(g_py).max()))
        r = float(np.abs(g_py - g_cpp).max()) / scale
        compared += 1
        if r > worst:
            worst, worst_name = r, name
    print("  gradients: %d tensors compared, worst relative %.2e (%s)" % (compared, worst, worst_name))
    ok = ok and compared > 0 and worst <= a.grad_tol
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
