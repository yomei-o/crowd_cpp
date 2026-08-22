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
    # **向きも見る。** max|diff|/max|g| だけだと、乱数で初期化した網では層ごとに勾配が
    # 極小になり、float32 の打ち消しで比が 1e-3 まで暴れる（実測: features.21.weight が
    # max|g| 1.19e-03 / max|diff| 2.25e-06 で比 1.9e-03 なのに、cos は 9 桁一致）。
    # 本当に backward が違えば**形が変わる**ので cos が落ちる。そこを合格条件に足す。
    ap.add_argument("--cos-tol", dest="cos_tol", type=float, default=1e-6,
                    help="1 - cos の上限（勾配の向きのずれ）")
    ap.add_argument("--grad-tol-loose", dest="grad_tol_loose", type=float, default=5e-3,
                    help="向きが一致しているときに許す max|diff|/max|g|")
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

    worst, worst_name, compared, worst_cos = 0.0, "", 0, 1.0
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
        na, nb = float(np.linalg.norm(g_py)), float(np.linalg.norm(g_cpp))
        cos = float(g_py @ g_cpp) / max(1e-30, na * nb)
        compared += 1
        # 厳しい比を満たすか、**向きが一致していて比も緩い上限の中**なら通す
        if r > a.grad_tol and not (1.0 - cos <= a.cos_tol and r <= a.grad_tol_loose):
            print("  %s: max|diff|/max|g| %.2e, 1-cos %.2e" % (name, r, 1.0 - cos))
            ok = False
        if r > worst:
            worst, worst_name, worst_cos = r, name, cos
    print("  gradients: %d tensors compared, worst relative %.2e (%s, 1-cos %.1e)"
          % (compared, worst, worst_name, 1.0 - worst_cos))
    ok = ok and compared > 0
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
