"""CSRNet in PyTorch, wired to load and save the same ONNX the C++ side writes.

The point of building it this way — rather than as an independent model — is that the two languages
then train *the same file*. `crowd init-csrnet` writes the graph and the initial weights; this loads
those exact tensors by name, trains, and writes the weights back into an ONNX with the same names. So
a model can be started in C++, trained in Python on a GPU, and evaluated in either.

  python tools/csrnet.py --onnx models/csrnet.onnx          # load, report, forward once
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# torchvision's VGG-16 layer indices for conv1_1..conv4_3 and the channel counts CSRNet keeps.
# `True` marks a max-pool after that layer; VGG-16's fourth pool is deliberately absent, which is what
# keeps the output at 1/8 instead of 1/16.
FRONT = [(0, 64, False), (2, 64, True),
         (5, 128, False), (7, 128, True),
         (10, 256, False), (12, 256, False), (14, 256, True),
         (17, 512, False), (19, 512, False), (21, 512, False)]
BACK = [512, 512, 512, 256, 128, 64]


class CSRNet(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()

        def ch(c):
            if width >= 0.999:
                return c
            return max(8, int(round(c * width)) // 8 * 8)

        # torch's ModuleDict rejects keys containing '.', and the ONNX names contain one; so the keys
        # are torch-safe and `onnx_name` does the translation. Keeping the ONNX name as the source of
        # truth is the whole point — that is what makes the same file trainable in either language.
        self.front = nn.ModuleDict()
        self.front_order = []
        cin = 3
        for idx, cout, pool in FRONT:
            key = "features_%d" % idx
            self.front[key] = nn.Conv2d(cin, ch(cout), 3, padding=1)
            self.front_order.append((key, "features.%d" % idx, pool))
            cin = ch(cout)
        self.back = nn.ModuleDict()
        self.back_order = []
        for i, c in enumerate(BACK):
            # dilation 2: the receptive field grows without another stride, so the map stays at 1/8
            key = "backend_%d" % i
            self.back[key] = nn.Conv2d(cin, ch(c), 3, padding=2, dilation=2)
            self.back_order.append((key, "backend.%d" % i))
            cin = ch(c)
        self.out = nn.Conv2d(cin, 1, 1)

    def forward(self, x):
        for key, _onnx, pool in self.front_order:
            x = torch.relu(self.front[key](x))
            if pool:
                x = torch.max_pool2d(x, 2, 2)
        for key, _onnx in self.back_order:
            x = torch.relu(self.back[key](x))
        return self.out(x)                      # no activation: the target is regressed directly

    # ---- ONNX <-> torch, by tensor name -------------------------------------------------------
    def named_onnx(self):
        """{onnx initializer name: parameter} — the mapping both languages agree on."""
        m = {}
        for key, onnx_name, _pool in self.front_order:
            m[onnx_name + ".weight"] = self.front[key].weight
            m[onnx_name + ".bias"] = self.front[key].bias
        for key, onnx_name in self.back_order:
            m[onnx_name + ".weight"] = self.back[key].weight
            m[onnx_name + ".bias"] = self.back[key].bias
        m["output_layer.weight"] = self.out.weight
        m["output_layer.bias"] = self.out.bias
        return m


def load_onnx(model, path, verbose=True):
    """Copy an ONNX's initializers into the module. Reports anything that did not line up rather than
    loading what it can and leaving the rest silently random."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(path).graph
    have = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    want = model.named_onnx()
    loaded, missing, mismatched = 0, [], []
    with torch.no_grad():
        for name, p in want.items():
            a = have.get(name)
            if a is None:
                missing.append(name)
                continue
            if tuple(a.shape) != tuple(p.shape):
                mismatched.append("%s onnx%s vs torch%s" % (name, tuple(a.shape), tuple(p.shape)))
                continue
            p.copy_(torch.from_numpy(a.copy()))
            loaded += 1
    if verbose:
        print("loaded %d/%d tensors from %s" % (loaded, len(want), os.path.basename(path)))
        for m in missing[:4]:
            print("  missing in onnx: %s" % m)
        for m in mismatched[:4]:
            print("  shape mismatch: %s" % m)
    extra = [k for k in have if k not in want]
    if verbose and extra:
        print("  onnx has %d tensor(s) this module does not use: %s" % (len(extra), extra[:3]))
    return loaded, missing, mismatched


def save_onnx(model, src_path, out_path):
    """Write the trained weights back into a copy of the source graph, so the topology stays the one
    C++ wrote (and stays loadable by pure/onnx_run.hpp) instead of whatever torch.onnx.export invents."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(src_path)
    want = {k: v.detach().cpu().numpy() for k, v in model.named_onnx().items()}
    n = 0
    for i, init in enumerate(m.graph.initializer):
        a = want.get(init.name)
        if a is None:
            continue
        m.graph.initializer[i].CopyFrom(numpy_helper.from_array(a.astype(np.float32), init.name))
        n += 1
    onnx.save(m, out_path)
    print("wrote %s (%d tensors updated)" % (out_path, n))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/csrnet.onnx")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--width", type=float, default=1.0)
    a = ap.parse_args()
    m = CSRNet(a.width)
    print("CSRNet: %d parameters" % sum(p.numel() for p in m.parameters()))
    if os.path.exists(a.onnx):
        load_onnx(m, a.onnx)
    m.eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, a.imgsz, a.imgsz))
    print("forward %dx%d -> %s, sum %.4f" % (a.imgsz, a.imgsz, tuple(y.shape), float(y.sum())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
