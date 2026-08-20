"""Parity test: the front end of the graph `crowd init-csrnet --from-pt` writes IS VGG-16's front end.

`crowd init-csrnet --from-pt vgg16.pth` reads a torch checkpoint in pure C++ (pure/ptio.hpp) and drops
its tensors into a graph the C++ side wrote. Two mistakes would survive that silently:

  * a wrong layer order or a wrong padding — the graph still runs, still trains, just not as VGG-16;
  * a mis-parsed tensor (transposed, fp16-mangled, off-by-one in the pickle walk) — same story.

So the check is direct: take the tensor after the last front-end ReLU (conv4_3) out of our ONNX and
compare it, element by element, with torchvision's `vgg16.features[:23]` on the same input.

  python tools/parity/vgg_front.py --pt <vgg16_features.pth> --onnx models/csrnet_vgg.onnx

The back end is expected to differ completely: those weights are freshly initialised here and have no
counterpart in the checkpoint (14 tensors, reported by init-csrnet).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(ROOT, "models", "csrnet_vgg.onnx"))
    ap.add_argument("--pt", required=True, help="the same checkpoint init-csrnet was given")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--tol", type=float, default=1e-4, help="max |diff| relative to the activation scale")
    a = ap.parse_args()

    import onnx
    import onnxruntime as ort
    import torch
    import torchvision

    m = onnx.load(a.onnx)
    # The last front-end activation is the input of the first dilated convolution — find it by asking
    # which node has a `dilations` attribute rather than by guessing a name.
    first_dil = next((n for n in m.graph.node
                      if any(at.name == "dilations" for at in n.attribute)), None)
    if first_dil is None:
        print("no dilated convolution in this graph — is it a CSRNet?")
        return 2
    front_out = first_dil.input[0]
    m.graph.output.extend([onnx.helper.make_empty_tensor_value_info(front_out)])
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])

    x = np.random.default_rng(0).random((1, 3, a.imgsz, a.imgsz), dtype=np.float32) * 2 - 1
    outs = s.run(None, {s.get_inputs()[0].name: x})
    ours = outs[[o.name for o in s.get_outputs()].index(front_out)]

    # the reference: torchvision's own layers, loaded with the same tensors
    vgg = torchvision.models.vgg16(weights=None)
    sd = torch.load(a.pt, map_location="cpu", weights_only=True)
    missing, unexpected = vgg.load_state_dict(sd, strict=False)
    front = torch.nn.Sequential(*list(vgg.features.children())[:23])   # conv1_1 .. relu4_3
    with torch.no_grad():
        ref = front(torch.from_numpy(x)).numpy()

    print("front-end tensor %s: ours %s   torchvision %s" % (front_out, ours.shape, ref.shape))
    if ours.shape != ref.shape:
        print("  shapes differ — the layer stack is not the same")
        return 1
    d = np.abs(ours - ref)
    scale = max(1e-9, float(np.abs(ref).max()))
    print("  max |diff| %.3e (activations up to %.2f) -> relative %.2e" % (d.max(), scale, d.max() / scale))
    print("  mean |diff| %.3e, non-zero fraction %.3f" % (d.mean(), float((ref != 0).mean())))
    ok = d.max() / scale <= a.tol
    print("PARITY %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
