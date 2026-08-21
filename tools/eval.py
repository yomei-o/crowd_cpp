"""Score a trained model on a split without training it — the Python half of `crowd eval`.

  python tools/eval.py --data <ShanghaiTech>/part_B --model models/csrnet_B.onnx
  python tools/eval.py --data <part>/part_B --model models/fidt_B.onnx --fidt --down 2 --sweep

Two reasons this exists rather than living inside the trainer:

  * **M6 asks whether the C++ runtime reproduces the MAE the GPU run reported.** Comparing that needed
    a way to evaluate a file, not a way to train one; before this the only evaluation was a step of
    `crowd train` away, which changes the weights it is measuring.
  * **`--sweep` separates "the peaks are wrong" from "the peaks are short".** The first FIDTM run
    scored F1 0.005 at FIDTM's threshold of 0.5 and F1 0.719 at 0.25, because the map only reached
    0.463 of the label's height at the annotated heads. A single number at a fixed threshold hid a
    model that was mostly right. The sweep and the amplitude ratio are printed together for that
    reason.

Output lines match pure/crowd.cpp's `cmd_eval` character for character where the numbers allow, so
the two implementations can be diffed directly.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import csrnet as C      # noqa: E402
import density as D     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SWEEP = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(part_dir, split, down, sigma, adaptive, fidt, limit):
    """The same layout `csrt::read_split` walks: IMG_n.jpg + GT_IMG_n.mat, in numeric order."""
    from PIL import Image
    img_dir = os.path.join(part_dir, split + "_data", "images")
    gt_dir = os.path.join(part_dir, split + "_data", "ground-truth")
    if not os.path.isdir(img_dir):
        raise SystemExit("no images under %s" % img_dir)
    names = sorted((f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")),
                   key=lambda f: int("".join(c for c in os.path.splitext(f)[0] if c.isdigit()) or 0))
    if limit > 0:
        names = names[:limit]
    out = []
    for f in names:
        gp = os.path.join(gt_dir, "GT_" + os.path.splitext(f)[0] + ".mat")
        if not os.path.exists(gp):
            continue
        img = np.asarray(Image.open(os.path.join(img_dir, f)).convert("RGB"), np.uint8)
        pts = D.load_points(gp)
        h, w, _ = img.shape
        h -= h % down
        w -= w % down
        lab = (D.fidt(pts, w, h, down=down) if fidt
               else D.density(pts, w, h, down=down, sigma=sigma, adaptive=adaptive))
        x = ((img[:h, :w].astype(np.float32) / 255.0 - MEAN) / SD).transpose(2, 0, 1)
        out.append((f, np.ascontiguousarray(x)[None], pts, lab, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="", help="not needed with --labels")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--down", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--fidt", action="store_true")
    ap.add_argument("--loc-thr", dest="loc_thr", type=float, default=8.0)
    ap.add_argument("--peak-thr", dest="peak_thr", type=float, default=0.0,
                    help="0 = the reference's LMDS (100/255 of each map's own maximum); "
                         "a positive value forces an absolute threshold")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--labels", action="store_true",
                    help="score the *label* maps instead of a model — the ceiling any model at this "
                         "output stride can reach. The gap between this and a trained model is what "
                         "training can still win; the gap between this and 1.0 is what the stride "
                         "costs (RESUME: 0.737 at 1/8, 0.933 at 1/4, 0.981 at 1/2, all at 8 px)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    items = load_split(a.data, a.split, a.down, a.sigma, a.adaptive, a.fidt, a.limit)
    if not items:
        raise SystemExit("no %s images under %s" % (a.split, a.data))
    if a.labels:
        print("labels (down %d) on %d %s images of %s" % (a.down, len(items), a.split, a.data))
        preds = [lab for _f, _x, _pts, lab, _w, _h in items]
    else:
        model = C.CSRNet(1.0, 8 // max(1, a.down))
        C.load_onnx(model, a.model, verbose=False)
        model.to(a.device).eval()
        print("%s on %d %s images of %s" % (a.model, len(items), a.split, a.data))
        preds = []
        with torch.no_grad():
            for _f, x, _pts, _lab, _w, _h in items:
                preds.append(model(torch.from_numpy(x).to(a.device))[0, 0].cpu().numpy())

    if not a.fidt:
        err = np.array([p.sum() - len(pts) for p, (_f, _x, pts, _l, _w, _h) in zip(preds, items)],
                       np.float64)
        print("count: MAE %.2f  RMSE %.2f  (%d images)"
              % (np.abs(err).mean(), np.sqrt((err ** 2).mean()), len(err)))
        return 0

    def score(thr):
        """`thr <= 0` scores with the reference's LMDS (relative to each map's own maximum)."""
        tp = fp = fn = 0
        for p, (_f, _x, pts, _l, _w, _h) in zip(preds, items):
            found = D.peaks(p, thr, 1) if thr > 0 else D.lmds(p, 1)
            pk = [(px * a.down, py * a.down) for px, py in found]
            r = D.match_points(pk, [tuple(q) for q in pts], a.loc_thr)
            tp += r["tp"]
            fp += r["fp"]
            fn += r["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)

    # The headline number always uses the rule the reference uses, unless an absolute threshold
    # was asked for.
    prec, rec, f1 = score(a.peak_thr)
    mx = float(np.mean([p.max() for p in preds]))
    if a.peak_thr > 0:
        print("localisation @%.2f absolute: F1 %.4f  precision %.4f  recall %.4f  map max %.3f  "
              "(%d images)" % (a.peak_thr, f1, prec, rec, mx, len(items)))
    else:
        print("localisation LMDS (100/255 x map max): F1 %.4f  precision %.4f  recall %.4f  "
              "map max %.3f  (%d images)" % (f1, prec, rec, mx, len(items)))
    if not a.sweep:
        return 0

    print("  absolute thresholds, for diagnosis only:")
    print("  peak_thr | precision | recall |    F1")
    for thr in SWEEP:
        prec, rec, f1 = score(thr)
        print("  %8.2f |   %7.3f | %6.3f | %5.3f" % (thr, prec, rec, f1))

    # amplitude at the annotated heads, label vs prediction — the number that explains the sweep
    lsum = psum = 0.0
    taken = 0
    for p, (_f, _x, pts, lab, _w, _h) in zip(preds, items):
        for px, py in pts:
            iy, ix = int(py) // a.down, int(px) // a.down
            if 0 <= iy < p.shape[0] and 0 <= ix < p.shape[1] \
                    and iy < lab.shape[0] and ix < lab.shape[1]:
                lsum += float(lab[iy, ix])
                psum += float(p[iy, ix])
                taken += 1
    if taken:
        print("  predicted amplitude at %d head positions: %.3f x the label's (label %.3f, pred %.3f)"
              % (taken, psum / max(1e-9, lsum), lsum / taken, psum / taken))
    return 0


if __name__ == "__main__":
    sys.exit(main())
