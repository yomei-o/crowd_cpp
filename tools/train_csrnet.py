"""Train CSRNet on ShanghaiTech and report the number the paper reports (MAE on whole test images).

  python tools/train_csrnet.py --data <ShanghaiTech>/part_B --init models/csrnet_vgg.onnx \
      --steps 4000 --batch 8 --crop 384 --export models/csrnet_B.onnx

Choices worth stating, because they are the ones that decide whether a reimplementation lands near the
paper (Part A MAE 68.2, Part B 10.6) or nowhere near it:

  * labels come from tools/density.py, i.e. the same generator the C++ side uses — fixed sigma 15 for
    Part B, adaptive (beta=0.3, k=3) for Part A, exactly as the paper splits them.
  * the loss is MSE *summed* over the map, averaged over the batch. Summing matters: with `mean` the
    gradient scales with the crop area and the learning rate stops meaning anything across crop sizes.
  * training is on **whole images** by default (`--crop 0`, batch forced to 1), because that is what
    the paper does and because the mismatch is not cosmetic. Measured with 384 crops instead: the loss
    fell fine (3.4 -> 1.2) while the whole-image MAE sat at 50-158 and swung wildly, and the diagnosis
    was that the model had learned to emit a near-uniform positive level — for IMG_1 (23 heads) it
    predicted 135.6, of which 0.0110 x 12288 cells = 135 was DC. Every 384 crop of Part B contains
    heads, so a constant is a good local minimum on crops and a disaster on a whole image five times
    the area. `--crop N` keeps the old behaviour for experiments.
  * Adam at 1e-5 rather than the paper's SGD at 1e-7. The original schedule needs its full 400-epoch
    run to get anywhere; Adam gets a usable model in a few thousand steps, which is the point of this
    exercise. The trade-off is stated rather than hidden.
  * the front end starts from VGG-16 (`--init` from `crowd init-csrnet --from-pt`). Without it the
    back end's ReLUs die and the map collapses to a constant — measured, and the reason CSRNet is a
    *pretrained* architecture rather than a from-scratch one.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import csrnet as C      # noqa: E402
import density as D     # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
SD = np.array([0.229, 0.224, 0.225], np.float32)


def list_split(part_dir, split):
    """ShanghaiTech's layout: <part>/<split>_data/images/IMG_n.jpg + ground-truth/GT_IMG_n.mat"""
    d = os.path.join(part_dir, split + "_data")
    img_dir, gt_dir = os.path.join(d, "images"), os.path.join(d, "ground-truth")
    if not os.path.isdir(img_dir):
        raise SystemExit("no images under %s" % img_dir)
    out = []
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith(".jpg"):
            continue
        gt = os.path.join(gt_dir, "GT_" + os.path.splitext(f)[0] + ".mat")
        if os.path.exists(gt):
            out.append((os.path.join(img_dir, f), gt))
    return out


class Set:
    """Images, points and the target map, cached in memory. ShanghaiTech is small enough (482 or 716
    images) that decoding once and keeping the arrays beats re-reading every epoch — and it makes the
    step time a function of the network alone, which is what the timings below mean."""

    def __init__(self, items, adaptive, down=8, sigma=15.0, verbose=True, fidt=False):
        from PIL import Image
        self.im, self.gt, self.n, self.pts = [], [], [], []
        self.fidt = fidt
        t0 = time.time()
        for k, (ip, gp) in enumerate(items):
            img = np.asarray(Image.open(ip).convert("RGB"), np.uint8)
            pts = D.load_points(gp)
            h, w, _ = img.shape
            self.im.append(img)
            self.pts.append(pts)
            self.gt.append(D.fidt(pts, w, h, down=down) if fidt
                           else D.density(pts, w, h, down=down, sigma=sigma, adaptive=adaptive))
            self.n.append(len(pts))
            if verbose and (k + 1) % 100 == 0:
                print("  prepared %d/%d (%.1fs)" % (k + 1, len(items), time.time() - t0), flush=True)
        self.down = down
        if verbose:
            print("  %d images, %d..%d heads (mean %.1f), %.1fs"
                  % (len(self.im), min(self.n), max(self.n), float(np.mean(self.n)), time.time() - t0))

    def __len__(self):
        return len(self.im)

    def crop(self, i, size, rng):
        """A random `size`x`size` crop with a random horizontal flip, and the matching target.
        `size <= 0` means the whole image (rectangular), which is the paper's protocol."""
        img, gt = self.im[i], self.gt[i]
        h, w, _ = img.shape
        if size <= 0:
            h -= h % self.down
            w -= w % self.down
            sub, m = img[:h, :w], gt[: h // self.down, : w // self.down]
            if rng.random() < 0.5:
                sub, m = sub[:, ::-1], m[:, ::-1]
            x = ((sub.astype(np.float32) / 255.0 - MEAN) / SD).transpose(2, 0, 1)
            return np.ascontiguousarray(x), np.ascontiguousarray(m)[None]
        s = min(size, h - h % self.down, w - w % self.down)
        y0 = int(rng.integers(0, max(1, h - s + 1)))
        x0 = int(rng.integers(0, max(1, w - s + 1)))
        y0 -= y0 % self.down                      # keep image and map aligned
        x0 -= x0 % self.down
        sub = img[y0:y0 + s, x0:x0 + s]
        m = gt[y0 // self.down:(y0 + s) // self.down, x0 // self.down:(x0 + s) // self.down]
        if rng.random() < 0.5:
            sub, m = sub[:, ::-1], m[:, ::-1]
        x = ((sub.astype(np.float32) / 255.0 - MEAN) / SD).transpose(2, 0, 1)
        return np.ascontiguousarray(x), np.ascontiguousarray(m)[None]

    def whole(self, i):
        img = self.im[i]
        h, w, _ = img.shape
        h -= h % self.down
        w -= w % self.down
        x = ((img[:h, :w].astype(np.float32) / 255.0 - MEAN) / SD).transpose(2, 0, 1)
        return np.ascontiguousarray(x)[None], self.n[i]


@torch.no_grad()
def evaluate(model, ds, device, limit=0):
    """MAE and RMSE of the predicted count against the annotation count, on whole images."""
    model.eval()
    err = []
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        x, cnt = ds.whole(i)
        pred = float(model(torch.from_numpy(x).to(device)).sum())
        err.append(pred - cnt)
    model.train()
    e = np.array(err)
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


@torch.no_grad()
def evaluate_loc(model, ds, device, down, thr=8.0, peak_thr=0.5, limit=0):
    """Localisation: peaks of the predicted FIDT map against the annotated points, as precision /
    recall / F1 at a distance threshold in *image* pixels. This is the number FIDTM exists for; a
    density model's count MAE says nothing about whether the positions are right."""
    model.eval()
    tp = fp = fn = 0
    pmax = []
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        x, _ = ds.whole(i)
        m = model(torch.from_numpy(x).to(device))[0, 0].cpu().numpy()
        # the map's peak height, reported alongside F1: FIDT targets are 1.0 at a head, so an F1 of
        # 0.0000 means something different when the map tops out at 0.03 (nothing is above the 0.5
        # peak threshold yet) than when it tops out at 0.9 (peaks exist but land in the wrong places)
        pmax.append(float(m.max()))
        pk = [(px * down, py * down) for px, py in D.peaks(m, peak_thr, 1)]
        r = D.match_points(pk, [tuple(p) for p in ds.pts[i]], thr)
        tp += r["tp"]
        fp += r["fp"]
        fn += r["fn"]
    model.train()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return (prec, rec, (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0),
            float(np.mean(pmax)) if pmax else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a ShanghaiTech part_A / part_B directory")
    ap.add_argument("--init", default="", help="ONNX to start from (crowd init-csrnet --from-pt ...)")
    ap.add_argument("--export", default="", help="write the trained weights back into an ONNX")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=0,
                    help="0 = whole images (the paper's protocol, batch forced to 1); N = NxN crops")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optim", default="adam", choices=["adam", "sgd"],
                    help="sgd is the paper's recipe (momentum 0.95); adam converges faster but its "
                         "steps move the map's DC level around, and the *count* is what that hurts")
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--lr-final", dest="lr_final", type=float, default=1.0,
                    help="cosine-decay the lr to lr*this by the last step (1.0 = constant, which is "
                         "what the reference implementation does)")
    ap.add_argument("--log", default="", help="write step,loss,lr and every eval to a CSV")
    ap.add_argument("--weight-decay", dest="wd", type=float, default=5e-4,
                    help="the reference implementation's value (leeyeehoo/CSRNet-pytorch)")
    ap.add_argument("--count-weight", dest="count_weight", type=float, default=0.0,
                    help="add this times (sum(pred)-sum(target))^2 / N to the loss. The summed MSE "
                         "barely constrains a uniform offset — 0.01 per cell over 12288 cells is 123 "
                         "heads of error but only 1.2 of loss — so the metric we report is nearly "
                         "unconstrained by the loss we train. This term constrains it directly.")
    ap.add_argument("--fidt", action="store_true",
                    help="train the FIDT target instead of a density map, and report localisation F1 "
                         "instead of count MAE. Use with a decoder graph (--decoder 2|4): the label "
                         "study measured F1 ceilings of 0.737 at 1/8, 0.933 at 1/4 and 0.981 at 1/2.")
    ap.add_argument("--down", type=int, default=8, help="the graph's output stride (decoder 2 -> 4, 4 -> 2)")
    ap.add_argument("--loc-thr", dest="loc_thr", type=float, default=8.0)
    ap.add_argument("--peak-thr", dest="peak_thr", type=float, default=0.5,
                    help="a local maximum counts as a detection above this. FIDTM's default "
                         "is 0.5, but a map that has not grown to 1.0 yet is limited by it "
                         "rather than by where its peaks are: at step 2000 the map topped out "
                         "at 0.446 and recall was 0.001 with precision 1.000")
    ap.add_argument("--adaptive", action="store_true", help="Part A's adaptive sigma (default: fixed 15)")
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--eval-every", dest="eval_every", type=int, default=250)
    ap.add_argument("--eval-limit", dest="eval_limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dump-loss", dest="dump_loss", action="store_true",
                    help="print only `step N loss X` — what the C++/Python parity test compares")
    ap.add_argument("--ckpt", default="",
                    help="write a resumable checkpoint here: weights, the optimiser's moments, both "
                         "RNG streams, the step count and the best metric so far. A Kaggle session "
                         "was measured dying after 1h50m-2h45m, so a 20k-step run has to survive one")
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=0,
                    help="steps between checkpoint writes (0 = only at each eval)")
    ap.add_argument("--resume", default="",
                    help="continue from a --ckpt file: the next step is the one after the one that "
                         "was saved, and the lr schedule keeps the run length it was computed from")
    a = ap.parse_args()

    if a.crop <= 0 and a.batch != 1:
        # whole images differ in size, so they cannot be stacked; the paper trains at batch 1
        print("whole-image training: forcing --batch 1 (was %d)" % a.batch)
        a.batch = 1
    if not a.dump_loss:
        print("train: %s, %s, batch %d, %s lr %g%s, %s sigma, device %s"
              % (a.data, ("whole images" if a.crop <= 0 else "crop %d" % a.crop), a.batch, a.optim,
                 a.lr, (", count weight %g" % a.count_weight) if a.count_weight > 0 else "",
                 "adaptive" if a.adaptive else "fixed", a.device))
    # --fidt / --down have to reach the label generator, or a decoder graph (output 1/2) gets fed
    # density labels at 1/8 and the run dies on a shape mismatch at step 1. The C++ side passes its
    # den::Cfg through; this call was dropping both, which is why the FIDT path had never actually run
    # here (found 2026-08-21, before the first FIDTM training run).
    train = Set(list_split(a.data, "train"), a.adaptive, down=a.down, sigma=a.sigma, fidt=a.fidt,
                verbose=not a.dump_loss)
    test = Set(list_split(a.data, "test"), a.adaptive, down=a.down, sigma=a.sigma, fidt=a.fidt,
               verbose=not a.dump_loss) if a.eval_every else None

    # seeded before the model is built, so that a run *without* --init still starts from the same
    # weights twice — which is what makes the resume check (tools/parity/resume.py) meaningful
    torch.manual_seed(a.seed)
    # the graph's output stride decides the decoder: --down 8 -> none, 4 -> 1/4, 2 -> 1/2
    model = C.CSRNet(decoder=8 // max(1, a.down))
    if a.init:
        C.load_onnx(model, a.init, verbose=not a.dump_loss)
    model.to(a.device).train()
    # The reference implementation (leeyeehoo/CSRNet-pytorch, the paper's first author) uses SGD at a
    # *constant* 1e-7 with momentum 0.95 and weight decay 5e-4, batch 1 on whole images, summed MSE —
    # all of which match what is here — for 400 epochs over a list repeated 4x, i.e. ~480,000 steps.
    # Its `adjust_learning_rate` has scales [1,1,1,1], so there is no decay to copy. The only real
    # differences from this file were the optimiser and the budget.
    opt = (torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
           if a.optim == "sgd" else torch.optim.Adam(model.parameters(), lr=a.lr))
    rng = np.random.default_rng(a.seed)

    # --- resume ---------------------------------------------------------------------------------
    # What has to be restored for "stop at K of N and resume" to equal an uninterrupted run, in the
    # order they were found to matter: the weights, the optimiser's moments (resuming from weights
    # alone restarts Adam's momentum from zero, which shows up as a bump in the loss), both RNG
    # streams (numpy draws the crops, torch is there for anything stochastic in the model), and the
    # run length the lr schedule was computed from — stopping a 6-step cosine at 3 and resuming is
    # *not* the same curve as running 3 then 3 unless the tail knows it is a 6-step run.
    best = (1e9, 0)
    start = 1
    sched_total = a.steps
    if a.resume:
        ck = torch.load(a.resume, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].to("cpu", torch.uint8))
        start = int(ck["step"]) + 1
        sched_total = int(ck["total"])
        best = (float(ck["best"]), int(ck["best_step"]))
        if not a.dump_loss:
            print("resume %s: step %d done, continuing at %d of %d (best %.4f at %d)"
                  % (a.resume, ck["step"], start, sched_total, best[0], best[1]), flush=True)
            if a.steps != sched_total:
                print("  note: --steps %d differs from the checkpoint's %d; keeping the schedule of "
                      "%d and stopping at %d" % (a.steps, sched_total, sched_total, a.steps),
                      flush=True)

    def save_ckpt(step, best):
        """Write the checkpoint via a temp file: a session that dies mid-write must not also take the
        previous checkpoint with it."""
        if not a.ckpt:
            return
        d = os.path.dirname(os.path.abspath(a.ckpt))
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save({"step": step, "total": sched_total, "model": model.state_dict(),
                    "opt": opt.state_dict(), "np_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "best": best[0], "best_step": best[1],
                    "args": vars(a)}, a.ckpt + ".tmp")
        os.replace(a.ckpt + ".tmp", a.ckpt)

    if test is not None and not a.dump_loss and not a.fidt:
        mae, rmse = evaluate(model, test, a.device, a.eval_limit)
        print("step %d: test MAE %.2f  RMSE %.2f  (%s)"
              % (start - 1, mae, rmse, "resumed" if a.resume else "before training"), flush=True)

    run = None
    t0 = time.time()
    # append on resume, so the curve of a run that was interrupted is not thrown away
    log = open(a.log, "a" if a.resume else "w", buffering=1) if a.log else None
    if log and not a.resume:
        log.write("step,loss,lr,test_mae,test_rmse,train_mae" + chr(10))
    for step in range(start, a.steps + 1):
        if a.lr_final < 1.0:
            f = a.lr_final + (1.0 - a.lr_final) * 0.5 * (1.0 + math.cos(math.pi * step / sched_total))
            for gp in opt.param_groups:
                gp["lr"] = a.lr * f
        idx = [int(rng.integers(0, len(train))) for _ in range(a.batch)]
        xs, ys = zip(*[train.crop(i, a.crop, rng) for i in idx])
        x = torch.from_numpy(np.stack(xs)).to(a.device)
        y = torch.from_numpy(np.stack(ys)).to(a.device)
        p = model(x)
        # summed MSE, averaged over the batch (see the module docstring)
        loss = ((p - y) ** 2).sum() / x.shape[0]
        if a.count_weight > 0:
            dc = (p.sum(dim=(1, 2, 3)) - y.sum(dim=(1, 2, 3))) ** 2
            loss = loss + a.count_weight * dc.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss)
        run = lv if run is None else 0.9 * run + 0.1 * lv
        if log:
            log.write("%d,%.6f,%.3e,,," % (step, lv, opt.param_groups[0]["lr"]) + chr(10))
        if a.dump_loss:
            print("step %d loss %.6f" % (step, lv), flush=True)
        elif step % 25 == 0 or step == 1:
            print("  step %5d/%d  loss %10.3f  %5.1fs" % (step, a.steps, run, time.time() - t0),
                  flush=True)
        if test is not None and a.eval_every and step % a.eval_every == 0 and not a.dump_loss and a.fidt:
            pr, rc, f1, pmax = evaluate_loc(model, test, a.device, a.down, a.loc_thr,
                                            peak_thr=a.peak_thr, limit=a.eval_limit)
            tag = ""
            if -f1 < best[0]:
                best = (-f1, step)
                tag = "  <- best"
                if a.export:
                    C.save_onnx(model, a.init or a.export, a.export)
            print("  eval @%d: F1 %.4f  (precision %.4f  recall %.4f, map max %.3f)%s"
                  % (step, f1, pr, rc, pmax, tag), flush=True)
            if log:
                log.write("%d,,%.3e,%.4f,%.4f,%.4f" % (step, opt.param_groups[0]["lr"], f1, pr, rc)
                          + chr(10))
        elif test is not None and a.eval_every and step % a.eval_every == 0 and not a.dump_loss:
            mae, rmse = evaluate(model, test, a.device, a.eval_limit)
            tr_mae, _ = evaluate(model, train, a.device, min(60, len(train)))
            tag = ""
            if mae < best[0]:
                best = (mae, step)
                tag = "  <- best"
                if a.export:
                    C.save_onnx(model, a.init or a.export, a.export)
            print("  eval @%d: test MAE %.2f  RMSE %.2f   (train MAE %.2f)%s"
                  % (step, mae, rmse, tr_mae, tag), flush=True)
            if log:
                log.write("%d,,%.3e,%.4f,%.4f,%.4f" % (step, opt.param_groups[0]["lr"], mae, rmse,
                                                       tr_mae) + chr(10))
        # After the eval, so the checkpoint carries the best-so-far the exported model belongs to.
        if a.ckpt and ((a.ckpt_every and step % a.ckpt_every == 0)
                       or (a.eval_every and step % a.eval_every == 0)):
            save_ckpt(step, best)

    if a.ckpt and a.steps >= start:
        save_ckpt(a.steps, best)
    if not a.dump_loss:
        if a.fidt:
            print("best F1 %.4f at step %d" % (-best[0], best[1]))
        else:
            print("best test MAE %.2f at step %d" % best)
        if a.export and best[1] == 0:
            C.save_onnx(model, a.init or a.export, a.export)
    return 0


if __name__ == "__main__":
    sys.exit(main())
