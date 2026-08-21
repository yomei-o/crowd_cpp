"""Point annotations -> target maps. The Python half of pure/density.hpp.

Every decision here is copied from that header on purpose, because a label generator that disagrees
across languages makes every parity number downstream meaningless:

  * the map is built directly at the output resolution (image size // down), with coordinates and
    sigma divided by `down`;
  * the Gaussian is truncated at 3 sigma and renormalised over the truncated window, so a density
    map's sum is exactly the number of points inside the image;
  * points outside the image are dropped *before* the kNN, so sigma never depends on a head that is
    not in the picture;
  * kNN uses a plain O(N^2) scan with ties left in index order.

  python tools/density.py --mat GT_IMG_1.mat --w 1024 --h 768 --adaptive
  python tools/parity/labels.py --mat GT_IMG_1.mat --w 1024 --h 768      # C++ と突き合わせる
"""
import argparse
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_points(path):
    """Read an Nx2 point list out of whatever the dataset wrapped it in (cell -> struct -> location)."""
    from scipy.io import loadmat
    m = loadmat(path)

    def walk(v, depth=0):
        if depth > 6:
            return None
        a = np.asarray(v)
        if a.dtype == object:
            if a.dtype.names:                      # a struct array
                for name in ("location",) + tuple(n for n in a.dtype.names if n != "location"):
                    if name in a.dtype.names:
                        for idx in np.ndindex(a.shape):
                            hit = walk(a[idx][name], depth + 1)
                            if hit is not None:
                                return hit
                return None
            for idx in np.ndindex(a.shape):        # a cell array
                hit = walk(a[idx], depth + 1)
                if hit is not None:
                    return hit
            return None
        if a.dtype.names:
            for name in a.dtype.names:
                hit = walk(a[name], depth + 1)
                if hit is not None:
                    return hit
            return None
        if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] > 0:
            return a.astype(np.float64)
        return None

    for k, v in m.items():
        if k.startswith("__"):
            continue
        hit = walk(v)
        if hit is not None:
            return hit
    return np.zeros((0, 2), np.float64)


def sigmas(pts, down=8, sigma=15.0, adaptive=False, knn=3, beta=0.3):
    n = len(pts)
    out = np.full(n, sigma / down, np.float64)
    if not adaptive or n < 2:
        return out
    k = max(1, min(knn, n - 1))
    for i in range(n):
        d2 = ((pts - pts[i]) ** 2).sum(1)
        take = np.sort(d2)[: k + 1]                # self is the single zero at index 0
        mean = np.sqrt(take[1 : k + 1]).mean()
        out[i] = max(0.5, beta * mean / down)
    return out


def density(pts, w, h, down=8, sigma=15.0, adaptive=False, knn=3, beta=0.3, trunc=3.0):
    mw, mh = max(1, w // down), max(1, h // down)
    m = np.zeros((mh, mw), np.float32)
    keep = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[keep]
    sg = sigmas(pts, down, sigma, adaptive, knn, beta)
    for i in range(len(pts)):
        cx, cy = pts[i, 0] / down, pts[i, 1] / down
        s = max(0.25, sg[i])
        r = max(1, int(np.ceil(trunc * s)))
        x0, x1 = max(0, int(np.floor(cx)) - r), min(mw - 1, int(np.floor(cx)) + r)
        y0, y1 = max(0, int(np.floor(cy)) - r), min(mh - 1, int(np.floor(cy)) + r)
        if x1 < x0 or y1 < y0:
            continue
        # float32 coordinates and a double exponent, in the same order as the C++ loop
        gx = (np.arange(x0, x1 + 1, dtype=np.float32) + np.float32(0.5) - np.float32(cx)).astype(np.float64)
        gy = (np.arange(y0, y1 + 1, dtype=np.float32) + np.float32(0.5) - np.float32(cy)).astype(np.float64)
        d2 = gy[:, None] ** 2 + gx[None, :] ** 2
        g = np.exp(-d2 / (2.0 * s * s))
        acc = g.sum()
        if acc <= 0:
            continue
        m[y0 : y1 + 1, x0 : x1 + 1] += (g / acc).astype(np.float32)
    return m


def fidt(pts, w, h, down=8, alpha=0.02, beta=0.75):
    mw, mh = max(1, w // down), max(1, h // down)
    m = np.zeros((mh, mw), np.float32)
    keep = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    p = pts[keep] / down
    if len(p) == 0:
        return m
    ys = np.arange(mh, dtype=np.float32) + np.float32(0.5)
    xs = np.arange(mw, dtype=np.float32) + np.float32(0.5)
    # distance to the nearest head, computed in blocks so a dense image does not need W*H*N floats
    D = np.empty((mh, mw), np.float64)
    for y0 in range(0, mh, 32):
        y1 = min(mh, y0 + 32)
        dy = ys[y0:y1, None, None].astype(np.float64) - p[None, None, :, 1]
        dx = xs[None, :, None].astype(np.float64) - p[None, None, :, 0]
        D[y0:y1] = np.sqrt((dx * dx + dy * dy).min(2))
    e = alpha * D + beta
    return (1.0 / (np.power(D, e) + 1.0)).astype(np.float32)


LMDS_REL = 100.0 / 255.0        # dk-liang/FIDTM test.py: threshold = 100/255 * the map's own maximum
LMDS_FLOOR = 0.1                # ... and nothing at all if that maximum is under 0.1


def lmds(m, radius=1, rel=LMDS_REL, floor=LMDS_FLOOR):
    """FIDTM's Local-Maxima-Detection-Strategy, as the reference implementation does it.

    The threshold is **relative to the map's own maximum**, not absolute — that is the part worth
    copying rather than inventing. Measured here 2026-08-21: a model whose map only reaches 0.47
    scores F1 0.005 at an absolute 0.5 and F1 0.72 under this rule, on the same weights. The
    `floor` is their negative-sample guard: a map that never gets above 0.1 predicts nothing.
    """
    mx = float(m.max()) if m.size else 0.0
    if mx < floor:
        return []
    # float32 on purpose: the C++ side computes `rel * mx` in float, and a peak sitting within a
    # float of the threshold would otherwise be kept by one implementation and dropped by the other.
    return peaks(m, float(np.float32(rel) * np.float32(mx)), radius)


def peaks(m, thr=0.5, radius=1):
    """Local maxima above a threshold, ties broken by scan order — the same rule as pure/density.hpp."""
    h, w = m.shape
    out = []
    for y in range(h):
        for x in range(w):
            v = m[y, x]
            if v < thr:
                continue
            top = True
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    u = m[ny, nx]
                    if u > v or (u == v and (ny < y or (ny == y and nx < x))):
                        top = False
                        break
                if not top:
                    break
            if top:
                out.append((x + 0.5, y + 0.5))
    return out


def match_points(pred, gt, thr):
    """Precision / recall / F1 with a distance threshold, matched greedily over the closest pairs.
    The same rule as pure/density.hpp: ties broken by index so both languages agree exactly."""
    pairs = []
    for i, (px, py) in enumerate(pred):
        for j, (gx, gy) in enumerate(gt):
            d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
            if d <= thr:
                pairs.append((d, i, j))
    pairs.sort()
    pu, gu, tp = set(), set(), 0
    for _d, i, j in pairs:
        if i in pu or j in gu:
            continue
        pu.add(i)
        gu.add(j)
        tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--img", default="")
    ap.add_argument("--w", type=int, default=0)
    ap.add_argument("--h", type=int, default=0)
    ap.add_argument("--down", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--fidt", action="store_true")
    ap.add_argument("--out", default="", help="write the map as int32 w,h then float32 data")
    a = ap.parse_args()

    pts = load_points(a.mat)
    w, h = a.w, a.h
    if a.img:
        from PIL import Image
        w, h = Image.open(a.img).size
    if w <= 0 or h <= 0:
        raise SystemExit("give --img or --w/--h")

    if a.fidt:
        m = fidt(pts, w, h, a.down)
        kind = "FIDT"
    else:
        m = density(pts, w, h, a.down, a.sigma, a.adaptive, a.knn, a.beta)
        kind = "density, adaptive sigma" if a.adaptive else "density, fixed sigma"
    print("%s: %d points, image %dx%d -> map %dx%d (%s)"
          % (a.mat, len(pts), w, h, m.shape[1], m.shape[0], kind))
    print("  sum %.6f   max %.6f" % (m.sum(dtype=np.float64), m.max()))
    if a.fidt:
        print("  peaks above 0.5 (radius 1): %d" % len(peaks(m)))
    else:
        inside = int(((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)).sum())
        print("  points inside the image %d -> sum error %.2e"
              % (inside, abs(float(m.sum(dtype=np.float64)) - inside)))
    if a.out:
        with open(a.out, "wb") as f:
            np.array([m.shape[1], m.shape[0]], np.int32).tofile(f)
            m.astype(np.float32).tofile(f)
        print("  wrote %s (%dx%d float32)" % (a.out, m.shape[1], m.shape[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
