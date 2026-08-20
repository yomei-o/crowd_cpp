// Point annotations -> the map a network is trained to regress.
//
// Two targets, one code path, because they differ only in what a single point contributes:
//
//   density (CSRNet)  each head becomes a normalised Gaussian, so the map's **sum is the count**.
//                     sigma is either fixed (the paper uses 15 px for ShanghaiTech Part B, whose
//                     crowds are sparse and roughly one scale) or adaptive (Part A: sigma_i is beta
//                     times the mean distance to the k nearest heads, beta=0.3, k=3 — dense regions
//                     get tight blobs, sparse ones get wide ones).
//   FIDT (FIDTM)      each pixel takes 1 / (D^(alpha*D+beta) + 1) where D is the distance to the
//                     *nearest* head. The sum means nothing; the **local maxima are the head
//                     positions**, and the transform is shaped so neighbouring peaks stay separate
//                     however dense the crowd is. This is the whole reason FIDTM can localise where a
//                     Gaussian density map cannot: two heads 3 px apart merge into one blob under a
//                     Gaussian, but stay two peaks under an inverse distance transform.
//
// Decisions that both languages must copy exactly, because a label generator that disagrees across
// languages makes every parity number downstream meaningless:
//
//   * The map is built directly at the output resolution (input/`down`), with point coordinates and
//     sigma divided by `down`. Generating at full resolution and pooling afterwards is equivalent up
//     to discretisation and costs 64x the memory.
//   * The Gaussian is truncated at 3 sigma and then **renormalised over the truncated window**, so
//     the sum is exactly the number of points that landed inside the image. (CSRNet's original code
//     normalises the untruncated kernel and quietly loses the tail mass; ours does not.)
//   * A point outside the image is dropped, and it is dropped *before* the kNN, so sigma never
//     depends on heads that are not in the picture.
//   * kNN distances are computed in original pixels with a plain O(N^2) scan, ties left in index
//     order. N is at most a few thousand per image.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace den {

struct Cfg {
  int down = 8;               // the network's output stride (CSRNet: 8)
  float sigma = 15.f;         // fixed-sigma mode, in original pixels
  bool adaptive = false;      // Part A style: sigma from the k nearest neighbours
  int knn = 3;
  float beta = 0.3f;
  float trunc = 3.f;          // Gaussian support, in sigma
  // FIDT: 1 / (D^(alpha*D + beta_f) + 1), the paper's default shape
  bool fidt = false;
  float fidt_alpha = 0.02f;
  float fidt_beta = 0.75f;
};

struct Map {
  int w = 0, h = 0;
  std::vector<float> v;
  double sum() const {
    double s = 0;
    for (float q : v) s += q;
    return s;
  }
  float max() const {
    float m = 0;
    for (float q : v) m = std::max(m, q);
    return m;
  }
};

// sigma per point, in *output* pixels
inline std::vector<float> sigmas(const std::vector<std::pair<float, float>>& pts, const Cfg& c) {
  std::vector<float> s(pts.size(), c.sigma / (float)c.down);
  if (!c.adaptive || pts.size() < 2) return s;
  const int k = std::max(1, std::min(c.knn, (int)pts.size() - 1));
  std::vector<float> d2(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    for (size_t j = 0; j < pts.size(); ++j) {
      const float dx = pts[i].first - pts[j].first, dy = pts[i].second - pts[j].second;
      d2[j] = dx * dx + dy * dy;
    }
    // The k smallest excluding self. Self contributes the single zero, so sort the first k+1 and skip
    // it. partial_sort rather than a full sort: N is a few thousand and this runs per point.
    const size_t take = std::min(d2.size(), (size_t)k + 1);
    std::partial_sort(d2.begin(), d2.begin() + take, d2.end());
    double mean = 0;
    int used = 0;
    for (int q = 1; q <= k && (size_t)q < d2.size(); ++q) { mean += std::sqrt((double)d2[(size_t)q]); ++used; }
    if (used) mean /= used;
    s[i] = std::max(0.5f, (float)(c.beta * mean) / (float)c.down);
  }
  return s;
}

// The density map: each point contributes a normalised, truncated Gaussian.
inline Map density(const std::vector<std::pair<float, float>>& pts_in, int img_w, int img_h,
                   const Cfg& c) {
  Map m;
  m.w = std::max(1, img_w / c.down);
  m.h = std::max(1, img_h / c.down);
  m.v.assign((size_t)m.w * m.h, 0.f);
  std::vector<std::pair<float, float>> pts;
  for (const auto& p : pts_in)
    if (p.first >= 0 && p.first < img_w && p.second >= 0 && p.second < img_h) pts.push_back(p);
  const std::vector<float> sg = sigmas(pts, c);

  for (size_t i = 0; i < pts.size(); ++i) {
    const float cx = pts[i].first / c.down, cy = pts[i].second / c.down;
    const float s = std::max(0.25f, sg[i]);
    const int r = std::max(1, (int)std::ceil(c.trunc * s));
    const int x0 = std::max(0, (int)std::floor(cx) - r), x1 = std::min(m.w - 1, (int)std::floor(cx) + r);
    const int y0 = std::max(0, (int)std::floor(cy) - r), y1 = std::min(m.h - 1, (int)std::floor(cy) + r);
    double acc = 0;
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        const float dx = (float)x + 0.5f - cx, dy = (float)y + 0.5f - cy;
        acc += std::exp(-(double)(dx * dx + dy * dy) / (2.0 * s * s));
      }
    if (acc <= 0) continue;                       // the point fell outside the map entirely
    for (int y = y0; y <= y1; ++y)
      for (int x = x0; x <= x1; ++x) {
        const float dx = (float)x + 0.5f - cx, dy = (float)y + 0.5f - cy;
        m.v[(size_t)y * m.w + x] += (float)(std::exp(-(double)(dx * dx + dy * dy) / (2.0 * s * s)) / acc);
      }
  }
  return m;
}

// The focal inverse distance transform: value at a pixel is a function of the distance to the nearest
// head only, so peaks do not merge. Computed by brute force over points per pixel — O(W*H*N) at 1/8
// resolution, which is a few million operations for a dense ShanghaiTech image.
inline Map fidt(const std::vector<std::pair<float, float>>& pts_in, int img_w, int img_h,
                const Cfg& c) {
  Map m;
  m.w = std::max(1, img_w / c.down);
  m.h = std::max(1, img_h / c.down);
  m.v.assign((size_t)m.w * m.h, 0.f);
  std::vector<std::pair<float, float>> pts;
  for (const auto& p : pts_in)
    if (p.first >= 0 && p.first < img_w && p.second >= 0 && p.second < img_h)
      pts.emplace_back(p.first / c.down, p.second / c.down);
  if (pts.empty()) return m;

  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      const float px = (float)x + 0.5f, py = (float)y + 0.5f;
      double best = 1e30;
      for (const auto& p : pts) {
        const double dx = px - p.first, dy = py - p.second;
        best = std::min(best, dx * dx + dy * dy);
      }
      const double D = std::sqrt(best);
      // 1 / (D^(alpha*D + beta) + 1): at D=0 this is 1, and it falls off faster where D is larger,
      // which is what keeps two nearby heads as two peaks instead of one plateau.
      const double e = c.fidt_alpha * D + c.fidt_beta;
      m.v[(size_t)y * m.w + x] = (float)(1.0 / (std::pow(D, e) + 1.0));
    }
  return m;
}

inline Map make(const std::vector<std::pair<float, float>>& pts, int w, int h, const Cfg& c) {
  return c.fidt ? fidt(pts, w, h, c) : density(pts, w, h, c);
}

// Local maxima above a threshold — how a FIDT map becomes head positions. `radius` is in output
// pixels; a peak must be the strict maximum of its window (ties broken by scan order, so the result
// is deterministic).
inline std::vector<std::pair<float, float>> peaks(const Map& m, float thr, int radius) {
  std::vector<std::pair<float, float>> out;
  for (int y = 0; y < m.h; ++y)
    for (int x = 0; x < m.w; ++x) {
      const float v = m.v[(size_t)y * m.w + x];
      if (v < thr) continue;
      bool top = true;
      for (int dy = -radius; dy <= radius && top; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
          const int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= m.w || ny >= m.h || (dx == 0 && dy == 0)) continue;
          const float u = m.v[(size_t)ny * m.w + nx];
          if (u > v || (u == v && (ny < y || (ny == y && nx < x)))) { top = false; break; }
        }
      if (top) out.emplace_back((float)x + 0.5f, (float)y + 0.5f);
    }
  return out;
}

}  // namespace den
