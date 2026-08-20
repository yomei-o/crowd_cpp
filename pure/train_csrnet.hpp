// Training CSRNet in C++, by differentiating the ONNX graph itself.
//
// The same approach the sibling repo uses for its recogniser: `onx::make_trainable` turns every float
// initializer into a live tensor, the graph runs forward through the ordinary interpreter, the loss is
// built on top of the output, and `write_back` puts the updated weights into the graph so it can be
// saved as ONNX again. There is no second definition of the architecture to drift out of sync — the
// file *is* the model.
//
// The loss is the one tools/train_csrnet.py uses: squared error **summed** over the density map and
// averaged over the batch. Summing rather than averaging over pixels matters, because with `mean` the
// gradient scales with the crop area and a learning rate stops transferring between crop sizes.
#pragma once
#include "density.hpp"
#include "matio.hpp"
#include "onnx_train.hpp"
#include "rng.hpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

// stb lives in the .cpp only (its implementation sits outside the include guard), so declare what is
// needed here rather than including it.
extern "C" unsigned char* stbi_load(const char* filename, int* x, int* y, int* comp, int req_comp);
extern "C" void stbi_image_free(void* retval_from_stbi_load);

namespace csrt {

struct Item {
  std::string img, gt;
  int w = 0, h = 0;
  std::vector<unsigned char> px;      // RGB, kept decoded: ShanghaiTech is small and this makes the
  den::Map target;                    // step time a function of the network alone
  int count = 0;
};

// ImageNet normalisation, because the front end is VGG-16's
inline void normalise(const unsigned char* px, int w, int h, int x0, int y0, int side,
                      bool flip, float* out) {
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x) {
        const int sx = flip ? (x0 + side - 1 - x) : (x0 + x);
        const float v = px[((size_t)(y0 + y) * w + sx) * 3 + c] / 255.f;
        out[((size_t)c * side + y) * side + x] = (v - mean[c]) / sd[c];
      }
}

inline std::vector<Item> read_split(const std::string& part_dir, const std::string& split,
                                    const den::Cfg& cfg, bool verbose = true) {
  std::vector<Item> out;
  const std::string d = part_dir + "/" + split + "_data";
  // ShanghaiTech numbers its files IMG_1..IMG_N with no gaps, so probing beats a directory walk (and
  // avoids std::filesystem, which the sibling repo found throws on non-ASCII paths on Windows).
  for (int i = 1; i <= 100000; ++i) {
    const std::string ip = d + "/images/IMG_" + std::to_string(i) + ".jpg";
    const std::string gp = d + "/ground-truth/GT_IMG_" + std::to_string(i) + ".mat";
    int w = 0, h = 0, ch = 0;
    unsigned char* px = stbi_load(ip.c_str(), &w, &h, &ch, 3);
    if (!px) {
      if (i > 1 && out.empty()) continue;
      if (out.empty()) continue;
      break;                            // the first miss after at least one hit ends the split
    }
    Item it;
    it.img = ip;
    it.gt = gp;
    it.w = w;
    it.h = h;
    it.px.assign(px, px + (size_t)w * h * 3);
    stbi_image_free(px);
    std::string why;
    const std::vector<std::pair<float, float>> pts = mat::load_points(gp, &why);
    it.count = (int)pts.size();
    it.target = den::make(pts, w, h, cfg);
    out.push_back(std::move(it));
    if (verbose && out.size() % 100 == 0) printf("  prepared %zu\n", out.size());
  }
  return out;
}

struct Batch {
  Tensor x;                            // [N,3,side,side]
  std::vector<float> y;                // [N,1,side/down,side/down]
  int mw = 0, mh = 0;
};

inline Batch make_batch(const std::vector<Item>& items, int side, int batch, Rng& rng,
                        const den::Cfg& cfg) {
  Batch b;
  const int ms = side / cfg.down;
  b.mw = b.mh = ms;
  b.x = make_tensor({batch, 3, side, side}, false);
  b.y.assign((size_t)batch * ms * ms, 0.f);
  for (int n = 0; n < batch; ++n) {
    const Item& it = items[(size_t)rng.below((uint64_t)items.size())];
    const int s = std::min(side, std::min(it.h - it.h % cfg.down, it.w - it.w % cfg.down));
    int x0 = (int)rng.below((uint64_t)std::max(1, it.w - s + 1));
    int y0 = (int)rng.below((uint64_t)std::max(1, it.h - s + 1));
    x0 -= x0 % cfg.down;                      // keep image and map aligned
    y0 -= y0 % cfg.down;
    const bool flip = rng.unit() < 0.5;
    normalise(it.px.data(), it.w, it.h, x0, y0, s, flip,
              b.x->data.data() + (size_t)n * 3 * side * side);
    for (int y = 0; y < s / cfg.down; ++y)
      for (int x = 0; x < s / cfg.down; ++x) {
        const int sx = flip ? (x0 / cfg.down + s / cfg.down - 1 - x) : (x0 / cfg.down + x);
        const int sy = y0 / cfg.down + y;
        if (sx < 0 || sy < 0 || sx >= it.target.w || sy >= it.target.h) continue;
        b.y[((size_t)n * ms + y) * ms + x] = it.target.v[(size_t)sy * it.target.w + sx];
      }
  }
  return b;
}

// summed squared error over the map, averaged over the batch
inline Tensor mse_sum(const Tensor& pred, const std::vector<float>& target, int batch) {
  Tensor t = make_tensor(pred->shape, false);
  t->data = target;
  Tensor d = sub(pred, t);
  return mul_scalar(sum(mul(d, d)), 1.f / (float)std::max(1, batch));
}

// MAE / RMSE of the predicted count against the annotation, on whole images — the number the CSRNet
// paper reports, and the reason the graph is written with dynamic H/W.
struct Eval { double mae = 0, rmse = 0; int n = 0; };

inline Eval evaluate(onx::Trainable& t, const std::vector<Item>& items, const den::Cfg& cfg,
                     int limit = 0) {
  Eval e;
  const size_t n = limit > 0 ? std::min((size_t)limit, items.size()) : items.size();
  double sa = 0, ss = 0;
  for (size_t i = 0; i < n; ++i) {
    const Item& it = items[i];
    const int w = it.w - it.w % cfg.down, h = it.h - it.h % cfg.down;
    Tensor x = make_tensor({1, 3, h, w}, false);
    // normalise() fills a square crop; a whole image is rectangular, so fill it directly here
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < h; ++y)
        for (int xx = 0; xx < w; ++xx) {
          const float v = it.px[((size_t)y * it.w + xx) * 3 + c] / 255.f;
          x->data[((size_t)c * h + y) * w + xx] = (v - mean[c]) / sd[c];
        }
    std::map<std::string, Tensor> vals = onx::forward(t, x);
    const Tensor& p = vals.at(t.g.outputs[0].name);
    double s = 0;
    for (int64_t k = 0; k < p->numel(); ++k) s += p->data[(size_t)k];
    const double err = s - it.count;
    sa += std::fabs(err);
    ss += err * err;
    free_graph(p);
  }
  e.n = (int)n;
  e.mae = n ? sa / n : 0;
  e.rmse = n ? std::sqrt(ss / n) : 0;
  return e;
}

}  // namespace csrt
