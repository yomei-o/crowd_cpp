// CSRNet as an ONNX graph, written in C++.
//
//   crowd init-csrnet --out models/csrnet.onnx [--from-pt vgg16.pth] [--imgsz 384]
//
// CSRNet (Li, Zhang, Chen, CVPR 2018) is two pieces:
//
//   front end : VGG-16's first ten convolutions (conv1_1 .. conv4_3) with three 2x2 max-pools, so the
//               feature map is 1/8 of the input. This is where the pretrained weights matter — the
//               paper's whole argument is that a deep, *pretrained* front end plus dilation beats the
//               multi-column architectures it replaced.
//   back end  : six 3x3 convolutions at dilation 2 (512,512,512,256,128,64), then a 1x1 to one
//               channel. Dilation is the point: the receptive field keeps growing without another
//               pool, so the density map stays at 1/8 instead of collapsing to 1/16 or 1/32.
//
// The output is a single-channel density map at 1/8 resolution whose sum is the head count. Nothing
// about this graph is specific to density: swap the training target and the same file predicts a
// Focal Inverse Distance Transform map instead (FIDTM), which is why this is the first thing built.
//
// Parameter names follow torchvision's VGG-16 state_dict (`features.0.weight`, `features.2.weight`,
// ...) for the front end, so `--from-pt vgg16.pth` maps tensor by tensor, and `backend.<n>.weight`
// for the rest. Initialisation is torch's: U(+-1/sqrt(fan_in)) for convolutions, matching what
// `nn.Conv2d` does when nobody calls an initialiser.
#pragma once
#include "onnx.hpp"
#include "ptio.hpp"
#include "rng.hpp"
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace csr {

struct Spec {
  int imgsz = 384;        // the size the graph *declares*; with `dynamic` it is only a hint
  bool dynamic = true;    // declare H and W as dynamic. CSRNet is convolutional throughout, so the
                          // declared size is metadata: our own runtime derives shapes from the tensor
                          // it is handed either way, but onnxruntime refuses a size the graph pins.
                          // Whole-image evaluation needs this, since ShanghaiTech images differ
                          // (1024x768, 1024x713, 1000x664 ...).
  bool bn = false;        // CSRNet has a batch-norm variant; the paper's headline numbers are without
  float width = 1.0f;     // 1.0 = the paper's channel counts; smaller makes the light variant
  uint64_t seed = 1234;
};

// torchvision VGG-16 layer indices for conv1_1..conv4_3, and the channel counts CSRNet keeps.
// (`M` marks a max-pool; the fourth pool of VGG-16 is deliberately dropped — that is what keeps the
// output at 1/8 rather than 1/16.)
struct FrontLayer { int torch_idx; int cout; bool pool_after; };
inline std::vector<FrontLayer> front_layers() {
  return {
      {0,  64,  false}, {2,  64,  true},                    // conv1_1, conv1_2, pool
      {5,  128, false}, {7,  128, true},                    // conv2_1, conv2_2, pool
      {10, 256, false}, {12, 256, false}, {14, 256, true},  // conv3_1..3_3, pool
      {17, 512, false}, {19, 512, false}, {21, 512, false}, // conv4_1..4_3  (no pool: stays at 1/8)
  };
}
inline std::vector<int> back_channels() { return {512, 512, 512, 256, 128, 64}; }

class Builder {
 public:
  Builder(const Spec& sp, const std::map<std::string, pt::Tensor>* src)
      : sp_(sp), rng_(sp.seed), src_(src) {}

  int taken = 0, made = 0;
  std::vector<std::string> missed;

  onx::Graph build() {
    g_.opset = 13;
    const int64_t dh = sp_.dynamic ? -1 : sp_.imgsz;
    g_.inputs.push_back({"input", {1, 3, dh, dh}});
    std::string x = "input";
    int cin = 3;
    for (const FrontLayer& f : front_layers()) {
      const int cout = scaled(f.cout);
      x = conv_relu(x, "features." + std::to_string(f.torch_idx), cin, cout, 3, 1, 1);
      cin = cout;
      if (f.pool_after) x = pool(x);
    }
    int i = 0;
    for (int c : back_channels()) {
      const int cout = scaled(c);
      // dilation 2 with padding 2 keeps the map at 1/8 while doubling the reach of every kernel
      x = conv_relu(x, "backend." + std::to_string(i++), cin, cout, 3, 1, 2, 2);
      cin = cout;
    }
    // the density head: 1x1 to a single channel, no activation (the target is non-negative but the
    // paper regresses it directly with MSE; clamping here would hide negative predictions instead of
    // letting the loss punish them)
    const std::string out = conv(x, "output_layer", cin, 1, 1, 1, 0, 1, "density");
    g_.outputs.push_back({out, {1, 1, sp_.dynamic ? -1 : sp_.imgsz / 8, sp_.dynamic ? -1 : sp_.imgsz / 8}});
    return g_;
  }

 private:
  Spec sp_;
  Rng rng_;
  const std::map<std::string, pt::Tensor>* src_;
  onx::Graph g_;
  int uid_ = 0;

  int scaled(int c) const {
    if (sp_.width >= 0.999f) return c;
    const int v = (int)std::round(c * sp_.width);
    return std::max(8, (v / 8) * 8);
  }
  std::string uniq(const std::string& b) { return b + "_" + std::to_string(uid_++); }

  void node(const std::string& op, const std::vector<std::string>& in, const std::string& out,
            const std::vector<onx::Attr>& attr = {}) {
    onx::Node n;
    n.op_type = op; n.name = out; n.input = in; n.output = {out}; n.attr = attr;
    g_.nodes.push_back(std::move(n));
  }
  static onx::Attr ai(const std::string& n, int64_t v) {
    onx::Attr a; a.name = n; a.type = onx::A_INT; a.i = v; return a;
  }
  static onx::Attr ais(const std::string& n, std::vector<int64_t> v) {
    onx::Attr a; a.name = n; a.type = onx::A_INTS; a.ints = std::move(v); return a;
  }

  void weight(const std::string& name, const std::vector<int64_t>& dims) {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    if (src_) {
      auto it = src_->find(name);
      if (it != src_->end() && (int64_t)it->second.data.size() == n) {
        bool same = it->second.shape.size() == dims.size();
        for (size_t i = 0; same && i < dims.size(); ++i) same = it->second.shape[i] == dims[i];
        if (same) {
          g_.init_f.push_back({name, dims, it->second.data});
          ++taken;
          return;
        }
      }
      missed.push_back(name);
    }
    int64_t fan_in = 1;
    for (size_t i = 1; i < dims.size(); ++i) fan_in *= dims[i];
    if (dims.size() == 1) fan_in = dims[0];            // bias: torch uses the weight's fan_in, close enough
    const float b = 1.f / std::sqrt((float)std::max<int64_t>(1, fan_in));
    std::vector<float> v((size_t)n);
    for (float& q : v) q = (float)rng_.range(-b, b);
    g_.init_f.push_back({name, dims, v});
    ++made;
  }

  std::string conv(const std::string& in, const std::string& mod, int cin, int cout, int k, int s,
                   int pad, int dil, const std::string& tag = "") {
    weight(mod + ".weight", {cout, cin, k, k});
    weight(mod + ".bias", {cout});
    const std::string o = tag.empty() ? uniq(mod) : tag;
    std::vector<onx::Attr> attr = {ais("kernel_shape", {k, k}), ais("strides", {s, s}),
                                  ais("pads", {pad, pad, pad, pad})};
    if (dil != 1) attr.push_back(ais("dilations", {dil, dil}));
    node("Conv", {in, mod + ".weight", mod + ".bias"}, o, attr);
    return o;
  }

  std::string conv_relu(const std::string& in, const std::string& mod, int cin, int cout, int k,
                        int s, int pad, int dil = 1) {
    const std::string c = conv(in, mod, cin, cout, k, s, pad, dil);
    const std::string r = uniq(mod + "/relu");
    node("Relu", {c}, r);
    return r;
  }

  std::string pool(const std::string& in) {
    const std::string o = uniq("pool");
    node("MaxPool", {in}, o, {ais("kernel_shape", {2, 2}), ais("strides", {2, 2}),
                              ais("pads", {0, 0, 0, 0})});
    return o;
  }
};

inline onx::Graph build(const Spec& sp, const std::map<std::string, pt::Tensor>* from_pt,
                        int* taken = nullptr, int* made = nullptr,
                        std::vector<std::string>* missed = nullptr) {
  Builder b(sp, from_pt);
  onx::Graph g = b.build();
  if (taken) *taken = b.taken;
  if (made) *made = b.made;
  if (missed) *missed = b.missed;
  return g;
}

}  // namespace csr
