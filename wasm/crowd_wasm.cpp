// WASM entry point — the browser runs the same graph, the same normalisation and the same
// local-maxima rule as `crowd eval`. The page fetches models/fidt_partB.onnx, hands over the bytes,
// then pushes RGBA frames in and reads back head positions.
//
// build: sh build/emcc.sh wasm/crowd_wasm.cpp -o wasm/crowd.js
//
// Two things here are easy to get wrong and both change what the demo shows:
//
//   * the input must be ImageNet-normalised (the front end is VGG-16's), and
//   * a peak counts only above `100/255 x the map's own maximum` — a *relative* threshold, the rule
//     dk-liang/FIDTM uses. With an absolute 0.5 this model finds almost nothing (measured: F1 0.005
//     instead of 0.80), which in a demo looks like a broken model rather than a wrong threshold.
#include "density.hpp"
#include "onnx_run.hpp"
#include <emscripten/emscripten.h>
#include <string>
#include <vector>

static onx::Graph g_graph;
static bool g_ok = false;
static std::string g_result = "{}";
static den::Map g_map;                 // the last predicted map, kept for the heat overlay

extern "C" {

// Returns the node count on success, -1 if the bytes are not a graph this runtime understands.
EMSCRIPTEN_KEEPALIVE int cw_load(const unsigned char* buf, int len) {
  g_graph = onx::parse_onnx(buf, (size_t)len);
  g_ok = !g_graph.nodes.empty();
  return g_ok ? (int)g_graph.nodes.size() : -1;
}

// One frame. `down` is the graph's output stride (2 for models/fidt_partB.onnx), `rel` the peak
// threshold as a fraction of the map's maximum (0 -> the reference's 100/255), `floor` the
// negative-sample guard. Returns the number of heads found, or -1.
EMSCRIPTEN_KEEPALIVE int cw_run(const unsigned char* rgba, int w, int h, int down, float rel,
                               float floor_) {
  if (!g_ok) { g_result = "{\"error\":\"model not loaded\"}"; return -1; }
  if (w <= 0 || h <= 0) { g_result = "{\"error\":\"empty frame\"}"; return -1; }
  const int cw = w - w % down, ch = h - h % down;      // the graph needs a multiple of the stride
  if (cw <= 0 || ch <= 0) { g_result = "{\"error\":\"frame smaller than the stride\"}"; return -1; }

  const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
  Tensor x = make_tensor({1, 3, ch, cw}, false);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < ch; ++y)
      for (int xx = 0; xx < cw; ++xx) {
        const float v = rgba[((size_t)y * w + xx) * 4 + c] / 255.f;   // RGBA in, RGB out
        x->data[((size_t)c * ch + y) * cw + xx] = (v - mean[c]) / sd[c];
      }

  std::map<std::string, Tensor> vals = onx::run_onnx(g_graph, x, {}, nullptr, false);
  auto it = vals.find(g_graph.outputs.empty() ? std::string() : g_graph.outputs[0].name);
  if (it == vals.end()) { g_result = "{\"error\":\"no output\"}"; return -1; }
  const Tensor& p = it->second;
  g_map.h = (int)p->shape[2];
  g_map.w = (int)p->shape[3];
  g_map.v.assign(p->data.begin(), p->data.begin() + (size_t)g_map.w * g_map.h);

  const float r = rel > 0.f ? rel : 100.f / 255.f;
  const float fl = floor_ > 0.f ? floor_ : 0.1f;
  std::vector<std::pair<float, float>> pk = den::lmds(g_map, 1, r, fl);

  // sum of the map is not the count for a FIDT map (that is the density model's property), so the
  // count reported here is the number of peaks
  std::string js = "{\"count\":" + std::to_string((int)pk.size()) +
                   ",\"map_w\":" + std::to_string(g_map.w) +
                   ",\"map_h\":" + std::to_string(g_map.h) +
                   ",\"map_max\":" + std::to_string(g_map.max()) +
                   ",\"points\":[";
  for (size_t i = 0; i < pk.size(); ++i) {
    // map coordinates -> input pixels
    const float px = pk[i].first * (float)down, py = pk[i].second * (float)down;
    js += (i ? ",[" : "[") + std::to_string(px) + "," + std::to_string(py) + "]";
  }
  js += "]}";
  g_result = js;
  free_graph(p);
  return (int)pk.size();
}

EMSCRIPTEN_KEEPALIVE const char* cw_result() { return g_result.c_str(); }

// The map itself, for drawing a heat overlay: float32, map_w * map_h, row major.
EMSCRIPTEN_KEEPALIVE const float* cw_map() { return g_map.v.empty() ? nullptr : g_map.v.data(); }
EMSCRIPTEN_KEEPALIVE int cw_map_w() { return g_map.w; }
EMSCRIPTEN_KEEPALIVE int cw_map_h() { return g_map.h; }

}  // extern "C"
