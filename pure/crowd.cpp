// crowd — the one CLI for this project (C++ side). Mirrors tools/crowd.py subcommand for subcommand;
// whatever one can do, the other must be able to do too.
//
//   crowd init-csrnet --out models/csrnet.onnx [--from-pt vgg16.pth] [--imgsz 384] [--width 1.0]
//   crowd labels      --mat <GT.mat> --img <image> [--fidt | --adaptive] [--out map.bin]
//   crowd infer       --img <file> --model <onnx> [--out heat.png]
//
// build: sh build/gcc.sh pure/crowd.cpp -o crowd.exe   |   sh build/cc.sh pure/crowd.cpp -o crowd.exe
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "make_csrnet.hpp"
#include "matio.hpp"
#include "density.hpp"
#include "onnx_run.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

static std::string arg_of(int argc, char** argv, const std::string& key, const std::string& def) {
  for (int i = 2; i + 1 < argc; ++i) if (key == argv[i]) return argv[i + 1];
  return def;
}
static bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 2; i < argc; ++i) if (key == argv[i]) return true;
  return false;
}

static int cmd_init_csrnet(int argc, char** argv) {
  csr::Spec sp;
  sp.imgsz = std::atoi(arg_of(argc, argv, "--imgsz", "384").c_str());
  sp.width = (float)atof(arg_of(argc, argv, "--width", "1.0").c_str());
  sp.seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  const std::string out = arg_of(argc, argv, "--out", "");
  const std::string from_pt = arg_of(argc, argv, "--from-pt", "");
  if (out.empty()) {
    printf("usage: crowd init-csrnet --out <onnx> [--imgsz 384] [--width 1.0]\n"
           "                         [--from-pt vgg16.pth] [--seed N]\n");
    return 1;
  }
  std::map<std::string, pt::Tensor> src;
  if (!from_pt.empty()) {
    std::vector<pt::Tensor> ts = pt::load_pt(from_pt);
    if (ts.empty()) ts = pt::load_pt_module(from_pt);
    for (pt::Tensor& t : ts) src[t.name] = std::move(t);
    printf("%s: %zu tensors\n", from_pt.c_str(), src.size());
  }
  int taken = 0, made = 0;
  std::vector<std::string> missed;
  onx::Graph g = csr::build(sp, from_pt.empty() ? nullptr : &src, &taken, &made, &missed);
  onx::save_onnx(g, out);
  size_t params = 0;
  for (const onx::Tensor64& t : g.init_f) params += t.data.size();
  printf("wrote %s: CSRNet imgsz=%d width=%.2f, %zu nodes, %zu tensors, %zu parameters\n",
         out.c_str(), sp.imgsz, sp.width, g.nodes.size(), g.init_f.size(), params);
  printf("  density map is %dx%d (input/8); its sum is the count\n", sp.imgsz / 8, sp.imgsz / 8);
  if (!from_pt.empty()) {
    printf("  %d tensors taken from the checkpoint, %d initialised here\n", taken, made);
    size_t shown = 0;
    for (const std::string& m : missed)
      if (shown++ < 6) printf("    fresh: %s\n", m.c_str());
    if (missed.size() > 6) printf("    ... and %zu more\n", missed.size() - 6);
  }
  return 0;
}

// crowd labels — read a .mat annotation and write the target map, plus the numbers that say whether it
// is right (the sum of a density map must be the head count).
static int cmd_labels(int argc, char** argv) {
  const std::string mat_p = arg_of(argc, argv, "--mat", "");
  const std::string img_p = arg_of(argc, argv, "--img", "");
  const std::string out = arg_of(argc, argv, "--out", "");
  den::Cfg c;
  c.down = std::atoi(arg_of(argc, argv, "--down", "8").c_str());
  c.sigma = (float)atof(arg_of(argc, argv, "--sigma", "15").c_str());
  c.adaptive = has_flag(argc, argv, "--adaptive");
  c.knn = std::atoi(arg_of(argc, argv, "--knn", "3").c_str());
  c.beta = (float)atof(arg_of(argc, argv, "--beta", "0.3").c_str());
  c.fidt = has_flag(argc, argv, "--fidt");
  int W = std::atoi(arg_of(argc, argv, "--w", "0").c_str());
  int H = std::atoi(arg_of(argc, argv, "--h", "0").c_str());
  if (mat_p.empty()) {
    printf("usage: crowd labels --mat <GT.mat> [--img <image>] [--w W --h H] [--down 8]\n"
           "                    [--sigma 15 | --adaptive [--knn 3] [--beta 0.3]] [--fidt]\n"
           "                    [--out map.bin]\n");
    return 1;
  }
  std::string why;
  std::vector<std::pair<float, float>> pts = mat::load_points(mat_p, &why);
  if (pts.empty()) { printf("%s: %s\n", mat_p.c_str(), why.c_str()); return 1; }
  if (!img_p.empty()) {
    int ch = 0;
    unsigned char* px = stbi_load(img_p.c_str(), &W, &H, &ch, 3);
    if (!px) { printf("cannot read %s\n", img_p.c_str()); return 1; }
    stbi_image_free(px);
  }
  if (W <= 0 || H <= 0) { printf("give --img or --w/--h\n"); return 1; }

  den::Map m = den::make(pts, W, H, c);
  printf("%s: %zu points, image %dx%d -> map %dx%d (%s)\n", mat_p.c_str(), pts.size(), W, H,
         m.w, m.h, c.fidt ? "FIDT" : (c.adaptive ? "density, adaptive sigma" : "density, fixed sigma"));
  printf("  sum %.6f   max %.6f\n", m.sum(), m.max());
  if (!c.fidt) {
    // the acceptance criterion for a density label: its sum is the number of points inside the image
    size_t inside = 0;
    for (const auto& p : pts)
      if (p.first >= 0 && p.first < W && p.second >= 0 && p.second < H) ++inside;
    printf("  points inside the image %zu -> sum error %.2e\n", inside,
           std::fabs(m.sum() - (double)inside));
  } else {
    std::vector<std::pair<float, float>> pk = den::peaks(m, 0.5f, 1);
    printf("  peaks above 0.5 (radius 1): %zu\n", pk.size());
  }
  if (!out.empty()) {
    FILE* f = fopen(out.c_str(), "wb");
    if (!f) { printf("cannot write %s\n", out.c_str()); return 1; }
    int32_t hdr[2] = {m.w, m.h};
    fwrite(hdr, 4, 2, f);
    fwrite(m.v.data(), 4, m.v.size(), f);
    fclose(f);
    printf("  wrote %s (%dx%d float32)\n", out.c_str(), m.w, m.h);
  }
  return 0;
}

// crowd infer — run a density model on one image and print the count.
static int cmd_infer(int argc, char** argv) {
  const std::string img = arg_of(argc, argv, "--img", "");
  const std::string model = arg_of(argc, argv, "--model", "models/csrnet.onnx");
  if (img.empty()) { printf("usage: crowd infer --img <file> [--model <onnx>]\n"); return 1; }

  int w = 0, h = 0, ch = 0;
  unsigned char* px = stbi_load(img.c_str(), &w, &h, &ch, 3);
  if (!px) { printf("cannot read %s\n", img.c_str()); return 1; }

  onx::Graph g = onx::load_onnx(model);
  int64_t iw = g.inputs.empty() || g.inputs[0].dims.size() < 4 ? 384 : g.inputs[0].dims[3];
  int64_t ih = g.inputs.empty() || g.inputs[0].dims.size() < 4 ? 384 : g.inputs[0].dims[2];

  // ImageNet normalisation, because the front end is VGG-16's and that is what it was trained with
  const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
  Tensor x = make_tensor({1, 3, ih, iw}, false);
  for (int64_t c = 0; c < 3; ++c)
    for (int64_t y = 0; y < ih; ++y)
      for (int64_t xx = 0; xx < iw; ++xx) {
        const int64_t sx = xx * w / iw, sy = y * h / ih;      // nearest resize, good enough to smoke-test
        const float v = px[(sy * w + sx) * 3 + c] / 255.f;
        x->data[(size_t)((c * ih + y) * iw + xx)] = (v - mean[c]) / sd[c];
      }
  stbi_image_free(px);

  std::map<std::string, Tensor> vals = onx::run_onnx(g, x, {}, nullptr, false);
  const Tensor& d = vals.at(g.outputs[0].name);
  double sum = 0, mx = -1e9;
  for (int64_t i = 0; i < d->numel(); ++i) { sum += d->data[(size_t)i]; mx = std::max(mx, (double)d->data[(size_t)i]); }
  printf("%s %dx%d -> density %lldx%lld, count %.2f (max %.4f)\n", img.c_str(), w, h,
         (long long)d->shape[2], (long long)d->shape[3], sum, mx);
  return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);
#endif
  if (argc < 2) {
    printf("usage: crowd <init-csrnet|labels|infer> ...\n");
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "init-csrnet") return cmd_init_csrnet(argc, argv);
  if (cmd == "labels") return cmd_labels(argc, argv);
  if (cmd == "infer") return cmd_infer(argc, argv);
  printf("crowd: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
