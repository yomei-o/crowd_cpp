// crowd — the one CLI for this project (C++ side). Mirrors tools/crowd.py subcommand for subcommand;
// whatever one can do, the other must be able to do too.
//
//   crowd init-csrnet --out models/csrnet.onnx [--from-pt vgg16.pth] [--imgsz 384] [--width 1.0]
//   crowd labels      --mat <GT.mat> --img <image> [--fidt | --adaptive] [--out map.bin]
//   crowd train       --data <ShanghaiTech/part_B> --init <onnx> [--steps N] [--export <onnx>]
//   crowd eval        --data <ShanghaiTech/part_B> --model <onnx> [--fidt --sweep]
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
#include "train_csrnet.hpp"
#include "optim.hpp"
#include "trainrt.hpp"
#include "onnx_run.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// mkdir -p. Without this, writing models/x.onnx into a fresh clone silently does nothing: the
// directory is not in git (gitignore keeps generated ONNX out, and git stores no empty directories),
// so fopen fails and the only symptom is a missing file several steps later. The sibling repo has the
// same rule written down.
static void make_dir(const std::string& d) {
  std::string acc;
  for (size_t i = 0; i <= d.size(); ++i) {
    if (i == d.size() || d[i] == '/' || d[i] == (char)0x5c) {
      if (!acc.empty() && acc != "." && acc != "..") {
#ifdef _WIN32
        _mkdir(acc.c_str());
#else
        mkdir(acc.c_str(), 0755);
#endif
      }
    }
    if (i < d.size()) acc += d[i];
  }
}

static void make_parent(const std::string& path) {
  const size_t sl = path.find_last_of("/" + std::string(1, (char)0x5c));
  if (sl != std::string::npos) make_dir(path.substr(0, sl));
}

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
  sp.dynamic = !has_flag(argc, argv, "--static");
  sp.decoder = std::atoi(arg_of(argc, argv, "--decoder", "0").c_str());   // 0=1/8, 2=1/4, 4=1/2, 8=1/1
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
  make_parent(out);
  onx::save_onnx(g, out);
  size_t params = 0;
  for (const onx::Tensor64& t : g.init_f) params += t.data.size();
  printf("wrote %s: CSRNet imgsz=%d width=%.2f decoder=%d, %zu nodes, %zu tensors, %zu parameters\n",
         out.c_str(), sp.imgsz, sp.width, sp.decoder, g.nodes.size(), g.init_f.size(), params);
  if (sp.dynamic)
    printf("  input is declared dynamic (any HxW); the density map is input/8 and its sum is the count\n");
  else
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

// crowd train — fine-tune a CSRNet ONNX in place. Mirrors tools/train_csrnet.py: random crops, summed
// MSE averaged over the batch, whole-image MAE for evaluation.
static int cmd_train(int argc, char** argv) {
  const std::string data = arg_of(argc, argv, "--data", "");
  const std::string init = arg_of(argc, argv, "--init", "models/csrnet.onnx");
  const std::string out = arg_of(argc, argv, "--export", "");
  const int steps = std::atoi(arg_of(argc, argv, "--steps", "100").c_str());
  const int batch = std::atoi(arg_of(argc, argv, "--batch", "4").c_str());
  int crop = std::atoi(arg_of(argc, argv, "--crop", "0").c_str());   // 0 = whole images (the paper's)
  const float lr = (float)atof(arg_of(argc, argv, "--lr", "1e-5").c_str());
  const std::string optim = arg_of(argc, argv, "--optim", "adam");
  const float momentum = (float)atof(arg_of(argc, argv, "--momentum", "0.95").c_str());
  const float wd = (float)atof(arg_of(argc, argv, "--weight-decay", "5e-4").c_str());
  const float count_w = (float)atof(arg_of(argc, argv, "--count-weight", "0").c_str());
  const int eval_every = std::atoi(arg_of(argc, argv, "--eval-every", "0").c_str());
  const int eval_limit = std::atoi(arg_of(argc, argv, "--eval-limit", "0").c_str());
  const uint64_t seed = strtoull(arg_of(argc, argv, "--seed", "1234").c_str(), nullptr, 10);
  const bool dump_loss = has_flag(argc, argv, "--dump-loss");
  const std::string fixture = arg_of(argc, argv, "--dump-fixture", "");
  // Same flags as tools/train_csrnet.py, because the two sides are meant to be interchangeable.
  const float lr_final = (float)atof(arg_of(argc, argv, "--lr-final", "1").c_str());
  const std::string ckpt = arg_of(argc, argv, "--ckpt", "");
  const int ckpt_every = std::atoi(arg_of(argc, argv, "--ckpt-every", "0").c_str());
  const std::string resume = arg_of(argc, argv, "--resume", "");
  const std::string logp = arg_of(argc, argv, "--log", "");
  den::Cfg cfg;
  cfg.down = std::atoi(arg_of(argc, argv, "--down", "8").c_str());
  cfg.sigma = (float)atof(arg_of(argc, argv, "--sigma", "15").c_str());
  cfg.adaptive = has_flag(argc, argv, "--adaptive");
  cfg.fidt = has_flag(argc, argv, "--fidt");
  const float loc_thr = (float)atof(arg_of(argc, argv, "--loc-thr", "8").c_str());
  // FIDTM's default is 0.5, but a map that has not grown to 1.0 yet is limited by the
  // threshold rather than by where its peaks are (measured: map max 0.446 at step 2000 gave
  // recall 0.001 at precision 1.000), so it is a flag on both sides.
  const float peak_thr = (float)atof(arg_of(argc, argv, "--peak-thr", "0.5").c_str());
  if (data.empty()) {
    printf("usage: crowd train --data <ShanghaiTech/part_B> [--init onnx] [--steps N] [--batch N]\n"
           "                   [--crop 0] [--lr 1e-5] [--lr-final 1] [--optim sgd] [--adaptive | --fidt]\n"
           "                   [--down 8] [--loc-thr 8] [--peak-thr 0.5] [--eval-every N] [--export onnx]\n"
           "                   [--log run.csv] [--ckpt run.ck] [--ckpt-every N] [--resume run.ck]\n");
    return 1;
  }

  std::vector<csrt::Item> train = csrt::read_split(data, "train", cfg, !dump_loss);
  if (train.empty()) { printf("no training images under %s\n", data.c_str()); return 1; }
  std::vector<csrt::Item> test;
  if (eval_every > 0) test = csrt::read_split(data, "test", cfg, !dump_loss);

  onx::Graph g = onx::load_onnx(init);
  onx::Trainable t = onx::make_trainable(g);
  if (!dump_loss) {
    printf("%s: %zu trainable tensors, %zu parameters\n", init.c_str(), t.params.size(),
           onx::param_count(t));
    printf("data: %zu train / %zu test images, %s, batch %d, %s lr %g%s, %s sigma\n",
           train.size(), test.size(), crop <= 0 ? "whole images" : "crops",
           crop <= 0 ? 1 : batch, optim.c_str(), lr,
           count_w > 0 ? " (+count term)" : "", cfg.adaptive ? "adaptive" : "fixed");
  }
  // SGD momentum 0.95 / wd 5e-4 is the reference implementation's recipe
  // (leeyeehoo/CSRNet-pytorch, lr 1e-7 constant); Adam converges faster on a small step budget but
  // moves the map's DC level around, and the count is what that hurts.
  const bool use_sgd = optim == "sgd";
  SGD sgd(t.params, lr, momentum, wd, false);
  Adam adam(t.params, lr, 0.9f, 0.999f, 1e-8f, 0.f, false);
  Rng rng(seed);
  double run = -1;
  double best = 1e9;
  int best_step = 0;

  // --- resume ---------------------------------------------------------------------------------
  // Restored: the weights, the optimiser's moments (weights alone would restart Adam's momentum from
  // zero, which shows as a bump in the loss), the sampler's RNG, and the run length the lr schedule
  // was computed from — a 6-step cosine stopped at 3 and resumed is only the same curve if the tail
  // still knows it is a 6-step run. tools/parity/resume.py holds this to the printed digits.
  int start = 1;
  int sched_total = steps;
  if (!resume.empty()) {
    rt::Ckpt ck;
    std::string why;
    if (!ck.load(resume, &why)) { printf("resume: %s\n", why.c_str()); return 1; }
    if (ck.slots.size() != t.params.size()) {
      printf("resume: %zu tensors in the checkpoint, %zu in this model\n", ck.slots.size(),
             t.params.size());
      return 1;
    }
    for (size_t i = 0; i < t.params.size(); ++i) {
      const rt::Slot& s = ck.slots[i];
      if (s.name != t.param_names[i] || s.data.size() != (size_t)t.params[i]->numel()) {
        printf("resume: checkpoint does not match this model at %s\n", s.name.c_str());
        return 1;
      }
      t.params[i]->data = s.data;
      if (use_sgd) { if (!s.s1.empty()) sgd.buf[i] = s.s1; }
      else { if (!s.s1.empty()) adam.m[i] = s.s1; if (!s.s2.empty()) adam.v[i] = s.s2; }
    }
    if (use_sgd) sgd.started = ck.opt_t != 0; else adam.t = ck.opt_t;
    rng.s = ck.rng;
    start = (int)ck.step + 1;
    sched_total = (int)ck.total;
    best = ck.best;
    if (!dump_loss) {
      printf("resume %s: step %lld done, continuing at %d of %d (best %.4f)\n", resume.c_str(),
             (long long)ck.step, start, sched_total, best);
      if (steps != sched_total)
        printf("  note: --steps %d differs from the checkpoint's %d; keeping the schedule of %d "
               "and stopping at %d\n", steps, sched_total, sched_total, steps);
      fflush(stdout);
    }
  }

  // Written via a temp file: a box that dies mid-write must not take the previous checkpoint with it.
  auto save_ckpt = [&](int step) {
    if (ckpt.empty()) return;
    rt::Ckpt ck;
    ck.step = step;
    ck.total = sched_total;
    ck.opt_kind = use_sgd ? 1 : 2;
    ck.opt_t = use_sgd ? (sgd.started ? 1 : 0) : adam.t;
    ck.best = best;
    ck.rng = rng.s;
    for (size_t i = 0; i < t.params.size(); ++i) {
      if (use_sgd) ck.add(t.param_names[i], t.params[i]->data, sgd.buf[i]);
      else ck.add(t.param_names[i], t.params[i]->data, adam.m[i], adam.v[i]);
    }
    make_parent(ckpt);
    const std::string tmp = ckpt + ".tmp";
    if (!ck.save(tmp)) { printf("cannot write %s\n", tmp.c_str()); return; }
    remove(ckpt.c_str());                       // rename() will not overwrite on Windows
    rename(tmp.c_str(), ckpt.c_str());
  };

  // Same columns as the Python side writes, so one plotting script reads either run.
  rt::Log log;
  if (!logp.empty()) {
    make_parent(logp);
    log.open(logp, "step,loss,lr,test_mae,test_rmse,train_mae", !resume.empty());
  }

  for (int step = start; step <= steps; ++step) {
    // cosine to lr*lr_final over sched_total steps; lr_final = 1 is the reference's constant lr
    const float cur_lr = lr_final < 1.f ? cosine_lr(step, sched_total, lr, 0, lr * lr_final) : lr;
    csrt::Batch b = csrt::make_batch(train, crop, batch, rng, cfg);
    std::map<std::string, Tensor> vals = onx::run_onnx(t.g, b.x, {}, &t.init, false);
    Tensor pred = vals.at(t.g.outputs[0].name);
    Tensor loss = csrt::mse_sum(pred, b.y, crop <= 0 ? 1 : batch, count_w);
    if (use_sgd) sgd.zero_grad(); else adam.zero_grad();
    backward(loss);
    if (use_sgd) { sgd.lr = cur_lr; sgd.step(); } else { adam.lr = cur_lr; adam.step(); }
    const double lv = loss->data[0];
    run = run < 0 ? lv : 0.9 * run + 0.1 * lv;
    free_graph(loss);
    if (!fixture.empty() && step == 1) {
      // Step 1's exact batch, the loss and every parameter gradient — what tools/parity/train.py
      // replays through the PyTorch model. Comparing two trainers on their *own* batches would only
      // ever measure the samplers; this measures the loss and the gradients.
      make_parent(fixture);
      FILE* f = fopen(fixture.c_str(), "wb");
      if (f) {
        fwrite("CSRFIX01", 1, 8, f);
        int32_t hdr[5] = {batch, 3, crop, b.mh, (int32_t)t.params.size()};
        fwrite(hdr, 4, 5, f);
        fwrite(b.x->data.data(), 4, b.x->data.size(), f);
        fwrite(b.y.data(), 4, b.y.size(), f);
        float lvf = (float)lv;
        fwrite(&lvf, 4, 1, f);
        for (size_t i = 0; i < t.params.size(); ++i) {
          const std::string& nm = t.param_names[i];
          int32_t nl = (int32_t)nm.size();
          fwrite(&nl, 4, 1, f);
          fwrite(nm.data(), 1, nm.size(), f);
          int32_t ne = (int32_t)t.params[i]->numel();
          fwrite(&ne, 4, 1, f);
          fwrite(t.params[i]->grad.data(), 4, (size_t)ne, f);
        }
        fclose(f);
        if (!dump_loss) printf("wrote %s (batch, loss and %zu parameter gradients of step 1)\n",
                               fixture.c_str(), t.params.size());
      }
    }
    log.row("%d,%.6f,%.3e,,,", step, lv, (double)cur_lr);
    if (dump_loss) printf("step %d loss %.6f\n", step, lv);
    else if (step % 5 == 0 || step == 1) printf("  step %5d/%d  loss %10.3f\n", step, steps, run);
    fflush(stdout);
    if (eval_every > 0 && step % eval_every == 0 && !test.empty() && cfg.fidt) {
      // FIDT: the metric is where the peaks are, not what the map sums to
      csrt::LocEval e = csrt::evaluate_loc(t, test, cfg.down, loc_thr, peak_thr, eval_limit);
      printf("  eval @%d: F1 %.4f  (precision %.4f  recall %.4f, map max %.3f, %d images)%s\n",
             step, e.f1, e.precision, e.recall, e.pmax, e.n, e.f1 > -best ? "  <- best" : "");
      if (e.f1 > -best) { best = -e.f1; best_step = step; }
      log.row("%d,,%.3e,%.4f,%.4f,", step, (double)cur_lr, e.f1, e.precision);
      fflush(stdout);
    } else if (eval_every > 0 && step % eval_every == 0 && !test.empty()) {
      csrt::Eval e = csrt::evaluate(t, test, cfg, eval_limit);
      printf("  eval @%d: test MAE %.2f  RMSE %.2f (%d images)%s\n", step, e.mae, e.rmse, e.n,
             e.mae < best ? "  <- best" : "");
      if (e.mae < best) { best = e.mae; best_step = step; }
      log.row("%d,,%.3e,%.4f,%.4f,", step, (double)cur_lr, e.mae, e.rmse);
      fflush(stdout);
    }
    // After the eval, so `best` in the checkpoint is the one the exported model belongs to.
    if (!ckpt.empty() && ((ckpt_every > 0 && step % ckpt_every == 0)
                          || (eval_every > 0 && step % eval_every == 0))) {
      save_ckpt(step);
      if (!dump_loss) { printf("  ckpt @%d -> %s\n", step, ckpt.c_str()); fflush(stdout); }
    }
  }
  if (!ckpt.empty() && steps >= start) save_ckpt(steps);
  if (!out.empty()) {
    onx::write_back(t);
    make_parent(out);
    onx::save_onnx(t.g, out);
    printf("wrote %s\n", out.c_str());
  }
  if (best_step) {
    if (cfg.fidt) printf("best F1 %.4f at step %d\n", -best, best_step);
    else printf("best test MAE %.2f at step %d\n", best, best_step);
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
    make_parent(out);
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

// crowd eval — score a trained model on a split without training it. Until this existed the only
// evaluation lived inside `crowd train`, so "does the C++ runtime reproduce the MAE the GPU run
// reported?" could not be asked without taking a training step first. Mirrors tools/eval.py.
//
// `--sweep` prints F1 against the peak threshold, and the predicted amplitude at the annotated head
// positions relative to the label's. That pair is what diagnosed the first FIDTM run: F1 0.005 at the
// paper's threshold of 0.5, F1 0.719 at 0.25, because the map only reached 0.463 of the label's height
// (RESUME has the table). A single F1 at a fixed threshold would have hidden it.
static const float kSweep[] = {0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.5f};

static int cmd_eval(int argc, char** argv) {
  const std::string data = arg_of(argc, argv, "--data", "");
  const std::string model = arg_of(argc, argv, "--model", "models/csrnet.onnx");
  const std::string split = arg_of(argc, argv, "--split", "test");
  const int limit = std::atoi(arg_of(argc, argv, "--limit", "0").c_str());
  const float loc_thr = (float)atof(arg_of(argc, argv, "--loc-thr", "8").c_str());
  const float peak_thr = (float)atof(arg_of(argc, argv, "--peak-thr", "0.5").c_str());
  const bool sweep = has_flag(argc, argv, "--sweep");
  den::Cfg cfg;
  cfg.down = std::atoi(arg_of(argc, argv, "--down", "8").c_str());
  cfg.sigma = (float)atof(arg_of(argc, argv, "--sigma", "15").c_str());
  cfg.adaptive = has_flag(argc, argv, "--adaptive");
  cfg.fidt = has_flag(argc, argv, "--fidt");
  if (data.empty()) {
    printf("usage: crowd eval --data <ShanghaiTech/part_B> --model <onnx> [--split test]\n"
           "                  [--limit N] [--down 8] [--adaptive | --fidt] [--peak-thr 0.5]\n"
           "                  [--loc-thr 8] [--sweep]\n");
    return 1;
  }
  std::vector<csrt::Item> items = csrt::read_split(data, split, cfg, false);
  if (items.empty()) { printf("no %s images under %s\n", split.c_str(), data.c_str()); return 1; }
  onx::Graph g = onx::load_onnx(model);
  onx::Trainable t = onx::make_trainable(g);      // only to reuse the eval helpers; nothing is trained
  const size_t n = limit > 0 ? std::min((size_t)limit, items.size()) : items.size();
  printf("%s on %zu %s images of %s\n", model.c_str(), n, split.c_str(), data.c_str());
  if (!cfg.fidt) {
    csrt::Eval e = csrt::evaluate(t, items, cfg, limit);
    printf("count: MAE %.2f  RMSE %.2f  (%d images)\n", e.mae, e.rmse, e.n);
    return 0;
  }
  if (!sweep) {
    csrt::LocEval e = csrt::evaluate_loc(t, items, cfg.down, loc_thr, peak_thr, limit);
    printf("localisation @%.2f: F1 %.4f  precision %.4f  recall %.4f  map max %.3f  (%d images)\n",
           peak_thr, e.f1, e.precision, e.recall, e.pmax, e.n);
    return 0;
  }
  // One forward pass per image, then every threshold scored off the stored maps. Scoring each
  // threshold with its own pass would have run the network seven times for one table, and 60 whole
  // images through the CPU runtime is minutes rather than seconds.
  std::vector<den::Map> maps;
  maps.reserve(n);
  double lsum = 0, psum = 0;
  size_t taken = 0;
  for (size_t i = 0; i < n; ++i) {
    const csrt::Item& it = items[i];
    const int w = it.w - it.w % cfg.down, h = it.h - it.h % cfg.down;
    Tensor x = make_tensor({1, 3, h, w}, false);
    const float mean[3] = {0.485f, 0.456f, 0.406f}, sd[3] = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < h; ++y)
        for (int xx = 0; xx < w; ++xx) {
          const float v = it.px[((size_t)y * it.w + xx) * 3 + c] / 255.f;
          x->data[((size_t)c * h + y) * w + xx] = (v - mean[c]) / sd[c];
        }
    std::map<std::string, Tensor> vals = onx::forward(t, x);
    const Tensor& p = vals.at(t.g.outputs[0].name);
    den::Map m;
    m.h = (int)p->shape[2];
    m.w = (int)p->shape[3];
    m.v.assign(p->data.begin(), p->data.begin() + (size_t)m.w * m.h);
    // amplitude at the annotated heads, label vs prediction — the number that explains the sweep
    for (const std::pair<float, float>& q : it.pts) {
      const int ix = (int)(q.first) / cfg.down, iy = (int)(q.second) / cfg.down;
      if (ix < 0 || iy < 0 || ix >= m.w || iy >= m.h) continue;
      if (ix >= it.target.w || iy >= it.target.h) continue;
      lsum += it.target.v[(size_t)iy * it.target.w + ix];
      psum += m.v[(size_t)iy * m.w + ix];
      ++taken;
    }
    maps.push_back(std::move(m));
    free_graph(p);
  }
  printf("  peak_thr | precision | recall |    F1\n");
  for (float thr : kSweep) {
    int tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < n; ++i) {
      std::vector<std::pair<float, float>> pk = den::peaks(maps[i], thr, 1);
      const float scale = (float)cfg.down;                  // map pixels -> image pixels
      for (std::pair<float, float>& q : pk) { q.first *= scale; q.second *= scale; }
      const den::Loc r = den::match_points(pk, items[i].pts, loc_thr);
      tp += r.tp;
      fp += r.fp;
      fn += r.fn;
    }
    const double prec = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    const double rec = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    const double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
    printf("  %8.2f |   %7.3f | %6.3f | %5.3f\n", thr, prec, rec, f1);
  }
  if (taken)
    printf("  predicted amplitude at %zu head positions: %.3f x the label's "
           "(label %.3f, pred %.3f)\n", taken, psum / std::max(1e-9, lsum),
           lsum / (double)taken, psum / (double)taken);
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
  // CSRNet is convolutional throughout, so the *declared* input size is only metadata; the runtime
  // derives every shape from the tensor it is handed. --imgsz proves that: the same file can be run at
  // a size it was never written for, which is what lets one graph evaluate whole images of any size.
  const int over = std::atoi(arg_of(argc, argv, "--imgsz", "0").c_str());
  if (over > 0) { iw = over; ih = over; }
  const int ow = std::atoi(arg_of(argc, argv, "--w", "0").c_str());
  const int oh = std::atoi(arg_of(argc, argv, "--h", "0").c_str());
  if (ow > 0 && oh > 0) { iw = ow; ih = oh; }

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
    printf("usage: crowd <init-csrnet|labels|train|eval|infer> ...\n");
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "init-csrnet") return cmd_init_csrnet(argc, argv);
  if (cmd == "labels") return cmd_labels(argc, argv);
  if (cmd == "train") return cmd_train(argc, argv);
  if (cmd == "eval") return cmd_eval(argc, argv);
  if (cmd == "infer") return cmd_infer(argc, argv);
  printf("crowd: '%s' is not implemented yet\n", cmd.c_str());
  return 1;
}
