// Gradient check for the engine ops this project adds — right now that means dilated convolution.
//
//   sh build/gcc.sh pure/gradcheck.cpp -o gradcheck.exe && ./gradcheck.exe
//
// Dilation is the whole point of CSRNet's back end, and it enters conv2d in exactly three places (the
// output size, im2col's input index, and the input-gradient scatter). Two of those are in the backward
// pass, where a mistake does not crash and does not show up in a forward comparison — it just trains
// to something slightly wrong. So the analytic gradient is checked against a central difference at
// dilation 1, 2 and 3, on both the input and the weights.
#include "rng.hpp"
#include "autograd.hpp"
#include <cmath>
#include <cstdio>

int main() {
  Rng rng(7);
  int fails = 0;
  for (int64_t dil : {(int64_t)1, (int64_t)2, (int64_t)3}) {
    const int64_t N = 2, Cin = 3, H = 11, W = 13, Cout = 4, k = 3;
    const int64_t pad = dil;                       // keeps H,W for k=3 at this dilation
    Tensor x = make_tensor({N, Cin, H, W}, true);
    Tensor w = make_tensor({Cout, Cin, k, k}, true);
    Tensor b = make_tensor({Cout}, true);
    for (Tensor* t : {&x, &w, &b})
      for (float& v : (*t)->data) v = (float)rng.range(-1.0, 1.0);

    Tensor y = conv2d(x, w, b, 1, pad, 1, dil);
    printf("dilation %lld: output %lldx%lldx%lldx%lld (H,W must stay %lld,%lld)\n", (long long)dil,
           (long long)y->shape[0], (long long)y->shape[1], (long long)y->shape[2],
           (long long)y->shape[3], (long long)H, (long long)W);
    if (y->shape[2] != H || y->shape[3] != W) { printf("  wrong output size\n"); ++fails; }

    // Differentiate ONE output element, not a weighted sum over all of them. With 1144 outputs the
    // sum is O(30) and a float32 central difference on it carries ~5e-3 of relative noise — enough to
    // hide or fake a bug. One element makes the loss O(1) and the difference clean.
    Rng mr(11);
    Tensor mask = make_tensor(y->shape, false);
    const int64_t pick = (int64_t)mr.below((uint64_t)y->numel());
    mask->data[(size_t)pick] = 1.f;
    auto loss_of = [&]() {
      Tensor yy = conv2d(x, w, b, 1, pad, 1, dil);
      const double s = (double)yy->data[(size_t)pick];
      free_graph(yy);
      return s;
    };
    Tensor loss = sum(mul(y, mask));
    backward(loss);

    double worst = 0;
    const double eps = 1e-2;   // the element is O(1), so a coarse step is the accurate choice here
    // Two entries are skipped on purpose. Ones with no gradient at all (they do not feed the picked
    // output) would make the ratio 0/0. Ones with a *tiny* gradient are worse than useless: the noise
    // of a float32 central difference is absolute (~1e-7 of the output, divided by the step), so an
    // analytic gradient of 1e-5 is compared against numerical noise of the same size and any
    // threshold becomes meaningless. Measured: allowing |g| >= 1e-6 reports 3.9e-03 at dilation 1 and
    // 1.9e-04 at dilation 2 for the *same* correct code.
    int compared = 0;
    for (int trial = 0; trial < 2000 && compared < 40; ++trial) {
      Tensor t = (trial % 2) ? w : x;
      const int64_t i = (int64_t)rng.below((uint64_t)t->numel());
      if (std::fabs(t->grad[(size_t)i]) < 1e-2) continue;
      ++compared;
      const float keep = t->data[(size_t)i];
      t->data[(size_t)i] = keep + (float)eps;
      const double lp = loss_of();
      t->data[(size_t)i] = keep - (float)eps;
      const double lm = loss_of();
      t->data[(size_t)i] = keep;
      const double num = (lp - lm) / (2 * eps), ana = t->grad[(size_t)i];
      const double rel = std::fabs(num - ana) / std::max(1e-6, std::max(std::fabs(num), std::fabs(ana)));
      if (rel > worst) worst = rel;
    }
    printf("  worst relative error over 40 entries (input and weights): %.3e  %s\n", worst,
           worst < 2e-3 ? "OK" : "FAIL");
    if (worst >= 2e-3) ++fails;
    free_graph(loss);
  }
  printf("gradcheck: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
