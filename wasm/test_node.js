// Headless check of the WASM build: load models/fidt_partB.onnx, push the same pixels the CLI sees
// (a .rgba fixture from `crowd rgba`), and assert on the heads it finds.
//
// A test that feeds a blank frame and expects zero peaks passes even when the model never loaded, and
// a test that only checks "it returned some JSON" passes when the threshold rule is wrong — which is
// the one mistake that actually happened here (an absolute 0.5 instead of a threshold relative to the
// map's maximum drops F1 from 0.80 to 0.005). So this asserts the count and the map's peak height
// against what onnxruntime + tools/density.py produce for the same image.
//
//   ./crowd.exe rgba --img wasm/samples/shibuya-crossing.jpg --out scratch/sample_crossing.rgba
//   node wasm/test_node.js
//
// The default fixture is built from a sample that *is* in the repository, so a fresh clone can run
// this with the one command above.
//
// The fixture is Part B test IMG_10 resized to 512x384 (bilinear, the way a canvas resizes), because
// **the full model does not fit in wasm32 at 1024x768**: our interpreter keeps every intermediate
// tensor, so the VGG front end alone wants 64 x 768 x 1024 x 4 B = 201 MB per layer and the run asks
// for 3.3 GB against a 2 GB limit. 512x384 is the largest size that fits (measured). The light model
// (M8, --width 0.25) is the fix for both memory and speed.
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const fixture = process.argv[2] || path.join(ROOT, 'scratch', 'sample_crossing.rgba');
const W = Number(process.argv[3] || 512);
const H = Number(process.argv[4] || 336);
const MODEL = process.env.MODEL || path.join(ROOT, 'models', 'fidt_partB.onnx');
const DOWN = Number(process.env.DOWN || 2);

// Reference values for the fixture through models/fidt_partB.onnx, computed with onnxruntime +
// tools/density.py (D.lmds) on the same pixels. The WASM build runs our own interpreter compiled by
// clang, so a peak sitting within a float of the relative threshold can flip: allow a few.
const EXPECT_PEAKS = Number(process.env.EXPECT_PEAKS || 218);
const PEAK_TOL = Number(process.env.PEAK_TOL || 5);
const EXPECT_MAX = Number(process.env.EXPECT_MAX || 0.6000);
const MAX_TOL = 0.01;

function fail(msg) { console.log('FAIL: ' + msg); process.exit(1); }

(async () => {
  if (!fs.existsSync(fixture)) {
    fail('no fixture at ' + fixture +
         '\n  make one: ./crowd.exe rgba --img scratch/sht/part_B/test_data/images/IMG_10.jpg' +
         ' --out scratch/sample.rgba');
  }
  if (!fs.existsSync(MODEL)) fail('no model at ' + MODEL);

  const createCrowd = require('./crowd.js');
  const M = await createCrowd();

  // hand over the ONNX bytes
  const onnx = fs.readFileSync(MODEL);
  const mp = M._malloc(onnx.length);
  M.HEAPU8.set(onnx, mp);
  const nodes = M.ccall('cw_load', 'number', ['number', 'number'], [mp, onnx.length]);
  M._free(mp);
  if (nodes <= 0) fail('cw_load returned ' + nodes + ' (not a graph this runtime understands)');
  console.log('loaded ' + path.basename(MODEL) + ': ' + nodes + ' nodes');

  const rgba = fs.readFileSync(fixture);
  if (rgba.length !== W * H * 4) fail('fixture is ' + rgba.length + ' bytes, expected ' + (W * H * 4));
  const fp = M._malloc(rgba.length);
  M.HEAPU8.set(rgba, fp);
  const t0 = Date.now();
  const n = M.ccall('cw_run', 'number',
                    ['number', 'number', 'number', 'number', 'number', 'number'],
                    [fp, W, H, DOWN, 0, 0]);
  const ms = Date.now() - t0;
  M._free(fp);
  if (n < 0) fail('cw_run returned ' + n + ': ' + M.UTF8ToString(M.ccall('cw_result', 'number', [], [])));

  const res = JSON.parse(M.UTF8ToString(M.ccall('cw_result', 'number', [], [])));
  console.log('found ' + res.count + ' heads in ' + ms + ' ms; map ' + res.map_w + 'x' + res.map_h +
              ', max ' + res.map_max.toFixed(4));

  // the crop is to a multiple of 8 (three 2x pools in the front end), so compare against that
  const CW = W - W % 8, CH = H - H % 8;
  if (res.map_w !== CW / DOWN || res.map_h !== CH / DOWN) {
    fail('map is ' + res.map_w + 'x' + res.map_h + ', expected ' + (CW / DOWN) + 'x' + (CH / DOWN) +
         ' for a ' + CW + 'x' + CH + ' crop');
  }
  if (Math.abs(res.map_max - EXPECT_MAX) > MAX_TOL) {
    fail('map max ' + res.map_max.toFixed(4) + ', expected ' + EXPECT_MAX + ' +-' + MAX_TOL +
         ' — the forward pass differs, not just the threshold');
  }
  if (Math.abs(res.count - EXPECT_PEAKS) > PEAK_TOL) {
    fail('found ' + res.count + ' peaks, expected ' + EXPECT_PEAKS + ' +-' + PEAK_TOL);
  }
  if (res.points.length !== res.count) fail('count says ' + res.count + ' but got ' +
                                            res.points.length + ' points');
  const bad = res.points.filter(p => !(p[0] >= 0 && p[0] <= W && p[1] >= 0 && p[1] <= H));
  if (bad.length) fail(bad.length + ' points outside the image, e.g. ' + JSON.stringify(bad[0]));

  // the heat map the page draws must be readable and finite
  const ptr = M.ccall('cw_map', 'number', [], []);
  if (!ptr) fail('cw_map returned null');
  const heat = M.HEAPF32.subarray(ptr / 4, ptr / 4 + res.map_w * res.map_h);
  let mx = -Infinity;
  for (let i = 0; i < heat.length; i++) {
    if (!Number.isFinite(heat[i])) fail('heat map has a non-finite value at ' + i);
    if (heat[i] > mx) mx = heat[i];
  }
  if (Math.abs(mx - res.map_max) > 1e-5) fail('heat max ' + mx + ' != reported ' + res.map_max);

  console.log('PASS');
})();
