// The model runs here, not on the UI thread: one frame of the full model is ~25 s at 512x384 in
// wasm (measured), and a page that freezes for that long looks broken rather than busy.
//
// Protocol: {type:'load', url} -> {type:'loaded', nodes, bytes} | {type:'error', message}
//           {type:'run', rgba, w, h, down, rel} -> {type:'result', ...} | {type:'error', message}
let M = null;
let loaded = false;

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') {
      if (!M) {
        // cache-busted: a stale crowd.wasm next to a fresh crowd.js is a confusing failure, and
        // browsers hold on to both harder than you expect
        importScripts('crowd.js?v=' + msg.stamp);
        M = await createCrowd();
      }
      const r = await fetch(msg.url + '?v=' + msg.stamp);
      if (!r.ok) throw new Error('fetch ' + msg.url + ': HTTP ' + r.status);
      const bytes = new Uint8Array(await r.arrayBuffer());
      const p = M._malloc(bytes.length);
      M.HEAPU8.set(bytes, p);
      const nodes = M.ccall('cw_load', 'number', ['number', 'number'], [p, bytes.length]);
      M._free(p);
      if (nodes <= 0) throw new Error('this file is not a graph the runtime understands');
      loaded = true;
      self.postMessage({type: 'loaded', nodes, bytes: bytes.length});
      return;
    }

    if (msg.type === 'run') {
      if (!loaded) throw new Error('model not loaded yet');
      const {rgba, w, h, down, rel} = msg;
      const p = M._malloc(rgba.length);
      M.HEAPU8.set(rgba, p);
      const t0 = performance.now();
      const n = M.ccall('cw_run', 'number',
                        ['number', 'number', 'number', 'number', 'number', 'number'],
                        [p, w, h, down, rel, 0]);
      const ms = performance.now() - t0;
      M._free(p);
      const res = JSON.parse(M.UTF8ToString(M.ccall('cw_result', 'number', [], [])));
      if (n < 0) throw new Error(res.error || ('cw_run returned ' + n));
      // copy the heat map out of the wasm heap before anything can move it
      const ptr = M.ccall('cw_map', 'number', [], []);
      const heat = new Float32Array(M.HEAPF32.subarray(ptr / 4, ptr / 4 + res.map_w * res.map_h));
      self.postMessage({type: 'result', ...res, ms, heat}, [heat.buffer]);
      return;
    }
  } catch (e) {
    self.postMessage({type: 'error', message: (e && e.message) || String(e)});
  }
};
