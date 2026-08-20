// A MATLAB level-5 .mat reader, enough of one to read crowd-counting ground truth without Python.
//
// Every crowd dataset ships its point annotations as .mat: ShanghaiTech's `GT_IMG_1.mat` holds
// `image_info` as a 1x1 cell containing a 1x1 struct with fields `location` (Nx2 double) and `number`.
// UCF-QNRF and NWPU use flatter layouts of the same primitives. So the reader needs numeric arrays,
// cells, structs and char arrays — and, because MATLAB compresses by default since v7, it needs to
// inflate: the top-level element of a real file is miCOMPRESSED, not the matrix itself.
//
//   mat::File f;
//   if (!mat::load(path, f, &why)) ...
//   const mat::Val* v = f.find("image_info");
//   std::vector<std::pair<float,float>> pts = mat::points(v);   // walks cell/struct to the Nx2 array
//
// The inflate comes from stb_image's zlib decoder, which is already vendored here for PNG — so this
// stays dependency-free. It is *declared* rather than included: stb_image.h puts its implementation
// outside the include guard, so a header that includes it breaks any translation unit that also
// defines STB_IMAGE_IMPLEMENTATION (the sibling repo has this written down as a rule — stb goes in
// the .cpp only). The caller must therefore compile one translation unit with the implementation.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

extern "C" char* stbi_zlib_decode_malloc_guesssize_headerflag(const char* buffer, int len,
                                                             int initial_size, int* outlen,
                                                             int parse_header);

namespace mat {

// data types (the tag's first word)
enum : uint32_t {
  miINT8 = 1, miUINT8 = 2, miINT16 = 3, miUINT16 = 4, miINT32 = 5, miUINT32 = 6,
  miSINGLE = 7, miDOUBLE = 9, miINT64 = 12, miUINT64 = 13, miMATRIX = 14,
  miCOMPRESSED = 15, miUTF8 = 16, miUTF16 = 17, miUTF32 = 18
};
// array classes (the low byte of the array-flags element)
enum : uint32_t {
  mxCELL = 1, mxSTRUCT = 2, mxOBJECT = 3, mxCHAR = 4, mxSPARSE = 5, mxDOUBLE = 6,
  mxSINGLE = 7, mxINT8 = 8, mxUINT8 = 9, mxINT16 = 10, mxUINT16 = 11, mxINT32 = 12,
  mxUINT32 = 13
};

struct Val;
using ValPtr = std::shared_ptr<Val>;

struct Val {
  uint32_t cls = 0;                        // mxDOUBLE, mxCELL, mxSTRUCT, mxCHAR ...
  std::string name;                        // variable or field name
  std::vector<int64_t> dims;
  std::vector<double> num;                 // numeric payload, column-major as MATLAB stores it
  std::string str;                         // mxCHAR payload
  std::vector<ValPtr> cells;               // mxCELL elements, in column-major order
  std::vector<std::pair<std::string, ValPtr>> fields;   // mxSTRUCT: (name, value) per element

  int64_t numel() const {
    int64_t n = dims.empty() ? 0 : 1;
    for (int64_t d : dims) n *= d;
    return n;
  }
  // MATLAB is column-major; crowd point lists are Nx2, so (r,c) needs the stride made explicit.
  double at(int64_t r, int64_t c) const {
    if (dims.size() < 2) return 0;
    const size_t i = (size_t)(c * dims[0] + r);
    return i < num.size() ? num[i] : 0;
  }
  const ValPtr field(const std::string& n) const {
    for (const auto& kv : fields) if (kv.first == n) return kv.second;
    return nullptr;
  }
};

struct File {
  std::string header;
  std::vector<ValPtr> vars;
  const Val* find(const std::string& n) const {
    for (const ValPtr& v : vars) if (v->name == n) return v.get();
    return nullptr;
  }
};

// ---------------------------------------------------------------- parsing

struct Reader {
  const uint8_t* p = nullptr;
  size_t n = 0, i = 0;
  bool swap = false;                       // "MI" instead of "IM": the file is big-endian
  std::string why;

  bool need(size_t k) { return i + k <= n; }
  uint32_t u32() {
    uint32_t v = 0;
    std::memcpy(&v, p + i, 4);
    i += 4;
    if (swap) v = ((v >> 24) & 0xff) | ((v >> 8) & 0xff00) | ((v << 8) & 0xff0000) | (v << 24);
    return v;
  }
  uint16_t u16_at(size_t off) const {
    uint16_t v = 0;
    std::memcpy(&v, p + off, 2);
    if (swap) v = (uint16_t)((v >> 8) | (v << 8));
    return v;
  }

  // A tag is normally 8 bytes (type, byte count). The "small data element" form packs a count of <=4
  // bytes into the upper half of the first word and the payload into the second — easy to miss, and
  // most of the tags in a real ShanghaiTech file are this form.
  bool tag(uint32_t& type, uint32_t& bytes, bool& small) {
    if (!need(8)) { why = "truncated tag"; return false; }
    const size_t at = i;
    uint32_t w = u32();
    const uint32_t upper = w >> 16;
    if (upper != 0) {                       // small element: [count:16][type:16] then 4 payload bytes
      type = w & 0xffff;
      bytes = upper;
      small = true;
      i = at + 4;
      return true;
    }
    type = w;
    bytes = u32();
    small = false;
    return true;
  }
};

inline size_t type_size(uint32_t t) {
  switch (t) {
    case miINT8: case miUINT8: case miUTF8: return 1;
    case miINT16: case miUINT16: case miUTF16: return 2;
    case miINT32: case miUINT32: case miSINGLE: case miUTF32: return 4;
    case miDOUBLE: case miINT64: case miUINT64: return 8;
    default: return 0;
  }
}

// Read `count` numbers of type `t` starting at `off`, widened to double.
inline void read_numbers(Reader& r, size_t off, uint32_t t, size_t count, std::vector<double>& out) {
  out.resize(count);
  const size_t sz = type_size(t);
  for (size_t k = 0; k < count; ++k) {
    const uint8_t* q = r.p + off + k * sz;
    switch (t) {
      case miDOUBLE: { double v; std::memcpy(&v, q, 8); out[k] = v; break; }
      case miSINGLE: { float v; std::memcpy(&v, q, 4); out[k] = v; break; }
      case miINT8:   out[k] = (double)(int8_t)q[0]; break;
      case miUINT8:  out[k] = (double)q[0]; break;
      case miINT16:  { int16_t v; std::memcpy(&v, q, 2); out[k] = v; break; }
      case miUINT16: { uint16_t v; std::memcpy(&v, q, 2); out[k] = v; break; }
      case miINT32:  { int32_t v; std::memcpy(&v, q, 4); out[k] = v; break; }
      case miUINT32: { uint32_t v; std::memcpy(&v, q, 4); out[k] = v; break; }
      case miINT64:  { int64_t v; std::memcpy(&v, q, 8); out[k] = (double)v; break; }
      case miUINT64: { uint64_t v; std::memcpy(&v, q, 8); out[k] = (double)v; break; }
      default: out[k] = 0; break;
    }
  }
}

inline bool parse_matrix(Reader& r, size_t end, ValPtr& out);

// One element: tag, then payload, then padding to the next 8-byte boundary.
inline bool parse_element(Reader& r, size_t limit, uint32_t& type, size_t& data_off, size_t& data_len) {
  uint32_t bytes = 0;
  bool small = false;
  if (!r.tag(type, bytes, small)) return false;
  data_off = r.i;
  data_len = bytes;
  // The *payload* must fit. The padding to the next 8-byte boundary need not: the last element of a
  // file is not padded, so a compressed 133-byte variable ends the file at 269 bytes and a strict
  // check on the padded length rejects a perfectly valid .mat (that was the first bug here).
  if (r.i + bytes > limit) { r.why = "element payload overruns its parent"; return false; }
  const size_t adv = small ? 4 : ((bytes + 7) / 8) * 8;
  r.i = std::min(limit, r.i + adv);
  return true;
}

inline bool parse_matrix(Reader& r, size_t end, ValPtr& out) {
  // array flags
  uint32_t t = 0;
  size_t off = 0, len = 0;
  if (!parse_element(r, end, t, off, len) || len < 8) { r.why = "no array flags"; return false; }
  uint32_t flags = 0;
  std::memcpy(&flags, r.p + off, 4);
  if (r.swap) flags = ((flags >> 24) & 0xff) | ((flags >> 8) & 0xff00) | ((flags << 8) & 0xff0000) | (flags << 24);
  out = std::make_shared<Val>();
  out->cls = flags & 0xff;

  // dimensions
  if (!parse_element(r, end, t, off, len)) return false;
  std::vector<double> dims;
  read_numbers(r, off, t, len / type_size(t ? t : miINT32), dims);
  for (double d : dims) out->dims.push_back((int64_t)d);

  // name
  if (!parse_element(r, end, t, off, len)) return false;
  out->name.assign((const char*)r.p + off, len);

  const int64_t ne = out->numel();
  if (out->cls == mxCELL) {
    for (int64_t k = 0; k < ne && r.i < end; ++k) {
      uint32_t st = 0;
      size_t so = 0, sl = 0;
      if (!parse_element(r, end, st, so, sl)) return false;
      if (st != miMATRIX) { r.why = "cell element is not a matrix"; return false; }
      Reader sub = r;
      sub.i = so;
      ValPtr child;
      if (!parse_matrix(sub, so + sl, child)) { r.why = sub.why; return false; }
      out->cells.push_back(child);
    }
    return true;
  }
  if (out->cls == mxSTRUCT) {
    if (!parse_element(r, end, t, off, len)) return false;          // field name length
    std::vector<double> fl;
    read_numbers(r, off, t, 1, fl);
    const size_t flen = (size_t)fl[0];
    if (!parse_element(r, end, t, off, len)) return false;          // the names, fixed width
    const size_t nf = flen ? len / flen : 0;
    std::vector<std::string> names;
    for (size_t k = 0; k < nf; ++k) {
      const char* q = (const char*)r.p + off + k * flen;
      names.push_back(std::string(q, strnlen(q, flen)));
    }
    for (int64_t e = 0; e < ne; ++e)
      for (size_t k = 0; k < nf; ++k) {
        uint32_t st = 0;
        size_t so = 0, sl = 0;
        if (!parse_element(r, end, st, so, sl)) return false;
        if (st != miMATRIX) { r.why = "struct field is not a matrix"; return false; }
        Reader sub = r;
        sub.i = so;
        ValPtr child;
        if (!parse_matrix(sub, so + sl, child)) { r.why = sub.why; return false; }
        out->fields.emplace_back(names[k], child);
      }
    return true;
  }
  // numeric or char: the real part (an imaginary part, if any, is ignored — no crowd label uses one)
  if (!parse_element(r, end, t, off, len)) return false;
  const size_t sz = type_size(t);
  if (out->cls == mxCHAR) {
    std::vector<double> cs;
    read_numbers(r, off, t, sz ? len / sz : 0, cs);
    for (double c : cs) out->str.push_back((char)(int)c);
  } else if (sz) {
    read_numbers(r, off, t, len / sz, out->num);
  }
  return true;
}

inline bool load(const std::string& path, File& f, std::string* why = nullptr) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) { if (why) *why = "cannot open " + path; return false; }
  fseek(fp, 0, SEEK_END);
  const long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> buf((size_t)std::max(0L, size));
  if (!buf.empty() && fread(buf.data(), 1, buf.size(), fp) != buf.size()) {
    fclose(fp);
    if (why) *why = "short read";
    return false;
  }
  fclose(fp);
  if (buf.size() < 132) { if (why) *why = "too small to be a .mat"; return false; }

  Reader r;
  r.p = buf.data();
  r.n = buf.size();
  f.header.assign((const char*)buf.data(), 116);
  r.swap = !(buf[126] == 'I' && buf[127] == 'M');
  r.i = 128;

  std::vector<std::vector<uint8_t>> keep;   // inflated blocks must outlive the parse
  while (r.i + 8 <= r.n) {
    uint32_t t = 0;
    size_t off = 0, len = 0;
    if (!parse_element(r, r.n, t, off, len)) break;
    if (t == miCOMPRESSED) {
      // MATLAB v7 wraps each variable in a zlib stream. stb_image's inflate is already here for PNG.
      int outn = 0;
      char* z = stbi_zlib_decode_malloc_guesssize_headerflag((const char*)r.p + off, (int)len,
                                                            (int)len * 8 + 1024, &outn, 1);
      if (!z) { if (why) *why = "zlib inflate failed"; return false; }
      keep.emplace_back((uint8_t*)z, (uint8_t*)z + outn);
      free(z);
      Reader sub;
      sub.p = keep.back().data();
      sub.n = keep.back().size();
      sub.swap = r.swap;
      sub.i = 0;
      uint32_t it = 0;
      size_t io = 0, il = 0;
      if (!parse_element(sub, sub.n, it, io, il) || it != miMATRIX) {
        if (why) *why = "compressed block does not hold a matrix";
        return false;
      }
      Reader m2 = sub;
      m2.i = io;
      ValPtr v;
      if (!parse_matrix(m2, io + il, v)) { if (why) *why = m2.why; return false; }
      f.vars.push_back(v);
    } else if (t == miMATRIX) {
      Reader sub = r;
      sub.i = off;
      ValPtr v;
      if (!parse_matrix(sub, off + len, v)) { if (why) *why = sub.why; return false; }
      f.vars.push_back(v);
    }
  }
  if (f.vars.empty() && why) *why = "no variables found";
  return !f.vars.empty();
}

// ---------------------------------------------------------------- crowd labels

// Walk whatever wrapping a dataset chose and return the Nx2 point list. ShanghaiTech nests
// cell -> struct('location') ; some mirrors store the Nx2 array directly under another name.
inline const Val* find_points(const Val* v, int depth = 0) {
  if (!v || depth > 6) return nullptr;
  if (v->cls == mxDOUBLE || v->cls == mxSINGLE) {
    if (v->dims.size() == 2 && v->dims[1] == 2 && v->dims[0] > 0) return v;   // Nx2
    return nullptr;
  }
  if (v->cls == mxSTRUCT) {
    if (const ValPtr loc = v->field("location")) {
      if (const Val* hit = find_points(loc.get(), depth + 1)) return hit;
    }
    for (const auto& kv : v->fields)
      if (const Val* hit = find_points(kv.second.get(), depth + 1)) return hit;
    return nullptr;
  }
  if (v->cls == mxCELL) {
    for (const ValPtr& c : v->cells)
      if (const Val* hit = find_points(c.get(), depth + 1)) return hit;
  }
  return nullptr;
}

inline std::vector<std::pair<float, float>> points(const Val* v) {
  std::vector<std::pair<float, float>> out;
  const Val* a = find_points(v);
  if (!a) return out;
  for (int64_t r = 0; r < a->dims[0]; ++r)
    out.emplace_back((float)a->at(r, 0), (float)a->at(r, 1));   // (x, y), as MATLAB stores them
  return out;
}

// The whole file: try every variable, take the first that yields points.
inline std::vector<std::pair<float, float>> load_points(const std::string& path,
                                                        std::string* why = nullptr) {
  File f;
  if (!load(path, f, why)) return {};
  for (const ValPtr& v : f.vars) {
    std::vector<std::pair<float, float>> p = points(v.get());
    if (!p.empty()) return p;
  }
  if (why) *why = "no Nx2 point array in " + path;
  return {};
}

}  // namespace mat
