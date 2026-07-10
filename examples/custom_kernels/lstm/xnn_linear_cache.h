// Process-wide cache of packed XnnLinear (XNNPACK fully-connected) operators,
// keyed by weight identity. Eliminates per-execute weight repacking
// (xnn_create_fully_connected_nc_f32 repacks: measured 92us/exec at H=128).
//
// SAFETY ARGUMENT (do not simplify away): the cache NEVER dereferences stored
// pointers. Fingerprints are always computed from the pointers the CALLER
// passes in — its own live kernel arguments — and compared as plain values
// against the stored fingerprint. An entry whose backing memory was freed
// (program unload) is inert: it can only be looked up again by a caller
// presenting the same pointer value, which then refers to the caller's OWN
// live memory, and the fingerprint decides reuse vs repack.
//
// CONCURRENCY: a global mutex guards the map. Each entry carries its own
// mutex, held across reshape+setup+run (XNNPACK operators are mutated by
// those calls) — two Runtime instances sharing an entry cannot race.
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "xnn_linear.h"

namespace etnp {

class XnnLinearCache {
 public:
  struct Entry {
    XnnLinear op;
    uint64_t fingerprint;
    uint64_t stamp = 0;    // LRU tick of last use
    std::mutex run_mutex;  // serializes reshape/setup/run on this operator
    Entry(XnnLinear&& o, uint64_t fp) : op(std::move(o)), fingerprint(fp) {}
  };
  struct Stats {
    std::size_t hits = 0, misses = 0, size = 0;
  };
  static constexpr std::size_t kMaxEntries = 16;

  // Packed FC for (in_ch, out_ch, weight, bias). Creates + inserts on miss or
  // on fingerprint mismatch (same address, different content -> repack).
  // Returns nullptr if XNNPACK operator creation fails.
  static std::shared_ptr<Entry> get(std::size_t in_ch, std::size_t out_ch,
                                    const float* weight, const float* bias) {
    State& s = state();
    const uint64_t fp = fingerprint(weight, in_ch * out_ch, bias, out_ch);
    const Key key{weight, bias, in_ch, out_ch};
    std::lock_guard<std::mutex> lock(s.mu);
    auto it = s.map.find(key);
    if (it != s.map.end()) {
      if (it->second->fingerprint == fp) {
        ++s.hits;
        it->second->stamp = ++s.tick;
        return it->second;
      }
      s.map.erase(it);  // same address, new content: stale packing
    }
    ++s.misses;
    auto op_r = XnnLinear::create(in_ch, out_ch, weight, bias);
    if (!op_r.ok()) return nullptr;
    auto entry = std::make_shared<Entry>(std::move(op_r.get()), fp);
    entry->stamp = ++s.tick;
    if (s.map.size() >= kMaxEntries) {
      auto oldest = s.map.begin();
      for (auto jt = s.map.begin(); jt != s.map.end(); ++jt)
        if (jt->second->stamp < oldest->second->stamp) oldest = jt;
      s.map.erase(oldest);  // shared_ptr keeps in-flight users alive
    }
    s.map.emplace(key, entry);
    return entry;
  }

  static Stats stats() {
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    return Stats{s.hits, s.misses, s.map.size()};
  }

  static void clear() {  // test hook
    State& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.map.clear();
    s.hits = s.misses = 0;
    s.tick = 0;
  }

 private:
  struct Key {
    const float* w;
    const float* b;
    std::size_t ic, oc;
    bool operator==(const Key& o) const {
      return w == o.w && b == o.b && ic == o.ic && oc == o.oc;
    }
  };
  struct KeyHash {
    std::size_t operator()(const Key& k) const {
      uint64_t x = reinterpret_cast<uintptr_t>(k.w);
      x = x * 1099511628211ULL ^ reinterpret_cast<uintptr_t>(k.b);
      x = x * 1099511628211ULL ^ k.ic;
      x = x * 1099511628211ULL ^ k.oc;
      return static_cast<std::size_t>(x);
    }
  };
  struct State {
    std::mutex mu;
    std::unordered_map<Key, std::shared_ptr<Entry>, KeyHash> map;
    uint64_t tick = 0;
    std::size_t hits = 0, misses = 0;
  };
  static State& state() {
    static State s;  // process lifetime; reachable at exit (not an LSan leak)
    return s;
  }

  // FNV-1a over a fixed sample of the weight (first 8 + last 8 floats) and
  // bias (first 4), plus the element count. Cheap and catches address reuse.
  static uint64_t fingerprint(const float* w, std::size_t n_w, const float* b,
                              std::size_t n_b) {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* p, std::size_t bytes) {
      const unsigned char* q = static_cast<const unsigned char*>(p);
      for (std::size_t i = 0; i < bytes; ++i) h = (h ^ q[i]) * 1099511628211ULL;
    };
    mix(&n_w, sizeof(n_w));
    const std::size_t head = n_w < 8 ? n_w : 8;
    mix(w, head * sizeof(float));
    if (n_w > 8) {
      const std::size_t tail = (n_w - 8) < 8 ? (n_w - 8) : 8;
      mix(w + n_w - tail, tail * sizeof(float));
    }
    if (b) {
      const std::size_t bh = n_b < 4 ? n_b : 4;
      mix(b, bh * sizeof(float));
    }
    return h;
  }
};

// Locked run on a cached entry (see CONCURRENCY note above).
inline Error run_cached(XnnLinearCache::Entry& e, std::size_t batch,
                        const float* input, float* out, pthreadpool_t tp) {
  std::lock_guard<std::mutex> lock(e.run_mutex);
  return e.op.run(batch, input, out, tp);
}

}  // namespace etnp
