// Fused LSTM gate/cell/hidden update for one timestep, SIMD-vectorized via
// Google Highway with runtime dynamic dispatch (best ISA chosen at run time
// from a baseline-flags build). Pure math: no XNNPACK/ExecuTorch types, so it
// is unit-testable standalone (native_tests/lstm_cell_test.cpp).
#pragma once
#include <cstddef>

namespace etnp {
// One timestep for all B batch rows.
//   g_ih_t, g_hh: gate pre-activations, row stride 4*H per batch row, gate
//                 block order [i|f|g|o], each block length H (PyTorch order).
//   c, h:         running cell/hidden state [B,H], updated in place.
//   out_t:        this timestep's output rows [B,H]; receives the new h.
// Per element: i,f,o = sigmoid(g_ih+g_hh), g = tanh(g_ih+g_hh);
//              c' = f*c + i*g;  h' = o*tanh(c').
void lstm_cell_update(std::size_t B, std::size_t H,
                      const float* g_ih_t, const float* g_hh,
                      float* c, float* h, float* out_t);
}  // namespace etnp
