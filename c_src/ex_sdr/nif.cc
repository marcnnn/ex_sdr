// Erlang NIF bindings for the SIMD kernels (module ExSdr.NIF).
// Inputs are read zero-copy from binaries; outputs are written directly
// into new binaries. Unless noted, functions run on dirty CPU schedulers;
// the byte-level channel-coding ones are also registered as `<name>_small`
// on regular schedulers (same code), for calls that stay well under the
// BEAM's 1 ms rule, where a dirty scheduler round trip costs more than the
// work and its wake latency dominates the tail.

#include <erl_nif.h>

#include <cstring>
#include <fine.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "kernels.h"

namespace {

// A fresh binary of `size` bytes and a pointer to write into it.
template <typename T>
std::tuple<fine::Term, T*> new_binary(ErlNifEnv* env, size_t count) {
  ERL_NIF_TERM term;
  auto* data = enif_make_new_binary(env, count * sizeof(T), &term);
  if (data == nullptr) throw std::runtime_error("cannot allocate binary");
  return {fine::Term(term), reinterpret_cast<T*>(data)};
}

template <typename T>
const T* view(const ErlNifBinary& bin, const char* what) {
  if (bin.size % sizeof(T) != 0) throw std::invalid_argument(std::string(what) + ": size is not a multiple of the element size");
  return reinterpret_cast<const T*>(bin.data);
}

}  // namespace

// Registers `name` a second time as `name_small` on a regular scheduler.
#define FINE_NIF_SMALL(name)                                                                  \
  static ERL_NIF_TERM name##_small_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    return fine::nif(env, argc, argv, name);                                                  \
  }                                                                                           \
  auto __nif_registration_##name##_small = fine::Registration::register_nif(                 \
      {#name "_small", fine::nif_arity(name), name##_small_nif, 0});                          \
  static_assert(true, "require a semicolon after the macro")

std::tuple<fine::Term, fine::Term, int64_t> ddc_q15(ErlNifEnv* env, ErlNifBinary in_i, ErlNifBinary in_q,
                                                    int64_t phase, int64_t phase_inc, int64_t gain,
                                                    int64_t lut_bits, bool round, ErlNifBinary table) {
  if (in_i.size != in_q.size) throw std::invalid_argument("ddc_q15: I and Q must have the same length");
  if (lut_bits < 2 || lut_bits > 14) throw std::invalid_argument("ddc_q15: lut_bits must be 2..14");
  if (table.size != (size_t{1} << lut_bits) * sizeof(int32_t))
    throw std::invalid_argument("ddc_q15: table must have 2^lut_bits int32 entries");

  const auto* t = view<int32_t>(table, "table");
  // The int32 range proof assumes |table| <= 32767 (amplitude 2^15 - 1).
  for (size_t k = 0; k < (size_t{1} << lut_bits); ++k)
    if (t[k] < -32767 || t[k] > 32767) throw std::invalid_argument("ddc_q15: table entries must be within +-32767");

  const size_t n = in_i.size / sizeof(int16_t);
  auto [out_i_term, out_i] = new_binary<int16_t>(env, n);
  auto [out_q_term, out_q] = new_binary<int16_t>(env, n);

  exhdl::DdcParams p{static_cast<uint16_t>(phase), static_cast<uint16_t>(phase_inc), static_cast<int16_t>(gain),
                     static_cast<int>(lut_bits), round, t};
  const uint16_t last = exhdl::DdcQ15(view<int16_t>(in_i, "i"), view<int16_t>(in_q, "q"), n, out_i, out_q, p);
  return {out_i_term, out_q_term, last};
}
FINE_NIF(ddc_q15, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// `history` = the previous n_taps - 1 samples (oldest first), `x` = new
// samples. Returns {y, new_history}. History and block are joined with one
// memcpy here; doing that on the Elixir side costs as much as the filter.
std::tuple<fine::Term, fine::Term> fir_q15(ErlNifEnv* env, ErlNifBinary history, ErlNifBinary x, ErlNifBinary taps,
                                           bool round) {
  const size_t n_taps = taps.size / sizeof(int32_t);
  const size_t hist = n_taps - 1;
  if (n_taps < 2) throw std::invalid_argument("fir_q15: need at least 2 taps");
  if (history.size != hist * sizeof(int16_t)) throw std::invalid_argument("fir_q15: history must have n_taps - 1 samples");

  // The int32 accumulator is exact while 32768 * sum(|taps|) < 2^31.
  const auto* t = view<int32_t>(taps, "taps");
  int64_t bound = 0;
  for (size_t k = 0; k < n_taps; ++k) {
    if (t[k] < -32767 || t[k] > 32767) throw std::invalid_argument("fir_q15: taps must be within +-32767");
    bound += t[k] < 0 ? -int64_t{t[k]} : t[k];
  }
  if (bound * 32768 >= (int64_t{1} << 31)) throw std::invalid_argument("fir_q15: taps too large for an int32 accumulator");

  const size_t n = x.size / sizeof(int16_t);
  std::vector<int16_t> ext(hist + n);
  std::memcpy(ext.data(), view<int16_t>(history, "history"), hist * sizeof(int16_t));
  std::memcpy(ext.data() + hist, view<int16_t>(x, "x"), n * sizeof(int16_t));

  auto [out_term, out] = new_binary<int16_t>(env, n);
  exhdl::FirQ15(ext.data(), n, t, n_taps, round, out);

  auto [hist_term, new_hist] = new_binary<int16_t>(env, hist);
  std::memcpy(new_hist, ext.data() + n, hist * sizeof(int16_t));
  return {out_term, hist_term};
}
FINE_NIF(fir_q15, ERL_NIF_DIRTY_JOB_CPU_BOUND);

std::tuple<fine::Term, int64_t> one_pole_q15(ErlNifEnv* env, ErlNifBinary x, int64_t y0, int64_t alpha, bool round) {
  const size_t n = x.size / sizeof(int16_t);
  auto [out_term, out] = new_binary<int16_t>(env, n);
  const int16_t last = exhdl::OnePoleQ15(view<int16_t>(x, "x"), n, static_cast<int16_t>(y0), static_cast<int16_t>(alpha), round, out);
  return {out_term, last};
}
FINE_NIF(one_pole_q15, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// Scalar reference kernels, exposed for testing the SIMD paths.
std::tuple<fine::Term, fine::Term, int64_t> ddc_q15_scalar(ErlNifEnv* env, ErlNifBinary in_i, ErlNifBinary in_q,
                                                           int64_t phase, int64_t phase_inc, int64_t gain,
                                                           int64_t lut_bits, bool round, ErlNifBinary table) {
  if (in_i.size != in_q.size || table.size != (size_t{1} << lut_bits) * sizeof(int32_t))
    throw std::invalid_argument("ddc_q15_scalar: bad arguments");
  const size_t n = in_i.size / sizeof(int16_t);
  auto [out_i_term, out_i] = new_binary<int16_t>(env, n);
  auto [out_q_term, out_q] = new_binary<int16_t>(env, n);
  exhdl::DdcParams p{static_cast<uint16_t>(phase), static_cast<uint16_t>(phase_inc), static_cast<int16_t>(gain),
                     static_cast<int>(lut_bits), round, view<int32_t>(table, "table")};
  const uint16_t last = exhdl::DdcQ15Scalar(view<int16_t>(in_i, "i"), view<int16_t>(in_q, "q"), n, out_i, out_q, p);
  return {out_i_term, out_q_term, last};
}
FINE_NIF(ddc_q15_scalar, ERL_NIF_DIRTY_JOB_CPU_BOUND);

namespace {
exhdl::ConvCode conv_code(int64_t k, const std::vector<int64_t>& polys) {
  if (k < 2 || k > 9 || polys.size() < 2 || polys.size() > 4)
    throw std::invalid_argument("viterbi: needs constraint length 2..9 and 2..4 polynomials");
  exhdl::ConvCode code{static_cast<unsigned>(k), static_cast<unsigned>(polys.size()), {0, 0, 0, 0}};
  for (size_t j = 0; j < polys.size(); ++j) {
    if (polys[j] <= 0 || polys[j] >= (int64_t{1} << k)) throw std::invalid_argument("viterbi: polynomial out of range");
    code.polys[j] = static_cast<unsigned>(polys[j]);
  }
  return code;
}

void check_rx(const exhdl::ConvCode& code, const ErlNifBinary& rx) {
  if (rx.size % code.n != 0) throw std::invalid_argument("viterbi: rx must hold whole steps");
  for (size_t i = 0; i < rx.size; ++i)
    if (rx.data[i] > 2) throw std::invalid_argument("viterbi: values must be 0, 1 or 2");
}

// Streaming Viterbi state as a binary: header, metrics, decision words.
struct StreamHeader {
  uint32_t k, simd, depth, metrics;
  uint64_t steps, emitted, words;
};

fine::Term encode_state(ErlNifEnv* env, const exhdl::ConvCode& code, const exhdl::ViterbiStreamState& st) {
  const StreamHeader h{code.k, st.simd, st.depth, static_cast<uint32_t>(st.metric.size()), st.steps, st.emitted,
                       st.decisions.size()};
  const size_t size = sizeof h + st.metric.size() * sizeof(uint32_t) + st.decisions.size() * sizeof(uint64_t);
  auto [term, out] = new_binary<uint8_t>(env, size);
  std::memcpy(out, &h, sizeof h);
  std::memcpy(out + sizeof h, st.metric.data(), st.metric.size() * sizeof(uint32_t));
  std::memcpy(out + sizeof h + st.metric.size() * sizeof(uint32_t), st.decisions.data(),
              st.decisions.size() * sizeof(uint64_t));
  return term;
}

exhdl::ViterbiStreamState decode_state(const exhdl::ConvCode& code, const ErlNifBinary& bin) {
  StreamHeader h;
  if (bin.size < sizeof h) throw std::invalid_argument("viterbi_stream: bad state");
  std::memcpy(&h, bin.data, sizeof h);
  const size_t S = size_t{1} << (code.k - 1), words = (S + 63) / 64;
  if (h.k != code.k || h.metrics != S || h.depth == 0 || h.words != (h.steps - h.emitted) * words ||
      bin.size != sizeof h + S * sizeof(uint32_t) + h.words * sizeof(uint64_t))
    throw std::invalid_argument("viterbi_stream: state does not match the code");
  exhdl::ViterbiStreamState st;
  st.simd = h.simd != 0;
  st.depth = h.depth;
  st.steps = h.steps;
  st.emitted = h.emitted;
  st.metric.resize(S);
  std::memcpy(st.metric.data(), bin.data + sizeof h, S * sizeof(uint32_t));
  st.decisions.resize(h.words);
  std::memcpy(st.decisions.data(), bin.data + sizeof h + S * sizeof(uint32_t), h.words * sizeof(uint64_t));
  return st;
}
}  // namespace

// rx: n bytes per step of the mother code (one per generator), values 0/1
// or 2 for an erasure. Returns one byte (0/1) per step.
fine::Term viterbi(ErlNifEnv* env, ErlNifBinary rx, int64_t k, std::vector<int64_t> polys, bool start_unknown, bool simd) {
  const auto code = conv_code(k, polys);
  check_rx(code, rx);
  const size_t n = rx.size / code.n;
  auto [out_term, out] = new_binary<uint8_t>(env, n);
  if (simd)
    exhdl::ViterbiSimd(code, rx.data, n, start_unknown, out);
  else
    exhdl::Viterbi(code, rx.data, n, start_unknown, out);
  return out_term;
}
FINE_NIF(viterbi, ERL_NIF_DIRTY_JOB_CPU_BOUND);

fine::Term viterbi_stream_new(ErlNifEnv* env, int64_t k, std::vector<int64_t> polys, bool start_unknown, bool simd,
                              int64_t depth) {
  const auto code = conv_code(k, polys);
  if (depth < 1 || depth > (int64_t{1} << 20)) throw std::invalid_argument("viterbi_stream_new: depth must be 1..2^20");
  return encode_state(env, code, exhdl::ViterbiStreamNew(code, start_unknown, simd, static_cast<uint32_t>(depth)));
}
FINE_NIF(viterbi_stream_new, 0);

// Returns {bits, new_state}; see exhdl::ViterbiStream.
std::tuple<fine::Term, fine::Term> viterbi_stream(ErlNifEnv* env, ErlNifBinary state, ErlNifBinary rx, int64_t k,
                                                  std::vector<int64_t> polys, bool flush) {
  const auto code = conv_code(k, polys);
  check_rx(code, rx);
  auto st = decode_state(code, state);
  std::vector<uint8_t> bits;
  exhdl::ViterbiStream(code, st, rx.data, rx.size / code.n, flush, bits);
  auto [bits_term, out] = new_binary<uint8_t>(env, bits.size());
  if (!bits.empty()) std::memcpy(out, bits.data(), bits.size());
  return {bits_term, encode_state(env, code, st)};
}
FINE_NIF(viterbi_stream, ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF_SMALL(viterbi_stream);

fine::Term conv_interleave(ErlNifEnv* env, ErlNifBinary in, int64_t branches, int64_t depth, bool inverse) {
  if (branches < 1 || depth < 0) throw std::invalid_argument("conv_interleave: needs branches >= 1, depth >= 0");
  auto [out_term, out] = new_binary<uint8_t>(env, in.size);
  exhdl::ConvInterleave(in.data, in.size, static_cast<int>(branches), static_cast<int>(depth), inverse, out);
  return out_term;
}
FINE_NIF(conv_interleave, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// history: the I * (I - 1) * M bytes before `in`; phase: bytes before `in`.
// Returns {out, new_history}.
std::tuple<fine::Term, fine::Term> conv_interleave_stream(ErlNifEnv* env, ErlNifBinary history, ErlNifBinary in,
                                                          int64_t branches, int64_t depth, bool inverse, int64_t phase) {
  if (branches < 1 || depth < 0 || phase < 0)
    throw std::invalid_argument("conv_interleave_stream: needs branches >= 1, depth >= 0, phase >= 0");
  const size_t hist = static_cast<size_t>(branches) * (branches - 1) * depth;
  if (history.size != hist) throw std::invalid_argument("conv_interleave_stream: history must hold I * (I - 1) * M bytes");
  std::vector<uint8_t> ext(hist + in.size);
  if (hist) std::memcpy(ext.data(), history.data, hist);
  if (in.size) std::memcpy(ext.data() + hist, in.data, in.size);
  auto [out_term, out] = new_binary<uint8_t>(env, in.size);
  exhdl::ConvInterleaveStream(ext.data(), in.size, static_cast<int>(branches), static_cast<int>(depth), inverse,
                              static_cast<uint64_t>(phase), out);
  auto [hist_term, new_hist] = new_binary<uint8_t>(env, hist);
  if (hist) std::memcpy(new_hist, ext.data() + in.size, hist);
  return {out_term, hist_term};
}
FINE_NIF(conv_interleave_stream, ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF_SMALL(conv_interleave_stream);

std::tuple<fine::Term, int64_t, int64_t> rs_decode(ErlNifEnv* env, ErlNifBinary in, int64_t field_poly, int64_t n, int64_t k,
                                                     int64_t fcr, int64_t prim, int64_t shortening) {
  if (n > 255 || k >= n || shortening >= k || (field_poly & 0x100) == 0)
    throw std::invalid_argument("rs_decode: needs a GF(256) code with 0 <= shortening < k < n <= 255");
  const exhdl::RsParams c{static_cast<unsigned>(field_poly), static_cast<int>(n), static_cast<int>(k),
                          static_cast<int>(fcr), static_cast<int>(prim), static_cast<int>(shortening)};
  const size_t len = c.n - c.shortening, msg = c.k - c.shortening;
  if (in.size % len != 0) throw std::invalid_argument("rs_decode: input must be whole codewords");
  const size_t count = in.size / len;
  auto [out_term, out] = new_binary<uint8_t>(env, count * msg);
  size_t corrected = 0, failed = 0;
  exhdl::RsDecode(c, in.data, count, out, &corrected, &failed);
  return {out_term, static_cast<int64_t>(corrected), static_cast<int64_t>(failed)};
}
FINE_NIF(rs_decode, ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF_SMALL(rs_decode);

fine::Term byte_affine(ErlNifEnv* env, ErlNifBinary in, ErlNifBinary keep, ErlNifBinary mask, int64_t phase) {
  if (keep.size != mask.size || mask.size == 0 || phase < 0)
    throw std::invalid_argument("byte_affine: masks of equal size, phase >= 0");
  auto [out_term, out] = new_binary<uint8_t>(env, in.size);
  exhdl::ByteAffine(in.data, in.size, keep.data, mask.data, mask.size, static_cast<size_t>(phase), out);
  return out_term;
}
FINE_NIF(byte_affine, ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF_SMALL(byte_affine);

// 0/1 bytes -> bytes, MSB first; a trailing partial byte is dropped.
fine::Term bits_pack(ErlNifEnv* env, ErlNifBinary bits) {
  const size_t n = bits.size / 8;
  auto [out_term, out] = new_binary<uint8_t>(env, n);
  exhdl::BitPack(bits.data, n, out);
  return out_term;
}
FINE_NIF(bits_pack, ERL_NIF_DIRTY_JOB_CPU_BOUND);
FINE_NIF_SMALL(bits_pack);

std::string simd_target(ErlNifEnv*) { return exhdl::SimdTarget(); }
FINE_NIF(simd_target, 0);

FINE_INIT("Elixir.ExSdr.NIF");
