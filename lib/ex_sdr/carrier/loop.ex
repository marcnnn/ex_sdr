defmodule ExSdr.Carrier.Loop do
  @moduledoc """
  The second order control loop of GNU Radio's `gr::blocks::control_loop`
  on 32 bit angle words (see `ExSdr.Analog.Math`; a full turn is 2^32), shared
  by the Costas loop, the band-edge FLL and the constellation receiver:

      freq  += beta * error
      phase += freq + alpha * error      (phase wraps, here exactly, as a word)
      freq   = clamp(freq, min_freq, max_freq)

  The order is GNU Radio's: the phase update uses the frequency *before*
  the limit. `alpha` and `beta` come from the loop bandwidth with damping
  sqrt(2) / 2 (`ExSdr.Analog.Kernels.loop_gains/1`).

  The error enters either as a real (`update_real/4`, the Costas and FLL
  detectors: radians per unit, so `error * alpha` radians) or as an angle
  word (`update_words/4`, the constellation receiver: the phase error is an
  angle already).
  """
  import ExSdr.Analog.Math

  alias ExSdr.Analog.Kernels, as: Analog

  @doc "The loop's options: `{alpha, beta, max_freq_word, min_freq_word}`."
  def params(opts) do
    {alpha, beta} = Analog.loop_gains(opts[:loop_bw])
    hi = min(round(opts[:max_freq] * words_per_rad()), 2_147_483_647)
    lo = max(round(opts[:min_freq] * words_per_rad()), -2_147_483_648)
    {alpha, beta, hi, lo}
  end

  @doc """
  One update from a real error (a fixed-point value with at most 32 raw bits,
  or an f32). Returns `{phase, freq}` as s32 words.
  """
  def update_real(err, {phase, freq}, opts, format) do
    {alpha, beta, hi, lo} = params(opts)
    ew_beta = words_from_real(err, beta * words_per_rad(), format)
    ew_alpha = words_from_real(err, alpha * words_per_rad(), format)
    finish(ew_beta, ew_alpha, phase, freq, hi, lo)
  end

  @doc "One update from an error that is an angle word (s64 tensor)."
  def update_words(err, {phase, freq}, opts) do
    {alpha, beta, hi, lo} = params(opts)
    finish(mul_real(err, beta), mul_real(err, alpha), phase, freq, hi, lo)
  end

  defp finish(d_freq, d_phase, phase, freq, hi, lo) do
    freq1 = Nx.add(Nx.as_type(freq, :s64), d_freq)
    phase1 = wrap32(Nx.add(Nx.add(Nx.as_type(phase, :s64), freq1), d_phase))
    freq2 = freq1 |> Nx.clip(i64(lo), i64(hi)) |> Nx.as_type(:s32)
    {phase1, freq2}
  end

  @doc "The initial loop state, phase and frequency words given in radians."
  def init(phase \\ 0.0, freq \\ 0.0) do
    {Nx.tensor(round(phase * words_per_rad()) |> wrap(), type: :s32),
     Nx.tensor(round(freq * words_per_rad()) |> max(-2_147_483_648) |> min(2_147_483_647),
       type: :s32
     )}
  end

  defp wrap(w), do: Integer.mod(w + 2_147_483_648, 4_294_967_296) - 2_147_483_648
end
