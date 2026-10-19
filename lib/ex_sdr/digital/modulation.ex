defmodule ExSdr.Digital.Modulation do
  @moduledoc """
  GNU Radio's modulators (`digital.generic_mod`, `psk_mod`, `qam_mod`,
  `gfsk_mod`, `gmsk_mod`) as compositions of the family's blocks, plus the
  tap generators they use (`firdes.root_raised_cosine`, `firdes.gaussian`).

  Each modulator is a `:host` block that takes packed bytes (a binary) and
  returns the complex baseband `{re, im}` of the block's format. The state
  is explicit: the bits of a partial symbol, the differential encoder and
  the pulse shaper's delay line, so the input can be cut anywhere.

  Pulse shaping is an interpolating FIR with the taps given (`:taps`).
  GNU Radio shapes with `pfb_arb_resampler_ccf` over a 32 arm root raised
  cosine; for an integer `samples_per_symbol` dividing 32 that is exactly
  an interpolating FIR with every `32 / sps`-th tap of the prototype, which
  `rrc_pulse/3` returns.
  """
  import Bitwise

  alias ExSdr.Comm.Bits
  alias ExSdr.Digital
  alias ExSdr.Digital.Constellation
  alias ExSdr.Filters.Firdes
  alias NxFixed.Block
  alias NxFixed.Num

  # -- taps ---------------------------------------------------------------------------------

  @doc """
  `firdes.root_raised_cosine(gain, sampling_freq, symbol_rate, alpha, ntaps)`:
  `ntaps` is made odd. Float64 (GNU Radio rounds to float32). Same as
  `ExSdr.Filters.Firdes.root_raised_cosine/5`.
  """
  def rrc_taps(gain, sampling_freq, symbol_rate, alpha, ntaps),
    do: Firdes.root_raised_cosine(gain, sampling_freq, symbol_rate, alpha, ntaps)

  @doc """
  The pulse GNU Radio's `generic_mod` effectively shapes with: the root
  raised cosine prototype of `nfilts * ntaps_per_filt * sps` taps at rate
  `nfilts` (gain `nfilts`), taken every `nfilts / sps` taps. `sps` must
  divide `nfilts` (default 32).
  """
  def rrc_pulse(sps, excess_bw, opts \\ []) do
    nfilts = Keyword.get(opts, :nfilts, 32)
    per = Keyword.get(opts, :ntaps_per_filt, 11)
    0 = rem(nfilts, sps)
    proto = rrc_taps(nfilts, nfilts, 1.0, excess_bw, nfilts * per * sps)
    proto |> Enum.take_every(div(nfilts, sps))
  end

  @doc "`firdes.gaussian(gain, spb, bt, ntaps)`, see `ExSdr.Filters.Firdes.gaussian/4`."
  def gaussian_taps(gain, spb, bt, ntaps), do: Firdes.gaussian(gain, spb, bt, ntaps)

  @doc """
  The Gaussian pulse of `gfsk_mod` / `gmsk_mod`: `firdes.gaussian(1, sps,
  bt, 4 * sps)` convolved with `sps` ones.
  """
  def gaussian_pulse(sps, bt) do
    g = gaussian_taps(1.0, sps, bt, 4 * sps)
    n = length(g) + sps - 1

    for k <- 0..(n - 1) do
      g
      |> Enum.with_index()
      |> Enum.reduce(0.0, fn {v, i}, acc -> if (k - i) in 0..(sps - 1), do: acc + v, else: acc end)
    end
  end

  # -- modulators ------------------------------------------------------------------------------

  @doc """
  `generic_mod` (and `psk_mod`, `qam_mod`, see `ExSdr.Digital.block/2`).
  Options: `:constellation`, `:samples_per_symbol`, `:taps` (pulse at the
  output rate, default `rrc_pulse(sps, excess_bw)`), `:excess_bw` (0.35),
  `:differential` (false), `:pre_diff_code` (true; only used if the
  constellation applies it), `:gain` (0.999, GNU Radio's output
  attenuation, folded into the taps) and `:format`.
  """
  def generic_mod(name, opts) do
    c = Keyword.fetch!(opts, :constellation)
    f = Keyword.get_lazy(opts, :format, &Num.float/0)
    sps = Keyword.get(opts, :samples_per_symbol, 2)
    true = sps >= 2

    taps =
      Keyword.get_lazy(opts, :taps, fn -> rrc_pulse(sps, Keyword.get(opts, :excess_bw, 0.35)) end)

    gain = Keyword.get(opts, :gain, 0.999)
    differential = Keyword.get(opts, :differential, false)
    bps = c.bits_per_symbol
    pre = Keyword.get(opts, :pre_diff_code, true) and c.apply_pre_diff_code

    map = if pre, do: Digital.block(:map_bb, map: c.pre_diff_code)
    diff = if differential, do: Digital.block(:diff_encoder_bb, modulus: 1 <<< bps)
    symbols = Digital.block(:chunks_to_symbols_bc, table: c.points, format: f)

    shaper =
      Digital.block(:interp_fir_filter_ccf,
        taps: Enum.map(taps, &(&1 * gain)),
        interpolation: sps,
        format: f
      )

    init = %{bits: <<>>, diff: diff && diff.init, shaper: shaper.init}

    fun = fn bytes, st ->
      bits = st.bits <> Bits.unpack(bytes)
      n = div(byte_size(bits), bps)
      used = n * bps
      <<now::binary-size(used), rest::binary>> = bits

      if n == 0 do
        {empty(f), %{st | bits: rest}}
      else
        words = for <<chunk::binary-size(bps) <- now>>, do: word(chunk)
        x = Nx.tensor(words, type: :u8)
        x = if map, do: elem(Digital.run(map, x), 0), else: x
        {x, diff_state} = if diff, do: Digital.run(diff, x, st.diff), else: {x, st.diff}
        {s, _} = Digital.run(symbols, x)
        {y, shaper_state} = Digital.run(shaper, s, st.shaper)
        {y, %{st | bits: rest, diff: diff_state, shaper: shaper_state}}
      end
    end

    %Block{
      name: name,
      kind: :host,
      fun: fun,
      init: init,
      rate: {bps, 8 * sps},
      item: {:byte, :complex},
      hardware: :feedback,
      latency: 0,
      gnuradio: "digital.#{name}"
    }
  end

  @doc """
  `gfsk_mod` (`sensitivity` 1 by default, output gain 0.999) and
  `gmsk_mod` (sensitivity `pi / 2 / sps`, no output gain): packed bytes
  (`do_unpack: true`, default) or bits to NRZ, Gaussian pulse (`:taps`,
  default `gaussian_pulse(sps, bt)`) and frequency modulation.
  """
  def fsk_mod(name, opts) do
    f = Keyword.get_lazy(opts, :format, &Num.float/0)
    sps = Keyword.get(opts, :samples_per_symbol, 2)
    bt = Keyword.get(opts, :bt, 0.35)
    taps = Keyword.get_lazy(opts, :taps, fn -> gaussian_pulse(sps, bt) end)

    {sens, gain} =
      case name do
        :gfsk_mod -> {Keyword.get(opts, :sensitivity, 1.0), Keyword.get(opts, :gain, 0.999)}
        :gmsk_mod -> {:math.pi() / 2 / sps, nil}
      end

    unpack = Keyword.get(opts, :do_unpack, true)
    nrz = Digital.block(:chunks_to_symbols_bf, table: [-1.0, 1.0], format: f)
    shaper = Digital.block(:interp_fir_filter_fff, taps: taps, interpolation: sps, format: f)
    fm = Digital.block(:frequency_modulator_fc, sensitivity: sens, format: f)
    amp = gain && Digital.block(:multiply_const_cc, gain: gain, format: f)

    fun = fn input, st ->
      bits = if unpack, do: Bits.unpack(input), else: input

      if bits == <<>> do
        {empty(f), st}
      else
        x = Nx.tensor(:binary.bin_to_list(bits), type: :u8)
        {s, _} = Digital.run(nrz, x)
        {g, shaper_state} = Digital.run(shaper, s, st.shaper)
        {y, fm_state} = Digital.run(fm, g, st.fm)
        y = if amp, do: elem(Digital.run(amp, y), 0), else: y
        {y, %{st | shaper: shaper_state, fm: fm_state}}
      end
    end

    %Block{
      name: name,
      kind: :host,
      fun: fun,
      init: %{shaper: shaper.init, fm: fm.init},
      rate: if(unpack, do: {1, 8 * sps}, else: {1, sps}),
      item: {if(unpack, do: :byte, else: :bit), :complex},
      hardware: :feedback,
      gnuradio: "digital.#{name}"
    }
  end

  defp word(chunk), do: for(<<b <- chunk>>, reduce: 0, do: (acc -> acc <<< 1 ||| b))

  defp empty(f), do: {Num.zeros({0}, f), Num.zeros({0}, f)}

  @doc "GNU Radio's `psk_mod` constellation for `m` points, see `ExSdr.Digital.Constellation.psk/3`."
  def psk_constellation(m, mod_code, differential),
    do: Constellation.psk(m, mod_code, differential)
end
