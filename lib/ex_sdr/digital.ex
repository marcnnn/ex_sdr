defmodule ExSdr.Digital do
  @moduledoc """
  Blocks of GNU Radio's gr-digital (modulation, symbol mapping, scrambling,
  framing), see `docs/blocks/digital.md` for the coverage table.

  `block(name, opts)` returns the `NxFixed.Block` for a block; `run/3`
  runs one. Sample level blocks take `format: NxFixed.Num.float()` (the
  default) or a fixed-point format.

  Items: bits, symbols and bytes are `u8` tensors (stream and frame blocks)
  or binaries (host blocks); complex samples are `{re, im}` tuples of
  tensors of the format. Stateless stream blocks carry a dummy `u8` state so
  they run like all others.
  """
  @behaviour NxFixed.Block

  alias ExSdr.Digital.Constellation
  alias ExSdr.Digital.Framing
  alias ExSdr.Digital.Kernels, as: K
  alias ExSdr.Digital.Modulation
  alias NxFixed.Block

  @doc """
  Runs `block` (see `NxFixed.Block.run/4`); returns `{output, state}`.
  Float instances of stream blocks are run with `allow_float: true` (the
  width check only applies to fixed-point instances). Stream blocks with
  several outputs per input (`rate: {1, n}`) return the items flattened.
  """
  def run(block, input, state \\ :init, opts \\ []) do
    float? = match?(%NxFixed.Num.Format{kind: :float}, block.opts[:format])
    opts = if float?, do: Keyword.put_new(opts, :allow_float, true), else: opts
    {out, state} = Block.run(block, input, state, opts)

    case {block.kind, block.rate} do
      {:stream, {_, o}} when o > 1 -> {flatten_items(out), state}
      _ -> {out, state}
    end
  end

  # Stream kernels with several outputs per step give `{n, o}`; the block's
  # output is the `n * o` items in order.
  defp flatten_items({re, im}), do: {flatten_items(re), flatten_items(im)}
  defp flatten_items(%NxFixed.Fixed{t: t} = x), do: %{x | t: Nx.reshape(t, {:auto})}
  defp flatten_items(t), do: Nx.reshape(t, {:auto})

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(:chunks_to_symbols_bf, opts) do
    table = Keyword.fetch!(opts, :table)

    stateless(:chunks_to_symbols_bf, &K.chunks_to_symbols_real/3, {:symbol, :real},
      gnuradio: "digital.chunks_to_symbols_bf",
      opts: [format: format(opts), table: table]
    )
  end

  def block(:chunks_to_symbols_bc, opts) do
    table = opts |> Keyword.fetch!(:table) |> Enum.map(&complex/1)

    stateless(:chunks_to_symbols_bc, &K.chunks_to_symbols_complex/3, {:symbol, :complex},
      gnuradio: "digital.chunks_to_symbols_bc",
      opts: [format: format(opts), table: table]
    )
  end

  def block(:constellation_decoder_cb, opts) do
    c = constellation(opts)

    stateless(:constellation_decoder_cb, &K.decide/3, {:complex, :symbol},
      gnuradio: "digital.constellation_decoder_cb",
      opts: [format: format(opts), points: c.points]
    )
  end

  def block(:constellation_soft_decoder_cf, opts) do
    c = constellation(opts)

    %Block{
      name: :constellation_soft_decoder_cf,
      kind: :frame,
      fun: &K.soft_decide/3,
      opts: [
        points: c.points,
        labels: Constellation.labels(c),
        npwr: Keyword.get(opts, :npwr, 1.0)
      ],
      rate: {1, c.bits_per_symbol},
      item: {:complex, :real},
      hardware: :stateless,
      gnuradio: "digital.constellation_soft_decoder_cf"
    }
  end

  def block(:map_bb, opts) do
    stateless(:map_bb, &K.map_bytes/3, {:byte, :byte},
      gnuradio: "digital.map_bb",
      opts: [map: Keyword.fetch!(opts, :map)]
    )
  end

  def block(:binary_slicer_fb, opts) do
    stateless(:binary_slicer_fb, &K.slice/3, {:real, :bit},
      gnuradio: "digital.binary_slicer_fb",
      opts: [format: format(opts)]
    )
  end

  def block(:diff_encoder_bb, opts) do
    %Block{
      name: :diff_encoder_bb,
      kind: :stream,
      fun: &K.diff_encode/3,
      init: Nx.tensor(0, type: :u8),
      opts: [modulus: Keyword.fetch!(opts, :modulus)],
      item: {:symbol, :symbol},
      hardware: :feedback,
      gnuradio: "digital.diff_encoder_bb"
    }
  end

  def block(:diff_decoder_bb, opts) do
    %Block{
      name: :diff_decoder_bb,
      kind: :stream,
      fun: &K.diff_decode/3,
      init: Nx.tensor(0, type: :u8),
      opts: [modulus: Keyword.fetch!(opts, :modulus)],
      item: {:symbol, :symbol},
      hardware: :feed_forward,
      warmup: 1,
      gnuradio: "digital.diff_decoder_bb"
    }
  end

  def block(:diff_phasor_cc, opts) do
    f = format(opts)

    %Block{
      name: :diff_phasor_cc,
      kind: :stream,
      fun: &K.diff_phasor/3,
      init: {NxFixed.Num.zeros({}, f), NxFixed.Num.zeros({}, f)},
      opts: [format: f],
      item: {:complex, :complex},
      hardware: :feed_forward,
      warmup: 1,
      gnuradio: "digital.diff_phasor_cc"
    }
  end

  def block(name, opts) when name in [:scrambler_bb, :descrambler_bb] do
    {taps, len, seed} = lfsr(opts)

    %Block{
      name: name,
      kind: :stream,
      fun: if(name == :scrambler_bb, do: &K.scramble/3, else: &K.descramble/3),
      init: Nx.tensor(seed, type: :u32),
      opts: [taps: taps, len: len],
      item: {:bit, :bit},
      hardware: :feedback,
      warmup: if(name == :descrambler_bb, do: len + 1, else: 0),
      gnuradio: "digital.#{name}"
    }
  end

  def block(:additive_scrambler_bb, opts) do
    {taps, len, seed} = lfsr(opts)
    bpb = Keyword.get(opts, :bits_per_byte, 8)
    true = bpb in 1..8

    %Block{
      name: :additive_scrambler_bb,
      kind: :stream,
      fun: &K.additive_scramble/3,
      init: {Nx.tensor(seed, type: :u32), Nx.tensor(0, type: :u32)},
      opts: [
        taps: taps,
        len: len,
        seed: seed,
        count: Keyword.get(opts, :count, 0),
        bits_per_byte: bpb
      ],
      item: {:byte, :byte},
      hardware: :feedback,
      gnuradio: "digital.additive_scrambler_bb"
    }
  end

  def block(:multiply_const_cc, opts) do
    stateless(:multiply_const_cc, &K.scale_complex/3, {:complex, :complex},
      gnuradio: "blocks.multiply_const_cc",
      opts: [format: format(opts), gain: Keyword.fetch!(opts, :gain)]
    )
  end

  def block(name, opts) when name in [:generic_mod, :psk_mod, :qam_mod] do
    opts =
      case name do
        :generic_mod -> opts
        :psk_mod -> psk_opts(opts)
        :qam_mod -> qam_opts(opts)
      end

    Modulation.generic_mod(name, opts)
  end

  def block(name, opts) when name in [:gfsk_mod, :gmsk_mod], do: Modulation.fsk_mod(name, opts)

  # -- framing and CRC (host blocks: packets are lists of binaries) ----------------------------

  def block(:crc32_bb, opts) do
    host(
      :crc32_bb,
      fn packets, st ->
        {for(p <- packets, {:ok, out} <- [Framing.crc32_packet(p, opts)], do: out), st}
      end,
      {:packet, :packet},
      nil,
      "digital.crc32_bb"
    )
  end

  def block(:crc_append, opts) do
    host(
      :crc_append,
      fn pdus, st -> {Enum.map(pdus, &Framing.crc_append(&1, opts)), st} end,
      {:packet, :packet},
      nil,
      "digital.crc_append"
    )
  end

  def block(:crc_check, opts),
    do: host(:crc_check, crc_check_fun(opts), {:packet, :packet}, nil, "digital.crc_check")

  def block(:crc32_async_bb, opts), do: async_crc(:crc32_async_bb, Framing.crc32_pdu_opts(), opts)
  def block(:crc16_async_bb, opts), do: async_crc(:crc16_async_bb, Framing.crc16_pdu_opts(), opts)

  def block(:correlate_access_code_bb, opts) do
    c = Framing.correlator(Keyword.fetch!(opts, :access_code), Keyword.fetch!(opts, :threshold))

    host(
      :correlate_access_code_bb,
      fn bits, st -> Framing.correlate_access_code(st, bits) end,
      {:bit, :bit},
      c,
      "digital.correlate_access_code_bb"
    )
  end

  def block(:correlate_access_code_tag_bb, opts) do
    c = Framing.correlator(Keyword.fetch!(opts, :access_code), Keyword.fetch!(opts, :threshold))

    host(
      :correlate_access_code_tag_bb,
      fn bits, st -> Framing.correlate_access_code_tag(st, bits) end,
      {:bit, :bit},
      c,
      "digital.correlate_access_code_tag_bb"
    )
  end

  def block(:protocol_formatter_bb, opts) do
    f =
      Framing.header_format(
        Keyword.get(opts, :kind, :default),
        Keyword.fetch!(opts, :access_code),
        Keyword.get(opts, :threshold, 0),
        Keyword.get(opts, :bps, 1)
      )

    fun = fn payloads, counter ->
      Enum.map_reduce(payloads, counter, fn p, n -> Framing.header(f, byte_size(p), n) end)
    end

    host(:protocol_formatter_bb, fun, {:packet, :packet}, 0, "digital.protocol_formatter_bb")
  end

  def block(:hdlc_framer_pb, _opts) do
    host(
      :hdlc_framer_pb,
      fn pdus, st -> {Enum.map(pdus, &Framing.hdlc_frame/1), st} end,
      {:packet, :bit},
      nil,
      "digital.hdlc_framer_pb"
    )
  end

  def block(:hdlc_deframer_bp, opts) do
    d =
      Framing.hdlc_deframer(
        Keyword.get(opts, :length_min, 32),
        Keyword.get(opts, :length_max, 500)
      )

    host(
      :hdlc_deframer_bp,
      fn bits, st -> Framing.hdlc_deframe(st, bits) end,
      {:bit, :packet},
      d,
      "digital.hdlc_deframer_bp"
    )
  end

  def block(:interp_fir_filter_fff, opts), do: interp_fir(:interp_fir_filter_fff, opts)
  def block(:interp_fir_filter_ccf, opts), do: interp_fir(:interp_fir_filter_ccf, opts)

  def block(:frequency_modulator_fc, opts) do
    f = format(opts)

    %Block{
      name: :frequency_modulator_fc,
      kind: :stream,
      fun: &K.frequency_modulate/3,
      init: if(f.kind == :float, do: Nx.tensor(0.0, type: :f32), else: Nx.tensor(0, type: :u16)),
      opts: [format: f, sensitivity: Keyword.fetch!(opts, :sensitivity)],
      item: {:real, :complex},
      hardware: :feedback,
      gnuradio: "analog.frequency_modulator_fc"
    }
  end

  # psk_mod(constellation_points, mod_code, differential, samples_per_symbol, excess_bw)
  defp psk_opts(opts) do
    c =
      Constellation.psk(
        Keyword.get(opts, :constellation_points, 4),
        Keyword.get(opts, :mod_code, :gray),
        Keyword.get(opts, :differential, true)
      )

    Keyword.merge([differential: true], opts) |> Keyword.put(:constellation, c)
  end

  # qam_mod(constellation_points, differential, mod_code, samples_per_symbol, excess_bw)
  defp qam_opts(opts) do
    differential = Keyword.get(opts, :differential, true)

    c =
      Constellation.qam(
        Keyword.get(opts, :constellation_points, 16),
        differential,
        Keyword.get(opts, :mod_code, :none)
      )

    Keyword.merge([differential: true], opts) |> Keyword.put(:constellation, c)
  end

  defp interp_fir(name, opts) do
    f = format(opts)
    taps = Keyword.fetch!(opts, :taps)
    sps = Keyword.fetch!(opts, :interpolation)
    n = K.line_length(length(taps), sps)
    zeros = fn -> NxFixed.Num.zeros({n}, f) end

    {fun, init, item} =
      if name == :interp_fir_filter_ccf,
        do: {&K.interp_fir_complex/3, {zeros.(), zeros.()}, {:complex, :complex}},
        else: {&K.interp_fir_real/3, zeros.(), {:real, :real}}

    %Block{
      name: name,
      kind: :stream,
      fun: fun,
      init: init,
      opts: [format: f, taps: taps, sps: sps],
      rate: {1, sps},
      item: item,
      latency: 0,
      warmup: n,
      hardware: :feed_forward,
      gnuradio: "filter.#{name}"
    }
  end

  defp host(name, fun, item, init, gnuradio) do
    %Block{
      name: name,
      kind: :host,
      fun: fun,
      init: init,
      item: item,
      hardware: :control,
      gnuradio: gnuradio
    }
  end

  # PDUs in, `%{ok: pdus, fail: pdus}` out (the check ports)
  defp crc_check_fun(opts) do
    fn pdus, st ->
      results = Enum.map(pdus, &Framing.crc_check(&1, opts))
      {%{ok: for({:ok, p} <- results, do: p), fail: for({:fail, p} <- results, do: p)}, st}
    end
  end

  defp async_crc(name, crc_opts, opts) do
    if Keyword.get(opts, :check, false),
      do:
        host(
          name,
          crc_check_fun(crc_opts ++ [discard_crc: true]),
          {:packet, :packet},
          nil,
          "digital.#{name}"
        ),
      else:
        host(
          name,
          fn pdus, st -> {Enum.map(pdus, &Framing.crc_append(&1, crc_opts)), st} end,
          {:packet, :packet},
          nil,
          "digital.#{name}"
        )
  end

  defp stateless(name, fun, item, extra) do
    struct!(
      %Block{
        name: name,
        kind: :stream,
        fun: fun,
        init: Nx.tensor(0, type: :u8),
        item: item,
        hardware: :stateless
      },
      extra
    )
  end

  # GNU Radio LFSR parameters: mask, seed, register length. The register
  # holds len + 1 bits; the taps are the positions of the mask's set bits.
  defp lfsr(opts) do
    mask = Keyword.fetch!(opts, :mask)
    len = Keyword.fetch!(opts, :len)
    taps = for k <- 0..len, Bitwise.band(Bitwise.bsr(mask, k), 1) == 1, do: k
    true = len in 1..31 and Bitwise.bsr(mask, len + 1) == 0
    {taps, len, Keyword.fetch!(opts, :seed)}
  end

  defp format(opts), do: Keyword.get_lazy(opts, :format, &NxFixed.Num.float/0)

  defp constellation(opts) do
    case Keyword.fetch!(opts, :constellation) do
      %Constellation{} = c -> c
      name when is_atom(name) -> apply(Constellation, name, [])
    end
  end

  defp complex({re, im}), do: {re, im}
  defp complex(%Complex{re: re, im: im}), do: {re, im}
  defp complex(re) when is_number(re), do: {re, 0.0}
end
