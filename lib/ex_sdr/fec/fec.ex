defmodule ExSdr.FEC do
  @moduledoc """
  The blocks of GNU Radio's gr-fec that are not LDPC: the generic encoder /
  decoder framework, convolutional, repetition, dummy and polar codes,
  puncturing and the BER meter. See `docs/blocks/fec.md` for the coverage
  table.

  `block(name, opts)` returns the `NxFixed.Block`. Code blocks take the
  options of their code (`ExSdr.FEC.Conv.encoder/1`, ...), the generic
  blocks (`:encoder`, `:decoder`, `:tagged_*`, `:async_*`) a code object
  (`code:`, see `ExSdr.FEC.Generic`). `run/3` runs a block like
  `NxFixed.Block.run/4`.

  Items are `u8` tensors for bits and 8 bit soft values, `f32` tensors for
  soft floats (positive: bit 1); streaming state is explicit and returned.
  The convolutional decoder in streaming mode withholds the last frame until
  the next call (it needs `rate * (K - 1)` items of look-ahead): its output is
  `nil` while no frame is complete.
  """
  @behaviour NxFixed.Block

  alias ExSdr.FEC.Generic
  alias NxFixed.Block

  # name => {kind, {module, function, leading args}, GNU Radio name}
  @code_blocks %{
    cc_encoder: {:encoder, {ExSdr.FEC.Conv, :encoder, []}, "fec.cc_encoder"},
    cc_decoder: {:decoder, {ExSdr.FEC.Conv, :decoder, []}, "fec.cc_decoder"},
    repetition_encoder:
      {:encoder, {ExSdr.FEC.Repetition, :encoder, []}, "fec.repetition_encoder"},
    repetition_decoder:
      {:decoder, {ExSdr.FEC.Repetition, :decoder, []}, "fec.repetition_decoder"},
    dummy_encoder: {:encoder, {ExSdr.FEC.Dummy, :encoder, []}, "fec.dummy_encoder"},
    dummy_decoder: {:decoder, {ExSdr.FEC.Dummy, :decoder, []}, "fec.dummy_decoder"},
    polar_encoder: {:encoder, {ExSdr.FEC.Polar, :encoder, []}, "fec.polar_encoder"},
    polar_encoder_systematic:
      {:encoder, {ExSdr.FEC.Polar, :encoder_systematic, []}, "fec.polar_encoder_systematic"},
    polar_decoder_sc:
      {:decoder, {ExSdr.FEC.PolarDecoder, :decoder, [:sc]}, "fec.polar_decoder_sc"},
    polar_decoder_sc_list:
      {:decoder, {ExSdr.FEC.PolarDecoder, :decoder, [:sc_list]}, "fec.polar_decoder_sc_list"},
    polar_decoder_sc_systematic:
      {:decoder, {ExSdr.FEC.PolarDecoder, :decoder, [:sc_systematic]},
       "fec.polar_decoder_sc_systematic"}
  }

  @generic ~w(encoder decoder tagged_encoder tagged_decoder async_encoder async_decoder
              extended_encoder extended_decoder)a
  @other ~w(puncture_bb puncture_ff depuncture_bb ber_bf)a

  @doc "The names `block/2` knows."
  def blocks, do: Map.keys(@code_blocks) ++ @generic ++ @other

  @doc """
  Runs `block` (see `NxFixed.Block.run/4`): `{output, state}`. Frame blocks
  of decoders may return `nil` as output when no frame completed.
  """
  def run(block, input, state \\ :init, opts \\ []), do: Block.run(block, input, state, opts)

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(name, opts) when is_map_key(@code_blocks, name) do
    {kind, {m, f, args}, gnuradio} = Map.fetch!(@code_blocks, name)
    code_block(name, kind, apply(m, f, args ++ [opts]), gnuradio)
  end

  def block(kind, opts) when kind in [:encoder, :decoder],
    do: code_block(kind, kind, Keyword.fetch!(opts, :code), "fec.#{kind}")

  def block(name, opts) when name in [:tagged_encoder, :tagged_decoder] do
    g = Keyword.fetch!(opts, :code)
    mtu = Keyword.get(opts, :mtu, 1500)

    fun =
      if name == :tagged_encoder,
        do: fn packets, st -> Generic.tagged_encode(g, packets, st, mtu) end,
        else: fn packets, st -> Generic.tagged_decode(g, packets, st, mtu) end

    host(name, fun, g, "fec.#{name}")
  end

  def block(name, opts) when name in [:async_encoder, :async_decoder] do
    g = Keyword.fetch!(opts, :code)
    # GNU Radio's defaults: bits are unpacked and packed LSB first
    popts = [
      packed: Keyword.get(opts, :packed, false),
      rev_unpack: Keyword.get(opts, :rev_unpack, true),
      rev_pack: Keyword.get(opts, :rev_pack, true),
      mtu: Keyword.get(opts, :mtu, 1500)
    ]

    fun =
      if name == :async_encoder,
        do: fn pdu, st -> Generic.async_encode(g, pdu, st, popts) end,
        else: fn pdu, st -> Generic.async_decode(g, pdu, st, popts) end

    host(name, fun, g, "fec.#{name}")
  end

  def block(:extended_encoder, opts) do
    g = Keyword.fetch!(opts, :code)

    fun = fn bits, st ->
      ExSdr.FEC.Extended.encode(g, bits, st, Keyword.get(opts, :puncpat, "11"))
    end

    host(:extended_encoder, fun, g, "fec.extended_encoder")
  end

  def block(:extended_decoder, opts) do
    g = Keyword.fetch!(opts, :code)

    fun = fn soft, st ->
      ExSdr.FEC.Extended.decode(g, soft, st, Keyword.get(opts, :puncpat, "11"))
    end

    %{host(:extended_decoder, fun, g, "fec.extended_decoder") | init: Generic.decoder_init(g)}
  end

  def block(:puncture_bb, opts), do: puncture(:puncture_bb, opts, {:byte, :byte})
  def block(:puncture_ff, opts), do: puncture(:puncture_ff, opts, {:real, :real})

  def block(:depuncture_bb, opts) do
    pat = ExSdr.FEC.Puncture.pattern(opts[:puncsize], opts[:puncpat], opts[:delay] || 0)
    kept = ExSdr.FEC.Puncture.kept(pat)

    %Block{
      name: :depuncture_bb,
      kind: :frame,
      fun: &ExSdr.FEC.Puncture.depuncture_frames/3,
      opts: Keyword.take(opts, [:puncsize, :puncpat, :delay, :symbol]),
      rate: {kept, opts[:puncsize]},
      item: {:byte, :byte},
      hardware: :stateless,
      gnuradio: "fec.depuncture_bb"
    }
  end

  def block(:ber_bf, opts) do
    %Block{
      name: :ber_bf,
      kind: :host,
      fun: &ExSdr.FEC.Ber.block_fun/2,
      init: ExSdr.FEC.Ber.new(opts),
      item: {:byte, :real},
      hardware: :control,
      gnuradio: "fec.ber_bf"
    }
  end

  defp puncture(name, opts, item) do
    pat = ExSdr.FEC.Puncture.pattern(opts[:puncsize], opts[:puncpat], opts[:delay] || 0)

    %Block{
      name: name,
      kind: :frame,
      fun: &ExSdr.FEC.Puncture.puncture_frames/3,
      opts: Keyword.take(opts, [:puncsize, :puncpat, :delay]),
      rate: {opts[:puncsize], ExSdr.FEC.Puncture.kept(pat)},
      item: item,
      hardware: :stateless,
      gnuradio: "fec.#{name}"
    }
  end

  defp code_block(name, :encoder, g, gnuradio) do
    %Block{
      name: name,
      kind: :frame,
      fun: &Generic.encode_frames/3,
      init: Generic.init(g),
      opts: [code: g],
      rate: {Generic.input_size(g), Generic.output_size(g)},
      latency: Generic.latency(g),
      warmup: Generic.warmup(g),
      item: {:bit, :bit},
      hardware: Generic.hardware(g),
      gnuradio: gnuradio
    }
  end

  defp code_block(name, :decoder, g, gnuradio) do
    %Block{
      name: name,
      kind: :frame,
      fun: &Generic.decode_frames/3,
      init: Generic.decoder_init(g),
      opts: [code: g],
      rate: {Generic.input_size(g), Generic.output_size(g)},
      latency: Generic.latency(g),
      warmup: Generic.warmup(g),
      item: {if(Generic.input_item(g) == :u8, do: :byte, else: :real), :bit},
      hardware: Generic.hardware(g),
      gnuradio: gnuradio
    }
  end

  defp host(name, fun, g, gnuradio) do
    %Block{
      name: name,
      kind: :host,
      fun: fun,
      init: Generic.init(g),
      item: {:bit, :bit},
      hardware: :control,
      gnuradio: gnuradio
    }
  end
end
