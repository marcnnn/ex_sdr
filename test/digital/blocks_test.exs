defmodule ExSdr.Digital.BlocksTest do
  @moduledoc """
  The block descriptions, and the state handling of the stream blocks: a
  stream cut in two pieces with the state carried gives the same output as
  one call, in the float and in fixed-point instances (which pass the width
  check when they run).
  """
  use ExUnit.Case, async: true

  alias ExSdr.Digital
  alias NxFixed.Block
  alias NxFixed.Num

  @kinds [:stream, :frame, :host]
  @hardware [:stateless, :feed_forward, :feedback, :memory, :transform, :control]

  defp example(name) do
    opts =
      case name do
        n when n in [:chunks_to_symbols_bf] ->
          [table: [-1.0, 1.0]]

        :chunks_to_symbols_bc ->
          [table: [{1.0, 0.0}, {0.0, 1.0}]]

        n when n in [:constellation_decoder_cb, :constellation_soft_decoder_cf] ->
          [constellation: :qpsk]

        :map_bb ->
          [map: [1, 0]]

        n when n in [:diff_encoder_bb, :diff_decoder_bb] ->
          [modulus: 4]

        n when n in [:scrambler_bb, :descrambler_bb, :additive_scrambler_bb] ->
          [mask: 0x8A, seed: 0x7F, len: 7]

        :multiply_const_cc ->
          [gain: 0.5]

        n when n in [:interp_fir_filter_fff, :interp_fir_filter_ccf] ->
          [taps: [0.5, 0.5, 0.25], interpolation: 2]

        :frequency_modulator_fc ->
          [sensitivity: 0.5]

        :generic_mod ->
          [constellation: :qpsk]

        :crc_append ->
          crc_opts()

        :crc_check ->
          crc_opts()

        n when n in [:correlate_access_code_bb, :correlate_access_code_tag_bb] ->
          [access_code: "1011", threshold: 1]

        :protocol_formatter_bb ->
          [access_code: "10101010"]

        _ ->
          []
      end

    Digital.block(name, opts)
  end

  defp crc_opts,
    do: [
      num_bits: 16,
      poly: 0x1021,
      initial_value: 0xFFFF,
      final_xor: 0,
      input_reflected: false,
      result_reflected: false
    ]

  test "every block describes itself as an NxFixed.Block" do
    for name <- Digital.blocks() do
      %Block{} = b = example(name)
      assert b.name == name
      assert b.kind in @kinds
      assert b.hardware in @hardware
      assert match?({i, o} when i > 0 and o > 0, b.rate), "#{name}"
      assert is_binary(b.gnuradio), "#{name}"
      assert match?({_, _}, b.item)
      if b.kind == :host, do: assert(b.hardware in [:control, :feedback])
    end
  end

  test "hardware classes follow the state" do
    classes = fn names -> for n <- names, do: example(n).hardware end

    assert classes.([:chunks_to_symbols_bc, :constellation_decoder_cb, :map_bb]) ==
             List.duplicate(:stateless, 3)

    assert classes.([:scrambler_bb, :additive_scrambler_bb, :diff_encoder_bb]) ==
             List.duplicate(:feedback, 3)

    assert classes.([:diff_decoder_bb, :interp_fir_filter_ccf]) ==
             List.duplicate(:feed_forward, 2)

    assert example(:hdlc_deframer_bp).hardware == :control
  end

  test "rates" do
    assert example(:interp_fir_filter_ccf).rate == {1, 2}
    assert example(:constellation_soft_decoder_cf).rate == {1, 2}
    assert Digital.block(:psk_mod, constellation_points: 8, samples_per_symbol: 4).rate == {3, 32}
    assert Digital.block(:gmsk_mod, samples_per_symbol: 2).rate == {1, 16}
  end

  # run in pieces: the output of both pieces equals the whole
  defp pieces(block, input, state \\ :init, cut) do
    {whole, _} = Digital.run(block, input, state)
    {a, b} = split(input, cut)
    {y1, st} = Digital.run(block, a, state)
    {y2, _} = Digital.run(block, b, st)
    assert flat(join(y1, y2)) == flat(whole)
  end

  defp split({re, im}, cut) do
    {ra, rb} = split(re, cut)
    {ia, ib} = split(im, cut)
    {{ra, ia}, {rb, ib}}
  end

  defp split(%NxFixed.Fixed{t: t} = x, cut),
    do: {%{x | t: t[0..(cut - 1)]}, %{x | t: t[cut..-1//1]}}

  defp split(t, cut), do: {t[0..(cut - 1)], t[cut..-1//1]}

  defp join({a, b}, {c, d}), do: {join(a, c), join(b, d)}
  defp join(%NxFixed.Fixed{t: a} = x, %NxFixed.Fixed{t: b}), do: %{x | t: Nx.concatenate([a, b])}
  defp join(a, b), do: Nx.concatenate([a, b])

  defp flat({a, b}), do: {flat(a), flat(b)}
  defp flat(%NxFixed.Fixed{t: t}), do: Nx.to_flat_list(t)
  defp flat(t), do: Nx.to_flat_list(t)

  defp bytes(n, mod), do: Nx.tensor(for(_ <- 1..n, do: :rand.uniform(mod) - 1), type: :u8)

  defp samples(n, f) do
    ints = fn -> Nx.tensor(for(_ <- 1..n, do: :rand.uniform(60_000) - 30_000), type: :s16) end
    {Num.from_int(ints.(), 15, f), Num.from_int(ints.(), 15, f)}
  end

  setup do
    :rand.seed(:exsss, {11, 12, 13})
    :ok
  end

  test "scramblers, differential coding and map keep their state" do
    for {name, opts, mod} <- [
          {:scrambler_bb, [mask: 0x8A, seed: 0x7F, len: 7], 2},
          {:descrambler_bb, [mask: 0x1021, seed: 0xACE1, len: 15], 2},
          {:additive_scrambler_bb, [mask: 0x8A, seed: 0x7F, len: 7, count: 7, bits_per_byte: 3],
           256},
          {:additive_scrambler_bb, [mask: 0x21, seed: 1, len: 5], 256},
          {:diff_encoder_bb, [modulus: 8], 8},
          {:diff_decoder_bb, [modulus: 8], 8},
          {:map_bb, [map: [3, 2, 1, 0]], 4}
        ],
        cut <- [1, 13, 40] do
      pieces(Digital.block(name, opts), bytes(60, mod), cut)
    end
  end

  test "differential phasor keeps the previous sample (float and Q15)" do
    for f <- [Num.float(), Num.q(16, 15), Num.q(16, 15, rounding: :truncate)] do
      pieces(Digital.block(:diff_phasor_cc, format: f), samples(50, f), 17)
    end
  end

  test "interpolating FIRs keep their delay lines (float and fixed)" do
    taps = for i <- 1..11, do: :math.sin(i) / 4

    for f <- [Num.float(), Num.q(16, 14), Num.q(18, 16)], sps <- [2, 3, 4] do
      pieces(
        Digital.block(:interp_fir_filter_ccf, taps: taps, interpolation: sps, format: f),
        samples(40, f),
        9
      )

      {re, _} = samples(40, f)

      pieces(
        Digital.block(:interp_fir_filter_fff, taps: taps, interpolation: sps, format: f),
        re,
        5
      )
    end
  end

  test "frequency modulator keeps the phase (float and fixed)" do
    for f <- [Num.float(), Num.q(16, 15), Num.q(16, 15, lut_bits: 12)] do
      {re, _} = samples(60, f)
      pieces(Digital.block(:frequency_modulator_fc, sensitivity: 0.7, format: f), re, 25)
    end
  end

  test "symbol tables work in narrow and wide formats and clip out of range symbols" do
    table = [{0.5, -0.25}, {-0.5, 0.25}, {0.125, 0.0}]

    for f <- [Num.q(8, 6), Num.q(12, 10), Num.q(32, 28)] do
      block = Digital.block(:chunks_to_symbols_bc, table: table, format: f)
      {{re, im}, _} = Digital.run(block, Nx.tensor([0, 1, 2, 2, 0], type: :u8))
      assert Num.to_float(re) == [0.5, -0.5, 0.125, 0.125, 0.5]
      assert Num.to_float(im) == [-0.25, 0.25, 0.0, 0.0, -0.25]
    end

    # a symbol beyond the table reads the last entry (GNU Radio reads out of bounds)
    {{re, _}, _} =
      Digital.run(Digital.block(:chunks_to_symbols_bc, table: table), Nx.tensor([7], type: :u8))

    assert Nx.to_flat_list(re) == [0.125]
  end

  test "the fixed-point decoder decides like the float one away from boundaries" do
    f = Num.q(16, 15)
    block = Digital.block(:constellation_decoder_cb, constellation: :psk8, format: f)
    fblock = Digital.block(:constellation_decoder_cb, constellation: :psk8)
    {re, im} = samples(300, f)
    {a, _} = Digital.run(block, {re, im})

    {b, _} =
      Digital.run(
        fblock,
        {Nx.divide(re.t, 32_768) |> Nx.as_type(:f32), Nx.divide(im.t, 32_768) |> Nx.as_type(:f32)}
      )

    differing =
      Enum.zip(Nx.to_flat_list(a), Nx.to_flat_list(b)) |> Enum.count(fn {x, y} -> x != y end)

    assert differing <= 2
  end

  test "the divided-by-modulus operations are modular for any byte" do
    # diff_decoder output for inputs above the modulus stays below it
    {out, _} =
      Digital.run(
        Digital.block(:diff_decoder_bb, modulus: 4),
        Nx.tensor([3, 200, 17, 0], type: :u8)
      )

    assert Enum.all?(Nx.to_flat_list(out), &(&1 in 0..3))
  end

  test "differential encoding then decoding returns the input" do
    x = bytes(100, 8)
    {enc, _} = Digital.run(Digital.block(:diff_encoder_bb, modulus: 8), x)
    {dec, _} = Digital.run(Digital.block(:diff_decoder_bb, modulus: 8), enc)
    assert Nx.to_flat_list(dec) == Nx.to_flat_list(x)
  end

  test "scrambling then descrambling: additive scrambler is its own inverse" do
    x = bytes(100, 256)
    block = Digital.block(:additive_scrambler_bb, mask: 0x8A, seed: 0x7F, len: 7, count: 11)
    {s, _} = Digital.run(block, x)
    {back, _} = Digital.run(block, s)
    assert Nx.to_flat_list(back) == Nx.to_flat_list(x)
    refute Nx.to_flat_list(s) == Nx.to_flat_list(x)
  end

  test "the scrambler agrees with ExSdr.Comm.LFSR (Fibonacci register, last stage output)" do
    # GNU Radio's register of len + 1 bits: taps are positions of the mask bits
    mask = 0x8A
    len = 7
    taps = for k <- 0..len, Bitwise.band(Bitwise.bsr(mask, k), 1) == 1, do: len + 1 - k
    lfsr = ExSdr.Comm.LFSR.new(length: len + 1, taps: taps, seed: 0x7F, output: :last)
    {bits, _} = ExSdr.Comm.LFSR.bits(lfsr, 64)

    zeros = Nx.broadcast(Nx.tensor(0, type: :u8), {64})
    # additive scrambler of zeros with bits_per_byte 1 is the LFSR output
    block =
      Digital.block(:additive_scrambler_bb, mask: mask, seed: 0x7F, len: len, bits_per_byte: 1)

    {out, _} = Digital.run(block, zeros)
    assert Nx.to_flat_list(out) == bits
  end
end
