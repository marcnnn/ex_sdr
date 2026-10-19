Code.require_file("test_kit.exs", __DIR__)

defmodule ExSdr.Filters.BlocksTest do
  @moduledoc """
  Declarations, split invariance and behaviours that do not need GNU Radio.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Filters
  alias ExSdr.Filters.TestKit
  alias ExSdr.Test.GnuRadioFixture, as: F
  alias NxFixed.{Block, Num}

  @moduletag timeout: 600_000

  @q15 Num.q(16, 15)

  describe "split invariance" do
    for {case, data} <- TestKit.family(), format <- [:float, :q15] do
      @case case
      @data data
      @format format

      test "#{case["name"]}: input in pieces equals one call (#{format})" do
        format = if @format == :float, do: Num.float(), else: @q15
        block = TestKit.block(@case, format)
        {unit, _} = block.rate
        total = min(length(@data.input), 400) |> div(unit) |> Kernel.*(unit)
        input = Enum.take(@data.input, total)
        whole = TestKit.run(block, input)

        # pieces of 7 and 13 units, the rest
        cuts = [7 * unit, 13 * unit]
        {first, rest} = Enum.split(input, hd(cuts))
        {second, third} = Enum.split(rest, Enum.at(cuts, 1))

        {a, s1} = TestKit.run(block, first, block.init)
        {b, s2} = TestKit.run(block, second, s1)
        {c, _} = TestKit.run(block, third, s2)
        pieces = a ++ b ++ c

        assert length(pieces) == length(whole)

        # fixed point: the same bits; float: the same up to the order of the f32 sums
        tolerance = if @format == :float, do: 1.0e-6, else: 0.0

        worst =
          Enum.zip(whole, pieces)
          |> Enum.map(fn
            {{a, b}, {c, d}} -> max(abs(a - c), abs(b - d))
            {a, b} -> abs(a - b)
          end)
          |> Enum.max()

        assert worst <= tolerance
      end
    end
  end

  describe "declarations" do
    test "every name builds, with a hardware class and GNU Radio name" do
      taps = [0.25, 0.5, 0.25]
      ctaps = [{0.25, 0.1}, {0.5, 0.0}, {0.25, -0.1}]

      opts = fn name ->
        s = Atom.to_string(name)

        t =
          if Regex.match?(
               ~r/_(fir|fft|resampler|filter)_?\w*[fc]c$|_fir_filter_..c$|filter_..c$/,
               s
             ) and String.ends_with?(s, "c") and name != :hilbert_fc,
             do: ctaps,
             else: taps

        [
          taps: t,
          decimation: 2,
          interpolation: 3,
          center_freq: 0.125,
          sampling_freq: 1.0,
          length: 4,
          alpha: 0.1,
          fftaps: [0.5],
          fbtaps: [1.0, 0.2],
          ntaps: 7
        ]
      end

      for name <- Filters.names() do
        b = Filters.block(name, opts.(name))
        assert %Block{hardware: hw, gnuradio: "filter." <> _} = b
        assert hw in [:feed_forward, :feedback]

        assert name in [:iir_filter_ffd, :single_pole_iir_filter_ff, :single_pole_iir_filter_cc] ==
                 (hw == :feedback)
      end
    end

    test "rates and output counts" do
      taps = Filters.Firdes.low_pass(1.0, 1.0, 0.1, 0.1)
      x = Enum.map(1..48, &(&1 * 100))

      for {name, extra, rate} <- [
            {:fir_filter_fff, [decimation: 4], {4, 1}},
            {:interp_fir_filter_fff, [interpolation: 3], {1, 3}},
            {:rational_resampler_fff, [interpolation: 3, decimation: 4], {4, 3}}
          ] do
        b = Filters.block(name, [taps: taps] ++ extra)
        assert b.rate == rate and b.kind == :frame
        assert length(TestKit.run(b, x)) == Block.output_items(b, 48)
      end

      assert Filters.block(:fir_filter_fff, taps: taps).kind == :stream
    end

    test "latency is the group delay of linear phase taps, warmup the delay line" do
      taps = Filters.Firdes.low_pass(1.0, 1.0, 0.1, 0.1)
      b = Filters.block(:fir_filter_fff, taps: taps)
      assert b.latency == div(length(taps) - 1, 2)
      assert b.warmup == length(taps) - 1
      d = Filters.block(:fir_filter_fff, taps: taps, decimation: 4)
      assert d.latency == div(length(taps) - 1, 8)
      assert Filters.block(:dc_blocker_ff, length: 8).latency == 14
      assert Filters.block(:dc_blocker_ff, length: 8, long_form: false).latency == 7
      assert Filters.block(:hilbert_fc, ntaps: 33).latency == 16
    end
  end

  describe "behaviour" do
    defp impulse(n, pos), do: for(i <- 0..(n - 1), do: if(i == pos, do: 16384, else: 0))

    test "FIR impulse response is the taps (float) and the Q15 taps (fixed)" do
      taps = [0.5, -0.25, 0.125]
      b = Filters.block(:fir_filter_fff, taps: taps)
      out = TestKit.run(b, impulse(6, 0))
      for {o, t} <- Enum.zip(out, taps ++ [0, 0, 0]), do: assert_in_delta(o, t * 0.5, 1.0e-6)

      q = Filters.block(:fir_filter_fff, taps: taps, format: @q15)
      out = TestKit.run(q, impulse(6, 0))
      assert Enum.take(out, 3) == [0.25, -0.125, 0.0625]
    end

    test "interpolator: zero stuffing then the taps" do
      taps = [1.0, 0.5, 0.25, 0.125, 0.0625]
      b = Filters.block(:interp_fir_filter_fff, taps: taps, interpolation: 2)
      out = TestKit.run(b, impulse(4, 0))
      for {o, t} <- Enum.zip(out, taps ++ [0, 0, 0]), do: assert_in_delta(o, t * 0.5, 1.0e-6)
    end

    test "DC blocker removes DC exactly in fixed point" do
      for long <- [true, false] do
        b = Filters.block(:dc_blocker_ff, length: 8, long_form: long, format: @q15)
        out = TestKit.run(b, List.duplicate(8000, 80))
        assert Enum.take(out, -10) == List.duplicate(0.0, 10)
      end
    end

    test "single pole has unit DC gain" do
      b = Filters.block(:single_pole_iir_filter_ff, alpha: 0.25, format: @q15)
      out = TestKit.run(b, List.duplicate(8000, 200))
      # the rounded update stops within 0.5 / alpha = 2 LSB of the input
      assert_in_delta List.last(out), 8000 / 32768, 2 / 32768
    end

    test "tap type must match the name" do
      assert_raise ArgumentError, fn ->
        Filters.block(:fir_filter_fff, taps: [{1.0, 0.0}, {0.0, 0.0}])
      end

      assert_raise ArgumentError, fn -> Filters.block(:fir_filter_ccc, taps: [1.0, 0.0]) end
    end

    test "fixed formats from a case run (smoke: F.format)" do
      assert %NxFixed.Num.Format{kind: :fixed} =
               F.format(%{"bits" => 16, "frac" => 15, "rounding" => "nearest"})
    end
  end
end
