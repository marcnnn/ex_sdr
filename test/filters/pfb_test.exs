Code.require_file("test_kit.exs", __DIR__)

defmodule ExSdr.Filters.PfbTest do
  @moduledoc """
  pfb_decimator_ccf against GNU Radio (test/fixtures/gnuradio/filters/pfb.json, made by
  test/gnuradio/filters_pfb.py, which feeds the block through stream_to_streams)
  and pfb_interpolator_ccf against the interpolating FIR it equals.

  Bound: f32 sums of 40-55 terms of products of values below 0.4 in magnitude;
  4e-6 absolute (about 0.13 Q15 LSB) covers rounding in any order.
  """
  use ExUnit.Case, async: true
  alias ExSdr.Filters
  alias ExSdr.Filters.TestKit
  alias NxFixed.Num

  @cases Path.expand("../fixtures/gnuradio/filters/pfb.json", __DIR__)
         |> File.read!()
         |> JSON.decode!()
         |> Map.fetch!("cases")

  for c <- @cases do
    @c c
    test "#{c["name"]}: float instance equals GNU Radio" do
      c = @c
      input = Enum.map(c["input"], fn [re, im] -> {round(re * 32768), round(im * 32768)} end)

      b =
        Filters.block(:pfb_decimator_ccf,
          taps: c["taps"],
          decimation: c["decim"],
          channel: c["channel"]
        )

      out = TestKit.run(b, input)
      assert length(out) == length(c["output"])

      worst =
        Enum.zip(out, c["output"])
        |> Enum.map(fn {{a, b}, [r, i]} -> max(abs(a - r), abs(b - i)) end)
        |> Enum.max()

      assert worst < 4.0e-6
    end

    test "#{c["name"]}: Q15 instance within the tap rounding bound and passes the width check" do
      c = @c
      f = Num.q(16, 15)
      input = Enum.map(c["input"], fn [re, im] -> {round(re * 32768), round(im * 32768)} end)
      opts = [taps: c["taps"], decimation: c["decim"], channel: c["channel"]]
      float = TestKit.run(Filters.block(:pfb_decimator_ccf, opts), input)
      b = Filters.block(:pfb_decimator_ccf, [format: f] ++ opts)
      fixed = TestKit.run(b, input)

      # tap rounding: sum over taps of 0.5 LSB (Q15) times the peak input 0.3 * 4.5 sigma,
      # re and im parts of the complex taps both act on a component; plus 0.5 LSB rounding
      peak = 1.4
      bound = (0.5 + 2 * length(c["taps"]) * 0.5 * peak) / 32768

      worst =
        Enum.zip(float, fixed)
        |> Enum.map(fn {{a, b}, {r, i}} -> max(abs(a - r), abs(b - i)) end)
        |> Enum.max()

      assert worst <= bound
      assert {:ok, _} = TestKit.width_check(b)
    end
  end

  @interpolators Path.expand("../fixtures/gnuradio/filters/pfb.json", __DIR__)
                 |> File.read!()
                 |> JSON.decode!()
                 |> Map.fetch!("interpolators")

  for c <- @interpolators do
    @c c
    test "#{c["name"]}: float instance equals GNU Radio" do
      c = @c
      input = Enum.map(c["input"], fn [re, im] -> {round(re * 32768), round(im * 32768)} end)
      b = Filters.block(:pfb_interpolator_ccf, taps: c["taps"], interpolation: c["interp"])
      out = TestKit.run(b, input)
      # GNU Radio stops a few items before the end of the input
      assert length(out) <= length(c["output"]) + 1 and length(c["output"]) <= length(out)

      worst =
        Enum.zip(out, c["output"])
        |> Enum.map(fn {{a, b}, [r, i]} -> max(abs(a - r), abs(b - i)) end)
        |> Enum.max()

      assert worst < 4.0e-6
    end
  end

  test "pfb_interpolator_ccf is the interpolating FIR" do
    taps = Filters.Firdes.low_pass(4.0, 4.0, 0.4, 0.3)
    x = for i <- 1..32, do: {i * 400, -i * 300}
    a = Filters.block(:pfb_interpolator_ccf, taps: taps, interpolation: 4)
    b = Filters.block(:interp_fir_filter_ccf, taps: taps, interpolation: 4)
    assert TestKit.run(a, x) == TestKit.run(b, x)
    assert a.gnuradio == "filter.pfb_interpolator_ccf"
  end
end
