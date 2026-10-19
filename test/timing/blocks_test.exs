Code.require_file("test_kit.exs", __DIR__)

defmodule ExSdr.Timing.BlocksTest do
  @moduledoc """
  Declarations of the timing blocks, the width check of their fixed-point
  instances, the interpolator tables and the fractional delay block.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Filters.Firdes
  alias ExSdr.Timing
  alias ExSdr.Timing.{Interp, Tables}
  alias ExSdr.Timing.TestKit, as: T
  alias NxFixed.{Block, Num}

  @moduletag timeout: 600_000

  @rrc Firdes.root_raised_cosine(32.0, 128.0, 1.0, 0.35, 32 * 4 * 11)
  @lowpass Firdes.low_pass(32.0, 32.0, 0.4, 0.2, :blackman)

  defp example(name) do
    opts =
      case name do
        n when n in [:fractional_delay_cc, :fractional_delay_ff] ->
          [mu: 0.3]

        n when n in [:mmse_resampler_cc, :mmse_resampler_ff] ->
          [resamp_ratio: 0.8]

        n when n in [:pfb_arb_resampler_ccf, :pfb_arb_resampler_fff] ->
          [rate: 1.25, taps: @lowpass]

        n when n in [:clock_recovery_mm_cc, :clock_recovery_mm_ff] ->
          [omega: 4.0, gain_omega: 0.0076, gain_mu: 0.175]

        n when n in [:symbol_sync_cc, :symbol_sync_ff] ->
          [ted_type: :gardner, sps: 4.0, loop_bw: 0.045]

        n when n in [:pfb_clock_sync_ccf, :pfb_clock_sync_fff] ->
          [sps: 4.0, loop_bw: 0.06, taps: @rrc]

        :msk_timing_recovery_cc ->
          [sps: 4.0, gain: 0.05, limit: 0.1]
      end

    {name, opts}
  end

  test "declarations" do
    for name <- Timing.names() do
      {name, opts} = example(name)
      b = Timing.block(name, opts)
      assert %Block{name: ^name, kind: :stream} = b

      complex? =
        name |> Atom.to_string() |> String.split("_") |> List.last() |> String.starts_with?("c")

      if name in [:fractional_delay_cc, :fractional_delay_ff] do
        assert b.rate == {1, 1}
        assert b.hardware == :feed_forward
        assert b.warmup == 7
        assert b.latency == 4
      else
        assert {:variable, max} = b.rate
        assert max >= 1
        assert b.hardware == :feedback
        assert b.warmup == :unbounded
        assert b.gnuradio =~ ~r/^(filter|digital)\.#{name}$/
      end

      item = if complex?, do: :complex, else: :real
      assert b.item == {item, item}
    end
  end

  test "unknown names raise" do
    assert_raise ArgumentError, fn -> Timing.block(:nope, []) end
  end

  describe "width check of the fixed-point instances" do
    for name <- Timing.names(),
        format <- [{16, 15, :nearest}, {18, 17, :nearest}, {12, 11, :truncate}] do
      @name name
      @format format

      test "#{name} #{inspect(format)}" do
        {bits, frac, rounding} = @format
        f = Num.q(bits, frac, rounding: rounding)
        {name, opts} = example(@name)
        b = Timing.block(name, opts ++ [format: f])
        x = Num.zeros({}, f)
        x = if elem(b.item, 0) == :complex, do: {x, x}, else: x
        assert {:ok, _graph} = NxFixed.analyze(b.fun, [x, b.init, b.opts])
      end
    end
  end

  describe "interpolators" do
    test "GNU Radio's MMSE table: 129 rows, delays at the ends, unit DC gain" do
      rows = Tables.mmse()
      assert length(rows) == 129 and Enum.all?(rows, &(length(&1) == 8))
      assert hd(rows) == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
      assert List.last(rows) == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
      for r <- rows, do: assert(abs(Enum.sum(r) - 1.0) < 5.0e-4)
      # the differentiator has zero DC gain
      for r <- Tables.mmse_diff(), do: assert(abs(Enum.sum(r)) < 2.0e-3)
    end

    test "pfb_no_mf picks every n-th MMSE row" do
      ip = Interp.pfb_no_mf(32)
      assert ip.arms == 32 and length(ip.rows) == 33
      assert Enum.at(ip.rows, 5) == Enum.at(Tables.mmse(), 20)
      # GNU Radio rounds up to a power of two
      assert Interp.pfb_no_mf(20).arms == 32
      assert Interp.pfb_no_mf(1000).arms == 128
    end

    test "pfb_mf arms and derivative" do
      ip = Interp.pfb_mf(@rrc, 32)
      # firdes makes the RRC odd: 1409 taps
      assert ip.arms == 32 and length(ip.rows) == 33 and ip.ntaps == 45
      assert Enum.at(ip.rows, 1) |> hd() == Enum.at(@rrc, 1)
      # derivative of the arms: absolute sum nfilts
      total = ip.drows |> Enum.take(32) |> List.flatten() |> Enum.map(&abs/1) |> Enum.sum()
      assert_in_delta total, 32.0, 1.0e-3
    end

    test "the MMSE differentiator gives the slope of a ramp" do
      f = Num.float()
      window = Nx.tensor(Enum.map(0..7, &(&1 * 0.01)), type: :f32)

      for mu <- [0.0, 0.25, 0.5, 0.9] do
        d =
          Interp.differentiate(Interp.mmse(), window, Nx.tensor(mu, type: :f32), f)
          |> Nx.to_number()

        assert_in_delta d, 0.01, 1.0e-4
      end
    end
  end

  describe "fractional delay" do
    defp tone(n),
      do:
        for(
          i <- 0..(n - 1),
          do: {round(9000 * :math.cos(0.21 * i)), round(9000 * :math.sin(0.21 * i))}
        )

    test "is the MMSE table row on the last 8 items" do
      x = tone(64)
      xs = Enum.map(x, fn {r, _} -> r / 32768 end)

      for mu <- [0.0, 0.3, 0.5, 1.0] do
        {out, _} = T.run(Timing.block(:fractional_delay_cc, mu: mu), x)
        row = Enum.at(Tables.mmse(), round(mu * 128))

        for n <- 7..63 do
          want = Enum.sum(for j <- 0..7, do: Enum.at(row, j) * Enum.at(xs, n - j))
          assert_in_delta elem(Enum.at(out, n), 0), want, 2.0e-6
        end
      end
    end

    test "equals mmse_resampler at ratio 1" do
      x = tone(80)
      {d, _} = T.run(Timing.block(:fractional_delay_cc, mu: 0.4), x)
      {r, _} = T.run(Timing.block(:mmse_resampler_cc, resamp_ratio: 1.0, phase_shift: 0.4), x)
      assert Enum.drop(d, 7) == Enum.take(r, 73)
    end

    test "with a polyphase matched filter, fixed point within the tap rounding" do
      x = tone(200)
      opts = [mu: 0.25, interpolator: {:pfb_mf, Enum.map(@rrc, &(&1 / 32)), 32}]
      {fl, _} = T.run(Timing.block(:fractional_delay_cc, opts), x)

      {fx, _} =
        T.run(
          Timing.block(:fractional_delay_cc, opts ++ [format: Num.q(16, 15), tap_frac: 17]),
          x
        )

      assert length(fl) == 200
      lsb = T.errors(fl, fx) |> Enum.max() |> Kernel.*(32768)
      # 44 taps rounded to Q17 (0.125 LSB each, on a peak of 0.39) and output rounding
      assert lsb <= 44 * 0.125 * 0.39 + 0.5
    end
  end

  test "mmse_resampler: output count follows the ratio" do
    x = tone(400)

    for ratio <- [0.25, 0.5, 1.75, 3.0] do
      {out, _} = T.run(Timing.block(:mmse_resampler_cc, resamp_ratio: ratio), x)
      # output k needs the window from floor(k * ratio), 8 items
      assert length(out) == ceil((400 - 7) / ratio)
    end
  end
end
