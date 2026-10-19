defmodule ExSdr.Analog.BlocksTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Analog
  alias NxFixed.Block
  alias NxFixed.Num

  @float Num.float()
  @q15 Num.q(16, 15, rounding: :nearest, lut_bits: 12)

  defp real_in(values, format),
    do: Num.from_int(Nx.tensor(Enum.map(values, &round(&1 * 32768)), type: :s16), 15, format)

  defp complex_in(values, format) do
    {re, im} = Enum.unzip(values)
    {real_in(re, format), real_in(im, format)}
  end

  defp floats(out) do
    case Num.to_float(out) do
      {re, im} -> Enum.zip(re, im)
      v -> v
    end
  end

  describe "metadata" do
    @min_opts %{
      frequency_modulator_fc: [sensitivity: 0.5],
      phase_modulator_fc: [sensitivity: 0.5],
      fm_deemph: [fs: 48_000.0],
      fm_preemph: [fs: 48_000.0],
      feedforward_agc_cc: [n: 8],
      simple_squelch_cc: [threshold_db: -20.0],
      pwr_squelch_cc: [threshold_db: -20.0],
      pwr_squelch_ff: [threshold_db: -20.0],
      rail_ff: [lo: -0.5, hi: 0.5],
      pll_carriertracking_cc: [loop_bw: 0.05],
      pll_freqdet_cf: [loop_bw: 0.05],
      pll_refout_cc: [loop_bw: 0.05],
      nbfm_tx: [quad_rate: 48_000.0],
      nbfm_rx_core: [quad_rate: 48_000.0]
    }

    test "every block declares kind, hardware class, item types and its GNU Radio name" do
      for name <- Analog.names(), format <- [@float, @q15] do
        b = Analog.block(name, Keyword.put(Map.get(@min_opts, name, []), :format, format))
        assert %Block{name: ^name, kind: :stream, rate: {1, 1}} = b
        assert b.hardware in [:stateless, :feed_forward, :feedback]
        assert {i, o} = b.item
        assert i in [:real, :complex] and o in [:real, :complex]
        assert is_binary(b.gnuradio)
        assert b.opts[:format] == format
      end
    end

    test "loops are feedback, filters and demodulators are not" do
      hw = fn name -> Analog.block(name, Map.get(@min_opts, name, [])).hardware end

      for n <-
            ~w(agc_cc agc_ff agc2_cc agc2_ff agc3_cc pll_carriertracking_cc pll_freqdet_cf pll_refout_cc
                  fm_deemph fm_preemph frequency_modulator_fc simple_squelch_cc)a,
          do: assert(hw.(n) == :feedback, "#{n}")

      assert hw.(:quadrature_demod_cf) == :feed_forward
      assert hw.(:feedforward_agc_cc) == :feed_forward
      assert hw.(:phase_modulator_fc) == :stateless
      assert hw.(:rail_ff) == :stateless
    end

    test "required options are enforced and unknown blocks rejected" do
      assert_raise ArgumentError, ~r/sensitivity/, fn ->
        Analog.block(:frequency_modulator_fc, [])
      end

      assert_raise ArgumentError, ~r/unknown/, fn -> Analog.block(:nope, []) end
    end

    test "feedforward_agc_cc declares its latency" do
      b = Analog.block(:feedforward_agc_cc, n: 16)
      assert b.latency == 15
    end
  end

  describe "emphasis taps" do
    test "have unit gain at DC like GNU Radio's design" do
      for [b0: b0, b1: b1, p1: p1] <- [
            Analog.deemph_taps(48_000.0, 75.0e-6),
            Analog.preemph_taps(48_000.0, 75.0e-6)
          ] do
        assert_in_delta (b0 + b1) / (1 - p1), 1.0, 1.0e-12
      end
    end

    test "de-emphasis is a pole at the bilinear image of 1/tau" do
      [b0: b0, b1: b1, p1: p1] = Analog.deemph_taps(48_000.0, 75.0e-6)
      assert_in_delta b0, b1, 1.0e-15
      # 3 dB at 1 / (2 pi tau) = 2122 Hz: |H|^2 = 1/2 (after prewarping)
      w = 2 * :math.pi() * 2122.1 / 48_000
      re = b0 + b1 * :math.cos(w)
      im = -b1 * :math.sin(w)
      dre = 1 - p1 * :math.cos(w)
      dim = p1 * :math.sin(w)
      gain = :math.sqrt((re * re + im * im) / (dre * dre + dim * dim))
      assert_in_delta gain, 1 / :math.sqrt(2), 5.0e-3
    end
  end

  describe "behaviour" do
    test "quadrature_demod_cf outputs 0 first, then gain * the phase step" do
      step = 0.2
      x = for n <- 0..49, do: {0.5 * :math.cos(step * n), 0.5 * :math.sin(step * n)}

      for format <- [@float, @q15] do
        b = Analog.block(:quadrature_demod_cf, format: format, gain: 2.0)
        {y, _} = Block.run(b, complex_in(x, format))
        [first | rest] = floats(y)
        # the zero history gives a zero product: atan2(0, 0) = 0
        assert first == 0.0
        # on Q15 the phase step is known to about 1/(0.25 * 32768) rad
        for v <- rest, do: assert_in_delta(v, 2 * step, 1.0e-3)
      end
    end

    test "frequency modulation then quadrature demodulation returns the audio" do
      audio = for n <- 0..399, do: 0.6 * :math.sin(0.07 * n) * :math.cos(0.011 * n)
      sens = 1.2

      for {format, tol} <- [{@float, 2.0e-4}, {@q15, 4.0e-3}] do
        mod = Analog.block(:frequency_modulator_fc, format: format, sensitivity: sens)
        demod = Analog.block(:quadrature_demod_cf, format: format, gain: 1 / sens)
        {iq, _} = Block.run(mod, real_in(audio, format))
        {y, _} = Block.run(demod, iq)

        for {a, v} <- Enum.zip(Enum.drop(audio, 1), Enum.drop(floats(y), 1)),
            do: assert(abs(a - v) < tol, "#{a} vs #{v}")
      end
    end

    test "phase_modulator_fc wraps: sensitivity times input beyond pi is the same point" do
      for format <- [@float, @q15] do
        b = Analog.block(:phase_modulator_fc, format: format, sensitivity: 6.0)
        {out, _} = Block.run(b, real_in([0.5, -0.5], format))
        [{c1, s1}, {c2, s2}] = floats(out)
        # 3 rad and -3 rad
        assert_in_delta c1, :math.cos(3.0), 2.0e-3
        assert_in_delta s1, :math.sin(3.0), 2.0e-3
        assert_in_delta c2, :math.cos(-3.0), 2.0e-3
        assert_in_delta s2, :math.sin(-3.0), 2.0e-3
      end
    end

    test "am_demod removes the carrier: |x| - 1" do
      x = [{0.5, 0.0}, {0.0, -0.25}, {0.6, 0.8}, {0.0, 0.0}]

      for format <- [@float, @q15] do
        b = Analog.block(:am_demod, format: format)
        {y, _} = Block.run(b, complex_in(x, format))

        for {v, want} <- Enum.zip(floats(y), [-0.5, -0.75, 0.0, -1.0]),
            do: assert_in_delta(v, want, 1.0e-4)
      end
    end

    test "rail_ff clips" do
      for format <- [@float, @q15] do
        b = Analog.block(:rail_ff, format: format, lo: -0.25, hi: 0.5)
        {y, _} = Block.run(b, real_in([-0.9, -0.25, 0.0, 0.3, 0.5, 0.9], format))

        for {v, w} <- Enum.zip(floats(y), [-0.25, -0.25, 0.0, 0.3, 0.5, 0.5]),
            do: assert_in_delta(v, w, 1 / 32768)
      end
    end

    test "sample_and_hold_ff latches data while control is not 0" do
      data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

      for format <- [@float, @q15] do
        b = Analog.block(:sample_and_hold_ff, format: format)

        input =
          {real_in(data, format),
           Num.from_int(Nx.tensor([0, 16384, 0, 0, 16384, 0], type: :s16), 15, format)}

        {y, state} = Block.run(b, input)
        # holds 0 until the first control pulse
        for {v, w} <- Enum.zip(floats(y), [0.0, 0.2, 0.2, 0.2, 0.5, 0.5]),
            do: assert_in_delta(v, w, 1.0e-4)

        assert_in_delta Num.to_float(state), 0.5, 1.0e-4
      end
    end

    test "AGC gain stops at max_gain, also with zero input" do
      for name <- [:agc_cc, :agc_ff, :agc2_cc, :agc2_ff], format <- [@float, @q15] do
        opts =
          if name in [:agc2_cc, :agc2_ff],
            do: [attack_rate: 0.5, decay_rate: 0.5, reference: 1.0, max_gain: 100.0],
            else: [rate: 0.5, reference: 1.0, max_gain: 100.0]

        b = Analog.block(name, [format: format] ++ opts)
        zeros = List.duplicate(0.0, 400)

        input =
          if b.item == {:complex, :complex},
            do: complex_in(Enum.map(zeros, &{&1, &1}), format),
            else: real_in(zeros, format)

        {_y, gain} = Block.run(b, input)
        assert_in_delta Num.to_float(gain), 100.0, 1.0e-6, "#{name}"
      end
    end

    test "squelch passes a loud signal after the average rises and mutes a quiet one" do
      loud = for n <- 0..399, do: {0.5 * :math.cos(0.3 * n), 0.5 * :math.sin(0.3 * n)}
      quiet = for n <- 0..399, do: {0.01 * :math.cos(0.3 * n), 0.01 * :math.sin(0.3 * n)}

      for format <- [@float, @q15] do
        b = Analog.block(:pwr_squelch_cc, format: format, threshold_db: -20.0, alpha: 0.01)
        {y, _} = Block.run(b, complex_in(loud ++ quiet, format))
        out = floats(y)
        assert Enum.at(out, 0) == {0.0, 0.0}

        # the average 0.25 * (1 - 0.99^n) crosses 0.01 after 5 samples, and decays below it again in the quiet part
        assert Enum.at(out, 100) != {0.0, 0.0}
        assert Enum.at(out, 399) != {0.0, 0.0}
        assert Enum.at(out, 799) == {0.0, 0.0}
        # the output is the input or zero
        for {o, i} <- Enum.zip(out, loud ++ quiet), o != {0.0, 0.0} do
          {oi, oq} = o
          {ii, iq} = i
          assert_in_delta oi, ii, 1 / 32768
          assert_in_delta oq, iq, 1 / 32768
        end
      end
    end

    test "feedforward_agc_cc: zeros for n - 1 samples, then the oldest sample scaled by reference over envelope" do
      n = 8
      x = for k <- 0..39, do: {0.25 * :math.cos(0.5 * k), 0.25 * :math.sin(0.5 * k)}

      for format <- [@float, @q15] do
        b = Analog.block(:feedforward_agc_cc, format: format, n: n, reference: 0.5)
        {y, _} = Block.run(b, complex_in(x, format))
        out = floats(y)
        assert Enum.take(out, n - 1) == List.duplicate({0.0, 0.0}, n - 1)

        env = fn window ->
          window
          |> Enum.map(fn {re, im} -> max(abs(re), abs(im)) + 0.4 * min(abs(re), abs(im)) end)
          |> Enum.max()
        end

        for k <- (n - 1)..39 do
          window = Enum.slice(x, (k - n + 1)..k)
          {re, im} = Enum.at(x, k - n + 1)
          g = 0.5 / env.(window)
          {oi, oq} = Enum.at(out, k)
          assert_in_delta oi, re * g, 2.0e-4
          assert_in_delta oq, im * g, 2.0e-4
        end
      end
    end

    test "the PLL locks to a carrier: freq estimate, tracked output and reference" do
      f0 = 0.03
      x = for n <- 0..999, do: {0.5 * :math.cos(f0 * n + 1.0), 0.5 * :math.sin(f0 * n + 1.0)}

      for format <- [@float, @q15] do
        opts = [format: format, loop_bw: 0.05, max_freq: 0.5, min_freq: -0.5]
        input = complex_in(x, format)

        {freq, _} = Block.run(Analog.block(:pll_freqdet_cf, opts), input)
        assert_in_delta List.last(floats(freq)), f0, 2.0e-3

        {tracked, _} = Block.run(Analog.block(:pll_carriertracking_cc, opts), input)
        # carrier removed: constant phase, here 0 after lock, amplitude 0.5
        {re, im} = tracked |> floats() |> List.last()
        assert_in_delta re, 0.5, 5.0e-3
        assert_in_delta im, 0.0, 5.0e-3

        {ref, _} = Block.run(Analog.block(:pll_refout_cc, opts), input)
        {c, s} = ref |> floats() |> List.last()
        a = f0 * 999 + 1.0
        assert_in_delta c, :math.cos(a), 5.0e-3
        assert_in_delta s, :math.sin(a), 5.0e-3
      end
    end

    test "the PLL limits its frequency to max_freq and min_freq" do
      # a carrier at 0.2 rad/sample with limits +-0.05
      x = for n <- 0..399, do: {0.5 * :math.cos(0.2 * n), 0.5 * :math.sin(0.2 * n)}

      for format <- [@float, @q15] do
        b =
          Analog.block(:pll_freqdet_cf,
            format: format,
            loop_bw: 0.05,
            max_freq: 0.05,
            min_freq: -0.05
          )

        {freq, _} = Block.run(b, complex_in(x, format))
        assert Enum.max(floats(freq)) <= 0.05 + 1.0e-4
        assert_in_delta List.last(floats(freq)), 0.05, 1.0e-4
      end
    end

    test "CORDIC iterations trade accuracy for hardware" do
      x =
        for n <- 0..299,
            do: {0.6 * :math.cos(0.02 * n * n / 40), 0.6 * :math.sin(0.02 * n * n / 40)}

      run = fn opts ->
        b = Analog.block(:quadrature_demod_cf, [format: @q15, gain: 1.0] ++ opts)
        {y, _} = Block.run(b, complex_in(x, @q15))
        floats(y)
      end

      {ref, _} =
        Block.run(
          Analog.block(:quadrature_demod_cf, format: @float, gain: 1.0),
          complex_in(x, @float)
        )

      ref = floats(ref)
      worst = fn y -> Enum.zip(ref, y) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max() end
      assert worst.(run.(iterations: 12)) > 3 * worst.(run.(iterations: 18))
      assert worst.(run.(iterations: 18)) < 1.0e-4
    end
  end

  property "fm_deemph in Q15: any split of the stream gives the same output" do
    b = Analog.block(:fm_deemph, format: @q15, fs: 48_000.0)

    check all(
            samples <- list_of(integer(-32_768..32_767), min_length: 40, max_length: 120),
            cut_a <- integer(1..15),
            cut_b <- integer(1..15),
            max_runs: 12
          ) do
      x = Num.from_int(Nx.tensor(samples, type: :s16), 15, @q15)
      {whole, _} = Block.run(b, x)
      n = length(samples)
      {p1, s1} = Block.run(b, slice(x, 0, cut_a))
      {p2, s2} = Block.run(b, slice(x, cut_a, cut_b), s1)
      {p3, _} = Block.run(b, slice(x, cut_a + cut_b, n - cut_a - cut_b), s2)
      pieces = Nx.concatenate([p1.t, p2.t, p3.t])
      assert Nx.to_list(pieces) == Nx.to_list(whole.t)
    end
  end

  defp slice(%NxFixed.Fixed{t: t} = x, from, len),
    do: %{x | t: Nx.slice_along_axis(t, from, len, axis: 0)}
end
