defmodule ExSdr.Sources.ChannelTest do
  @moduledoc """
  Channel models: structure (split invariance, delays, frequency scaling),
  and for the random parts (noise, fading) statistics against theory.
  Agreement with GNU Radio for the deterministic parts is in
  gnuradio_test.exs.
  """
  use ExUnit.Case, async: true

  Code.require_file("support.exs", __DIR__)
  alias ExSdr.SourcesCase, as: C
  alias ExSdr.Sources
  alias ExSdr.Sources.{Channel, Resampler}
  alias NxFixed.Block
  alias NxFixed.Num

  @moduletag timeout: 300_000

  @float Num.float()
  @q15 Num.q(16, 15)
  @pi :math.pi()

  # complex tone of `n` samples at `freq` cycles per sample, amplitude `amp`, in `format`
  defp tone(n, freq, amp, f) do
    re = for k <- 0..(n - 1), do: round(amp * 32768 * :math.cos(2 * @pi * freq * k))
    im = for k <- 0..(n - 1), do: round(amp * 32768 * :math.sin(2 * @pi * freq * k))
    {C.samples(re, f), C.samples(im, f)}
  end

  defp cut({re, im}, start, len), do: {cut1(re, start, len), cut1(im, start, len)}

  defp cut1(%NxFixed.Fixed{t: t, frac: f}, s, l),
    do: %NxFixed.Fixed{t: Nx.slice(t, [s], [l]), frac: f}

  defp cut1(t, s, l), do: Nx.slice(t, [s], [l])

  defp run(block, input, state \\ :init), do: Block.run(block, input, state, allow_float: true)

  # a pair of vectors (or nil) as a list of {re, im}
  defp items(nil), do: []
  defp items(out), do: C.to_floats(out)

  # `pieces` of `input` through `block` against one call
  defp assert_split_invariant(block, input, sizes) do
    {whole, _} = run(block, input)
    total = Enum.sum(sizes)

    {parts, _} =
      Enum.map_reduce(Enum.zip(sizes, Enum.scan(sizes, 0, &+/2)), block.init, fn {len, upto},
                                                                                 state ->
        {out, next} = run(block, cut(input, upto - len, len), state)
        {items(out), next}
      end)

    assert items(whole) == Enum.concat(parts),
           "#{block.name}: pieces #{inspect(sizes)} of #{total}"
  end

  describe "resampler" do
    test "interpolation taps: delta at phase 0 and 128, mirror symmetric, DC gain within 3e-4 of 1" do
      t = Resampler.taps()
      assert length(t) == 129

      assert Enum.at(t, 0) |> Enum.map(&Float.round(&1, 12)) == [
               0.0,
               0.0,
               0.0,
               1.0,
               0.0,
               0.0,
               0.0,
               0.0
             ]

      assert Enum.at(t, 128) |> Enum.map(&Float.round(&1, 12)) == [
               0.0,
               0.0,
               0.0,
               0.0,
               1.0,
               0.0,
               0.0,
               0.0
             ]

      for k <- 0..128 do
        assert_in_delta Enum.sum(Enum.at(t, k)), 1.0, 3.0e-4

        for {a, b} <- Enum.zip(Enum.at(t, k), Enum.reverse(Enum.at(t, 128 - k))),
            do: assert_in_delta(a, b, 1.0e-9)
      end
    end

    test "ratio 1.0 is the input advanced by 3 samples, bit exact in float" do
      b = Resampler.block(format: @float, ratio: 1.0)
      x = tone(40, 0.05, 0.5, @float)
      {y, _} = run(b, x)
      # every output whose 8 sample window fits: 40 - 8 + 1
      assert items(y) == Enum.drop(items(x), 3) |> Enum.take(33)
    end

    for {ratio, label} <- [{1.0017000436782837, "slow"}, {0.9983000159263611, "fast"}] do
      @ratio ratio
      test "#{label} (#{ratio}): tone frequency scales by the ratio, rate and split invariance" do
        for f <- [@float, @q15] do
          b = Resampler.block(format: f, ratio: @ratio)
          x = tone(300, 0.05, 0.5, f)
          {y, _} = run(b, x)
          out = items(y)
          # every input sample used but the last 8, at ratio per output
          {rin, rout} = b.rate
          assert abs(length(out) - 292 * rout / rin) < 2
          # phase advance per output sample = 2 pi f ratio
          steps =
            out
            |> Enum.chunk_every(2, 1, :discard)
            |> Enum.map(fn [{a, b}, {c, d}] -> :math.atan2(a * d - b * c, a * c + b * d) end)

          assert_in_delta Enum.sum(steps) / length(steps), 2 * @pi * 0.05 * @ratio, 1.0e-3

          assert_split_invariant(b, x, [1, 2, 5, 17, 80, 195])
          assert_split_invariant(b, x, [150, 150])
        end
      end
    end

    test "block fields are honest" do
      b = Resampler.block(ratio: 1.0017000436782837)
      assert b.kind == :host and b.item == {:complex, :complex}
      {rin, rout} = b.rate
      # 2^32 output items per step input items, reduced
      assert rin / rout == 1.0017000436782837
      assert b.gnuradio == "filter.mmse_resampler_cc"
    end
  end

  describe "fir_ccc" do
    test "impulse response is the taps (float), real and complex" do
      taps = [{1.0, 0.0}, {0.0, 0.5}, {-0.25, 0.125}]
      b = Sources.block(:fir_ccc, format: @float, taps: taps)
      impulse = {Nx.tensor([1.0, 0, 0, 0, 0], type: :f32), Nx.broadcast(Nx.tensor(0.0), {5})}
      {y, _} = run(b, impulse)
      assert items(y) == taps ++ [{0.0, 0.0}, {0.0, 0.0}]

      real = Sources.block(:fir_ccc, format: @float, taps: [0.5, 0.25])
      {y, _} = run(real, impulse)
      assert items(y) |> Enum.take(3) == [{0.5, 0.0}, {0.25, 0.0}, {0.0, 0.0}]
    end

    test "single tap, and split invariance in both formats" do
      for f <- [@float, @q15], taps <- [[0.5], [0.5, {0.0, 0.25}, 0.125]] do
        b = Sources.block(:fir_ccc, format: f, taps: taps)
        assert_split_invariant(b, tone(100, 0.07, 0.5, f), [1, 49, 50])
      end
    end
  end

  describe "noise in channel_model" do
    test "complex gaussian noise_voltage: noise power is its square, both components equal" do
      v = 0.05

      b =
        Sources.block(:channel_model, format: @float, noise_voltage: v, seed: 4, resample: false)

      zero = {Nx.broadcast(Nx.tensor(0.0), {20_000}), Nx.broadcast(Nx.tensor(0.0), {20_000})}
      {y, _} = run(b, zero)
      xs = items(y)
      power = Enum.sum(Enum.map(xs, fn {a, c} -> a * a + c * c end)) / length(xs)
      assert_in_delta power / (v * v), 1.0, 0.04
      re = Enum.map(xs, &elem(&1, 0))
      im = Enum.map(xs, &elem(&1, 1))
      assert_in_delta C.variance(re) / C.variance(im), 1.0, 0.06
    end
  end

  describe "channel_model" do
    test "frequency offset turns the phase by 2 pi f per sample" do
      b = Sources.block(:channel_model, format: @float, frequency_offset: 0.01, resample: false)
      dc = {Nx.broadcast(Nx.tensor(0.5), {50}), Nx.broadcast(Nx.tensor(0.0), {50})}
      {y, _} = run(b, dc)
      [{a, c}, {d, e}] = items(y) |> Enum.slice(10, 2)
      assert_in_delta :math.atan2(a * e - c * d, a * d + c * e), 2 * @pi * 0.01, 1.0e-5
    end

    test "multipath: taps are the impulse response" do
      b = Sources.block(:channel_model, format: @float, taps: [1.0, {0.0, 0.5}], resample: false)
      impulse = {Nx.tensor([1.0, 0, 0, 0], type: :f32), Nx.broadcast(Nx.tensor(0.0), {4})}
      {y, _} = run(b, impulse)
      assert items(y) |> Enum.take(3) == [{1.0, 0.0}, {0.0, 0.5}, {0.0, 0.0}]
    end

    test "all parts together, split anywhere, in float and Q15" do
      for f <- [@float, @q15] do
        b =
          Sources.block(:channel_model,
            format: f,
            noise_voltage: 0.01,
            frequency_offset: 0.013,
            epsilon: 0.9990234375,
            taps: [1.0, {0.0, 0.25}, 0.125],
            seed: 9
          )

        {rin, rout} = b.rate
        assert rin / rout == 0.9990234375
        assert b.hardware == :feedback
        assert_split_invariant(b, tone(200, 0.04, 0.4, f), [1, 3, 30, 66, 100])
      end
    end

    test "sequence of one block is the block; hardware class is the strictest of the parts" do
      fir = Sources.block(:fir_ccc, format: @float, taps: [1.0, 0.5])
      assert Channel.sequence([fir], :x) == fir
      seq = Channel.sequence([fir, Sources.block(:rotator_cc, frequency: 0.1)], :x)
      assert seq.hardware == :feedback
      assert seq.warmup == :unbounded
    end
  end

  # -- fading ---------------------------------------------------------------------------------

  # h[0..n-1] of a fading block: a constant 1 passed through it
  defp gain(opts, n) do
    b = Sources.block(:fading_model, [format: @float] ++ opts)
    one = {Nx.broadcast(Nx.tensor(1.0), {n}), Nx.broadcast(Nx.tensor(0.0), {n})}
    {h, _} = run(b, one)
    items(h)
  end

  defp j0(x) do
    # series, adequate for |x| < 10
    Enum.reduce(0..30, {0.0, 1.0}, fn k, {sum, term} ->
      {sum + term, term * -(x * x / 4) / ((k + 1) * (k + 1))}
    end)
    |> elem(0)
  end

  describe "fading_model" do
    test "Rayleigh over seeds: E|h|^2 = 1, power exponentially distributed, correlation J0" do
      lag = 20
      hs = for seed <- 1..300, do: gain([n: 16, fdts: 0.01, seed: seed], lag + 1)
      p = for h <- hs, do: (fn {a, b} -> a * a + b * b end).(hd(h))
      assert_in_delta C.mean(p), 1.0, 0.12
      # CDF of an exponential variate at three points (standard error 0.03)
      for x <- [0.3, 1.0, 2.0] do
        frac = Enum.count(p, &(&1 <= x)) / length(p)
        assert_in_delta frac, 1 - :math.exp(-x), 0.08
      end

      # E[h(0) conj h(lag)] is real and equals J0(2 pi fd lag)
      corr =
        for h <- hs do
          {a, b} = hd(h)
          {c, d} = List.last(h)
          {a * c + b * d, b * c - a * d}
        end

      re = C.mean(Enum.map(corr, &elem(&1, 0)))
      im = C.mean(Enum.map(corr, &elem(&1, 1)))
      assert_in_delta re, j0(2 * @pi * 0.01 * lag), 0.12
      assert_in_delta im, 0.0, 0.12
    end

    test "Rician K = 4 over seeds: E|h|^2 = 1 and the power spreads less (variance (2K+1)/(K+1)^2)" do
      p =
        for seed <- 1..300,
            do:
              (fn {a, b} -> a * a + b * b end).(
                hd(gain([n: 16, fdts: 0.01, los: true, k: 4.0, seed: seed], 1))
              )

      assert_in_delta C.mean(p), 1.0, 0.07
      assert_in_delta C.variance(p), 9 / 25, 0.12
    end

    test "time average of the power over one long realisation is 1" do
      for opts <- [[los: false], [los: true, k: 3.0]] do
        h = gain([n: 16, fdts: 0.05, seed: 3] ++ opts, 20_000)
        power = Enum.sum(Enum.map(h, fn {a, b} -> a * a + b * b end)) / length(h)
        assert_in_delta power, 1.0, 0.1
      end
    end

    test "fdts = 0 freezes the channel; Doppler never exceeds fdts" do
      h = gain([n: 8, fdts: 0.0, seed: 2], 10)
      assert Enum.uniq(h) |> length() == 1

      # the phase of a single LOS-dominated path rotates at most 2 pi fdts per sample
      h = gain([n: 4, fdts: 0.02, los: true, k: 1.0e6, seed: 5], 100)

      steps =
        h
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [{a, b}, {c, d}] -> abs(:math.atan2(a * d - b * c, a * c + b * d)) end)

      assert Enum.max(steps) <= 2 * @pi * 0.02 + 1.0e-3
    end

    test "split invariance in float and Q15; fixed point stays close to float" do
      opts = [n: 16, fdts: 0.02, los: true, k: 2.0, seed: 6]

      for f <- [@float, Num.q(18, 17, lut_bits: 14)] do
        b = Sources.block(:fading_model, [format: f] ++ opts)
        assert_split_invariant(b, tone(120, 0.03, 0.4, f), [7, 50, 63])
      end

      x = tone(2000, 0.03, 0.4, @float)
      {fl, _} = run(Sources.block(:fading_model, [format: @float] ++ opts), x)
      f17 = Num.q(18, 17, lut_bits: 14)

      {fx, _} =
        run(Sources.block(:fading_model, [format: f17] ++ opts), tone(2000, 0.03, 0.4, f17))

      r = C.compare(items(fl), items(fx))
      # phase rounding to 14 table bits: |h| * pi / 2^14 per component of h (16 components of
      # amplitude 1/4 each in the worst case) times the 0.4 signal: about 8 LSB; plus the
      # input rounding in Q17 (the tone is made in Q15)
      assert r.worst_lsb <= 12, "worst #{r.worst_lsb}"
      assert r.snr >= 60, "snr #{r.snr}"
    end
  end

  describe "fading against GNU Radio's output" do
    for name <- ["fading_rayleigh", "fading_rician"] do
      @name name
      test "#{name}: time averaged power equals GNU Radio's (both near 1)" do
        {case, stats} = C.stats_case_named(@name)
        # input is a constant 0.5: output power / 0.25 is the fading power
        gr = stats["power_mean"]
        mine = C.run_stats_case(case, @float) |> Enum.map(fn {a, b} -> (a * a + b * b) / 0.25 end)
        assert_in_delta C.mean(mine), 1.0, 0.1
        assert_in_delta gr, 1.0, 0.1
        assert_in_delta C.mean(mine) / gr, 1.0, 0.12
        # Rician power spreads less than Rayleigh's (variance 1 for Rayleigh, 0.36 for K = 4)
        spread = fn xs -> C.variance(xs) end
        expected = if @name == "fading_rayleigh", do: 1.0, else: 0.36
        # one realisation of 8 or 16 sinusoids is not ergodic: only the class of the spread
        assert spread.(mine) < expected * 3 + 0.5
        assert stats["power_variance"] < expected * 3 + 0.5
      end
    end
  end

  describe "selective_fading_model" do
    test "fractional delay taps: whole delays are one tap, fractions interpolate with DC gain within 3e-4 of 1" do
      t = Channel.fractional_delay_taps(2, 8)
      assert Enum.at(t, 6) == 1.0
      assert Enum.sum(t) == 1.0
      assert length(Enum.filter(t, &(&1 != 0.0))) == 1

      half = Channel.fractional_delay_taps(1.5, 8)
      assert_in_delta Enum.sum(half), 1.0, 0.02
      # symmetric about the centre 5.5
      assert_in_delta Enum.at(half, 5), Enum.at(half, 6), 1.0e-12
    end

    test "average power is the sum of the squared path gains" do
      b =
        Sources.block(:selective_fading_model,
          format: @float,
          n: 16,
          fdts: 0.02,
          seed: 12,
          delays: [0.0, 3.5],
          mags: [1.0, 0.5],
          ntaps: 8
        )

      white =
        Sources.generate(
          Sources.block(:noise_source_c, type: :gaussian, amplitude: 0.5, seed: 3),
          40_000
        )
        |> elem(0)

      {y, _} = run(b, white)
      px = power(items(white))
      py = power(items(y))
      # paths 1 and 0.5: 1.25, with the time averaged power of two independent fading
      # processes (a few percent) and the interpolator's gain at the band edge
      assert_in_delta py / px, 1.25, 0.15
      assert b.gnuradio == "channels.selective_fading_model"
    end

    test "split invariance in float and fixed point" do
      for f <- [@float, @q15] do
        b =
          Sources.block(:selective_fading_model,
            format: f,
            n: 8,
            fdts: 0.01,
            seed: 3,
            delays: [0.0, 2.25],
            mags: [0.8, 0.4]
          )

        assert_split_invariant(b, tone(90, 0.02, 0.4, f), [11, 30, 49])
      end
    end

    defp power(xs), do: Enum.sum(Enum.map(xs, fn {a, b} -> a * a + b * b end)) / length(xs)
  end
end
