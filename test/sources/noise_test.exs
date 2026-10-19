defmodule ExSdr.Sources.NoiseTest do
  @moduledoc """
  Noise sources: deterministic given the seed, split invariant, and
  statistically right (mean, variance, kurtosis, distribution, spectral
  flatness), also against GNU Radio's output for the same distribution
  (different random numbers, same statistics).

  Tolerances come from the sample size: with N = 100000 samples the
  relative standard error of a variance is sqrt((kurtosis - 1) / N), 0.45
  percent for a Gaussian, so the bounds below are about 3 sigma.
  """
  use ExUnit.Case, async: true

  Code.require_file("support.exs", __DIR__)
  alias ExSdr.SourcesCase, as: C
  alias ExSdr.Sources
  alias NxFixed.Num

  @moduletag timeout: 300_000

  # components of a real or complex stream, as a list of real streams
  defp components([{_, _} | _] = xs), do: [Enum.map(xs, &elem(&1, 0)), Enum.map(xs, &elem(&1, 1))]
  defp components(xs), do: [xs]

  defp amp(case), do: case["block"]["opts"]["amplitude"]

  # variance of one component for the distribution, amplitude 1
  defp theory("uniform"), do: 1 / 3
  defp theory("gaussian"), do: 1.0
  defp theory("laplacian"), do: 2.0

  setup_all do
    runs =
      for {case, stats} <- C.stats_cases(),
          String.starts_with?(case["name"], ["noise_", "fastnoise_"]),
          into: %{} do
        {case["name"],
         %{
           case: case,
           mine: C.run_stats_case(case, Num.float()) |> components(),
           gnuradio: stats["components"]
         }}
      end

    %{runs: runs}
  end

  for {case, _} <- C.stats_cases(), String.starts_with?(case["name"], ["noise_", "fastnoise_"]) do
    @name case["name"]
    @type_ case["block"]["opts"]["type"]
    @complex String.ends_with?(case["name"], "_c")

    describe case["name"] do
      if @type_ != "impulse" do
        test "mean, variance, kurtosis match theory and GNU Radio", %{runs: runs} do
          %{case: case, mine: mine, gnuradio: gr} = runs[@name]
          a = amp(case)
          # complex noise: each component carries half of the power
          scale = if @complex, do: 0.5, else: 1.0
          expected = theory(@type_) * a * a * scale
          fastnoise? = String.starts_with?(@name, "fastnoise")
          tol = if fastnoise?, do: 0.05, else: if(@type_ == "laplacian", do: 0.03, else: 0.02)

          for {m, g} <- Enum.zip(mine, gr) do
            sigma = :math.sqrt(expected)

            assert abs(C.mean(m)) <
                     4 * sigma / :math.sqrt(length(m)) * if(fastnoise?, do: 3, else: 1)

            assert_in_delta C.variance(m) / expected, 1.0, tol
            # against GNU Radio's own stream: their sampling error adds
            assert_in_delta C.variance(m) / g["variance"], 1.0, tol + 0.01

            kurt = %{"uniform" => 1.8, "gaussian" => 3.0, "laplacian" => 6.0}[@type_]
            ktol = %{"uniform" => 0.05, "gaussian" => 0.1, "laplacian" => 0.6}[@type_]
            assert_in_delta C.kurtosis(m), kurt, ktol * if(fastnoise?, do: 3, else: 1)
          end
        end
      end

      if @type_ != "impulse" and not String.starts_with?(@name, "fastnoise") do
        test "distribution (Kolmogorov-Smirnov) and range", %{runs: runs} do
          %{case: case, mine: mine} = runs[@name]
          a = amp(case) * if(@complex, do: :math.sqrt(0.5), else: 1.0)

          for m <- mine do
            n = length(m)

            cdf =
              case @type_ do
                "uniform" -> fn x -> min(max((x / a + 1) / 2, 0.0), 1.0) end
                "gaussian" -> fn x -> C.normal_cdf(x / a) end
                "laplacian" -> fn x -> C.laplace_cdf(x / a) end
              end

            # 99.9 percent critical value 1.95 / sqrt(n)
            assert C.ks_distance(m, cdf) < 1.95 / :math.sqrt(n)

            if @type_ == "uniform", do: assert(Enum.all?(m, &(abs(&1) <= a)))
          end
        end
      end

      test "spectrum is flat", %{runs: runs} do
        %{mine: mine, gnuradio: gr} = runs[@name]

        for m <- mine do
          # 390 averaged periodograms: white noise gives 1 - 1/(2 * 390) = 0.9987
          # for a flat spectrum; impulse noise has fewer events per segment
          assert C.spectral_flatness(m) >= if(@type_ == "impulse", do: 0.5, else: 0.99)
        end

        # as flat as GNU Radio's (to within the estimator's scatter)
        for {m, g} <- Enum.zip(mine, gr) do
          assert abs(C.spectral_flatness(m) - g["flatness"]) < 0.02
        end
      end

      if @complex do
        test "components are uncorrelated and equally strong", %{runs: runs} do
          %{mine: [re, im]} = runs[@name]
          n = length(re)

          corr =
            Enum.sum(Enum.zip_with(re, im, &(&1 * &2))) / n /
              :math.sqrt(C.variance(re) * C.variance(im))

          assert abs(corr) < 4 / :math.sqrt(n)
          assert_in_delta C.variance(re) / C.variance(im), 1.0, 0.03
        end
      end
    end
  end

  describe "impulse noise" do
    test "sparse positive spikes like GNU Radio's", %{runs: runs} do
      %{case: case, mine: [m], gnuradio: [g]} = runs["noise_impulse_f"]
      a = amp(case)
      nm = Enum.filter(m, &(&1 != 0.0))

      # about 1 in 560 (e^-6.3), counted over 400000 samples (about 700 spikes)
      assert length(nm) / length(m) > 0.0013 and length(nm) / length(m) < 0.0024
      assert_in_delta length(nm) / length(m) / (g["nonzero"] / g["n"]), 1.0, 0.25
      assert Enum.all?(nm, &(&1 >= 9 * a * 0.999)), "spikes start at factor * amplitude"
      # mean of a spike: 9 + 1.43 times the amplitude
      assert_in_delta C.mean(nm) / a, g["nonzero_mean"] / a, 0.8
      assert_in_delta C.mean(nm) / a, 10.43, 0.4

      assert_in_delta C.mean(Enum.map(nm, &(&1 * &1))) / g["nonzero_mean_square"], 1.0, 0.15
    end
  end

  describe "generator" do
    test "xorshift32 in Nx equals the integer reference" do
      state = ExSdr.Sources.Noise.seed_state(5)

      {words, _} =
        Enum.map_reduce(1..50, state, fn _, s ->
          n = ExSdr.Sources.Noise.next(s)
          {Nx.to_number(n), n}
        end)

      ref = fn x ->
        import Bitwise
        x = bxor(x, x <<< 13 &&& 0xFFFFFFFF)
        x = bxor(x, x >>> 17)
        bxor(x, x <<< 5 &&& 0xFFFFFFFF)
      end

      {expected, _} =
        Enum.map_reduce(1..50, Nx.to_number(state), fn _, x -> {ref.(x), ref.(x)} end)

      assert words == expected
    end

    test "seeds are never the zero state and differ" do
      states = for s <- 0..200, do: Nx.to_number(ExSdr.Sources.Noise.seed_state(s))
      assert Enum.all?(states, &(&1 != 0))
      assert length(Enum.uniq(states)) == length(states)
    end

    test "same seed, same stream; other seed, other stream" do
      for f <- [Num.float(), Num.q(16, 15)] do
        mk = fn seed ->
          Sources.block(:noise_source_c, format: f, type: :gaussian, amplitude: 0.1, seed: seed)
        end

        {a, _} = Sources.generate(mk.(1), 64)
        {b, _} = Sources.generate(mk.(1), 64)
        {c, _} = Sources.generate(mk.(2), 64)
        assert Num.to_float(a) == Num.to_float(b)
        refute Num.to_float(a) == Num.to_float(c)
      end
    end
  end

  describe "split invariance" do
    for name <- [:noise_source_f, :noise_source_c, :fastnoise_source_c],
        type <- [:uniform, :gaussian, :laplacian, :impulse] do
      @name name
      @type_ type
      test "#{name} #{type}: pieces of 1, 20 and 79 give the same bits as one call of 100" do
        for f <- [Num.float(), Num.q(16, 15)] do
          opts = [format: f, type: @type_, amplitude: 0.05, seed: 11, samples: 64]
          block = Sources.block(@name, opts)
          {whole, _} = Sources.generate(block, 100)
          {p1, s1} = Sources.generate(block, 1)
          {p2, s2} = Sources.generate(block, 20, s1)
          {p3, _} = Sources.generate(block, 79, s2)
          cat = &Enum.concat([Num.to_float(&1) |> flat()])

          assert flat(Num.to_float(whole)) ==
                   flat(Num.to_float(p1)) ++ flat(Num.to_float(p2)) ++ flat(Num.to_float(p3))

          _ = cat
        end
      end
    end

    defp flat({re, im}), do: Enum.zip(re, im)
    defp flat(xs) when is_list(xs), do: xs
    defp flat(x), do: [x]
  end

  describe "fixed point against the float instance" do
    for {type, bound} <- [uniform: 1.6, gaussian: 0.6, laplacian: 0.6, impulse: 0.6] do
      @type_ type
      @bound bound
      test "#{type}: Q15 values are the float values rounded (bound #{bound} LSB)" do
        mk = fn f ->
          Sources.block(:noise_source_f, format: f, type: @type_, amplitude: 0.05, seed: 3)
        end

        {fl, _} = Sources.generate(mk.(Num.float()), 4000)
        {fx, _} = Sources.generate(mk.(Num.q(16, 15)), 4000)

        worst =
          Enum.zip(Num.to_float(fl), Num.to_float(fx))
          |> Enum.map(fn {a, b} -> abs(a - b) * 32768 end)
          |> Enum.max()

        assert worst <= @bound
      end
    end

    test "fixed point noise passes the width check for gaussian at Q17 and the Q15 uniform default" do
      for f <- [Num.q(18, 17), Num.q(16, 15, rounding: :truncate)],
          type <- [:uniform, :gaussian] do
        b = Sources.block(:noise_source_c, format: f, type: type, amplitude: 0.1, seed: 1)
        {_, _} = Sources.generate(b, 8)
      end
    end
  end
end
