defmodule ExSdr.Carrier.BlocksTest do
  @moduledoc """
  What the blocks declare about themselves, option checks, hand computed
  steps of the adaptive algorithms, and stream splitting at awkward cuts.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Carrier
  alias ExSdr.Digital.Constellation
  alias NxFixed.Num

  @qpsk Constellation.new([{0.5, 0.5}, {-0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}],
          normalization: :none
        )

  defp complex_f32(list) do
    {re, im} = Enum.unzip(list)
    {Nx.tensor(re, type: :f32), Nx.tensor(im, type: :f32)}
  end

  describe "block fields" do
    test "every block states kind, hardware class, item types and its GNU Radio name" do
      fixed = Num.q(16, 15)

      blocks = [
        Carrier.block(:costas_loop_cc, loop_bw: 0.05, order: 4, format: fixed),
        Carrier.block(:fll_band_edge_cc,
          samples_per_symbol: 4,
          rolloff: 0.35,
          filter_size: 21,
          loop_bw: 0.05
        ),
        Carrier.block(:constellation_receiver_cb,
          constellation: @qpsk,
          loop_bw: 0.05,
          fmin: -0.1,
          fmax: 0.1
        ),
        Carrier.block(:linear_equalizer,
          num_taps: 5,
          algorithm: :lms,
          mu: 0.01,
          constellation: @qpsk
        ),
        Carrier.block(:decision_feedback_equalizer,
          num_taps_forward: 5,
          num_taps_feedback: 2,
          algorithm: :nlms,
          mu: 0.1,
          constellation: @qpsk
        ),
        Carrier.block(:corr_est_cc, symbols: [{1.0, 0.0}, {0.0, 1.0}]),
        Carrier.block(:mpsk_snr_est_cc, type: :m2m4),
        Carrier.block(:probe_density_b, alpha: 0.1),
        Carrier.block(:probe_signal)
      ]

      for b <- blocks do
        assert b.kind == :stream
        assert b.gnuradio =~ "."
        assert is_function(b.fun, 3)
        assert match?({_, _}, b.item)
      end

      by_name = Map.new(blocks, &{&1.name, &1})

      for name <- [
            :costas_loop_cc,
            :fll_band_edge_cc,
            :constellation_receiver_cb,
            :linear_equalizer,
            :decision_feedback_equalizer
          ] do
        assert by_name[name].hardware == :feedback
        assert by_name[name].warmup == :unbounded
      end

      assert by_name.fll_band_edge_cc.latency == 21
      assert by_name.corr_est_cc.latency == 2
      assert by_name.corr_est_cc.hardware == :control
      assert by_name.probe_signal.hardware == :feed_forward
    end

    test "equalizers decimating by sps declare a variable rate of at most one output" do
      b =
        Carrier.block(:linear_equalizer,
          num_taps: 5,
          algorithm: :lms,
          mu: 0.01,
          sps: 2,
          constellation: @qpsk
        )

      assert b.rate == {:variable, 1}
      assert Carrier.block(:cma_equalizer_cc, num_taps: 3, mu: 0.01).rate == {1, 1}
    end

    test "the two older equalizer names are linear_equalizer with the algorithm preset" do
      assert Carrier.block(:cma_equalizer_cc, num_taps: 3, mu: 0.01).opts[:algorithm] == :cma

      b = Carrier.block(:lms_dd_equalizer_cc, num_taps: 3, mu: 0.01, constellation: @qpsk)
      assert b.opts[:algorithm] == :lms
    end

    test "bad options are rejected" do
      assert_raise ArgumentError, ~r/order/, fn ->
        Carrier.block(:costas_loop_cc, loop_bw: 0.1, order: 3)
      end

      assert_raise ArgumentError, ~r/missing option :loop_bw/, fn ->
        Carrier.block(:costas_loop_cc, order: 4)
      end

      assert_raise ArgumentError, ~r/unknown carrier block/, fn -> Carrier.block(:nope) end

      assert_raise ArgumentError, fn ->
        Carrier.block(:linear_equalizer,
          num_taps: 1,
          algorithm: :lms,
          mu: 0.1,
          constellation: @qpsk
        )
      end

      assert_raise ArgumentError, fn -> Carrier.block(:mpsk_snr_est_cc, type: :bogus) end
    end

    test "corr_est_cc clamps mark_delay below the sync word length as GNU Radio does" do
      b = Carrier.block(:corr_est_cc, symbols: List.duplicate({1.0, 0.0}, 8), mark_delay: 100)
      assert b.opts[:mark_delay] == 7
    end
  end

  describe "control loop" do
    test "gains are those of gr::blocks::control_loop (damping sqrt(2)/2)" do
      {alpha, beta} = ExSdr.Analog.Kernels.loop_gains(0.05)
      denom = 1.0 + 2 * (:math.sqrt(2) / 2) * 0.05 + 0.05 * 0.05
      assert_in_delta alpha, 4 * (:math.sqrt(2) / 2) * 0.05 / denom, 1.0e-12
      assert_in_delta beta, 4 * 0.05 * 0.05 / denom, 1.0e-12
      {a, b, hi, lo} = ExSdr.Carrier.Loop.params(loop_bw: 0.05, max_freq: 1.0, min_freq: -1.0)
      assert {a, b} == {alpha, beta}
      assert hi == round(1.0 * 2_147_483_648 / :math.pi())
      assert lo == -hi
    end

    test "Costas on a clean rotated QPSK stream locks, float and Q15, order 4" do
      pts =
        for i <- 0..1999,
            do:
              Enum.at(
                [{0.7, 0.7}, {-0.7, 0.7}, {-0.7, -0.7}, {0.7, -0.7}],
                rem(i * 7 + div(i, 3), 4)
              )

      offset = 0.003

      rotated =
        for {{a, b}, i} <- Enum.with_index(pts) do
          phi = 2 * :math.pi() * offset * i + 0.4
          {a * :math.cos(phi) - b * :math.sin(phi), a * :math.sin(phi) + b * :math.cos(phi)}
        end

      {re, im} = Enum.unzip(rotated)

      for fmt <- [Num.float(), Num.q(16, 15, lut_bits: 12)] do
        input =
          case fmt.kind do
            :float ->
              complex_f32(rotated)

            :fixed ->
              q = fn l ->
                Num.from_int(Nx.tensor(Enum.map(l, &round(&1 * 32768)), type: :s16), 15, fmt)
              end

              {q.(re), q.(im)}
          end

        b = Carrier.block(:costas_loop_cc, format: fmt, loop_bw: 0.04, order: 4, outputs: [:freq])
        {{_y, {freq}}, _} = Carrier.run(b, input)
        tail = freq |> Nx.to_flat_list() |> Enum.take(-200)
        mean = Enum.sum(tail) / length(tail) * :math.pi() / 2_147_483_648
        assert_in_delta mean, 2 * :math.pi() * offset, 1.0e-3
      end
    end
  end

  describe "adaptive algorithms, one step by hand" do
    # two taps, window [0, x0]; w starts at zero (LMS, NLMS) or the unit tap on the newest sample (CMA)
    @x0 {0.3, -0.2}
    @train {0.5, 0.5}

    defp first_taps(alg, extra \\ []) do
      opts =
        [
          format: Num.float(),
          num_taps: 2,
          algorithm: alg,
          mu: 0.1,
          constellation: @qpsk,
          training: [@train, @train],
          outputs: [:taps]
        ] ++ extra

      b = Carrier.block(:linear_equalizer, opts)
      input = {complex_f32([@x0, @x0, @x0]), Nx.tensor([1, 0, 0], type: :u8)}
      {{_y, {taps}}, _} = Carrier.run(b, input)
      {wr, wi} = taps
      {Nx.to_list(wr), Nx.to_list(wi)}
    end

    test "LMS: w += mu conj(x) (d - y) after the first training symbol" do
      {wr, wi} = first_taps(:lms)
      # taps before the update of output 0 are zero, those before output 1 are the updated ones
      assert Enum.at(wr, 0) == [0.0, 0.0]
      # y = 0, e = d; w[1] = mu * conj(x0) * d (x0 is the newest sample of the window)
      {xr, xi} = @x0
      {dr, di} = @train
      expect_r = 0.1 * (xr * dr + xi * di)
      expect_i = 0.1 * (xr * di - xi * dr)
      assert_in_delta Enum.at(wr, 1) |> Enum.at(1), expect_r, 1.0e-6
      assert_in_delta Enum.at(wi, 1) |> Enum.at(1), expect_i, 1.0e-6
      # the older tap saw a zero sample
      assert Enum.at(wr, 1) |> Enum.at(0) == 0.0
    end

    test "NLMS: the same step divided by the window energy" do
      {wr, _wi} = first_taps(:nlms)
      {xr, xi} = @x0
      {dr, di} = @train
      energy = xr * xr + xi * xi
      assert_in_delta Enum.at(wr, 1) |> Enum.at(1), 0.1 * (xr * dr + xi * di) / energy, 1.0e-5
    end

    test "CMA: w -= mu conj(x) e with e = y (|y|^2 - modulus) clipped, starting from the unit tap" do
      {wr, wi} = first_taps(:cma, modulus: 0.25)
      {xr, xi} = @x0
      p = xr * xr + xi * xi - 0.25
      {er, ei} = {xr * p, xi * p}
      # w[1] starts at 1 and moves by -mu conj(x) e
      expect_r = 1.0 - 0.1 * (xr * er + xi * ei)
      expect_i = 0.0 - 0.1 * (xr * ei - xi * er)
      assert_in_delta wr |> Enum.at(1) |> Enum.at(1), expect_r, 1.0e-6
      assert_in_delta wi |> Enum.at(1) |> Enum.at(1), expect_i, 1.0e-6
    end
  end

  describe "stream splitting" do
    test "a decimating equalizer cut at an odd item carries its phase (fixed point, bit-exact)" do
      fmt = Num.q(16, 15, lut_bits: 10)

      b =
        Carrier.block(:linear_equalizer,
          format: fmt,
          num_taps: 4,
          algorithm: :lms,
          mu: 0.02,
          sps: 2,
          constellation: @qpsk
        )

      :rand.seed(:exsss, {1, 2, 3})
      ints = fn -> for _ <- 1..60, do: round(:rand.normal() * 8000) end
      q = fn l -> Num.from_int(Nx.tensor(l, type: :s16), 15, fmt) end
      {re, im} = {ints.(), ints.()}
      whole = {q.(re), q.(im)}
      {w, _} = Carrier.run(b, whole)

      cut = 21
      {a1, s1} = Carrier.run(b, {q.(Enum.take(re, cut)), q.(Enum.take(im, cut))})
      {a2, _} = Carrier.run(b, {q.(Enum.drop(re, cut)), q.(Enum.drop(im, cut))}, s1)

      {wr, wi} = w
      {r1, i1} = a1
      {r2, i2} = a2
      assert Nx.to_flat_list(wr.t) == Nx.to_flat_list(r1.t) ++ Nx.to_flat_list(r2.t)
      assert Nx.to_flat_list(wi.t) == Nx.to_flat_list(i1.t) ++ Nx.to_flat_list(i2.t)
      assert length(Nx.to_flat_list(wr.t)) == 30
    end
  end
end
