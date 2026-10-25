defmodule ExSdr.DSP.FFTTest do
  @moduledoc """
  The radix-2 Stockham FFT against `Nx.fft`: exact to float precision in the
  float format, about 55 dB SNR in q(16, 12) with q(16, 15) twiddles, and
  bit-identical between nx_simd and the Nx evaluator.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.DSP.FFT
  alias NxFixed.{Fixed, Num}

  defp random(shape, amplitude, seed) do
    key = Nx.Random.key(seed)
    {re, key} = Nx.Random.uniform(key, -amplitude, amplitude, shape: shape)
    {im, _} = Nx.Random.uniform(key, -amplitude, amplitude, shape: shape)
    {re, im}
  end

  defp reference({re, im}), do: Nx.fft(Nx.complex(re, im))

  # worst error relative to the largest output magnitude
  defp max_error({re, im}, expected) do
    got = Nx.complex(re, im)
    peak = expected |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    got
    |> Nx.subtract(expected)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
    |> Kernel./(peak)
  end

  defp snr_db({re, im}, expected) do
    got = Nx.complex(re, im)
    noise = got |> Nx.subtract(expected) |> Nx.abs() |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
    signal = expected |> Nx.abs() |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
    10 * :math.log10(signal / noise)
  end

  defp fixed_fft(re, im, n_shifts) do
    f = Num.q(16, 12)
    {r, i} = FFT.fft({q(re, f), q(im, f)}, format: f, twiddle: Num.q(16, 15), shifts: n_shifts)
    {fixed_to_float(r), fixed_to_float(i)}
  end

  defp q(x, f),
    do: %Fixed{t: x |> Nx.multiply(2 ** f.frac) |> Nx.round() |> Nx.as_type(:s16), frac: f.frac}

  defp fixed_to_float(%Fixed{t: t, frac: frac}),
    do: t |> Nx.as_type(:f32) |> Nx.divide(2 ** frac)

  describe "float format" do
    property "matches Nx.fft for random sizes and batch shapes" do
      check all(
              log <- integer(1..11),
              batch <- member_of([{}, {1}, {3}, {2, 2}]),
              seed <- integer(0..1_000_000),
              max_runs: 25
            ) do
        n = Integer.pow(2, log)
        x = random(Tuple.insert_at(batch, tuple_size(batch), n), 1.0, seed)
        got = FFT.fft(x, format: Num.float())

        assert Nx.shape(elem(got, 0)) == Nx.shape(elem(x, 0))
        assert max_error(got, reference(x)) < 1.0e-6
      end
    end

    test "n = 2048 error stays at float precision" do
      x = random({2048}, 1.0, 7)
      assert max_error(FFT.fft(x, format: Num.float()), reference(x)) < 1.0e-6
    end

    property "shifts halve the result once per shift" do
      check all(
              log <- integer(1..9),
              shifts <- integer(0..log),
              seed <- integer(0..1_000_000),
              max_runs: 20
            ) do
        n = Integer.pow(2, log)
        x = random({2, n}, 1.0, seed)
        got = FFT.fft(x, format: Num.float(), shifts: shifts)
        expected = Nx.divide(reference(x), 2 ** shifts)
        assert max_error(got, expected) < 1.0e-6
      end
    end
  end

  describe "fixed point" do
    test "q(16, 12), q(16, 15) twiddles, 5 shifts at n = 2048 is above 50 dB SNR" do
      for seed <- 1..3 do
        {re, im} = x = random({2048}, 0.2, seed)
        got = fixed_fft(re, im, 5)
        expected = Nx.divide(reference(x), 32)
        snr = snr_db(got, expected)
        assert snr > 50, "seed #{seed}: #{Float.round(snr, 1)} dB"
      end
    end
  end

  describe "errors" do
    test "length not a power of two" do
      for n <- [3, 6, 100, 2047] do
        x = random({n}, 1.0, 1)
        assert_raise ArgumentError, ~r/power of two/, fn -> FFT.fft(x, format: Num.float()) end
      end
    end

    test "shifts out of range" do
      x = random({16}, 1.0, 1)

      for shifts <- [-1, 5] do
        assert_raise ArgumentError, fn -> FFT.fft(x, format: Num.float(), shifts: shifts) end
      end
    end
  end

  describe "nx_simd" do
    test "compiled fixed FFT is bit-identical to the evaluator" do
      if Code.ensure_loaded?(NxSimd) and NxSimd.available?() do
        f = Num.q(16, 12)

        for n <- [2, 16, 256, 1024] do
          shifts = min(3, round(:math.log2(n)))

          fun = fn re, im ->
            {r, i} =
              FFT.fft({%Fixed{t: re, frac: 12}, %Fixed{t: im, frac: 12}},
                format: f,
                twiddle: Num.q(16, 15),
                shifts: shifts
              )

            {r.t, i.t}
          end

          {re, im} = random({2, n}, 0.4, n)
          args = [q(re, f).t, q(im, f).t]

          simd =
            Nx.Defn.jit_apply(fun, args, compiler: NxSimd.Compiler, fallback: false)

          eval = Nx.Defn.jit_apply(fun, args, compiler: Nx.Defn.Evaluator)

          assert Nx.to_binary(elem(simd, 0)) == Nx.to_binary(elem(eval, 0)), "re, n = #{n}"
          assert Nx.to_binary(elem(simd, 1)) == Nx.to_binary(elem(eval, 1)), "im, n = #{n}"
        end
      else
        IO.puts("nx_simd unavailable, skipping")
      end
    end
  end
end
