defmodule ExSdr.Sources.ImpairmentsTest do
  @moduledoc """
  Impairment blocks. The deterministic ones (`iqbal_gen`, `distortion_*`,
  `quantizer`) are compared with GNU Radio in gnuradio_test.exs; here their
  structure, the CFO random walk's statistics and split invariance.
  """
  use ExUnit.Case, async: true

  Code.require_file("support.exs", __DIR__)
  alias ExSdr.SourcesCase, as: C
  alias ExSdr.Sources
  alias NxFixed.Block
  alias NxFixed.Num

  @float Num.float()
  @q15 Num.q(16, 15)

  defp run(block, input, state \\ :init), do: Block.run(block, input, state, allow_float: true)

  defp dc(n, v, f),
    do: {C.samples(List.duplicate(round(v * 32768), n), f), C.samples(List.duplicate(0, n), f)}

  describe "quantizer" do
    test "levels are multiples of 2^-(bits-1), ties go to even" do
      b = Sources.block(:quantizer, bits: 3)

      # step 0.25: 0.125 is a tie between 0 and 0.25 (even: 0), 0.375 between 0.25 and 0.5 (even: 0.5)
      x = Nx.tensor([0.125, 0.375, -0.125, -0.375, 0.2, 0.9, -1.2], type: :f32)
      {y, _} = run(b, x)
      assert Num.to_float(y) == [0.0, 0.5, 0.0, -0.5, 0.25, 1.0, -1.25]
    end

    test "fixed point: same levels, finer converter than the format is a no-op" do
      b = Sources.block(:quantizer, format: @q15, bits: 3)
      x = C.samples([4096, 12288, -4096, 6554], @q15)
      {y, _} = run(b, x)
      assert Num.to_float(y) == [0.0, 0.5, 0.0, 0.25]
      fine = Sources.block(:quantizer, format: @q15, bits: 20)
      {y, _} = run(fine, x)
      assert Num.to_float(y) == Num.to_float(x)
    end

    test "complex quantizer works per component" do
      b = Sources.block(:quantizer, bits: 3, complex: true)
      {y, _} = run(b, {Nx.tensor([0.2], type: :f32), Nx.tensor([-0.6], type: :f32)})
      assert Num.to_float(y) == {[0.25], [-0.5]}
    end
  end

  describe "iqbal_gen and distortion" do
    test "no imbalance and no distortion are the identity" do
      x = {C.samples([1000, -2000, 30000], @q15), C.samples([-500, 700, -30000], @q15)}

      for b <- [
            Sources.block(:iqbal_gen, format: @q15, magnitude: 0.0, phase: 0.0),
            Sources.block(:iqbal_gen, format: @q15, magnitude: 0.0, phase: 0.0, mode: 1),
            Sources.block(:distortion_2_gen, format: @q15, beta: 0.0),
            Sources.block(:distortion_3_gen, format: @q15, beta: 0.0)
          ] do
        {y, _} = run(b, x)
        # the unity gain constant is 1 - 2^-15 in Q15: at most 1 LSB
        for {{a, b2}, {c, d}} <- Enum.zip(C.to_floats(y), C.to_floats(x)),
            do: assert(abs(a - c) * 32768 <= 1.0 and abs(b2 - d) * 32768 <= 1.0)
      end
    end

    test "constants beyond the format are rejected" do
      assert_raise ArgumentError, fn ->
        Sources.block(:iqbal_gen, format: @q15, magnitude: 3.0, phase: 0.0)
      end
    end

    test "split invariance (stateless)" do
      for name <- [:iqbal_gen, :distortion_2_gen, :distortion_3_gen] do
        opts = if name == :iqbal_gen, do: [magnitude: -1.0, phase: 4.0], else: [beta: 0.1]
        b = Sources.block(name, [format: @q15] ++ opts)

        x =
          {C.samples(Enum.to_list(-50..49) |> Enum.map(&(&1 * 400)), @q15),
           C.samples(Enum.to_list(0..99) |> Enum.map(&(&1 * 300 - 15_000)), @q15)}

        {whole, _} = run(b, x)
        cut = fn {re, im}, s, l -> {Nx.slice(re.t, [s], [l]), Nx.slice(im.t, [s], [l])} end
        {a, st} = run(b, wrap(cut.(x, 0, 30), @q15))
        {c, _} = run(b, wrap(cut.(x, 30, 70), @q15), st)
        assert C.to_floats(whole) == C.to_floats(a) ++ C.to_floats(c)
      end
    end

    defp wrap({re, im}, f),
      do: {%NxFixed.Fixed{t: re, frac: f.frac}, %NxFixed.Fixed{t: im, frac: f.frac}}
  end

  describe "cfo_model" do
    # instantaneous frequency in Hz from the phase step between samples of a rotated constant
    defp frequencies(y, rate) do
      C.to_floats(y)
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [{a, b}, {c, d}] ->
        :math.atan2(a * d - b * c, a * c + b * d) / (2 * :math.pi()) * rate
      end)
    end

    test "frequency steps have the requested standard deviation; the walk is clipped" do
      rate = 1.0e6
      n = 20_000

      free =
        Sources.block(:cfo_model, sample_rate: rate, std_dev_hz: 20.0, max_dev_hz: 1.0e5, seed: 1)

      {y, _} = run(free, dc(n, 0.5, @float))
      f = frequencies(y, rate)
      steps = Enum.zip_with(f, tl(f), &(&2 - &1))
      # the Gaussian table (4096 levels) has 0.1 percent less variance than the exact one
      assert_in_delta :math.sqrt(C.variance(steps)) / 20.0, 1.0, 0.03
      assert abs(C.mean(steps)) < 4 * 20.0 / :math.sqrt(n)

      clipped =
        Sources.block(:cfo_model, sample_rate: rate, std_dev_hz: 20.0, max_dev_hz: 300.0, seed: 1)

      {y, _} = run(clipped, dc(n, 0.5, @float))
      f = frequencies(y, rate)
      assert Enum.max(Enum.map(f, &abs/1)) <= 300.0 * 1.001
      assert Enum.max(Enum.map(f, &abs/1)) > 270.0, "a random walk of this size reaches the limit"
    end

    test "split invariance in float and Q15, and the walk starts at zero frequency" do
      for f <- [@float, @q15] do
        b =
          Sources.block(:cfo_model,
            format: f,
            sample_rate: 1.0e6,
            std_dev_hz: 50.0,
            max_dev_hz: 2000.0,
            seed: 5
          )

        x = dc(120, 0.4, f)
        {whole, _} = run(b, x)

        cut = fn start, len ->
          {re, im} = x

          g = fn v ->
            if f.kind == :fixed,
              do: %NxFixed.Fixed{t: Nx.slice(v.t, [start], [len]), frac: v.frac},
              else: Nx.slice(v, [start], [len])
          end

          {g.(re), g.(im)}
        end

        {a, st} = run(b, cut.(0, 45))
        {c, _} = run(b, cut.(45, 75), st)
        assert C.to_floats(whole) == C.to_floats(a) ++ C.to_floats(c)
        # first sample uses phase 0
        {x0, _} = hd(C.to_floats(whole))
        assert_in_delta x0, 0.4, 1.0e-3
      end
    end
  end
end
