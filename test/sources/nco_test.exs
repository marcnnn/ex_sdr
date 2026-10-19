defmodule ExSdr.Sources.NCOTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Sources
  alias ExSdr.Sources.NCO
  alias NxFixed.Num

  @moduletag timeout: 300_000

  @waveforms [:sin, :cos, :square, :triangle, :sawtooth, :constant]
  @formats [Num.float(), Num.q(16, 15), Num.q(18, 17, rounding: :truncate, lut_bits: 12)]

  defp floats(out) do
    case Num.to_float(out) do
      {re, im} -> Enum.zip(re, im)
      xs -> xs
    end
  end

  describe "phase words" do
    test "frequency resolution and wrap" do
      # 1 Hz at 1 MHz with 32 bits: 4294.967 words, rounded
      assert NCO.phase_inc(1.0, 1.0e6) == 4295
      assert NCO.phase_inc(1.0, 1.0e6, 16) == 0
      assert NCO.phase_inc(0.25, 1.0, 16) == 16384
      assert NCO.phase_inc(-0.25, 1.0, 16) == 49152
      assert NCO.phase_inc(1.25, 1.0, 16) == 16384
      assert NCO.phase_from_radians(:math.pi(), 32) == 2_147_483_648
    end

    property "the state after n samples is phase + n * inc modulo 2^bits, exactly" do
      check all(
              bits <- member_of([16, 24, 32]),
              inc <- integer(0..65_535),
              p0 <- integer(0..65_535),
              n <- member_of([1, 7, 40]),
              max_runs: 12
            ) do
        block = Sources.block(:sig_source_c, phase_bits: bits, format: Num.q(16, 15))
        block = %{block | init: NCO.tensor(p0), params: [NCO.tensor(inc)]}
        {_, state} = Sources.generate(block, n)
        assert Nx.to_number(state) == rem(p0 + n * inc, Bitwise.bsl(1, bits))
      end
    end

    test "a 16 bit accumulator gives the same signal as a 32 bit one at the same frequency" do
      for f <- @formats, wf <- [:sin, :triangle] do
        mk = fn bits ->
          b =
            Sources.block(:sig_source_c,
              format: f,
              waveform: wf,
              frequency: 1 / 64,
              phase: :math.pi() / 2,
              phase_bits: bits
            )

          {out, _} = Sources.generate(b, 130)
          floats(out)
        end

        assert mk.(16) == mk.(32)
      end
    end
  end

  describe "split invariance" do
    for name <- [:sig_source_f, :sig_source_c], wf <- @waveforms do
      @name name
      @wf wf
      test "#{name} #{wf}: 37 + 63 samples equal 100, in every format" do
        for f <- @formats do
          block =
            Sources.block(@name,
              format: f,
              waveform: @wf,
              frequency: 0.0123,
              amplitude: 0.7,
              offset: 0.1,
              phase: 0.3
            )

          {whole, _} = Sources.generate(block, 100)
          {a, s} = Sources.generate(block, 37)
          {b, _} = Sources.generate(block, 63, s)
          assert floats(whole) == floats(a) ++ floats(b)
        end
      end
    end

    test "rotator_cc and vco_c are split invariant" do
      for f <- @formats do
        rot = Sources.block(:rotator_cc, format: f, frequency: 0.031, phase: 0.2)

        x =
          {Num.from_int(Nx.iota({100}, type: :s16), 8, f),
           Num.from_int(Nx.negate(Nx.iota({100}, type: :s16)), 8, f)}

        {whole, _} = NxFixed.Block.run(rot, x, rot.init, allow_float: true)
        {a, s} = NxFixed.Block.run(rot, slice(x, 0, 41), rot.init, allow_float: true)
        {b, _} = NxFixed.Block.run(rot, slice(x, 41, 59), s, allow_float: true)
        assert floats(whole) == floats(a) ++ floats(b)
      end
    end

    defp slice({re, im}, start, len), do: {slice1(re, start, len), slice1(im, start, len)}

    defp slice1(%NxFixed.Fixed{t: t, frac: f}, s, l),
      do: %NxFixed.Fixed{t: Nx.slice(t, [s], [l]), frac: f}

    defp slice1(t, s, l), do: Nx.slice(t, [s], [l])
  end

  describe "rotator" do
    test "keeps the magnitude and turns a constant into a tone of the right frequency" do
      f = Num.float()
      n = 64
      rot = Sources.block(:rotator_cc, format: f, frequency: 1 / 16)
      x = {Nx.broadcast(Nx.tensor(0.5), {n}), Nx.broadcast(Nx.tensor(0.0), {n})}
      {y, _} = NxFixed.Block.run(rot, x, rot.init, allow_float: true)
      {re, im} = Num.to_float(y)

      for {r, i} <- Enum.zip(re, im), do: assert_in_delta(:math.sqrt(r * r + i * i), 0.5, 1.0e-6)
      # sample 4 is a quarter turn: 0.5j
      assert_in_delta Enum.at(re, 4), 0.0, 1.0e-6
      assert_in_delta Enum.at(im, 4), 0.5, 1.0e-6
      # the first sample uses the initial phase (0): no advance before use
      assert_in_delta Enum.at(re, 0), 0.5, 1.0e-6
    end
  end

  describe "spurs" do
    # a tone of 2^16 samples, 4099 turns: the phase word repeats only after the whole record,
    # so the table truncation error is one clean spur
    defp sfdr_db(format) do
      n = 65_536
      b = Sources.block(:sig_source_c, format: format, frequency: 4099 / n, amplitude: 0.99)
      {out, _} = Sources.generate(b, n)
      {re, im} = Num.to_float(out)
      x = Nx.complex(Nx.tensor(re, type: :f64), Nx.tensor(im, type: :f64))
      m = x |> Nx.fft() |> Nx.abs() |> Nx.to_list()
      carrier = Enum.at(m, 4099)

      spur =
        m
        |> Enum.with_index()
        |> Enum.reject(fn {_, i} -> i == 4099 end)
        |> Enum.map(&elem(&1, 0))
        |> Enum.max()

      20 * :math.log10(carrier / spur)
    end

    test "worst spur is 6.02 dB per table bit, truncated or rounded" do
      for {l, rounding} <- [{8, :truncate}, {10, :nearest}, {12, :truncate}] do
        sfdr = sfdr_db(Num.q(18, 17, rounding: rounding, lut_bits: l))
        assert_in_delta sfdr, 6.02 * l, 1.5, "#{l} table bits, #{rounding}"
      end
    end

    test "float instance has no visible spur" do
      assert sfdr_db(Num.float()) > 140
    end
  end

  describe "waveform values (float instance)" do
    test "levels at the quarter points" do
      at = fn wf, phase ->
        b = Sources.block(:sig_source_f, waveform: wf, phase: phase)
        {out, _} = Sources.generate(b, 1)
        hd(floats(out))
      end

      pi = :math.pi()
      assert_in_delta at.(:sin, pi / 2), 1.0, 1.0e-6
      assert_in_delta at.(:cos, 0.0), 1.0, 1.0e-6
      # GNU Radio: square is amplitude where the phase is negative, else 0
      assert at.(:square, 0.1) == 0.0
      assert at.(:square, -0.1 + 2 * pi) == 1.0
      assert_in_delta at.(:triangle, 0.0), 1.0, 1.0e-6
      assert_in_delta at.(:triangle, pi - 1.0e-4), 0.0, 1.0e-4
      assert_in_delta at.(:sawtooth, 0.0), 0.5, 1.0e-6
      assert_in_delta at.(:constant, 0.0), 1.0, 1.0e-6
    end

    test "amplitude and offset must fit a fixed-point format" do
      assert_raise ArgumentError, fn ->
        Sources.block(:sig_source_f, format: Num.q(16, 15), amplitude: 1.5)
      end

      assert Sources.block(:sig_source_f, format: Num.q(16, 15), amplitude: 1.0)

      assert Sources.block(:sig_source_f, format: Num.q(18, 15), amplitude: 3.5)
    end
  end

  describe "vco" do
    test "zero control gives the initial phase forever, control moves the phase" do
      f = Num.float()

      vco =
        Sources.block(:vco_c,
          format: f,
          sensitivity: 2 * :math.pi(),
          sample_rate: 1.0,
          phase: 1.0
        )

      zero = Nx.broadcast(Nx.tensor(0.0), {8})
      {y, _} = NxFixed.Block.run(vco, zero, vco.init, allow_float: true)
      {re, im} = Num.to_float(y)
      for r <- re, do: assert_in_delta(r, :math.cos(1.0), 1.0e-6)
      for i <- im, do: assert_in_delta(i, :math.sin(1.0), 1.0e-6)

      # control 0.25 with sensitivity 2 pi: a quarter turn per sample
      quarter = Nx.broadcast(Nx.tensor(0.25), {4})
      {y, _} = NxFixed.Block.run(vco, quarter, vco.init, allow_float: true)
      {re, _} = Num.to_float(y)
      assert_in_delta Enum.at(re, 1), :math.cos(1.0 + :math.pi() / 2), 1.0e-5
    end
  end
end
