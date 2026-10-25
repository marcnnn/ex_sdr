defmodule ExSdr.Comm.OFDM.FixedFrontEndTest do
  @moduledoc """
  The integer OFDM front end against the float `ExSdr.Comm.OFDM.FrontEnd` on
  a DVB-T 2k QPSK signal, chunk continuity of the frequency correction, and
  bit-identical results across compilers.

  The comparison on the long impaired signal is tagged `:dvbt`
  (`mix test --only dvbt`).
  """
  use ExUnit.Case, async: false

  alias ExSdr.Comm.OFDM.{FixedFrontEnd, FrontEnd}
  alias ExSdr.DVBT.{Channel, Receiver, Sync, TestSignal, Transmitter}
  alias NxFixed.Num

  @moduletag timeout: 900_000

  @opts [constellation: :qpsk, code_rate: {1, 2}, guard_interval: {1, 32}, cell_id: 0]
  @sym 2112
  @delay 3333
  @cfo 1.37
  @gain 0.47

  setup_all do
    clean = TestSignal.ts_packets(1008) |> Transmitter.transmit(@opts)
    impaired = clean |> Channel.delay(@delay, 3) |> Channel.frequency_offset(@cfo)
    %{clean: clean, impaired: impaired, spec: Receiver.front_end_spec(:qpsk)}
  end

  # synchronised samples of n symbols and the estimated frequency offset
  defp acquired(x, n) do
    {:ok, s} = Sync.acquire(x)
    {Nx.slice(x, [s.start], [n * @sym]), s.freq_offset}
  end

  defp differing(a, b) do
    assert byte_size(a) == byte_size(b)

    Enum.zip(:binary.bin_to_list(a), :binary.bin_to_list(b))
    |> Enum.count(fn {x, y} -> x != y end)
  end

  defp float_bits(y, freq, spec), do: FrontEnd.run(y, freq, spec)

  defp chunk(sc16, symbols, first) do
    sc16 |> Nx.slice([first * @sym * 2], [symbols * @sym * 2])
  end

  describe "frequency correction" do
    test "phase increment is the 32-bit word of -offset / fft" do
      assert FixedFrontEnd.phase_increment(0.0, 2048) == 0
      # one subcarrier spacing is 1/2048 of a turn per sample
      assert FixedFrontEnd.phase_increment(-1.0, 2048) == 4_294_967_296 / 2048
      assert FixedFrontEnd.phase_increment(1.0, 2048) == 4_294_967_296 - 4_294_967_296 / 2048
      assert FixedFrontEnd.phase_increment(2048.0, 2048) == 0
    end

    test "chunks with start offsets join exactly", %{impaired: x, spec: spec} do
      {y, freq} = acquired(x, 8)
      sc16 = FixedFrontEnd.to_sc16(y, @gain)

      fe8 = FixedFrontEnd.prepare(spec, 8)
      fe4 = FixedFrontEnd.prepare(spec, 4)

      whole = FixedFrontEnd.run(fe8, sc16, freq)

      joined =
        FixedFrontEnd.run(fe4, chunk(sc16, 4, 0), freq, 0) <>
          FixedFrontEnd.run(fe4, chunk(sc16, 4, 4), freq, 4 * @sym)

      assert whole == joined
    end
  end

  describe "stages" do
    test "the stage outputs compose to run/4", %{impaired: x, spec: spec} do
      {y, freq} = acquired(x, 8)
      sc16 = FixedFrontEnd.to_sc16(y, @gain)
      fe = FixedFrontEnd.prepare(spec, 8)
      st = FixedFrontEnd.stages(fe, sc16, freq, 0)

      assert st.bits == FixedFrontEnd.run(fe, sc16, freq)
      assert {re, _im} = st.spectrum
      assert Nx.shape(re.t) == {8, 2048}
      assert {rc, _} = st.corrected
      assert Nx.shape(rc.t) == {8 * @sym}
      assert Nx.shape(st.words) == Nx.shape(st.cells)
      assert Nx.type(st.words) == {:u, 8}

      # the stages also hold with a start offset
      assert FixedFrontEnd.stages(fe, sc16, freq, 5 * @sym).bits ==
               FixedFrontEnd.run(fe, sc16, freq, 5 * @sym)
    end
  end

  describe "against the float front end" do
    test "clean signal: decisions agree", %{clean: clean, spec: spec} do
      y = Nx.slice(clean, [0], [8 * @sym])
      sc16 = FixedFrontEnd.to_sc16(y, @gain)
      fixed = FixedFrontEnd.run(FixedFrontEnd.prepare(spec, 8), sc16, 0.0)

      assert byte_size(fixed) == 8 * 1512 * 2
      assert differing(fixed, float_bits(y, 0.0, spec)) <= 2
    end

    @tag :dvbt
    test "impaired signal, 68 symbols: under 0.5% of code bits differ", %{
      impaired: x,
      spec: spec
    } do
      noisy = Channel.awgn(x, 8.0, 42)
      {y, freq} = acquired(noisy, 68)

      fixed =
        FixedFrontEnd.run(FixedFrontEnd.prepare(spec, 68), FixedFrontEnd.to_sc16(y, @gain), freq)

      count = differing(fixed, float_bits(y, freq, spec))
      assert count < byte_size(fixed) * 0.005, "#{count} of #{byte_size(fixed)} code bits differ"
    end

    test "impaired signal without noise, 8 symbols", %{impaired: x, spec: spec} do
      {y, freq} = acquired(x, 8)

      fixed =
        FixedFrontEnd.run(FixedFrontEnd.prepare(spec, 8), FixedFrontEnd.to_sc16(y, @gain), freq)

      count = differing(fixed, float_bits(y, freq, spec))
      assert count < byte_size(fixed) * 0.001, "#{count} of #{byte_size(fixed)} code bits differ"
    end

    test "float format instance runs and is close to FrontEnd", %{impaired: x, spec: spec} do
      {y, freq} = acquired(x, 4)
      sc16 = FixedFrontEnd.to_sc16(y, @gain)

      fe = FixedFrontEnd.prepare(spec, 4, format: Num.float(), compiler: Nx.Defn.Evaluator)
      bits = FixedFrontEnd.run(fe, sc16, freq)

      assert byte_size(bits) == 4 * 1512 * 2
      assert bits |> :binary.bin_to_list() |> Enum.all?(&(&1 in [0, 1]))
      assert differing(bits, float_bits(y, freq, spec)) < byte_size(bits) * 0.001
    end
  end

  describe "compilers" do
    test "nx_simd, EXLA and the evaluator give identical bits", %{impaired: x, spec: spec} do
      {y, freq} = acquired(x, 4)
      sc16 = FixedFrontEnd.to_sc16(y, @gain)

      run = fn compiler, extra ->
        fe = FixedFrontEnd.prepare(spec, 4, [compiler: compiler] ++ extra)
        FixedFrontEnd.run(fe, sc16, freq)
      end

      reference = run.(Nx.Defn.Evaluator, [])

      simd? = Code.ensure_loaded?(NxSimd) and NxSimd.available?()
      if simd?, do: assert(run.(NxSimd.Compiler, fallback: false) == reference, "nx_simd")
      if Code.ensure_loaded?(EXLA), do: assert(run.(EXLA, []) == reference, "EXLA")
      unless simd? or Code.ensure_loaded?(EXLA), do: IO.puts("no other compiler available")
    end
  end
end
