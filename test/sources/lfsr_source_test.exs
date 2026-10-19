defmodule ExSdr.Sources.LfsrSourceTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Comm.LFSR
  alias ExSdr.Sources
  alias ExSdr.Sources.LfsrSource

  describe "default masks" do
    test "GNU Radio's mask for degree 5 is the Galois form of x^5 + x^3 + 1" do
      assert LfsrSource.default_mask(5) == 0x12
      assert LfsrSource.default_mask(8) == 0x8E
    end

    test "every default register (degrees 2-16) has the full period 2^degree - 1" do
      for degree <- 2..16 do
        reg = LfsrSource.new(degree, LfsrSource.default_mask(degree), 1)
        {bits, back} = LFSR.bits(reg, 2 ** degree - 1)
        assert back.state == reg.state, "degree #{degree} period"
        assert Enum.sum(bits) == 2 ** (degree - 1), "degree #{degree} balance"
      end
    end
  end

  describe "Fibonacci form equals GNU Radio's Galois register" do
    property "any mask and seed, 200 bits" do
      check all(
              degree <- integer(3..12),
              mask <- integer(1..(2 ** degree - 1)),
              seed <- integer(1..(2 ** degree - 1)),
              max_runs: 60
            ) do
        {bits, _} = LFSR.bits(LfsrSource.new(degree, mask, seed), 200)
        assert bits == LfsrSource.galois_bits(seed, mask, 200)
      end
    end
  end

  describe "block" do
    test "contract" do
      b = Sources.block(:glfsr_source_b, degree: 7)
      assert %NxFixed.Block{kind: :host, item: {:none, :bit}, hardware: :feedback} = b
      assert b.gnuradio == "digital.glfsr_source_b"
    end

    test "pieces give the same bits as one call; the seed matters" do
      b = Sources.block(:glfsr_source_b, degree: 11, seed: 77)
      {whole, _} = Sources.generate(b, 300)
      {p1, s} = Sources.generate(b, 123)
      {p2, _} = Sources.generate(b, 177, s)
      assert whole == p1 <> p2
      {other, _} = Sources.generate(Sources.block(:glfsr_source_b, degree: 11, seed: 78), 300)
      refute whole == other
    end
  end
end
