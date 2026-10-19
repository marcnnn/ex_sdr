defmodule ExSdr.FEC.PunctureTest do
  @moduledoc """
  puncture_bb, puncture_ff, depuncture_bb and ber_bf against GNU Radio
  (test/fixtures/gnuradio/fec/misc.json): bytes and floats must be equal.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.FEC.Ber
  alias ExSdr.FEC.Puncture

  @fixture Path.expand("../fixtures/gnuradio/fec/misc.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!()}
  end

  defp hex(s), do: Base.decode16!(s, case: :lower)
  defp enc(b), do: Base.encode16(b, case: :lower)

  describe "puncture and depuncture" do
    test "equal GNU Radio for patterns, delays and symbols", %{gr: gr} do
      for c <- gr["puncture"] do
        size = c["puncsize"]
        pat = c["puncpat"]
        delay = c["delay"]

        data = Nx.from_binary(hex(c["data"]), :u8)
        punctured = Puncture.puncture(data, size, pat, delay)
        assert enc(Nx.to_binary(punctured)) == c["punctured"], "#{size} #{pat} #{delay}"
        assert Puncture.kept(Puncture.pattern(size, pat, delay)) == c["kept"]

        floats = Nx.from_binary(hex(c["float_in"]), :f32)
        assert enc(Nx.to_binary(Puncture.puncture(floats, size, pat, delay))) == c["float_out"]

        assert enc(Nx.to_binary(Puncture.depuncture(punctured, size, pat, delay))) ==
                 c["depunctured"]

        assert enc(Nx.to_binary(Puncture.depuncture(punctured, size, pat, delay, 55))) ==
                 c["depunctured_55"]
      end
    end

    test "as blocks: rate, item types and the stream in pieces", %{gr: gr} do
      c = hd(gr["puncture"])
      b = ExSdr.FEC.block(:puncture_bb, puncsize: c["puncsize"], puncpat: c["puncpat"], delay: c["delay"])
      assert b.rate == {c["puncsize"], c["kept"]}
      assert b.hardware == :stateless
      data = Nx.from_binary(hex(c["data"]), :u8)
      {out, _} = ExSdr.FEC.run(b, data)
      assert enc(Nx.to_binary(out)) == c["punctured"]

      d = ExSdr.FEC.block(:depuncture_bb, puncsize: c["puncsize"], puncpat: c["puncpat"], delay: c["delay"])
      assert d.rate == {c["kept"], c["puncsize"]}
      {back, _} = ExSdr.FEC.run(d, out)
      assert enc(Nx.to_binary(back)) == c["depunctured"]
    end

    property "puncture then depuncture restores the kept positions" do
      check all(
              size <- StreamData.integer(2..12),
              pat <- StreamData.integer(1..4095),
              delay <- StreamData.integer(0..14),
              groups <- StreamData.integer(1..5),
              seed <- StreamData.integer(),
              max_runs: 60
            ) do
        pattern = Puncture.pattern(size, pat, delay)
        keep? = Puncture.kept(pattern) > 0

        if keep? do
          :rand.seed(:exsss, {seed, 3, 4})
          items = for _ <- 1..(size * groups), do: :rand.uniform(200)
          t = Nx.tensor(items, type: :u8)
          back = t |> Puncture.puncture(size, pat, delay) |> Puncture.depuncture(size, pat, delay, 0)
          expected = for {x, i} <- Enum.with_index(items), do: if(Enum.at(pattern, rem(i, size)) == 1, do: x, else: 0)
          assert Nx.to_flat_list(back) == expected
        end
      end
    end
  end

  describe "ber_bf" do
    test "streaming value equals GNU Radio's log10 BER", %{gr: gr} do
      for c <- gr["ber"] do
        s = Ber.new()
        {v, s} = Ber.update(s, hex(c["a"]), hex(c["b"]))
        assert Ber.bit_errors(hex(c["a"]), hex(c["b"])) == c["errors"]
        assert s.errors == c["errors"]
        # GNU Radio's output is a float
        assert_in_delta v, c["last"], 1.0e-6
      end
    end

    test "the value does not depend on how the stream is cut", %{gr: gr} do
      c = hd(gr["ber"])
      {a, b} = {hex(c["a"]), hex(c["b"])}

      {v, _} =
        Enum.reduce(0..(byte_size(a) - 1)//7, {nil, Ber.new()}, fn i, {_, s} ->
          n = min(7, byte_size(a) - i)
          Ber.update(s, binary_part(a, i, n), binary_part(b, i, n))
        end)

      assert_in_delta v, c["last"], 1.0e-6
    end

    test "no errors give minus infinity, no data gives nothing" do
      assert {:neg_infinity, _} = Ber.update(Ber.new(), <<1, 2>>, <<1, 2>>)
      assert {nil, _} = Ber.update(Ber.new(), <<>>, <<>>)
    end

    test "test mode waits for the minimum number of errors, then stops" do
      s = Ber.new(test_mode: true, berminerrors: 20)
      {nil, s} = Ber.update(s, <<0xFF, 0>>, <<0, 0>>)
      assert s.errors == 8
      {v, s} = Ber.update(s, <<0xFF, 0xFF>>, <<0, 0>>)
      assert_in_delta v, :math.log10(24 / 32), 1.0e-12
      assert {:done, _} = Ber.update(s, <<1>>, <<0>>)
    end

    test "test mode returns the limit when the rate is below it" do
      s = Ber.new(test_mode: true, berminerrors: 1000, ber_limit: -2.0)
      {v, s} = Ber.update(s, :binary.copy(<<0>>, 200) <> <<1>>, :binary.copy(<<0>>, 201))
      assert v == -2.0
      assert {:done, _} = Ber.update(s, <<0>>, <<0>>)
    end

    test "as a block" do
      b = ExSdr.FEC.block(:ber_bf, [])
      {v, _} = ExSdr.FEC.run(b, {<<0b1010>>, <<0b0110>>})
      assert_in_delta v, :math.log10(2 / 8), 1.0e-12
    end
  end
end
