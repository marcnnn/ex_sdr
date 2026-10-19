defmodule ExSdr.Digital.ModulationTest do
  @moduledoc """
  Tap generators against GNU Radio's firdes, and the stream properties of
  the modulators (the sample level agreement with GNU Radio is in
  gnuradio_cases_test.exs).
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Digital
  alias ExSdr.Digital.Modulation
  alias NxFixed.Num

  @fixture Path.expand("../fixtures/gnuradio/digital/framing.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!() |> Map.fetch!("taps")}
  end

  test "rrc_taps equals firdes.root_raised_cosine", %{gr: gr} do
    for %{"args" => [gain, fs, sr, alpha, ntaps], "taps" => taps} <- gr["rrc"] do
      mine = Modulation.rrc_taps(gain, fs, sr, alpha, ntaps)
      assert length(mine) == length(taps)
      worst = Enum.zip_with(mine, taps, &abs(&1 - &2)) |> Enum.max()

      assert worst < 2.0e-7 * max(1, gain),
             "rrc #{inspect([gain, fs, sr, alpha, ntaps])}: #{worst}"
    end
  end

  test "gaussian_taps equals firdes.gaussian", %{gr: gr} do
    for %{"args" => [gain, spb, bt, ntaps], "taps" => taps} <- gr["gaussian"] do
      mine = Modulation.gaussian_taps(gain, spb, bt, ntaps)
      worst = Enum.zip_with(mine, taps, &abs(&1 - &2)) |> Enum.max()

      assert worst < 1.0e-7 * max(1, gain),
             "gaussian #{inspect([gain, spb, bt, ntaps])}: #{worst}"
    end
  end

  test "the pulse of generic_mod is every (32 / sps)-th tap of the prototype" do
    proto = Modulation.rrc_taps(32, 32, 1.0, 0.35, 32 * 11 * 4)
    assert Modulation.rrc_pulse(4, 0.35) == Enum.take_every(proto, 8)
    assert length(Modulation.rrc_pulse(2, 0.35)) == 11 * 2 * 2 + 1
  end

  test "gaussian_pulse sums to sps" do
    for sps <- [2, 4, 8] do
      assert_in_delta Enum.sum(Modulation.gaussian_pulse(sps, 0.35)), sps, 1.0e-12
    end
  end

  defp concat({a, b}, {c, d}), do: {Nx.concatenate([a, c]), Nx.concatenate([b, d])}

  defp same?({a, b}, {c, d}),
    do: Nx.to_flat_list(a) == Nx.to_flat_list(c) and Nx.to_flat_list(b) == Nx.to_flat_list(d)

  describe "modulators in pieces" do
    for {name, opts} <- [
          {:psk_mod, [constellation_points: 8, samples_per_symbol: 4]},
          {:psk_mod, [constellation_points: 4, differential: false]},
          {:qam_mod, [constellation_points: 16, samples_per_symbol: 2]},
          {:gfsk_mod, [samples_per_symbol: 4]},
          {:gmsk_mod, [samples_per_symbol: 2]}
        ] do
      @name name
      @opts opts
      property "#{name} #{inspect(opts)}: cutting the input anywhere gives the same samples" do
        block = Digital.block(@name, @opts)

        check all(
                data <- binary(min_length: 3, max_length: 30),
                cut <- integer(1..29),
                max_runs: 12
              ) do
          cut = min(cut, byte_size(data) - 1)
          <<a::binary-size(cut), b::binary>> = data
          {whole, _} = Digital.run(block, data)
          {y1, st} = Digital.run(block, a)
          {y2, _} = Digital.run(block, b, st)
          assert same?(concat(y1, y2), whole)
        end
      end
    end

    test "fixed point modulators are split invariant too" do
      f = Num.q(16, 14)
      block = Digital.block(:psk_mod, constellation_points: 4, samples_per_symbol: 4, format: f)
      data = :crypto.strong_rand_bytes(20)
      <<a::binary-size(7), b::binary>> = data
      {{wr, wi}, _} = Digital.run(block, data)
      {{r1, i1}, st} = Digital.run(block, a)
      {{r2, i2}, _} = Digital.run(block, b, st)
      assert Nx.to_flat_list(Nx.concatenate([r1.t, r2.t])) == Nx.to_flat_list(wr.t)
      assert Nx.to_flat_list(Nx.concatenate([i1.t, i2.t])) == Nx.to_flat_list(wi.t)
    end

    test "8-PSK keeps the bits of a partial symbol between calls" do
      block = Digital.block(:psk_mod, constellation_points: 8, samples_per_symbol: 2)
      # 1 byte = 8 bits = 2 symbols + 2 bits pending
      {y1, st} = Digital.run(block, <<0xAB>>)
      assert Nx.size(elem(y1, 0)) == 4
      assert byte_size(st.bits) == 2
    end
  end

  test "gfsk_mod with unpacked input equals packed input unpacked" do
    block = Digital.block(:gfsk_mod, samples_per_symbol: 2)
    bits = block_bits = ExSdr.Comm.Bits.unpack(<<0xA5, 0x3C>>)
    unpacked = Digital.block(:gfsk_mod, samples_per_symbol: 2, do_unpack: false)
    {a, _} = Digital.run(block, <<0xA5, 0x3C>>)
    {b, _} = Digital.run(unpacked, bits)
    assert block_bits == bits
    assert same?(a, b)
  end
end
