defmodule ExSdr.FEC.PolarTest do
  @moduledoc """
  gr-fec's polar codes against GNU Radio's output
  (test/fixtures/gnuradio/fec/polar.json, made by test/gnuradio/fec_polar_golden.py):
  frozen bit constructions, encoders (unpacked, packed, systematic) and the
  three decoders on the same soft input; bits must be equal. Then properties
  of the codes and a seeded error rate comparison of the two list modes.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.FEC.Generic
  alias ExSdr.FEC.Polar
  alias ExSdr.FEC.PolarDecoder

  @fixture Path.expand("../fixtures/gnuradio/fec/polar.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!()}
  end

  defp bits(s), do: for(<<c <- s>>, do: c - ?0)
  defp bit_string(t), do: t |> Generic.to_binary() |> :binary.bin_to_list() |> Enum.join()
  defp hex(s), do: Base.decode16!(s, case: :lower)
  defp f32s(hex), do: hex |> hex() |> Nx.from_binary(:f32)

  describe "constructions" do
    test "frozen positions, Z parameters and capacities equal gr-fec's", %{gr: gr} do
      for c <- gr["constructions"] do
        assert Polar.frozen_bit_positions(c["n"], c["k"], design_snr: c["design_snr"]) ==
                 c["frozen"],
               "n #{c["n"]} k #{c["k"]}"

        eta = Polar.design_snr_to_bec_eta(c["design_snr"])
        z = Polar.z_parameters(eta, c["n"])
        cap = Polar.bec_capacities(eta, c["n"])

        # (the parameters are stored for the small codes only; zip stops at the shorter)
        for {a, b} <- Enum.zip(z, c["z"]), do: assert_in_delta(a, b, 1.0e-12)
        for {a, b} <- Enum.zip(cap, c["capacities"]), do: assert_in_delta(a, b, 1.0e-12)

        # the Bhattacharyya choice is the same set up to rounding
        by_z =
          Polar.frozen_bit_positions(c["n"], c["k"],
            design_snr: c["design_snr"],
            method: :bhattacharyya
          )

        assert length(by_z -- c["frozen"]) <= 2
      end
    end

    test "DEGA freezes almost the same channels as the BEC rule and the worst first" do
      for {n, k} <- [{64, 32}, {256, 128}, {1024, 512}] do
        bec = Polar.frozen_bit_positions(n, k, design_snr: 0.0)
        dega = Polar.frozen_bit_positions(n, k, design_snr: 0.0, method: :dega)
        assert length(dega) == n - k
        assert length(dega -- bec) <= div(n, 8)
        # channel 0 is the worst, channel n - 1 the best
        assert hd(dega) == 0
        refute (n - 1) in dega
      end

      means = Polar.dega_means(0.0, 64)
      assert Enum.max(means) == List.last(means)
      assert Enum.min(means) == hd(means)
    end
  end

  describe "encoders" do
    test "equal GNU Radio (unpacked, packed, systematic)", %{gr: gr} do
      for c <- gr["encoders"] do
        opts = [
          block_size: c["n"],
          num_info_bits: c["k"],
          frozen_bit_positions: c["frozen"],
          frozen_bit_values: c["values"],
          packed: c["packed"]
        ]

        enc = if c["systematic"], do: Polar.encoder_systematic(opts), else: Polar.encoder(opts)

        assert Generic.input_size(enc) == c["input_size"]
        assert Generic.output_size(enc) == c["output_size"]
        assert Generic.rate(enc) == c["rate"]
        assert Generic.input_conversion(enc) == c["input_conversion"]
        assert Generic.output_conversion(enc) == c["output_conversion"]

        for {i, o} <- Enum.zip(c["inputs"], c["outputs"]) do
          input =
            if c["packed"],
              do: Nx.from_binary(hex(i), :u8),
              else: Nx.tensor(bits(i), type: :u8)

          {out, _} = Generic.work(enc, input, nil)

          got =
            if c["packed"],
              do: out |> Generic.to_binary() |> Base.encode16(case: :lower),
              else: bit_string(out)

          assert got == o, "n #{c["n"]} packed #{c["packed"]} systematic #{c["systematic"]}"
        end
      end
    end
  end

  describe "decoders" do
    test "decide like GNU Radio on the same input", %{gr: gr} do
      for c <- gr["decoders"] do
        kind =
          case c["kind"] do
            "sc" -> :sc
            "list" -> :sc_list
            "systematic" -> :sc_systematic
          end

        dec =
          PolarDecoder.decoder(kind,
            block_size: c["n"],
            num_info_bits: c["k"],
            frozen_bit_positions: c["frozen"],
            frozen_bit_values: c["values"],
            max_list_size: max(c["list"], 1)
          )

        assert Generic.input_size(dec) == c["input_size"]
        assert Generic.output_size(dec) == c["output_size"]
        assert Generic.rate(dec) == c["rate"]

        for {i, o} <- Enum.zip(c["inputs"], c["outputs"]) do
          {out, _} = Generic.work(dec, f32s(i), nil)

          assert bit_string(out) == o,
                 "#{c["kind"]} n #{c["n"]} list #{c["list"]} sigma #{c["sigma"]}"
        end
      end
    end
  end

  describe "properties" do
    defp code(n, k, systematic) do
      frozen = Polar.frozen_bit_positions(n, k)
      opts = [block_size: n, num_info_bits: k, frozen_bit_positions: frozen]

      {if(systematic, do: Polar.encoder_systematic(opts), else: Polar.encoder(opts)),
       PolarDecoder.decoder(if(systematic, do: :sc_systematic, else: :sc), opts),
       PolarDecoder.decoder(:sc_list, [max_list_size: 4] ++ opts)}
    end

    property "noiseless codewords decode to the info bits (SC, list, systematic)" do
      check all(
              power <- StreamData.integer(3..8),
              rate <- StreamData.member_of([0.25, 0.5, 0.75]),
              systematic <- StreamData.boolean(),
              seed <- StreamData.integer(),
              max_runs: 25
            ) do
        n = Bitwise.<<<(1, power)
        k = max(1, round(n * rate))
        {enc, dec, list} = code(n, k, systematic)
        :rand.seed(:exsss, {seed, 1, 2})
        info = for _ <- 1..k, do: :rand.uniform(2) - 1
        {x, _} = Generic.work(enc, Nx.tensor(info, type: :u8), nil)
        soft = x |> Nx.as_type(:f32) |> Nx.multiply(2) |> Nx.subtract(1)

        for d <- if(systematic, do: [dec], else: [dec, list]) do
          {out, _} = Generic.work(d, soft, nil)
          assert Nx.to_flat_list(out) == info
        end
      end
    end

    test "a systematic codeword carries the info bits" do
      n = 64
      k = 32
      {enc, _, _} = code(n, k, true)
      info = for i <- 1..k, do: rem(i * 7, 3) |> min(1)
      {x, _} = Generic.work(enc, Nx.tensor(info, type: :u8), nil)
      x = Nx.to_flat_list(x) |> List.to_tuple()

      positions =
        (Enum.to_list(0..(n - 1)) -- Polar.frozen_bit_positions(n, k))
        |> Enum.map(&Polar.bit_reverse(&1, 6))

      assert Enum.map(positions, &elem(x, &1)) == info
    end

    test "the transform is its own inverse" do
      for n <- [8, 64, 512] do
        v = Nx.tensor(for(i <- 1..n, do: rem(i * i + 3, 2)), type: :u8)
        assert Polar.transform(Polar.transform(v)) == v
      end
    end
  end

  describe "list decoder modes" do
    test "the exact list decoder differs from GNU Radio only in rare frames", %{gr: gr} do
      {frames, differing} =
        for c <- gr["decoders"],
            c["kind"] == "list",
            {i, o} <- Enum.zip(c["inputs"], c["outputs"]),
            reduce: {0, 0} do
          {n, d} ->
            dec =
              PolarDecoder.decoder(:sc_list,
                block_size: c["n"],
                num_info_bits: c["k"],
                frozen_bit_positions: c["frozen"],
                frozen_bit_values: c["values"],
                max_list_size: c["list"],
                list_mode: :exact
              )

            {out, _} = Generic.work(dec, f32s(i), nil)
            {n + 1, if(bit_string(out) == o, do: d, else: d + 1)}
        end

      # GNU Radio's buffer takeover flaw shows in a few frames (see docs/blocks/fec.md)
      assert differing >= 1
      assert differing * 20 <= frames
    end

    test "the textbook list decoder is not worse than GNU Radio's bookkeeping (seeded)" do
      n = 128
      k = 64
      frozen = Polar.frozen_bit_positions(n, k)
      opts = [block_size: n, num_info_bits: k, frozen_bit_positions: frozen, max_list_size: 4]
      enc = Polar.encoder(opts)
      gr_mode = PolarDecoder.decoder(:sc_list, opts)
      exact = PolarDecoder.decoder(:sc_list, [list_mode: :exact] ++ opts)
      sc = PolarDecoder.decoder(:sc, opts)
      :rand.seed(:exsss, {2024, 5, 6})

      {sc_err, gr_err, ex_err} =
        Enum.reduce(1..120, {0, 0, 0}, fn _, {a, b, c} ->
          info = for _ <- 1..k, do: :rand.uniform(2) - 1
          {x, _} = Generic.work(enc, Nx.tensor(info, type: :u8), nil)

          soft =
            x
            |> Nx.to_flat_list()
            |> Enum.map(fn bit -> 2 * bit - 1 + :rand.normal() * 0.8 end)
            |> Nx.tensor(type: :f32)

          wrong = fn d ->
            {out, _} = Generic.work(d, soft, nil)
            if Nx.to_flat_list(out) == info, do: 0, else: 1
          end

          {a + wrong.(sc), b + wrong.(gr_mode), c + wrong.(exact)}
        end)

      # list decoding gains over SC, and the exact list decoder over the quirk
      assert ex_err < sc_err
      assert ex_err <= gr_err
      assert gr_err <= sc_err + 3
    end
  end
end
