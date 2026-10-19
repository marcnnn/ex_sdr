defmodule ExSdr.FEC.LDPCDVBS2Test.Support do
  @moduledoc false
  import Bitwise

  alias ExSdr.FEC.LDPC.{Code, DVBS2}

  @ref Path.expand("../fixtures/gnuradio/fec/ldpc_dvb_s2/dvb.json", __DIR__)

  # gr-dtv's dvb_ldpc_bb output for every code (test/gnuradio/fec_ldpc_golden.py dvb)
  def reference, do: @ref |> File.read!() |> JSON.decode!() |> Map.fetch!("codes")

  # The information bits the reference was made with: bit i is bit 27 of
  # (i + 1) * 2654435761 mod 2^32.
  def bits(k), do: for(i <- 1..k, do: (i * 2_654_435_761 &&& 0xFFFFFFFF) >>> 27 &&& 1)

  def code(r) do
    std = if r["standard"] == "all", do: :dvbs2, else: String.to_atom(r["standard"])
    DVBS2.code(String.to_atom(r["frame"]), r["rate"], std)
  end

  def check(r) do
    code = code(r)

    assert_equal = fn a, b, what ->
      if a != b, do: raise("#{r["frame"]} #{r["rate"]}: #{what}")
    end

    assert_equal.({code.n, code.k}, {r["n"], r["k"]}, "n, k")
    s = bits(code.k)
    cw = Code.encode(code, s)
    sha = :crypto.hash(:sha256, :binary.list_to_bin(cw)) |> Base.encode16(case: :lower)
    assert_equal.(sha, r["sha256"], "codeword differs from gr-dtv")
    assert_equal.(Code.codeword?(code, cw), true, "H c != 0")

    if r["parity_hex"] do
      parity = for <<b::1 <- Base.decode16!(r["parity_hex"], case: :lower)>>, do: b
      assert_equal.(Enum.drop(cw, code.k), Enum.take(parity, code.n - code.k), "parity")
    end

    :ok
  end
end

defmodule ExSdr.FEC.LDPCDVBS2Test do
  @moduledoc """
  Structured repeat-accumulate codes from parameters, configured as
  DVB-S2/S2X/T2 (ExSdr.FEC.LDPC.DVBS2): encoders equal to gr-dtv's
  dvb_ldpc_bb, H c = 0, the tensor encoder, decoding a DVB-S2 frame.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.FEC.LDPC
  alias ExSdr.FEC.LDPC.{Code, DVBS2, IRA}
  alias ExSdr.FEC.LDPCDVBS2Test.Support
  alias NxFixed.Num

  test "the tables: 53 codes, rates and sizes consistent" do
    codes = DVBS2.codes()
    assert length(codes) == 53
    assert length(Support.reference()) == 53

    for c <- codes do
      assert c.n == if(c.frame == :normal, do: 64800, else: 16200)
      assert c.q * 360 == c.n - c.k
    end

    assert DVBS2.params(:normal, "2/3", :dvbt2).table != DVBS2.params(:normal, "2/3").table
  end

  test "short frames: equal to gr-dtv's dvb_ldpc_bb, codewords satisfy H" do
    for r <- Support.reference(), r["frame"] == "short", do: assert(Support.check(r) == :ok)
  end

  test "the tensor encoder equals the host encoder (several frames)" do
    for {frame, rate, std} <- [
          {:short, "1/2", :dvbs2},
          {:short, "3/5", :dvbt2},
          {:short, "8/9", :dvbs2}
        ] do
      code = DVBS2.code(frame, rate, std)
      b = LDPC.block(:dvb_ldpc_bb, frame: frame, rate: rate, standard: std)
      assert b.gnuradio == "dtv.dvb_ldpc_bb"
      :rand.seed(:exsss, 3)
      msgs = for _ <- 1..3, do: for(_ <- 1..code.k, do: :rand.uniform(2) - 1)
      {out, _} = LDPC.run(b, Nx.tensor(List.flatten(msgs), type: :u8))
      assert Nx.to_flat_list(out) == Enum.flat_map(msgs, &Code.encode(code, &1))
    end
  end

  property "any IRA parameters: codewords satisfy H, tensor equals host" do
    check all group <- integer(1..6),
              rows <- integer(1..4),
              q <- integer(1..5),
              seed <- integer(0..100_000),
              max_runs: 25 do
      :rand.seed(:exsss, seed)
      m = q * group
      table = for _ <- 1..rows, do: Enum.uniq(for _ <- 1..3, do: :rand.uniform(m) - 1)
      code = IRA.code(n: rows * group + m, k: rows * group, q: q, group: group, table: table)
      msgs = for _ <- 1..3, do: for(_ <- 1..code.k, do: :rand.uniform(2) - 1)

      {out, _} =
        LDPC.run(
          LDPC.block(:ldpc_ira_encoder, code: code),
          Nx.tensor(List.flatten(msgs), type: :u8)
        )

      words = out |> Nx.to_flat_list() |> Enum.chunk_every(code.n)

      for {s, c} <- Enum.zip(msgs, words) do
        assert c == Code.encode(code, s)
        assert Code.codeword?(code, c)
        assert Enum.take(c, code.k) == s
      end
    end
  end

  test "width check of the IRA encoder" do
    code = DVBS2.code(:short, "1/2")
    b = LDPC.block(:dvb_ldpc_bb, frame: :short, rate: "1/2")
    assert {:ok, _} = NxFixed.analyze(b.fun, [Nx.template({2 * code.k}, :u8), {}, b.opts])
  end

  @tag timeout: 300_000
  test "a DVB-S2 short frame (rate 1/2) decodes at 2 dB with 6-bit normalised min-sum" do
    # The short 1/2 code (k = 7200, n = 16200) needs about 1.5 dB with
    # 30 min-sum iterations (measured: 2 of 16 frames wrong at 1.5 dB, none
    # at 2.0 dB); at 2.0 dB a frame error is rare, so allow one in 16.
    code = DVBS2.code(:short, "1/2")
    frames = 16
    :rand.seed(:exsss, 5)
    bits = for _ <- 1..(frames * code.k), do: :rand.uniform(2) - 1
    {cw, _} = LDPC.run(LDPC.block(:ldpc_ira_encoder, code: code), Nx.tensor(bits, type: :u8))
    sigma = :math.sqrt(1 / (2 * code.k / code.n * :math.pow(10, 2.0 / 10)))

    llr =
      Nx.tensor(
        for(c <- Nx.to_flat_list(cw), do: 2 * (2 * c - 1 + sigma * :rand.normal()) / sigma ** 2),
        type: :f32
      )

    f = Num.q(6, 1)
    b = LDPC.block(:ldpc_min_sum_decoder, code: code, max_iter: 30, scale: 0.75, format: f)
    {out, _} = LDPC.run(b, LDPC.quantize_llr(llr, f))

    fe =
      Enum.zip(Nx.to_flat_list(out), bits)
      |> Enum.chunk_every(code.k)
      |> Enum.count(fn fr -> Enum.any?(fr, fn {a, b} -> a != b end) end)

    assert fe <= 1
  end
end

defmodule ExSdr.FEC.LDPCDVBS2NormalTest do
  @moduledoc "Normal frames (n = 64800): every code equal to gr-dtv's dvb_ldpc_bb."
  use ExUnit.Case, async: true

  alias ExSdr.FEC.LDPCDVBS2Test.Support

  @tag timeout: 300_000
  test "normal frames: equal to gr-dtv's dvb_ldpc_bb, codewords satisfy H" do
    refs = for r <- Support.reference(), r["frame"] == "normal", do: r

    results =
      refs
      |> Task.async_stream(&Support.check/1, timeout: :infinity, max_concurrency: 4)
      |> Enum.map(fn {:ok, v} -> v end)

    assert results == List.duplicate(:ok, length(refs))
  end
end
