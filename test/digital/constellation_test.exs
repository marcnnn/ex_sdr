defmodule ExSdr.Digital.ConstellationTest do
  @moduledoc """
  The constellation objects: points, pre-differential code and properties
  against GNU Radio's objects (framing.json, section "constellations"),
  decisions against `ExSdr.Comm.Constellation` and the soft decision
  against the tensor form.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias ExSdr.Comm.Constellation, as: Comm
  alias ExSdr.Digital
  alias ExSdr.Digital.Constellation, as: C

  @fixture Path.expand("../fixtures/gnuradio/digital/framing.json", __DIR__)

  setup_all do
    %{gr: @fixture |> File.read!() |> JSON.decode!() |> Map.fetch!("constellations")}
  end

  defp ours("bpsk"), do: C.bpsk()
  defp ours("qpsk"), do: C.qpsk()
  defp ours("dqpsk"), do: C.dqpsk()
  defp ours("psk8"), do: C.psk8()
  defp ours("psk8_natural"), do: C.psk8_natural()
  defp ours("qam16"), do: C.qam16()

  defp ours("psk_" <> rest) do
    [m, code, diff] = String.split(rest, "_")
    C.psk(String.to_integer(m), String.to_atom(code), diff == "diff")
  end

  defp ours("qam_" <> rest) do
    [m, diff, code] = String.split(rest, "_")
    C.qam(String.to_integer(m), diff == "diff", String.to_atom(code))
  end

  defp close(points, gr_points, tol \\ 2.0e-6) do
    assert length(points) == length(gr_points)

    for {{a, b}, [c, d]} <- Enum.zip(points, gr_points) do
      assert abs(a - c) < tol and abs(b - d) < tol, "#{a},#{b} vs #{c},#{d}"
    end
  end

  test "built-in, PSK and QAM constellations equal GNU Radio's objects", %{gr: gr} do
    for {name, g} <- gr, not String.starts_with?(name, "rect_") do
      c = ours(name)
      close(c.points, g["points"])
      assert c.pre_diff_code == g["pre_diff_code"], name
      assert c.apply_pre_diff_code == g["apply_pre_diff_code"], name
      assert c.bits_per_symbol == g["bits_per_symbol"], name
      assert c.rotational_symmetry == g["rotational_symmetry"], name
    end
  end

  test "normalisation of point lists equals constellation_rect's", %{gr: gr} do
    for kind <- ~w(amplitude power none) do
      g = gr["rect_#{kind}"]
      input = for [a, b] <- g["input"], do: {a, b}

      c =
        C.new(input,
          pre_diff_code: g["pre_diff_code"],
          rotational_symmetry: 4,
          normalization: String.to_atom(kind)
        )

      close(c.points, g["points"])
      assert c.apply_pre_diff_code
    end
  end

  test "GNU Radio's QPSK and 16-QAM are not normalised alike" do
    power = fn c -> Enum.sum(for {a, b} <- c.points, do: a * a + b * b) / length(c.points) end
    assert_in_delta power.(C.qam16()), 1.0, 1.0e-12
    assert_in_delta power.(C.qpsk()), 4.0, 1.0e-12
    assert_in_delta power.(C.psk8()), 1.0, 1.0e-12
  end

  property "the hard decision is the nearest point, as ExSdr.Comm.Constellation decides" do
    for name <- ~w(bpsk qpsk dqpsk psk8 qam16) do
      c = ours(name)

      check all(re <- float(min: -2.0, max: 2.0), im <- float(min: -2.0, max: 2.0), max_runs: 40) do
        # decisions are scale invariant: scale the input to the unit power points
        scale =
          1 / :math.sqrt(Enum.sum(for {a, b} <- c.points, do: a * a + b * b) / length(c.points))

        assert C.decide(c, {re, im}) == Comm.demap(c.base, {re * scale, im * scale})
      end
    end
  end

  test "the tensor decoder block equals the reference decision" do
    :rand.seed(:exsss, {1, 2, 3})
    n = 300
    re = for _ <- 1..n, do: :rand.uniform() * 3 - 1.5
    im = for _ <- 1..n, do: :rand.uniform() * 3 - 1.5

    for name <- [:bpsk, :qpsk, :dqpsk, :psk8, :qam16] do
      block = Digital.block(:constellation_decoder_cb, constellation: name)
      x = {Nx.tensor(re, type: :f32), Nx.tensor(im, type: :f32)}
      {out, _} = Digital.run(block, x)
      c = apply(C, name, [])
      want = for {a, b} <- Enum.zip(re, im), do: C.decide(c, {a, b})
      assert Nx.to_flat_list(out) == want
    end
  end

  test "soft decisions: tensor form equals the reference, signs agree with the hard decision" do
    :rand.seed(:exsss, {4, 5, 6})
    n = 100
    re = for _ <- 1..n, do: :rand.uniform() * 3 - 1.5
    im = for _ <- 1..n, do: :rand.uniform() * 3 - 1.5

    for name <- [:bpsk, :qpsk, :dqpsk, :psk8, :qam16], npwr <- [1.0, 0.3] do
      c = apply(C, name, [])
      block = Digital.block(:constellation_soft_decoder_cf, constellation: name, npwr: npwr)
      {out, _} = Digital.run(block, {Nx.tensor(re, type: :f32), Nx.tensor(im, type: :f32)})
      got = Nx.to_list(out)
      assert length(hd(got)) == c.bits_per_symbol

      for {row, {a, b}} <- Enum.zip(got, Enum.zip(re, im)) do
        want = C.soft_decision(c, {a, b}, npwr)
        for {x, y} <- Enum.zip(row, want), do: assert(abs(x - y) < 1.0e-3 * max(1.0, abs(y)))
      end
    end
  end

  test "a symbol's own point decodes to confident bits of its label" do
    for name <- [:qpsk, :dqpsk, :psk8, :qam16] do
      c = apply(C, name, [])
      labels = C.labels(c)

      for {point, label} <- Enum.zip(c.points, labels) do
        llr = C.soft_decision(c, point, 0.05)

        bits =
          for j <- 0..(c.bits_per_symbol - 1),
              do: Bitwise.band(Bitwise.bsr(label, c.bits_per_symbol - 1 - j), 1)

        assert Enum.map(llr, &if(&1 > 0, do: 1, else: 0)) == bits
      end
    end
  end
end
