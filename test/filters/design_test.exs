defmodule ExSdr.Filters.DesignTest do
  @moduledoc """
  Windows and firdes taps against GNU Radio 3.10 (test/fixtures/gnuradio/filters/taps.json,
  made by test/gnuradio/filters_taps.py).

  Bound: GNU Radio returns f32 taps, so 3e-7 absolute (taps are below 1.1;
  f32 spacing at 1 is 1.2e-7, plus its f32 intermediate sums). Complex band
  pass: GNU Radio advances the rotation phase in f32 (`phase += freq`), so its
  error grows with the tap index; 5e-6 covers up to 105 taps.
  """
  use ExUnit.Case, async: true
  alias ExSdr.Filters.{Firdes, Window}

  @entries Path.expand("../fixtures/gnuradio/filters/taps.json", __DIR__)
           |> File.read!()
           |> JSON.decode!()
           |> Map.fetch!("entries")

  defp t(name), do: String.to_atom(name)

  defp design("window", [type, n, beta]), do: Window.build(t(type), n, beta)

  defp design("low_pass", [g, fs, c, tw, w, b]), do: Firdes.low_pass(g, fs, c, tw, t(w), b)
  defp design("high_pass", [g, fs, c, tw, w, b]), do: Firdes.high_pass(g, fs, c, tw, t(w), b)

  defp design("band_pass", [g, fs, l, h, tw, w, b]),
    do: Firdes.band_pass(g, fs, l, h, tw, t(w), b)

  defp design("band_reject", [g, fs, l, h, tw, w, b]),
    do: Firdes.band_reject(g, fs, l, h, tw, t(w), b)

  defp design("complex_band_pass", [g, fs, l, h, tw, w, b]),
    do: Firdes.complex_band_pass(g, fs, l, h, tw, t(w), b) |> Enum.map(&Tuple.to_list/1)

  defp design("hilbert", [n, w, b]), do: Firdes.hilbert(n, t(w), b)

  defp design("root_raised_cosine", [g, fs, r, a, n]),
    do: Firdes.root_raised_cosine(g, fs, r, a, n)

  defp design("gaussian", [g, spb, bt, n]), do: Firdes.gaussian(g, spb, bt, n)

  for {entry, i} <- Enum.with_index(@entries) do
    @entry entry
    test "#{i}: #{entry["fun"]} #{inspect(entry["args"])} equals GNU Radio" do
      %{"fun" => fun, "args" => args, "taps" => reference} = @entry
      got = design(fun, args)
      assert length(List.flatten(got)) == length(List.flatten(reference))
      bound = if fun == "complex_band_pass", do: 5.0e-6, else: 3.0e-7

      worst =
        Enum.zip(List.flatten(got), List.flatten(reference))
        |> Enum.map(fn {a, b} -> abs(a - b) end)
        |> Enum.max()

      assert worst <= bound, "#{fun} #{inspect(args)}: off by #{worst}"
    end
  end

  test "design properties: gains" do
    lp = Firdes.low_pass(2.0, 1.0, 0.1, 0.05)
    assert_in_delta Enum.sum(lp), 2.0, 1.0e-12
    hp = Firdes.high_pass(1.0, 1.0, 0.2, 0.05)
    nyq = hp |> Enum.with_index() |> Enum.map(fn {v, i} -> v * :math.pow(-1, i) end) |> Enum.sum()
    assert_in_delta nyq, 1.0, 1.0e-12
    assert_in_delta Enum.sum(Firdes.root_raised_cosine(3.0, 4.0, 1.0, 0.35, 33)), 3.0, 1.0e-12
    assert length(Firdes.low_pass(1.0, 1.0, 0.1, 0.05)) == 49
  end
end
