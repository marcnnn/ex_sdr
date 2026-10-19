Code.require_file("support/helper.exs", __DIR__)

defmodule ExSdr.Carrier.EstimatorsTest do
  @moduledoc """
  Correlation estimator, SNR estimators and probes against GNU Radio's
  tags and probe values (cases `"compare": "corr_est" | "snr" | "probe_snr" | "probe"`
  in test/dsp_cases/carrier), against the float instance for the fixed-point
  variants, and split invariance. The skew estimator, whose GNU Radio value
  depends on its buffer sizes, is checked against an independent double
  precision implementation of the same recurrence.
  """
  use ExUnit.Case, async: true

  alias ExSdr.Carrier.Estimators
  alias ExSdr.Carrier.TestHelper, as: H
  alias NxFixed.Num

  @moduletag timeout: 600_000

  defp to_floats(%NxFixed.Fixed{t: t} = x), do: Num.to_float(%{x | t: Nx.reshape(t, {:auto})})
  defp to_floats(%Nx.Tensor{} = t), do: t |> Nx.reshape({:auto}) |> Nx.to_flat_list()

  # -- corr_est_cc ---------------------------------------------------------------------

  # `{detections, corr, delayed, raw}`: detections are the items that report a peak
  defp run_corr(case, data, format, state \\ :init, input \\ nil) do
    b = H.block(case, format)
    {out, state} = ExSdr.Carrier.run(b, input || H.input(case, data, format), state)
    {delayed, corr, det, phase, time_est, mag} = out
    flags = to_floats(det)
    pick = fn list -> for {v, 1} <- Enum.zip(list, flags), do: v end

    detections =
      Enum.zip([
        for({1, i} <- Enum.with_index(flags), do: i - 1),
        pick.(H.words(phase)),
        pick.(to_floats(time_est)),
        pick.(to_floats(mag))
      ])

    {detections, H.items(corr), H.items(delayed), {out, state}}
  end

  setup_all do
    runs =
      for {path, _} <- H.cases_of("corr_est"), into: %{} do
        {case, data} = H.load(path)
        {det, corr, delayed, _} = run_corr(case, data, Num.float())

        fixed =
          for {v, f} <- H.fixed_formats(case), into: %{} do
            {v["variant"], run_corr(case, data, f)}
          end

        {case["name"], %{case: case, data: data, float: {det, corr, delayed}, fixed: fixed}}
      end

    %{corr_runs: runs}
  end

  for {_path, case} <- H.cases_of("corr_est") do
    @name case["name"]

    test "#{case["name"]}: float instance vs GNU Radio tags (#{case["description"]})", %{
      corr_runs: runs
    } do
      %{case: case, data: data, float: {det, corr, delayed}} = runs[@name]
      b = case["float_vs_gnuradio"]
      md = case["params"]["mark_delay"]
      # GNU Radio stops at a multiple of its FFT filter's block size
      n_gr = length(Enum.at(data.ports, 0))
      assert n_gr > 0.9 * length(data.input)
      det = Enum.filter(det, fn {p, _, _, _} -> p < n_gr end)
      tags = data.tags["tags"]
      by_key = fn key -> for t <- tags, t["key"] == key, do: t end

      assert Enum.map(det, &elem(&1, 0)) == Enum.map(by_key.("corr_start"), & &1["offset"])
      assert length(det) > 0

      for key <- ~w(phase_est time_est corr_est) do
        assert Enum.map(by_key.(key), & &1["offset"]) == Enum.map(det, &(elem(&1, 0) + md))
      end

      for {{_, phase, time_est, mag}, ph, te, ce} <-
            Enum.zip([det, by_key.("phase_est"), by_key.("time_est"), by_key.("corr_est")]) do
        assert abs(phase - ph["value"]) <= b["phase"]
        assert abs(time_est - te["value"]) <= b["time_est"]
        assert abs(mag - ce["value"]) / ce["value"] <= b["corr_est_rel"]
      end

      assert H.max_diff(Enum.at(data.ports, 1), Enum.take(corr, n_gr)) <= b["corr"]
      assert Enum.at(data.ports, 0) == Enum.take(delayed, n_gr)
    end

    for {variant, _f} <- H.fixed_formats(case) do
      @variant variant
      test "#{case["name"]}: #{variant["variant"]} vs float instance", %{corr_runs: runs} do
        %{float: {det, corr, delayed}, fixed: fixed} = runs[@name]
        {fdet, fcorr, fdelayed, _} = fixed[@variant["variant"]]

        assert Enum.map(fdet, &elem(&1, 0)) == Enum.map(det, &elem(&1, 0))
        assert H.max_diff(corr, fcorr) <= @variant["corr"]
        assert fdelayed == delayed

        for {{_, p1, t1, m1}, {_, p2, t2, m2}} <- Enum.zip(det, fdet) do
          assert abs(:math.atan2(:math.sin(p1 - p2), :math.cos(p1 - p2))) <= @variant["phase"]
          assert abs(t1 - t2) <= @variant["time_est"]
          assert abs(m1 - m2) / m1 <= @variant["corr_est_rel"]
        end
      end
    end

    test "#{case["name"]}: split into pieces gives the same output", %{corr_runs: runs} do
      %{case: case, data: data, fixed: fixed} = runs[@name]
      v = hd(case["fixed_vs_float"])
      f = H.format(v["format"])
      {_, _, _, {whole, _}} = fixed[v["variant"]]
      n = length(data.input)
      input = H.input(case, data, f)
      cut = div(n, 3)

      {_, _, _, {a, state}} = run_corr(case, data, f, :init, slice(input, 0, cut))
      {_, _, _, {b, _}} = run_corr(case, data, f, state, slice(input, cut, n - cut))

      joined =
        Tuple.to_list(a)
        |> Enum.zip(Tuple.to_list(b))
        |> Enum.map(fn {x, y} -> concat(x, y) end)
        |> List.to_tuple()

      assert flat(joined) == flat(whole)
    end
  end

  test "the hold-off after a peak is round(sps) samples" do
    {path, _} = hd(for {p, c} <- H.cases_of("corr_est"), c["params"]["sps"] == 4, do: {p, c})
    {case, data} = H.load(path)
    {det, _, _, _} = run_corr(case, data, Num.float())
    peaks = Enum.map(det, &elem(&1, 0))

    assert peaks |> Enum.chunk_every(2, 1, :discard) |> Enum.all?(fn [a, b] -> b - a >= 4 end)
  end

  defp slice({a, b}, from, len), do: {slice(a, from, len), slice(b, from, len)}
  defp slice(%NxFixed.Fixed{t: t} = x, from, len), do: %{x | t: Nx.slice_along_axis(t, from, len)}
  defp slice(%Nx.Tensor{} = t, from, len), do: Nx.slice_along_axis(t, from, len)

  defp concat({a, b}, {c, d}), do: {concat(a, c), concat(b, d)}

  defp concat(%NxFixed.Fixed{t: t, frac: f}, %NxFixed.Fixed{t: u}),
    do: %NxFixed.Fixed{t: Nx.concatenate([t, u]), frac: f}

  defp concat(%Nx.Tensor{} = t, %Nx.Tensor{} = u), do: Nx.concatenate([t, u])

  defp flat(out), do: out |> Tuple.to_list() |> Enum.map(&flat_part/1)
  defp flat_part({a, b}), do: {flat_part(a), flat_part(b)}
  defp flat_part(%NxFixed.Fixed{t: t}), do: Nx.to_flat_list(t)
  defp flat_part(%Nx.Tensor{} = t), do: Nx.to_flat_list(t)

  # -- SNR estimators ------------------------------------------------------------------------

  # the registers after each item, as lists of floats per register
  defp run_snr(case, data, format, state \\ :init, input \\ nil) do
    b = H.block(case, format)
    {regs, state} = ExSdr.Carrier.run(b, input || H.input(case, data, format), state)
    {regs |> Tuple.to_list() |> Enum.map(&to_floats/1), regs, state}
  end

  defp regs_at(regs, i), do: regs |> Enum.map(&Enum.at(&1, i)) |> List.to_tuple()

  setup_all do
    runs =
      for {path, _} <- H.cases_of("snr"), into: %{} do
        {case, data} = H.load(path)
        {float, _, _} = run_snr(case, data, Num.float())

        fixed =
          for {v, f} <- H.fixed_formats(case), into: %{} do
            {regs, raw, _} = run_snr(case, data, f)
            {v["variant"], {regs, raw}}
          end

        {case["name"], %{case: case, data: data, float: float, fixed: fixed}}
      end

    %{snr_runs: runs}
  end

  for {_path, case} <- H.cases_of("snr") do
    @name case["name"]

    test "#{case["name"]}: float instance vs GNU Radio's snr tags", %{snr_runs: runs} do
      %{case: case, data: data, float: float} = runs[@name]
      type = String.to_atom(case["params"]["type"])
      tags = for t <- data.tags["tags"], t["key"] == "snr", do: t
      assert length(tags) >= 2

      for t <- tags do
        # the tag at offset n holds the estimate after items 0..n-1
        mine = Estimators.snr_db(type, regs_at(float, t["offset"] - 1))

        assert abs(mine - t["value"]) <= case["float_vs_gnuradio"]["db"],
               "#{@name} at #{t["offset"]}: #{mine} dB vs #{t["value"]} dB"
      end
    end

    for {variant, _f} <- H.fixed_formats(case) do
      @variant variant
      test "#{case["name"]}: #{variant["variant"]} vs float instance", %{snr_runs: runs} do
        %{case: case, float: float, fixed: fixed} = runs[@name]
        {regs, _} = fixed[@variant["variant"]]
        type = String.to_atom(case["params"]["type"])

        for i <- [499, 999, 1999, 2999] do
          a = Estimators.snr_db(type, regs_at(float, i))
          b = Estimators.snr_db(type, regs_at(regs, i))
          assert abs(a - b) <= @variant["db"], "#{@name} at #{i}: #{a} dB vs #{b} dB"
        end
      end
    end

    test "#{case["name"]}: split into pieces gives the same registers", %{snr_runs: runs} do
      %{case: case, data: data, fixed: fixed} = runs[@name]
      v = hd(case["fixed_vs_float"])
      f = H.format(v["format"])
      {_, whole} = fixed[v["variant"]]
      n = length(data.input)
      input = H.input(case, data, f)

      {_, a, state} = run_snr(case, data, f, :init, slice(input, 0, 1000))
      {_, b, _} = run_snr(case, data, f, state, slice(input, 1000, n - 1000))

      joined =
        a
        |> Tuple.to_list()
        |> Enum.zip(Tuple.to_list(b))
        |> Enum.map(fn {x, y} -> concat(x, y) end)

      assert flat_regs(joined) == flat_regs(Tuple.to_list(whole))
    end
  end

  defp flat_regs(list), do: Enum.map(list, &flat_part/1)

  for {path, case} <- H.cases_of("probe_snr") do
    @path path

    test "#{case["name"]}: snr, signal and noise vs GNU Radio, float and fixed" do
      {case, data} = H.load(@path)
      type = String.to_atom(case["params"]["type"])
      probe = data.tags["probe"]
      n = length(data.input)

      formats =
        [{"float", Num.float(), case["float_vs_gnuradio"]["db"]}] ++
          for {v, f} <- H.fixed_formats(case),
              do: {v["variant"], f, case["float_vs_gnuradio"]["db"] + v["db"]}

      for {label, f, bound} <- formats do
        {regs, _, _} = run_snr(case, data, f)
        r = regs_at(regs, n - 1)
        assert abs(Estimators.snr_db(type, r) - probe["snr"]) <= bound, "#{label} snr"

        if type != :svr do
          {s, ns} = Estimators.signal_noise_db(type, r, [])
          assert abs(s - probe["signal"]) <= bound, "#{label} signal"
          assert abs(ns - probe["noise"]) <= bound, "#{label} noise"
        end
      end
    end
  end

  # QPSK with noise and a phase offset, deterministic
  defp skew_input do
    :rand.seed(:exsss, {7, 8, 9})
    pts = [{0.5, 0.5}, {-0.5, 0.5}, {-0.5, -0.5}, {0.5, -0.5}]

    for _ <- 1..2000 do
      {a, b} = Enum.random(pts)
      {a + 0.04 * :rand.normal(), b + 0.04 * :rand.normal()}
    end
  end

  # GNU Radio's mpsk_snr_est_skew::update in double precision, one item at a time,
  # on the input delayed by one (history), with i the index since the start
  defp skew_reference(samples, alpha) do
    {_, regs} =
      samples
      |> Enum.with_index()
      |> Enum.reduce({{0.0, 0.0}, {0.0, 0.0, 0.0}}, fn {x, i}, {prev, {y1, y2, _y3}} ->
        {zr, zi} = prev
        mag = :math.sqrt(zr * zr + zi * zi)
        y1n = alpha * mag + (1 - alpha) * y1
        y2n = alpha * (zr * zr - zi * zi) + (1 - alpha) * y2
        d = mag - y1n
        d_i = d / (i + 1)
        y3n = d * d_i * i * d_i * (i - 1) - 3.0 * d_i * y2n
        {x, {y1n, y2n, y3n}}
      end)

    regs
  end

  test "float and fixed registers follow the double precision recurrence" do
    samples = skew_input()
    alpha = 0.01
    {r1, r2, r3} = skew_reference(samples, alpha)

    {re, im} = Enum.unzip(samples)
    ints = fn xs -> Enum.map(xs, &round(&1 * 32768)) end

    for {fmt, bound} <- [{Num.float(), 1.0e-4}, {Num.q(16, 15), 2.0e-3}, {Num.q(18, 17), 2.0e-3}] do
      input =
        {Num.from_int(Nx.tensor(ints.(re), type: :s16), 15, fmt),
         Num.from_int(Nx.tensor(ints.(im), type: :s16), 15, fmt)}

      b = ExSdr.Carrier.block(:mpsk_snr_est_cc, format: fmt, type: :skew, alpha: alpha)
      {regs, _} = ExSdr.Carrier.run(b, input)
      [y1, y2, y3, _] = regs |> Tuple.to_list() |> Enum.map(&to_floats/1)

      assert abs(List.last(y1) - r1) <= bound, "#{inspect(fmt.kind)} y1"
      assert abs(List.last(y2) - r2) <= bound, "#{inspect(fmt.kind)} y2"
      assert abs(List.last(y3) - r3) <= bound, "#{inspect(fmt.kind)} y3 #{List.last(y3)} vs #{r3}"
    end
  end

  # -- probes -----------------------------------------------------------------------------------

  for {path, case} <- H.cases_of("probe") do
    @path path
    test "#{case["name"]}: final value vs GNU Radio, float and fixed" do
      {case, data} = H.load(@path)
      probe = data.tags["probe"]
      name = String.to_atom(case["block"]["name"])
      bound = case["float_vs_gnuradio"]["abs"]

      formats =
        [{Num.float(), bound}] ++ for {v, f} <- H.fixed_formats(case), do: {f, bound + v["abs"]}

      for {f, bound} <- formats do
        b = H.block(case, f)
        {out, state} = ExSdr.Carrier.run(b, H.input(case, data, f))

        case name do
          :probe_density_b ->
            assert abs(List.last(H.items(out)) - probe["density"]) <= bound
            assert abs(hd(to_floats(state)) - probe["density"]) <= bound

          :probe_signal ->
            level = hd(H.items(state))

            case level do
              {re, im} ->
                [pre, pim] = probe["level"]
                assert abs(re - pre) <= bound and abs(im - pim) <= bound

              v ->
                assert abs(v - probe["level"]) <= bound
            end

            # the block passes the stream on unchanged
            assert H.items(out) == H.items(H.input(case, data, f))
        end
      end
    end
  end
end
