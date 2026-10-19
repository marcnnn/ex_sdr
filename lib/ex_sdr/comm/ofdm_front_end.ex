defmodule ExSdr.Comm.OFDM.FrontEnd do
  @moduledoc """
  OFDM receiver front end compiled into one computation (`Nx.Defn`, EXLA
  when available): frequency correction, FFT, carrier extraction, pilot
  channel estimate and equalisation, hard decisions, cell and bit
  deinterleaving, bit unpacking. Output: one `u8` (0/1) per code bit, the
  input of a Viterbi decoder.

  A spec is a map:

    * `:ofdm` - `ExSdr.Comm.OFDM` configuration
    * `:constellation` - `ExSdr.Comm.Constellation`
    * `:cells` - `ExSdr.Comm.Permutation` applied to the words of each
      symbol (the deinterleaver; block index = symbol index), or `nil`
    * `:bits` - `ExSdr.Comm.Permutation` over the code bits (the
      deinterleaver), or `nil`

  The samples start at a symbol of class 0; whole multiples of the class
  period (and of the cell permutation's table count) are used.
  """
  import Nx.Defn
  alias ExSdr.Comm.{Constellation, OFDM, Permutation}

  @pi :math.pi()

  @doc """
  Runs the front end; returns the code bits as a binary of 0/1 bytes.
  `phase` is the phase (radians) of the frequency correction at the first
  sample: a stream processed in chunks passes
  `start_phase(spec, freq_offset, samples before the chunk)` to continue it.
  """
  def run(samples, freq_offset, spec, phase \\ 0.0) do
    n = div(div(Nx.size(samples), OFDM.symbol_length(spec.ofdm)), group(spec)) * group(spec)
    x = Nx.slice(samples, [0], [n * OFDM.symbol_length(spec.ofdm)])
    spec |> prepare(n) |> run_prepared(x, freq_offset, phase)
  end

  @doc """
  The compiled front end for exactly `symbols` symbols, with its tables on
  the device: for a stream of equal chunks (`run_prepared/4`), so the
  per-call cost is only the computation.
  """
  def prepare(spec, symbols) do
    t = tables(spec)

    %{
      fun: compiled(symbols, spec, t),
      tables: t,
      samples: symbols * OFDM.symbol_length(spec.ofdm)
    }
  end

  @doc "Runs a `prepare/2`d front end on exactly its number of samples."
  def run_prepared(%{fun: f, tables: t, samples: n}, samples, freq_offset, phase \\ 0.0) do
    ^n = Nx.size(samples)

    f.(samples, Nx.tensor(freq_offset, type: :f32), Nx.tensor(phase, type: :f32), t)
    |> Nx.to_binary()
  end

  @doc "Symbols per call must be a multiple of this (pilot classes and cell tables)."
  def group(spec), do: lcm(OFDM.period(spec.ofdm), cell_tables(spec))

  @doc "Phase of the frequency correction after `samples` samples, reduced mod 2 pi."
  def start_phase(spec, freq_offset, samples) do
    fft = spec.ofdm.fft
    # exact integer part first: samples * offset / fft turns, in float64
    turns = freq_offset * rem(samples, fft) / fft + freq_offset * div(samples, fft)
    -2 * @pi * (turns - Float.floor(turns))
  end

  defp cell_tables(%{cells: nil}), do: 1
  defp cell_tables(%{cells: p}), do: length(p.tables)

  defp lcm(a, b), do: div(a * b, Integer.gcd(a, b))

  defp compiled(n, spec, t) do
    key = {__MODULE__, n, :erlang.phash2(spec)}

    case :persistent_term.get(key, nil) do
      nil ->
        o = spec.ofdm

        static = [
          fft: o.fft,
          cp: o.cp,
          period: OFDM.period(o),
          constellation: spec.constellation,
          cells?: spec.cells != nil,
          bits?: spec.bits != nil
        ]

        templates = [
          Nx.template({n * OFDM.symbol_length(o)}, :c64),
          Nx.template({}, :f32),
          Nx.template({}, :f32),
          t
        ]

        compiler = if Code.ensure_loaded?(EXLA), do: EXLA, else: Nx.Defn.Evaluator
        f = Nx.Defn.compile(&front_end/5, templates ++ [static], compiler: compiler)
        :persistent_term.put(key, f)
        f

      f ->
        f
    end
  end

  # all tables as tensor arguments (not constants folded into the graph)
  defp tables(spec) do
    key = {__MODULE__, :tables, :erlang.phash2(spec)}

    case :persistent_term.get(key, nil) do
      nil ->
        t = %{
          bins: Nx.tensor(spec.ofdm.bins),
          ofdm: OFDM.tables(spec.ofdm),
          demap: Constellation.tensor_tables(spec.constellation),
          cells: if(spec.cells, do: Permutation.tensor(spec.cells), else: Nx.tensor(0)),
          bits: if(spec.bits, do: Permutation.tensor(spec.bits), else: Nx.tensor(0))
        }

        # on the device once, not copied on every call
        t =
          if Code.ensure_loaded?(EXLA.Backend), do: Nx.backend_transfer(t, EXLA.Backend), else: t

        :persistent_term.put(key, t)
        t

      t ->
        t
    end
  end

  defn front_end(x, freq_offset, phase0, t, opts \\ []) do
    opts = keyword!(opts, [:fft, :cp, :period, :constellation, :cells?, :bits?])
    fft = opts[:fft]
    cp = opts[:cp]
    period = opts[:period]
    sym = fft + cp
    n = div(Nx.size(x), sym)

    phase =
      Nx.complex(0.0, -2 * @pi / fft) * freq_offset * Nx.iota({n * sym}, type: :f32) +
        Nx.complex(0.0, phase0)

    carriers =
      (x * Nx.exp(phase))
      |> Nx.reshape({n, sym})
      |> Nx.slice_along_axis(cp, fft, axis: 1)
      |> Nx.fft()
      |> Nx.take(t.bins, axis: 1)

    # symbol i has class i mod period: group the classes with a reshape
    eq = equalise(carriers, t.ofdm, period)
    words = demap(eq, t.demap, opts[:constellation])
    words = cells(words, t.cells, opts[:cells?])
    code_bits(words, t.bits, opts[:constellation], opts[:bits?])
  end

  deftransformp equalise(carriers, t, period) do
    {n, k} = Nx.shape(carriers)
    carriers |> Nx.reshape({div(n, period), period, k}) |> OFDM.equalise_grouped(t, period)
  end

  deftransformp demap(points, t, constellation),
    do: Constellation.demap_tensor(constellation, points, t)

  deftransformp cells(words, t, apply?) do
    if apply?, do: Permutation.apply_tensor(words, t), else: words
  end

  deftransformp code_bits(words, t, constellation, apply?) do
    v = constellation.bits
    shifts = Nx.tensor(Enum.to_list((v - 1)..0//-1), type: :u8)

    bits =
      words
      |> Nx.reshape({:auto, 1})
      |> Nx.right_shift(shifts)
      |> Nx.bitwise_and(1)
      |> Nx.as_type(:u8)

    if apply? do
      {_, size} = Nx.shape(t)
      bits |> Nx.reshape({:auto, size}) |> Permutation.apply_tensor(t) |> Nx.flatten()
    else
      Nx.flatten(bits)
    end
  end
end
