defmodule ExSdr.Sources do
  @moduledoc """
  Signal sources and channel models (GNU Radio's gr-analog sources,
  gr-blocks sources and gr-channels). See `docs/blocks/sources.md` for the
  coverage table and the differences from GNU Radio.

  `block(name, opts)` returns a `%NxFixed.Block{}`. Source blocks have no
  input: they are stream blocks whose input sample is ignored, so they run
  through the same scan loop as every other block. `generate/3` builds the
  dummy input for you:

      block = ExSdr.Sources.block(:sig_source_c, frequency: 1000.0, sample_rate: 48_000.0)
      {samples, state} = ExSdr.Sources.generate(block, 4096)
      {more, state} = ExSdr.Sources.generate(block, 4096, state)   # continues

  Common options: `:format` (`NxFixed.Num.float()` by default, or
  `NxFixed.Num.q/3`).

  ## Blocks

  | name | GNU Radio | options |
  |---|---|---|
  | `:sig_source_f`, `:sig_source_c` | `analog.sig_source_f/c` | `:waveform`, `:frequency`, `:sample_rate`, `:amplitude`, `:offset`, `:phase`, `:phase_bits` |
  | `:nco` | (`fxpt_nco`) | like `:sig_source_c`, sine, unit amplitude |
  | `:rotator_cc` | `blocks.rotator_cc` | `:frequency`, `:sample_rate`, `:phase`, `:phase_bits` |
  | `:vco_c` | `blocks.vco_c` | `:sensitivity`, `:sample_rate`, `:amplitude`, `:phase_bits` |
  | `:noise_source_f`, `:noise_source_c` | `analog.noise_source_f/c` | `:type`, `:amplitude`, `:seed` |
  | `:fastnoise_source_f`, `:fastnoise_source_c` | `analog.fastnoise_source_f/c` | as above, `:samples` |
  | `:vector_source_f`, `:vector_source_c` | `blocks.vector_source_f/c` | `:data`, `:repeat` |
  | `:glfsr_source_b` | `digital.glfsr_source_b` | `:degree`, `:mask`, `:seed` |

  | `:channel_model` | `channels.channel_model` | `:noise_voltage`, `:frequency_offset`, `:epsilon`, `:taps`, `:seed` |
  | `:fading_model` | `channels.fading_model` | `:n`, `:fdts`, `:los`, `:k`, `:seed` |
  | `:selective_fading_model` | `channels.selective_fading_model` | as above plus `:delays`, `:mags`, `:ntaps` |
  | `:mmse_resampler_cc`, `:fir_ccc`, `:add_noise` | `filter.mmse_resampler_cc`, `filter.fir_filter_ccc`, - | `:ratio`; `:taps`; `:amplitude` |
  | `:iqbal_gen`, `:distortion_2_gen`, `:distortion_3_gen`, `:quantizer`, `:cfo_model` | `channels.*` | see `ExSdr.Sources.Impairments` |

  Channel models are in `ExSdr.Sources.Channel`.
  """
  alias ExSdr.Sources.{Channel, Noise, NCO, Vector}
  alias NxFixed.Block
  alias NxFixed.Num.Format

  @behaviour NxFixed.Block

  @impl true
  def block(name, opts \\ [])

  def block(name, opts) when name in [:sig_source_f, :sig_source_c, :nco] do
    f = format(opts)
    bits = Keyword.get(opts, :phase_bits, 32)
    complex? = name != :sig_source_f
    waveform = if name == :nco, do: :sin, else: Keyword.get(opts, :waveform, :sin)
    amplitude = Keyword.get(opts, :amplitude, 1.0)
    offset = Keyword.get(opts, :offset, 0.0)
    check_range!(f, [amplitude, offset, amplitude + offset])

    unless waveform in [:sin, :cos, :square, :triangle, :sawtooth, :constant],
      do: raise(ArgumentError, "unknown waveform #{inspect(waveform)}")

    %Block{
      name: name,
      kind: :stream,
      fun: &NCO.sig_source/4,
      init: NCO.tensor(NCO.phase_from_radians(Keyword.get(opts, :phase, 0.0), bits)),
      params: [NCO.tensor(inc(opts, bits))],
      opts: [
        format: f,
        waveform: waveform,
        complex: complex?,
        amplitude: amplitude,
        offset: offset,
        phase_bits: bits
      ],
      item: {:none, if(complex?, do: :complex, else: :real)},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: if(name == :nco, do: nil, else: "analog.#{name}")
    }
  end

  def block(:rotator_cc, opts) do
    bits = Keyword.get(opts, :phase_bits, 32)

    %Block{
      name: :rotator_cc,
      kind: :stream,
      fun: &NCO.rotator/4,
      init: NCO.tensor(NCO.phase_from_radians(Keyword.get(opts, :phase, 0.0), bits)),
      params: [NCO.tensor(inc(opts, bits))],
      opts: [format: format(opts), phase_bits: bits],
      item: {:complex, :complex},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "blocks.rotator_cc"
    }
  end

  def block(:vco_c, opts) do
    bits = Keyword.get(opts, :phase_bits, 32)
    rate = Keyword.get(opts, :sample_rate, 1.0)
    sensitivity = Keyword.fetch!(opts, :sensitivity)
    f = format(opts)
    amplitude = Keyword.get(opts, :amplitude, 1.0)
    check_range!(f, [amplitude])

    %Block{
      name: :vco_c,
      kind: :stream,
      fun: &NCO.vco/3,
      init: NCO.tensor(NCO.phase_from_radians(Keyword.get(opts, :phase, 0.0), bits)),
      opts: [
        format: f,
        turns_per_unit: sensitivity / (2 * :math.pi() * rate),
        amplitude: amplitude,
        phase_bits: bits
      ],
      item: {:real, :complex},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "blocks.vco_c"
    }
  end

  def block(name, opts) when name in [:noise_source_f, :noise_source_c] do
    f = format(opts)
    amplitude = Keyword.get(opts, :amplitude, 1.0)
    check_range!(f, [amplitude])
    type = Keyword.get(opts, :type, :gaussian)

    unless type in [:uniform, :gaussian, :laplacian, :impulse],
      do: raise(ArgumentError, "unknown noise type #{inspect(type)}")

    complex? = name == :noise_source_c

    %Block{
      name: name,
      kind: :stream,
      fun: &Noise.noise_source/3,
      init: Noise.seed_state(Keyword.get(opts, :seed, 0)),
      opts:
        [format: f, type: type, complex: complex?, amplitude: amplitude] ++
          Keyword.take(opts, [:table_bits, :impulse_factor]),
      item: {:none, if(complex?, do: :complex, else: :real)},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "analog.#{name}"
    }
  end

  def block(name, opts) when name in [:fastnoise_source_f, :fastnoise_source_c] do
    source =
      block(if(name == :fastnoise_source_f, do: :noise_source_f, else: :noise_source_c), opts)

    samples = Keyword.get(opts, :samples, 4096)
    true = Bitwise.band(samples, samples - 1) == 0 and samples >= 2

    {table, _} =
      generate(%{source | init: Noise.seed_state(Keyword.get(opts, :seed, 0) + 0x5EED)}, samples)

    table = Nx.Defn.Composite.traverse(table, &Nx.backend_copy(&1, Nx.BinaryBackend))

    %Block{
      source
      | name: name,
        fun: &Noise.fastnoise_source/4,
        params: [table],
        opts: Keyword.put(source.opts, :table_bits, log2(samples)),
        hardware: :memory,
        gnuradio: "analog.#{name}"
    }
  end

  def block(name, opts) when name in [:vector_source_f, :vector_source_c] do
    f = format(opts)
    data = Keyword.fetch!(opts, :data)

    %Block{
      name: name,
      kind: :stream,
      fun: &Vector.vector_source/4,
      init: Nx.tensor(0, type: :u32),
      params: [Vector.table(data, f)],
      opts: [format: f, repeat: Keyword.get(opts, :repeat, true)],
      item: {:none, if(name == :vector_source_c, do: :complex, else: :real)},
      hardware: :memory,
      warmup: :unbounded,
      gnuradio: "blocks.#{name}"
    }
  end

  def block(:glfsr_source_b, opts), do: ExSdr.Sources.LfsrSource.block(opts)

  def block(name, opts) when name in [:channel_model, :fading_model, :selective_fading_model],
    do: Channel.block(name, opts)

  def block(name, opts)
      when name in [:iqbal_gen, :distortion_2_gen, :distortion_3_gen, :quantizer, :cfo_model],
      do: ExSdr.Sources.Impairments.block(name, opts)

  def block(name, opts), do: Channel.block(name, opts)

  @doc """
  `n` items from a source `block`, continuing from `state` (default: the
  block's initial state): `{output, next_state}`.
  """
  def generate(%Block{} = block, n, state \\ :init) do
    input =
      case block.item do
        {:none, _} -> dummy(block, n)
        _ -> raise ArgumentError, "#{block.name} has an input; use NxFixed.Block.run/4"
      end

    Block.run(block, input, state)
  end

  defp dummy(%Block{kind: :host}, n), do: n

  defp dummy(%Block{opts: opts}, n) do
    case opts[:format] do
      %Format{kind: :float} -> Nx.broadcast(Nx.tensor(0.0, type: :f32), {n})
      _ -> Nx.broadcast(Nx.tensor(0, type: :u8), {n})
    end
  end

  defp format(opts), do: Keyword.get(opts, :format, NxFixed.Num.float())

  defp inc(opts, bits) do
    rate = Keyword.get(opts, :sample_rate, 1.0)
    NCO.phase_inc(Keyword.get(opts, :frequency, 0.0), rate, bits)
  end

  defp log2(n), do: n |> :math.log2() |> round()

  # A constant beyond the range would silently saturate (1.0 in Q15 is 1 - 2^-15).
  defp check_range!(%Format{kind: :float}, _values), do: :ok

  defp check_range!(%Format{kind: :fixed, bits: bits, frac: frac}, values) do
    limit = :math.pow(2, bits - 1 - frac)

    for v <- values, v > limit or v < -limit do
      raise ArgumentError, "#{v} does not fit the format (|x| <= #{limit})"
    end

    :ok
  end
end
