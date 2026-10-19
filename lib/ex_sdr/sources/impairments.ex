defmodule ExSdr.Sources.Impairments do
  @moduledoc """
  Receiver and transmitter impairments of gr-channels: `iqbal_gen` (IQ
  gain and phase imbalance), `distortion_2_gen` and `distortion_3_gen`
  (memoryless second and third order distortion), `quantizer` (a converter
  with `bits` bits) and `cfo_model` (carrier frequency offset as a clipped
  random walk).

  All are complex sample blocks except `quantizer` (real, or per component
  with `complex: true`). The first four are stateless and follow GNU Radio's
  formulas exactly; constants must be representable in the format
  (`|constant| <= 2^(bits - 1 - frac)`).

    * `iqbal_gen(magnitude_db, phase_deg, mode)`, `g = 10^(db / 20)`:
      transmitter (`mode: 0`) `re = g cos(p) I, im = g sin(p) I + Q`;
      receiver (`mode: 1`) `re = g (cos(p) I + sin(p) Q), im = Q`
    * `distortion_2_gen(beta)`: `y = x + beta (x^2 + |x|^2)`
    * `distortion_3_gen(beta)`: `y = x + beta |x|^2 x`
    * `quantizer(bits)`: `round(x 2^(bits - 1)) / 2^(bits - 1)` with ties to
      even and saturation to 16 bit integers, like GNU Radio's
      float-to-short round trip
  """
  import Nx.Defn
  import Bitwise
  import NxFixed.Num

  alias ExSdr.Sources.{NCO, Noise}
  alias NxFixed.Block
  alias NxFixed.Fixed
  alias NxFixed.Num
  alias NxFixed.Num.Format

  # -- kernels -----------------------------------------------------------------------------------

  @doc "IQ imbalance stream kernel `step(x, state, opts)` (state unused, `nil` or `{}`)."
  defn iqbal(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :magnitude_db, :phase_deg, mode: 0])
    {iqbal_value(x, opts), state}
  end

  deftransform iqbal_value({i, q}, opts) do
    f = opts[:format]
    g = :math.pow(10, opts[:magnitude_db] / 20)
    p = opts[:phase_deg] * :math.pi() / 180
    {a, b} = {g * :math.cos(p), g * :math.sin(p)}

    if opts[:mode] == 0 do
      {scale(i, a, f), q |> add(scale(i, b, f)) |> quantize(f)}
    else
      {add(scale(i, a, f), scale(q, b, f)) |> quantize(f), q}
    end
  end

  defp scale(v, k, f), do: v |> mul(constant(k, f)) |> rescale(f) |> quantize(f)

  @doc "Second order distortion, `y = x + beta (x^2 + |x|^2)`."
  defn distortion_2(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :beta])
    {distort(x, opts, 2), state}
  end

  @doc "Third order distortion, `y = x + beta |x|^2 x`."
  defn distortion_3(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :beta])
    {distort(x, opts, 3), state}
  end

  deftransform distort(x, opts, order) do
    f = opts[:format]
    # clip the input to the format's width: the width check then knows the squares fit
    {re, im} = x = quantize(x, f)
    beta = constant(opts[:beta], f)
    mag2 = add(mul(re, re), mul(im, im)) |> rescale(f)

    inner =
      case order do
        2 ->
          {p_re, p_im} = cmul(x, x)
          {add(p_re, mag2), p_im} |> rescale(f)

        3 ->
          cscale(x, mag2) |> rescale(f)
      end

    inner
    |> cscale(beta)
    |> rescale(f)
    |> then(fn {a, b} -> {add(a, re), add(b, im)} end)
    |> quantize(f)
  end

  @doc """
  Quantizer: `step(x, state, opts)`; real samples, or `{re, im}` with
  `complex: true`. Options `:format`, `:bits`.
  """
  defn quantizer(x, state, opts \\ []) do
    opts = keyword!(opts, [:format, :bits, complex: false])
    {quantize_value(x, opts), state}
  end

  deftransform quantize_value({re, im}, opts),
    do: {quantize_value(re, opts), quantize_value(im, opts)}

  deftransform quantize_value(x, opts) do
    f = opts[:format]
    k = opts[:bits] - 1

    case x do
      %Fixed{t: t, frac: frac} when frac > k ->
        shift = frac - k
        t = Nx.as_type(t, :s64)
        # round half to even: add half less one, plus the bit that becomes the LSB
        q =
          t
          |> Nx.add((1 <<< (shift - 1)) - 1)
          |> Nx.add(Nx.bitwise_and(Nx.right_shift(t, shift), 1))
          |> Nx.right_shift(shift)
          |> Nx.clip(-32768, 32767)

        quantize(%Fixed{t: Nx.left_shift(q, shift), frac: frac}, f)

      %Fixed{} ->
        # the converter is finer than the format: nothing to round
        x

      _float ->
        scaled = Nx.multiply(x, :math.pow(2, k))
        up = Nx.floor(Nx.add(scaled, 0.5))
        tie = Nx.equal(up, Nx.add(scaled, 0.5))
        odd = Nx.not_equal(Nx.remainder(up, 2), 0)
        q = Nx.select(Nx.logical_and(tie, odd), Nx.subtract(up, 1), up) |> Nx.clip(-32768, 32767)
        Nx.divide(q, :math.pow(2, k)) |> Nx.as_type(:f32)
    end
  end

  @doc """
  CFO random walk: `step(x, {generator, cfo, phase}, table, opts)`. `cfo` is
  the frequency as a signed phase word step, changed every sample by a
  Gaussian step from `table` (phase word units; `cfo_table/3`) and clipped
  to `+-:max_word`; the sample is rotated by the phase before the update.
  """
  defn cfo(x, state, table, opts \\ []) do
    opts = keyword!(opts, [:format, :max_word, :table_bits, phase_bits: 32])
    cfo_step(x, state, table, opts)
  end

  deftransform cfo_step(x, {rng, cfo, phase}, table, opts) do
    f = opts[:format]
    bits = opts[:phase_bits]
    {c, s} = NCO.sin_cos(NCO.widen(phase, bits), f)
    y = x |> cmul({c, s}) |> rescale(f) |> quantize(f)

    next_rng = Noise.next(rng)
    idx = next_rng |> Nx.as_type(:s64) |> Nx.right_shift(32 - opts[:table_bits])
    step = Nx.take(table, idx) |> Nx.as_type(:s64)
    m = opts[:max_word]

    next_cfo =
      cfo
      |> Nx.as_type(:s64)
      |> Nx.add(step)
      |> Nx.clip(-m, m)
      |> Nx.as_type(:s32)

    {y, {next_rng, next_cfo, NCO.advance(phase, next_cfo, bits)}}
  end

  @doc "Gaussian step table of `2^bits` entries, standard deviation `std_word` phase words."
  def cfo_table(std_word, bits, _unused \\ nil) do
    n = 1 <<< bits
    values = for i <- 0..(n - 1), do: round(std_word * Noise.quantile(:gaussian, (i + 0.5) / n))
    Nx.tensor(values, type: :s32, backend: Nx.BinaryBackend)
  end

  # -- blocks ---------------------------------------------------------------------------------------

  @doc "Blocks of this module, see `ExSdr.Sources.block/2`."
  def block(name, opts) when name in [:iqbal_gen, :distortion_2_gen, :distortion_3_gen] do
    f = Keyword.get(opts, :format, Num.float())

    {fun, kernel_opts, gnuradio, ranges} =
      case name do
        :iqbal_gen ->
          mag = Keyword.get(opts, :magnitude, 0.0)
          phase = Keyword.get(opts, :phase, 0.0)
          g = :math.pow(10, mag / 20)

          {&__MODULE__.iqbal/3,
           [magnitude_db: mag, phase_deg: phase, mode: Keyword.get(opts, :mode, 0)],
           "channels.iqbal_gen",
           [g * :math.cos(phase * :math.pi() / 180), g * :math.sin(phase * :math.pi() / 180)]}

        :distortion_2_gen ->
          {&__MODULE__.distortion_2/3, [beta: Keyword.fetch!(opts, :beta)],
           "channels.distortion_2_gen", [Keyword.fetch!(opts, :beta)]}

        :distortion_3_gen ->
          {&__MODULE__.distortion_3/3, [beta: Keyword.fetch!(opts, :beta)],
           "channels.distortion_3_gen", [Keyword.fetch!(opts, :beta)]}
      end

    check_range!(f, ranges)

    %Block{
      name: name,
      kind: :stream,
      fun: fun,
      init: Nx.tensor(0, type: :u8),
      opts: [format: f] ++ kernel_opts,
      item: {:complex, :complex},
      hardware: :stateless,
      warmup: 0,
      gnuradio: gnuradio
    }
  end

  def block(:quantizer, opts) do
    f = Keyword.get(opts, :format, Num.float())
    complex? = Keyword.get(opts, :complex, false)

    %Block{
      name: :quantizer,
      kind: :stream,
      fun: &__MODULE__.quantizer/3,
      init: Nx.tensor(0, type: :u8),
      opts: [format: f, bits: Keyword.fetch!(opts, :bits), complex: complex?],
      item: if(complex?, do: {:complex, :complex}, else: {:real, :real}),
      hardware: :stateless,
      warmup: 0,
      gnuradio: if(complex?, do: nil, else: "channels.quantizer")
    }
  end

  def block(:cfo_model, opts) do
    f = Keyword.get(opts, :format, Num.float())
    bits = Keyword.get(opts, :phase_bits, 32)
    rate = Keyword.fetch!(opts, :sample_rate)
    word_per_hz = (1 <<< bits) / rate
    std_word = Keyword.fetch!(opts, :std_dev_hz) * word_per_hz
    max_word = round(Keyword.fetch!(opts, :max_dev_hz) * word_per_hz)
    true = max_word < 1 <<< 30
    table_bits = Keyword.get(opts, :table_bits, 12)

    %Block{
      name: :cfo_model,
      kind: :stream,
      fun: &__MODULE__.cfo/4,
      init:
        {Noise.seed_state(Keyword.get(opts, :seed, 0)), Nx.tensor(0, type: :s32),
         Nx.tensor(0, type: :u32)},
      params: [cfo_table(std_word, table_bits)],
      opts: [format: f, max_word: max_word, table_bits: table_bits, phase_bits: bits],
      item: {:complex, :complex},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "channels.cfo_model"
    }
  end

  defp check_range!(%Format{kind: :float}, _), do: :ok

  defp check_range!(%Format{bits: bits, frac: frac}, values) do
    limit = :math.pow(2, bits - 1 - frac)

    for v <- values,
        abs(v) > limit,
        do: raise(ArgumentError, "#{v} does not fit the format (|x| <= #{limit})")

    :ok
  end
end
