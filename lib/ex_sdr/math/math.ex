defmodule ExSdr.Math do
  @moduledoc """
  GNU Radio stream math and type conversion blocks (gr-blocks) as
  `NxFixed.Block` descriptions. See `docs/blocks/math.md` for the coverage
  table, the deviations from GNU Radio and the hardware classes.

      ExSdr.Math.block(:multiply_const_ff, format: NxFixed.Num.q(16, 15), k: 0.7)

  Options: `:format` (`NxFixed.Num.float/0` by default, or `q/3`) and the
  block's parameters: `:k`, `:c`, `:lo`, `:hi`, `:scale`, `:n`, `:m`, `:k`
  and `:l` (bit blocks), `:decim`, `:interp`, `:delay`, `:length`, `:vlen`,
  `:endianness`, and `:item` (`:real` or `:complex`) for the generic blocks.
  Constants are quantized by `param/2`.
  """
  @behaviour NxFixed.Block

  alias ExSdr.Math.{Bits, Stream, Vector}
  alias NxFixed.Block
  alias NxFixed.Num
  alias NxFixed.Num.Format

  @doc """
  Quantizes a real or `[re, im]` constant for a format. The fraction keeps
  as many bits as the value's integer part allows, so a constant of 127 in
  Q15 becomes Q(16, 8) instead of saturating. Float: `f32`.
  """
  def param([re, im], %Format{} = f), do: {param(re, f), param(im, f)}

  def param(value, %Format{kind: :float}) when is_number(value),
    do: Num.constant(value, Num.float())

  def param(value, %Format{kind: :fixed} = f) when is_number(value) do
    frac = f.bits - 1 - integer_bits(abs(value))

    if frac < 0,
      do: raise(ArgumentError, "constant #{value} does not fit a #{f.bits}-bit format")

    Num.constant(value, Num.q(f.bits, frac, rounding: f.rounding, lut_bits: f.lut_bits))
  end

  defp integer_bits(v), do: Enum.find(0..64, &(v < :math.pow(2, &1)))

  @impl true
  def block(name, opts \\ []) do
    format = Keyword.get(opts, :format, Num.float())
    spec(name, format, opts)
  end

  # -- sample domain ------------------------------------------------------------------

  defp spec(name, format, opts) when name in [:multiply_const_ff, :multiply_const_cc] do
    k = param(Keyword.fetch!(opts, :k), format)
    item = if name == :multiply_const_cc, do: :complex, else: :real
    stream(name, &Stream.multiply_const/4, [k], format, item, "blocks.#{name}")
  end

  defp spec(name, format, opts) when name in [:add_const_ff, :add_const_cc] do
    c = param(Keyword.fetch!(opts, :c), format)
    item = if name == :add_const_cc, do: :complex, else: :real
    stream(name, &Stream.add_const/4, [c], format, item, "blocks.#{name}")
  end

  defp spec(:conjugate_cc, format, _opts),
    do: stream(:conjugate_cc, &Stream.conjugate_cc/3, [], format, :complex, "blocks.conjugate_cc")

  defp spec(:complex_to_mag_squared, format, _opts),
    do:
      stream(
        :complex_to_mag_squared,
        &Stream.complex_to_mag_squared/3,
        [],
        format,
        {:complex, :real},
        "blocks.complex_to_mag_squared"
      )

  defp spec(:complex_to_real, format, _opts),
    do:
      stream(
        :complex_to_real,
        &Stream.complex_to_real/3,
        [],
        format,
        {:complex, :real},
        "blocks.complex_to_real"
      )

  defp spec(:complex_to_imag, format, _opts),
    do:
      stream(
        :complex_to_imag,
        &Stream.complex_to_imag/3,
        [],
        format,
        {:complex, :real},
        "blocks.complex_to_imag"
      )

  defp spec(:abs_ff, format, _opts),
    do: stream(:abs_ff, &Stream.abs_ff/3, [], format, :real, "blocks.abs_ff")

  defp spec(name, format, _opts) when name in [:complex_to_mag, :complex_to_arg] do
    float_only!(name, format)
    fun = if name == :complex_to_mag, do: &Stream.complex_to_mag/3, else: &Stream.complex_to_arg/3
    stream(name, fun, [], format, {:complex, :real}, "blocks.#{name}")
  end

  defp spec(:nlog10_ff, format, opts) do
    float_only!(:nlog10_ff, format)
    n = Keyword.get(opts, :n, 1.0)
    k = Keyword.get(opts, :k, 0.0)

    stream(
      :nlog10_ff,
      &Stream.nlog10_ff/5,
      [param(n, format), param(k, format)],
      format,
      :real,
      "blocks.nlog10_ff"
    )
  end

  defp spec(:threshold_ff, format, opts) do
    lo = param(Keyword.fetch!(opts, :lo), format)
    hi = param(Keyword.fetch!(opts, :hi), format)

    %Block{
      name: :threshold_ff,
      kind: :stream,
      fun: &Stream.threshold_ff/5,
      init: Nx.tensor(0, type: :u8),
      gnuradio: "blocks.threshold_ff",
      params: [lo, hi],
      opts: [format: format],
      item: {:real, :bit},
      hardware: :feedback,
      # hysteresis: the output forgets its initial state only after the
      # input leaves the band, which need not happen (see docs/blocks/math.md)
      warmup: :unbounded
    }
  end

  defp spec(:moving_average_ff, format, opts),
    do: moving_average(:moving_average_ff, format, opts, :real, 1)

  defp spec(:moving_average_cc, format, opts),
    do: moving_average(:moving_average_cc, format, opts, :complex, 2)

  defp spec(:delay, format, opts) do
    d = Keyword.fetch!(opts, :delay)
    item = Keyword.get(opts, :item, :real)
    if d < 1, do: raise(ArgumentError, "delay must be at least 1 (0 is the identity)")

    zeros = Num.zeros({d}, format)
    init = if item == :complex, do: {zeros, zeros}, else: zeros

    stream(:delay, &Stream.delay/3, [], format, item, "blocks.delay",
      init: init,
      opts: [delay: d],
      latency: d,
      warmup: d,
      hardware: :feed_forward
    )
  end

  defp spec(:float_to_complex, format, _opts),
    do:
      stream(
        :float_to_complex,
        &Stream.float_to_complex/3,
        [],
        format,
        {:real_pair, :complex},
        "blocks.float_to_complex"
      )

  defp spec(:magphase_to_complex, format, _opts) do
    float_only!(:magphase_to_complex, format)

    stream(
      :magphase_to_complex,
      &Stream.magphase_to_complex/3,
      [],
      format,
      {:real_pair, :complex},
      "blocks.magphase_to_complex"
    )
  end

  defp spec(name, format, _opts) when name in [:add_ff, :sub_ff, :mul_ff, :div_ff] do
    kernel = pair_kernel(name)
    stream(name, kernel, [], format, {:real_pair, :real}, "blocks.#{name}")
  end

  defp spec(name, format, _opts) when name in [:add_cc, :sub_cc, :mul_cc, :div_cc] do
    kernel = pair_kernel(name)
    stream(name, kernel, [], format, {:complex_pair, :complex}, "blocks.#{name}")
  end

  defp spec(:char_to_float, format, opts) do
    # GNU Radio divides by scale: the kernel multiplies by its reciprocal
    scale = param(1.0 / Keyword.get(opts, :scale, 1.0), format)

    stream(
      :char_to_float,
      &Stream.char_to_float/4,
      [scale],
      format,
      {:byte, :real},
      "blocks.char_to_float"
    )
  end

  defp spec(:float_to_char, format, opts) do
    scale = param(Keyword.get(opts, :scale, 1.0), format)

    stream(
      :float_to_char,
      &Stream.float_to_char/4,
      [scale],
      format,
      {:real, :byte},
      "blocks.float_to_char"
    )
  end

  defp spec(name, format, opts) when name in [:stream_to_vector, :vector_to_stream] do
    item = Keyword.get(opts, :item, :real)

    stream(name, &Stream.identity/3, [], format, item, "blocks.#{name}")
  end

  # -- frame and host ------------------------------------------------------------------

  defp spec(name, format, opts) when name in [:integrate_ff, :integrate_cc] do
    decim = Keyword.fetch!(opts, :decim)
    item = if name == :integrate_cc, do: :complex, else: :real

    %Block{
      name: name,
      kind: :frame,
      fun: &Vector.integrate/3,
      init: {},
      gnuradio: "blocks.#{name}",
      opts: [format: format, decim: decim],
      rate: {decim, 1},
      item: {item, item},
      hardware: :feedback
    }
  end

  defp spec(:repeat, format, opts) do
    interp = Keyword.fetch!(opts, :interp)
    item = Keyword.get(opts, :item, :real)

    %Block{
      name: :repeat,
      kind: :frame,
      fun: &Vector.repeat/3,
      init: {},
      gnuradio: "blocks.repeat",
      opts: [format: format, interp: interp],
      rate: {1, interp},
      item: {item, item}
    }
  end

  defp spec(name, format, opts) when name in [:max_ff, :min_ff] do
    vlen = Keyword.fetch!(opts, :vlen)
    fun = if name == :max_ff, do: &Vector.max_ff/3, else: &Vector.min_ff/3

    %Block{
      name: name,
      kind: :frame,
      fun: fun,
      init: {},
      gnuradio: "blocks.#{name}",
      opts: [format: format, vlen: vlen],
      rate: {vlen, 1},
      item: {:real, :real}
    }
  end

  defp spec(name, _format, opts) when name in [:unpack_k_bits_bb, :packed_to_unpacked_bb] do
    k = Keyword.fetch!(opts, :k)

    %Block{
      name: name,
      kind: :frame,
      fun: &Bits.unpack_k_bits/3,
      init: {},
      gnuradio: "blocks.#{name}",
      opts: [k: k, endianness: Keyword.get(opts, :endianness, :msb)],
      rate: {1, k},
      item: {:byte, :bit}
    }
  end

  defp spec(name, _format, opts) when name in [:pack_k_bits_bb, :unpacked_to_packed_bb] do
    k = Keyword.fetch!(opts, :k)

    %Block{
      name: name,
      kind: :frame,
      fun: &Bits.pack_k_bits/3,
      init: {},
      gnuradio: "blocks.#{name}",
      opts: [k: k, endianness: Keyword.get(opts, :endianness, :msb)],
      rate: {k, 1},
      item: {:bit, :byte}
    }
  end

  defp spec(:keep_one_in_n, _format, opts) do
    n = Keyword.fetch!(opts, :n)
    keep_host(:keep_one_in_n, Vector.keep_one_in_n(n), {n, 1}, opts)
  end

  defp spec(:keep_m_in_n, _format, opts) do
    m = Keyword.fetch!(opts, :m)
    n = Keyword.fetch!(opts, :n)
    keep_host(:keep_m_in_n, Vector.keep_m_in_n(m, n), {n, m}, opts)
  end

  defp spec(:repack_bits_bb, _format, opts) do
    k = Keyword.fetch!(opts, :k)
    l = Keyword.fetch!(opts, :l)

    %Block{
      name: :repack_bits_bb,
      kind: :host,
      fun: Bits.repack_bits(k, l),
      init: <<>>,
      gnuradio: "blocks.repack_bits_bb",
      rate: {k, l},
      item: {:bit, :byte},
      hardware: :control
    }
  end

  defp spec(name, _format, _opts) do
    raise ArgumentError, "no math block #{inspect(name)} (see docs/blocks/math.md)"
  end

  # -- builders -----------------------------------------------------------------------

  defp stream(name, fun, params, format, item, gnuradio, extra \\ []) do
    {item_in, item_out} = if is_tuple(item), do: item, else: {item, item}
    {init, extra} = Keyword.pop(extra, :init, {})
    {kernel_opts, extra} = Keyword.pop(extra, :opts, [])
    {warmup, extra} = Keyword.pop(extra, :warmup, 0)
    {latency, extra} = Keyword.pop(extra, :latency, 0)
    {hardware, _extra} = Keyword.pop(extra, :hardware, :stateless)

    %Block{
      name: name,
      kind: :stream,
      fun: fun,
      init: init,
      gnuradio: gnuradio,
      params: params,
      opts: [format: format] ++ kernel_opts,
      rate: {1, 1},
      latency: latency,
      warmup: warmup,
      item: {item_in, item_out},
      hardware: hardware
    }
  end

  defp keep_host(name, fun, rate, opts) do
    item = Keyword.get(opts, :item, :real)

    %Block{
      name: name,
      kind: :host,
      fun: fun,
      init: 0,
      gnuradio: "blocks.#{name}",
      rate: rate,
      item: {item, item},
      hardware: :control
    }
  end

  defp float_only!(_name, %Format{kind: :float}), do: :ok

  defp float_only!(name, _format),
    do: raise(ArgumentError, "#{name} has a float instance only (see docs/blocks/math.md)")

  defp moving_average(name, format, opts, item, lines) do
    length = Keyword.fetch!(opts, :length)
    scale = Keyword.get(opts, :scale, 1.0)
    z = Num.zeros({length - 1}, format)
    init = if lines == 2, do: {z, z}, else: z

    stream(name, &Stream.moving_average/3, [], format, item, "blocks.#{name}",
      init: init,
      opts: [taps: List.duplicate(scale, length)],
      warmup: length - 1,
      hardware: :feed_forward
    )
  end

  defp pair_kernel(name) do
    case name do
      n when n in [:add_ff, :add_cc] -> &Stream.add_pair/3
      n when n in [:sub_ff, :sub_cc] -> &Stream.sub_pair/3
      n when n in [:mul_ff, :mul_cc] -> &Stream.mul_pair/3
      n when n in [:div_ff, :div_cc] -> &Stream.div_pair/3
    end
  end
end
