defmodule ExSdr.Sources.NCO do
  @moduledoc """
  Phase accumulator oscillators: `sig_source` (sine, cosine, square,
  triangle, sawtooth, constant; real or complex), `rotator` (phase-continuous
  complex mixer) and `vco` (phase driven by an input).

  ## Phase word

  The state is a phase word `p` of `:phase_bits` bits (default 32, at most
  32) in a `u32` tensor; the full circle is `2^phase_bits`. The step is a
  phase word too, `phase_inc/3`. The frequency resolution is
  `rate / 2^phase_bits` (0.0002 Hz at 1 MHz with 32 bits), and the phase
  wraps by construction, so a stream of any length keeps its phase exactly.
  Output `n` uses the phase before the `n`-th increment, like GNU Radio's
  `sig_source` and `rotator`: with initial phase `p0` the first sample is
  `wave(p0)`.

  Internally the word is widened to 32 bits (`p << (32 - phase_bits)`) so
  that all waveforms see the same signed phase `s` in `[-2^31, 2^31)`, the
  analogue of GNU Radio's phase in `[-pi, pi)`.

  ## Waveforms (GNU Radio conventions, `amplitude` a, `offset` o)

    * `:sin`, `:cos` (real): `a sin(phi) + o`, `a cos(phi) + o`; complex:
      `a (cos phi + j sin phi) + o` for both (as in GNU Radio)
    * `:square`: `a` where `phi < 0`, else `0`, plus `o`; complex imaginary
      part: the same function of `phi - pi/2`
    * `:triangle`: `a (1 - |phi| / pi) + o`; imaginary: of `phi - pi/2`
    * `:sawtooth`: `a (1/2 + phi / (2 pi)) + o`; imaginary: of `phi - pi/2`
    * `:constant`: `a + o` (imaginary 0)

  The offset is added to the real part only.

  ## Number format and spurs

  Float: exact `sin`/`cos` of the phase word in `f64`, rounded to `f32`.
  Fixed point: a sine table of `2^lut_bits` entries (amplitude `2^frac - 1`,
  `lut_bits` from the format, default 10) addressed by the top bits of the
  phase (truncated or rounded per the format). Dropping the low phase bits
  makes a periodic phase error: a sawtooth of one table step, `2^-lut_bits`
  of a turn. For a tone whose truncated phase repeats with a long period it
  is a single spur of level

      -6.02 * lut_bits dBc

  (measured on a tone of 2^16 samples: 48.1 dBc at 8 bits, 60.2 at 10, 72.2
  at 12, in Q17; the 14 bit table is limited by the Q17 amplitude at 83
  dBc). Rounding the phase instead of truncating it removes the mean of that
  error (the error power against the ideal sine falls by 6 dB, so the SNR
  rises by 6 dB) but not the spur. The default 10 table bits give 60 dBc
  spurs, 12 bits 72 dBc, 14 bits 84 dBc, 16 bits 96 dBc. The width of the
  phase accumulator itself only sets the frequency resolution (it must not
  be narrower than the table address). The triangle and sawtooth read the
  top `frac + 1` bits of the phase directly and have no table: only
  amplitude quantisation.

  Waves that peak at exactly 1.0 (square, triangle, sawtooth with
  `amplitude` 1) saturate to the largest value of the format
  (`1 - 2^-frac` in Q15).

  `amplitude` and `offset` are compile time constants of the kernel; they
  must be representable in the format.
  """
  import Nx.Defn
  import Bitwise
  import NxFixed.Num

  alias NxFixed.Fixed
  alias NxFixed.Num.Format
  alias NxFixed.Primitives, as: P

  @doc "The phase word for `cycles` (fraction of a turn, may be negative or above 1)."
  def phase_word(cycles, bits \\ 32) when bits in 2..32 do
    size = 1 <<< bits
    round(cycles * size) |> Integer.mod(size)
  end

  @doc "The phase increment for `freq` at sample rate `rate` (same unit)."
  def phase_inc(freq, rate, bits \\ 32), do: phase_word(freq / rate, bits)

  @doc "The phase word for an angle in radians."
  def phase_from_radians(radians, bits \\ 32), do: phase_word(radians / (2 * :math.pi()), bits)

  @doc "A phase word as a `u32` tensor (state or parameter)."
  def tensor(word), do: Nx.tensor(word, type: :u32)

  # -- kernels --------------------------------------------------------------------------

  @doc """
  `sig_source` stream kernel `step(_x, phase, phase_inc, opts)`: the input
  sample is ignored (feed zeros); the output is the real or complex sample
  at `phase`, the state the advanced phase.

  Options: `:format`, `:waveform`, `:complex` (default false), `:amplitude`
  (default 1.0), `:offset` (0.0), `:phase_bits` (32).
  """
  defn sig_source(_x, phase, phase_inc, opts \\ []) do
    opts =
      keyword!(opts, [
        :format,
        :waveform,
        complex: false,
        amplitude: 1.0,
        offset: 0.0,
        phase_bits: 32
      ])

    {source_value(phase, opts), advance(phase, phase_inc, opts[:phase_bits])}
  end

  deftransform source_value(phase, opts) do
    format = opts[:format]
    w = widen(phase, opts[:phase_bits])
    kind = opts[:waveform]
    {amp, off} = {opts[:amplitude], opts[:offset]}

    cond do
      kind == :constant and opts[:complex] ->
        {constant(amp + off, format), constant(0.0, format)}

      kind == :constant ->
        constant(amp + off, format)

      opts[:complex] ->
        {re, im} =
          case kind do
            k when k in [:sin, :cos] -> sin_cos(w, format)
            k -> {wave(k, w, format), wave(k, shift(w, -(1 <<< 30)), format)}
          end

        {scale_offset(re, amp, off, format), scale_offset(im, amp, 0.0, format)}

      true ->
        scale_offset(wave(kind, w, format), amp, off, format)
    end
  end

  @doc """
  Phase-continuous complex rotator `y = x * exp(j phase)`, state and
  parameters as in `sig_source`. Options: `:format`, `:phase_bits`.
  """
  defn rotator(x, phase, phase_inc, opts \\ []) do
    opts = keyword!(opts, [:format, phase_bits: 32])
    format = opts[:format]
    {c, s} = sin_cos(widen(phase, opts[:phase_bits]), format)
    y = x |> cmul({c, s}) |> rescale(format) |> quantize(format)
    {y, advance(phase, phase_inc, opts[:phase_bits])}
  end

  @doc """
  Voltage controlled oscillator `exp(j phase)` times `:amplitude`; the real
  input sample `v` adds `v * sensitivity / rate` turns per sample
  (`:turns_per_unit`, a constant: GNU Radio's `sensitivity / (2 pi rate)`).
  The state is the phase word. Output is the sample at the phase before the
  update.
  """
  defn vco(v, phase, opts \\ []) do
    opts = keyword!(opts, [:format, :turns_per_unit, amplitude: 1.0, phase_bits: 32])
    format = opts[:format]
    {c, s} = sin_cos(widen(phase, opts[:phase_bits]), format)

    y =
      {scale_offset(c, opts[:amplitude], 0.0, format),
       scale_offset(s, opts[:amplitude], 0.0, format)}

    {y,
     advance(
       phase,
       vco_inc(v, format, opts[:turns_per_unit], opts[:phase_bits]),
       opts[:phase_bits]
     )}
  end

  # Phase increment word from a real sample: v * k turns, rounded.
  deftransform vco_inc(v, format, turns_per_unit, bits) do
    extra = 16
    k = round(turns_per_unit * (1 <<< (bits + extra)))
    {raw, frac} = real_parts(v, format)
    prod = Nx.multiply(Nx.as_type(raw, :s64), c64(k))
    inc = if frac + extra > 0, do: P.shift_round(prod, frac + extra), else: prod
    Nx.bitwise_and(inc, c64((1 <<< bits) - 1)) |> Nx.as_type(:u32)
  end

  defp real_parts(%Fixed{t: t, frac: frac}, _format), do: {t, frac}
  defp real_parts(t, %Format{kind: :float}), do: {Nx.round(Nx.multiply(t, 1 <<< 15)), 15}

  # -- phase arithmetic ---------------------------------------------------------------------

  @doc "`phase + inc` modulo `2^bits`."
  deftransform advance(phase, inc, bits) do
    Nx.add(Nx.as_type(phase, :s64), Nx.as_type(inc, :s64))
    |> Nx.bitwise_and(c64((1 <<< bits) - 1))
    |> Nx.as_type(:u32)
  end

  # The phase as a 32 bit word in an s64 container.
  deftransform widen(phase, bits),
    do:
      phase
      |> Nx.as_type(:s64)
      |> Nx.bitwise_and(c64((1 <<< bits) - 1))
      |> Nx.left_shift(32 - bits)

  defp shift(w, delta), do: w |> Nx.add(c64(delta)) |> Nx.bitwise_and(c64(0xFFFFFFFF))

  # -- waveforms --------------------------------------------------------------------------------

  @doc "`{cos, sin}` of the 32 bit phase word `w` in `format`."
  deftransform sin_cos(w, %Format{kind: :float}) do
    angle = Nx.multiply(Nx.as_type(w, :f64), 2 * :math.pi() / 4_294_967_296)
    {angle |> Nx.cos() |> Nx.as_type(:f32), angle |> Nx.sin() |> Nx.as_type(:f32)}
  end

  deftransform sin_cos(w, %Format{kind: :fixed} = f) do
    l = f.lut_bits
    n = 1 <<< l
    shift = 32 - l

    idx =
      if f.rounding == :nearest,
        do: w |> P.shift_round(shift) |> Nx.bitwise_and(n - 1),
        else: Nx.right_shift(w, shift)

    values = P.sine_values(n, (1 <<< f.frac) - 1)
    type = if Enum.all?(values, &(&1 in -32768..32767)), do: :s16, else: :s32
    table = Nx.tensor(values, type: type, backend: Nx.BinaryBackend)
    cos_idx = idx |> Nx.add(div(n, 4)) |> Nx.bitwise_and(n - 1)

    {%Fixed{t: Nx.take(table, cos_idx), frac: f.frac},
     %Fixed{t: Nx.take(table, idx), frac: f.frac}}
  end

  @doc "Unit amplitude waveform of the 32 bit phase word `w`."
  deftransform wave(:sin, w, f), do: w |> sin_cos(f) |> elem(1)
  deftransform wave(:cos, w, f), do: w |> sin_cos(f) |> elem(0)

  deftransform wave(:square, w, %Format{kind: :float}),
    do: w |> Nx.right_shift(31) |> Nx.as_type(:f32)

  deftransform wave(:square, w, %Format{kind: :fixed} = f),
    do: w |> Nx.right_shift(31) |> Nx.multiply(1 <<< f.frac) |> fixed(f)

  deftransform wave(:triangle, w, %Format{kind: :float}),
    do:
      Nx.subtract(1.0, Nx.divide(Nx.as_type(Nx.abs(signed(w)), :f64), 2_147_483_648.0))
      |> Nx.as_type(:f32)

  deftransform wave(:triangle, w, %Format{kind: :fixed} = f) do
    m = Nx.subtract(c64(2_147_483_648), Nx.abs(signed(w)))
    m |> to_frac(31, f) |> fixed(f)
  end

  deftransform wave(:sawtooth, w, %Format{kind: :float}),
    do:
      w
      |> Nx.add(c64(2_147_483_648))
      |> Nx.bitwise_and(c64(0xFFFFFFFF))
      |> Nx.as_type(:f64)
      |> Nx.divide(4_294_967_296.0)
      |> Nx.as_type(:f32)

  deftransform wave(:sawtooth, w, %Format{kind: :fixed} = f) do
    u = w |> Nx.add(c64(2_147_483_648)) |> Nx.bitwise_and(c64(0xFFFFFFFF))
    u |> to_frac(32, f) |> fixed(f)
  end

  defp c64(n), do: Nx.tensor(n, type: :s64)

  defp signed(w), do: Nx.subtract(w, Nx.multiply(Nx.right_shift(w, 31), c64(4_294_967_296)))

  # `x` has `from` fractional bits; bring it to the format's.
  defp to_frac(x, from, %Format{frac: frac, rounding: r}) do
    k = from - frac
    if r == :nearest, do: P.shift_round(x, k), else: Nx.right_shift(x, k)
  end

  defp fixed(t, %Format{} = f), do: quantize(%Fixed{t: t, frac: f.frac}, f)

  @doc false
  deftransform scale_offset(v, amp, off, %Format{} = f) do
    v = if amp == 1.0, do: v, else: v |> mul(constant(amp, f)) |> rescale(f)
    v = if off == 0.0, do: v, else: add(v, constant(off, f))
    quantize(v, f)
  end
end
