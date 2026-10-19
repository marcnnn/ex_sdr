defmodule ExSdr.Sources.LfsrSource do
  @moduledoc """
  `glfsr_source_b`: the output of a maximal length (or any) shift register
  as bits, built on `ExSdr.Comm.LFSR`.

  GNU Radio's register is the Galois form `state = (state >> 1) ^ (bit0 ?
  mask : 0)` with output `bit0`. `ExSdr.Comm.LFSR` is the Fibonacci form;
  both produce the same sequences for matching masks and seeds, and this
  module converts: with `m_k` the bits of the bit-reversed mask (over
  `degree` bits) the output obeys `s[n + degree] = xor of s[n + k]` for the
  `k` with `m_k = 1`, and the first `degree` outputs from the Galois seed
  are the Fibonacci register's initial contents.

  `mask: 0` (default) selects the same primitive polynomial as GNU Radio's
  default (`default_mask/1`, degrees 2-32), `seed` defaults to 1.

  The block is a `:host` block: input is a count (or a binary whose size is
  the count), output a binary of `0`/`1` bytes; the state is the register.
  """
  alias ExSdr.Comm.LFSR
  alias NxFixed.Block
  import Bitwise

  # Bit-reversed (recurrence) form of GNU Radio's default masks, degrees 2-32.
  @recurrence %{
    2 => 0x3,
    3 => 0x5,
    4 => 0x9,
    5 => 0x9,
    6 => 0x21,
    7 => 0x41,
    8 => 0x71,
    9 => 0x21,
    10 => 0x81,
    11 => 0x201,
    12 => 0x941,
    13 => 0x1601,
    14 => 0x2A01,
    15 => 0x4001,
    16 => 0x6801,
    17 => 0x4001,
    18 => 0x32001,
    19 => 0x64001,
    20 => 0x20001,
    21 => 0x80001,
    22 => 0x200001,
    23 => 0x40001,
    24 => 0xB00001,
    25 => 0x400001,
    26 => 0x3100001,
    27 => 0x6400001,
    28 => 0x2000001,
    29 => 0x8000001,
    30 => 0x25000001,
    31 => 0x10000001,
    32 => 0xEA000001
  }

  @doc "GNU Radio's default (Galois) mask for `degree` (2-32)."
  def default_mask(degree), do: reverse(Map.fetch!(@recurrence, degree), degree)

  @doc "A `NxFixed.Block` from `:degree`, `:mask` (0), `:seed` (1)."
  def block(opts) do
    degree = Keyword.fetch!(opts, :degree)
    mask = Keyword.get(opts, :mask, 0)
    mask = if mask == 0, do: default_mask(degree), else: mask
    seed = Keyword.get(opts, :seed, 1)

    %Block{
      name: :glfsr_source_b,
      kind: :host,
      fun: &generate/2,
      init: new(degree, mask, seed),
      item: {:none, :bit},
      hardware: :feedback,
      warmup: :unbounded,
      gnuradio: "digital.glfsr_source_b"
    }
  end

  @doc "The Fibonacci register equivalent to GNU Radio's `(degree, mask, seed)`."
  def new(degree, mask, seed) do
    window = galois_bits(seed, mask, degree)
    state = window |> Enum.with_index() |> Enum.reduce(0, fn {b, k}, acc -> acc ||| b <<< k end)
    rec = reverse(mask, degree)
    taps = for k <- 0..(degree - 1), (rec >>> k &&& 1) == 1, do: degree - k
    LFSR.new(length: degree, taps: taps, output: :last, seed: state)
  end

  # `count` outputs of GNU Radio's register (reference for `new/3`).
  @doc false
  def galois_bits(seed, mask, count) do
    {bits, _} =
      Enum.map_reduce(1..count//1, seed, fn _, s ->
        out = s &&& 1
        {out, if(out == 1, do: bxor(s >>> 1, mask), else: s >>> 1)}
      end)

    bits
  end

  defp reverse(x, bits),
    do: Enum.reduce(0..(bits - 1), 0, fn k, acc -> acc ||| (x >>> k &&& 1) <<< (bits - 1 - k) end)

  @doc "`n` bits (`n` or a binary of `n` bytes as input) as a binary of 0/1 bytes."
  def generate(input, lfsr) do
    n = if is_integer(input), do: input, else: byte_size(input)
    {bits, lfsr} = LFSR.bits(lfsr, n)
    {:erlang.list_to_binary(bits), lfsr}
  end
end
