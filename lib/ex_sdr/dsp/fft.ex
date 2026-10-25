defmodule ExSdr.DSP.FFT do
  @moduledoc """
  Radix-2 decimation-in-time FFT along the last axis, written once against
  `NxFixed.Num` (float reference or fixed point) with views only: slices,
  reshapes, broadcasts and concatenation, no gathers.

  The data are held as `{batch, m, n / m}`: column c of row block m is the
  m-point DFT of the subsequence `x[c], x[c + n/m], ...`. Each stage joins
  the left and right halves of the columns (the even and odd
  subsequences) with one butterfly `a +- w^k b` per row, doubling m, so
  the output comes out in natural order without bit reversal (Stockham
  order). The butterflies, twiddles and rounding are those of any radix-2
  DIT FFT, e.g. a pipelined one in hardware.

  Fixed point: twiddles `exp(-i pi k / m)` in the `:twiddle` format, each
  twiddle product rescaled to the data format (rounding per format), each
  stage quantized (saturated) to it. The last `:shifts` stages halve their
  outputs (rounded), so the result is DFT / 2^shifts.
  """
  import Nx.Defn
  import Bitwise
  alias NxFixed.{Fixed, Num}

  @doc """
  FFT of the complex value `{re, im}` along its last axis (a power of two).
  Options: `:format` (data), `:twiddle` (default: `:format`), `:shifts`
  (default 0).
  """
  deftransform fft({re, im}, opts) do
    opts = Keyword.validate!(opts, [:format, twiddle: nil, shifts: 0])
    f = opts[:format]
    tw = opts[:twiddle] || f
    shape = Nx.shape(value(re))
    rank = tuple_size(shape)
    n = elem(shape, rank - 1)
    stages = round(:math.log2(n))

    unless 1 <<< stages == n and opts[:shifts] in 0..stages,
      do: raise(ArgumentError, "fft: length must be a power of two, shifts 0..log2(length)")

    batch = div(Tuple.product(shape), n)
    x = {view(re, &Nx.reshape(&1, {batch, 1, n})), view(im, &Nx.reshape(&1, {batch, 1, n}))}

    {re, im} =
      Enum.reduce(0..(stages - 1)//1, x, fn s, x ->
        stage(x, 1 <<< s, n, s >= stages - opts[:shifts], f, tw)
      end)

    {view(re, &Nx.reshape(&1, shape)), view(im, &Nx.reshape(&1, shape))}
  end

  # rows m -> 2m: top row k = a + w^k b, bottom row k = a - w^k b
  defp stage({re, im}, m, n, halve?, f, tw) do
    half = div(n, 2 * m)
    left = fn x -> view(x, &Nx.slice_along_axis(&1, 0, half, axis: 2)) end
    right = fn x -> view(x, &Nx.slice_along_axis(&1, half, half, axis: 2)) end
    a = {left.(re), left.(im)}
    b = {right.(re), right.(im)}
    # w^0 = 1 in the first stage: no multiplication
    {tr, ti} = if m == 1, do: b, else: b |> Num.cmul(twiddles(m, tw)) |> Num.rescale(f)
    {ar, ai} = a

    finish = fn v -> v |> halve(halve?, f) |> Num.quantize(f) end
    top = {finish.(Num.add(ar, tr)), finish.(Num.add(ai, ti))}
    bottom = {finish.(Num.sub(ar, tr)), finish.(Num.sub(ai, ti))}
    {concat(elem(top, 0), elem(bottom, 0)), concat(elem(top, 1), elem(bottom, 1))}
  end

  defp twiddles(m, format) do
    angles = for k <- 0..(m - 1), do: -:math.pi() * k / m

    {constants(Enum.map(angles, &:math.cos/1), m, format),
     constants(Enum.map(angles, &:math.sin/1), m, format)}
  end

  defp constants(values, m, %Num.Format{kind: :float}),
    do: Nx.tensor(values, type: :f32, backend: Nx.BinaryBackend) |> Nx.reshape({1, m, 1})

  defp constants(values, m, %Num.Format{kind: :fixed} = f) do
    type = if f.bits <= 16, do: :s16, else: :s32
    t = Num.quantize_constants(values, f) |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
    %Fixed{t: Nx.reshape(t, {1, m, 1}), frac: f.frac}
  end

  defp halve(x, false, _f), do: x
  defp halve(%Fixed{} = x, true, f), do: Num.rescale(%{x | frac: x.frac + 1}, f)
  defp halve(x, true, _f), do: Nx.multiply(x, 0.5)

  defp concat(%Fixed{} = a, %Fixed{} = b), do: %{a | t: Nx.concatenate([a.t, b.t], axis: 1)}
  defp concat(a, b), do: Nx.concatenate([a, b], axis: 1)

  defp value(%Fixed{t: t}), do: t
  defp value(t), do: t

  defp view(%Fixed{} = x, fun), do: %{x | t: fun.(x.t)}
  defp view(t, fun), do: fun.(t)
end
