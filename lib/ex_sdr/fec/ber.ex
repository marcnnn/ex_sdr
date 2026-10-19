defmodule ExSdr.FEC.Ber do
  @moduledoc """
  gr-fec's `ber_bf`: the bit error rate between two streams of packed bytes
  (every bit of a byte counts), as `log10(errors / (8 * bytes))`.

  `update/3` is one call of GNU Radio's `general_work`: it takes the bytes
  that are available on both inputs and returns `{value, state}`, `value`
  being `nil` when the block produces nothing (GNU Radio's block produces one
  value per call, so in streaming mode how many values there are depends on
  how the scheduler cuts the stream; the last value is always the BER of
  everything seen so far). Zero errors give `:neg_infinity`.

  Test mode (`test_mode: true`) is for BER curves: it counts until
  `berminerrors` errors have been seen and then gives one value; if the
  rate falls below `ber_limit` (with at least one error) it gives the limit
  instead; afterwards it returns `:done`.
  """
  import Bitwise

  defstruct test_mode: false, min_errors: 100, ber_limit: -7.0, errors: 0, total: 0

  @doc "Options `:test_mode` (false), `:berminerrors` (100), `:ber_limit` (-7.0)."
  def new(opts \\ []) do
    %__MODULE__{
      test_mode: Keyword.get(opts, :test_mode, false),
      min_errors: Keyword.get(opts, :berminerrors, 100),
      ber_limit: Keyword.get(opts, :ber_limit, -7.0)
    }
  end

  @doc "Number of differing bits between two binaries of the same length."
  def bit_errors(a, b) when byte_size(a) == byte_size(b) do
    a
    |> :binary.bin_to_list()
    |> Enum.zip(:binary.bin_to_list(b))
    |> Enum.reduce(0, fn {x, y}, acc -> acc + popcount(bxor(x, y)) end)
  end

  defp popcount(0), do: 0
  defp popcount(x), do: (x &&& 1) + popcount(x >>> 1)

  @doc "`log10(errors / (8 * bytes))`, `:neg_infinity` for no errors."
  def log_ber(0, _total), do: :neg_infinity
  def log_ber(errors, total), do: :math.log10(errors / (total * 8.0))

  @doc "One work call over the common length of `a` and `b`: `{value | nil | :done, state}`."
  def update(%__MODULE__{test_mode: true, errors: e, min_errors: m} = s, _a, _b) when e >= m,
    do: {:done, s}

  def update(%__MODULE__{} = s, a, b) do
    items = min(byte_size(a), byte_size(b))
    a = binary_part(a, 0, items)
    b = binary_part(b, 0, items)
    s = if items > 0, do: %{s | errors: s.errors + bit_errors(a, b), total: s.total + items}, else: s

    cond do
      not s.test_mode and items > 0 ->
        {log_ber(s.errors, s.total), s}

      not s.test_mode ->
        {nil, s}

      s.errors >= s.min_errors ->
        {log_ber(s.errors, s.total), s}

      s.errors > 0 and below_limit?(s) ->
        {s.ber_limit, %{s | errors: s.min_errors + 1}}

      true ->
        {nil, s}
    end
  end

  defp below_limit?(s), do: log_ber(s.errors, s.total) < s.ber_limit

  @doc "Host block function: `{a, b}` binaries in, `{value, state}` out."
  def block_fun({a, b}, state), do: update(state, a, b)
end
