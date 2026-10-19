defmodule ExSdr.FEC.LDPC.IRA do
  @moduledoc """
  Structured irregular repeat-accumulate LDPC codes given by parameters,
  the construction of DVB-S2/S2X, DVB-T2, DVB-C2 and ATSC 3.0:

    * `n`, `k` - codeword and information length; `m = n - k` parity bits
    * `group` - information bits per table row (360 in DVB)
    * `q` - address step within a group (`m / group` in DVB)
    * `table` - one list of parity addresses per group of `group`
      information bits

  Information bit `i = group * r + c` takes part in the parity checks
  `(a + c * q) mod m` for every address `a` of row `r`; parity bit `j` in
  checks `j` and `j + 1` (a staircase). Encoding accumulates every
  information bit into its parity addresses, then
  `p[j] = p[j] xor p[j - 1]` for `j = 1 .. m-1` (gr-dtv's `dvb_ldpc_bb`).
  Codewords are `[information bits, parity bits]`.

  Each group of `group` bits is a cyclic shift pattern, so H is
  quasi-cyclic up to a row permutation; the decoders use H as a sparse
  matrix like any other.
  """
  alias ExSdr.FEC.LDPC.{Code, Matrix}

  @doc """
  The code (`ExSdr.FEC.LDPC.Code`, kind `:ira`) for `n:`, `k:`, `q:`,
  `table:` and `group:` (default 360).
  """
  def code(opts) do
    n = Keyword.fetch!(opts, :n)
    k = Keyword.fetch!(opts, :k)
    q = Keyword.fetch!(opts, :q)
    group = Keyword.get(opts, :group, 360)
    table = Keyword.fetch!(opts, :table)
    m = n - k

    unless length(table) * group == k and m > 0,
      do: raise(ArgumentError, "ira: #{length(table)} table rows x #{group} must be k = #{k}")

    lut = lut(table, group, q, m)

    h_rows =
      lut
      |> Enum.with_index()
      |> Enum.map(fn {infos, j} -> infos ++ if(j > 0, do: [k + j - 1, k + j], else: [k]) end)

    %Code{
      kind: :ira,
      h: Matrix.from_row_lists(h_rows, n),
      n: n,
      k: k,
      info: Enum.to_list(0..(k - 1)),
      ru: %{lut: lut, q: q, group: group}
    }
  end

  @doc """
  Host reference encoder (gr-dtv's algorithm): information bits (0/1 list,
  `k`) to the codeword (`n`).
  """
  def encode(%Code{kind: :ira, ru: %{lut: lut}}, bits) do
    s = List.to_tuple(bits)

    acc =
      Enum.map(lut, fn infos ->
        Enum.reduce(infos, 0, fn i, a -> Bitwise.bxor(a, elem(s, i)) end)
      end)

    {parity, _} =
      Enum.map_reduce(acc, 0, fn a, prev -> {Bitwise.bxor(a, prev), Bitwise.bxor(a, prev)} end)

    bits ++ parity
  end

  # For each parity bit, the information bits accumulated into it. An
  # information bit that hits the same parity bit twice cancels (XOR).
  defp lut(table, group, q, m) do
    entries =
      for {row, r} <- Enum.with_index(table), c <- 0..(group - 1), a <- row do
        {rem(a + c * q, m), group * r + c}
      end

    by_parity = Enum.group_by(entries, &elem(&1, 0), &elem(&1, 1))

    for j <- 0..(m - 1) do
      by_parity
      |> Map.get(j, [])
      |> Enum.frequencies()
      |> Enum.filter(fn {_, count} -> rem(count, 2) == 1 end)
      |> Enum.map(&elem(&1, 0))
      |> Enum.sort()
    end
  end
end
