defmodule ExSdr.FEC.LDPC.Code do
  @moduledoc """
  A binary linear block code given by a parity-check matrix, plus how
  information bits map into codewords. The three constructors mirror
  gr-fec's matrix objects:

    * `h_matrix/2` - `ldpc_H_matrix(alist, gap)`: H in approximate lower
      triangular form for Richardson-Urbanke encoding. Codeword
      `[p1 (n-k-gap), p2 (gap), s (k)]`: parity bits first, information
      bits last.
    * `g_matrix/1` - `ldpc_G_matrix(alist)`: a systematic generator
      `G = [I P]`; `H = [P' I]`, information bits first.
    * `cldpc/1` - the code of `ldpc_encoder`/`ldpc_decoder`: H as given,
      a generator from Gaussian elimination with column swaps (gr-fec's
      `GF2Mat::get_G`), information bits where that elimination puts
      them. `k = n - rank(H)`.

  Fields: `h` (`ExSdr.FEC.LDPC.Matrix`), `n`, `k`, `info` (codeword
  position of information bit `j`), `generator` (for each codeword bit
  the information bits it is the XOR of, as a bit set), `kind`, and `ru`
  (the Richardson-Urbanke submatrices of an `h_matrix`).

  The tensor encoder uses `generator`; `encode/2` here is the host
  reference with gr-fec's algorithm (back substitution for
  `h_matrix`, `G' s` for `g_matrix`, `cldpc::encode`).
  """
  import Bitwise
  alias ExSdr.FEC.LDPC.Matrix

  defstruct [:kind, :h, :n, :k, :info, :generator, :ru]

  @type t :: %__MODULE__{}

  @doc """
  `ldpc_H_matrix`: `alist` (path, alist text via `{:alist, text}`, or a
  `Matrix`) with `n - k` rows and the gap `g`. H is partitioned as

      [ T  A  B ]   rows 0 .. t-1      (t = n - k - g)
      [ E  C  D ]   rows t .. n-k-1
        t  g  k     columns

  T must be upper triangular with a unit diagonal and
  `phi = E T^-1 A + C` invertible (gr-fec checks the same; it fails the
  construction otherwise).
  """
  def h_matrix(alist, gap) when is_integer(gap) and gap >= 0 do
    h = matrix(alist)
    n = h.cols
    m = h.rows
    k = n - m
    t = m - gap
    if t <= 0 or k <= 0, do: raise(ArgumentError, "h_matrix: bad gap #{gap} for #{m}x#{n}")

    tm = Matrix.sub_bits(h, 0, t, 0, t)

    for {row, i} <- Enum.with_index(tm) do
      unless (row >>> i &&& 1) == 1 and (row &&& (1 <<< i) - 1) == 0,
        do:
          raise(
            ArgumentError,
            "h_matrix: T (#{t}x#{t}) is not upper triangular with unit diagonal at row #{i}; check the gap"
          )
    end

    ru = %{
      t: t,
      gap: gap,
      tm: tm,
      a: Matrix.sub_bits(h, 0, t, t, gap),
      b: Matrix.sub_bits(h, 0, t, t + gap, k),
      e: Matrix.sub_bits(h, t, gap, 0, t),
      c: Matrix.sub_bits(h, t, gap, t, gap),
      d: Matrix.sub_bits(h, t, gap, t + gap, k)
    }

    phi_inv =
      if gap == 0 do
        []
      else
        # T^-1 A, bit-sliced: row i holds the gap columns of (T^-1 A)[i].
        t_inv_a = back_solve(ru.tm, ru.a)
        phi = Enum.zip_with(Matrix.mul_bits(ru.e, t_inv_a), ru.c, &bxor/2)

        case Matrix.inverse_bits(phi, gap) do
          {:ok, inv} -> inv
          :singular -> raise ArgumentError, "h_matrix: phi is singular; check the gap"
        end
      end

    ru = Map.put(ru, :phi_inv, phi_inv)

    # Encoding all unit vectors at once: input bit j is bit j of every value.
    generator = ru_encode(ru, for(j <- 0..(k - 1), do: 1 <<< j))

    %__MODULE__{
      kind: :h_matrix,
      h: h,
      n: n,
      k: k,
      info: Enum.to_list((n - k)..(n - 1)),
      generator: List.to_tuple(generator),
      ru: ru
    }
  end

  @doc """
  `ldpc_G_matrix`: `alist` holds a `k x n` generator `[I P]` (the
  identity is checked, as gr-fec does).
  """
  def g_matrix(alist) do
    g = matrix(alist)
    k = g.rows
    n = g.cols

    for {row, i} <- Enum.with_index(Tuple.to_list(g.row_lists)) do
      unless Enum.filter(row, &(&1 < k)) == [i],
        do: raise(ArgumentError, "g_matrix: G is not of the form [I P] (row #{i})")
    end

    # H = [P' I]: row r has P[j][r] for j < k, and a 1 at column k + r.
    h_rows =
      for r <- 0..(n - k - 1) do
        for(j <- elem(g.col_lists, k + r), do: j) ++ [k + r]
      end

    %__MODULE__{
      kind: :g_matrix,
      h: Matrix.from_row_lists(h_rows, n),
      n: n,
      k: k,
      info: Enum.to_list(0..(k - 1)),
      generator: g.col_lists |> Tuple.to_list() |> Enum.map(&Matrix.bits_of/1) |> List.to_tuple()
    }
  end

  @doc """
  The code of gr-fec's `ldpc_encoder`/`ldpc_decoder` (`cldpc`) for the
  parity-check matrix in `alist`.
  """
  def cldpc(alist) do
    h = matrix(alist)
    n = h.cols
    {g_rows, perm, rank} = get_g(Matrix.row_bits(h), h.rows, n)
    k = n - rank
    perm_t = List.to_tuple(perm)

    # x[i] = s[i - rank] for i >= rank; x[i] = G[i][rank..n-1] . s for i < rank,
    # where G's columns are in permuted order; codeword y[perm[i]] = x[i].
    parity_gen =
      for i <- 0..(rank - 1)//1 do
        row = Enum.at(g_rows, i)

        for(j <- 0..(k - 1)//1, (row >>> elem(perm_t, rank + j) &&& 1) == 1, do: 1 <<< j)
        |> Enum.reduce(0, &bor/2)
      end

    x_gen = parity_gen ++ for(j <- 0..(k - 1)//1, do: 1 <<< j)

    generator =
      x_gen
      |> Enum.with_index()
      |> Enum.map(fn {g, i} -> {elem(perm_t, i), g} end)
      |> Enum.sort()
      |> Enum.map(&elem(&1, 1))

    %__MODULE__{
      kind: :cldpc,
      h: h,
      n: n,
      k: k,
      info: for(j <- 0..(k - 1)//1, do: elem(perm_t, rank + j)),
      generator: List.to_tuple(generator)
    }
  end

  @doc """
  Host reference encoder: the information bits (0/1 list of length `k`)
  to a codeword (0/1 list of length `n`), with gr-fec's algorithm for
  the code's kind.
  """
  def encode(%__MODULE__{kind: :h_matrix, ru: ru}, bits), do: ru_encode(ru, bits)

  def encode(%__MODULE__{} = c, bits) do
    s = Matrix.bits_of(for {1, j} <- Enum.with_index(bits), do: j)
    for i <- 0..(c.n - 1), do: Matrix.parity(elem(c.generator, i) &&& s)
  end

  @doc "The generator as a dense `n x k` list of 0/1 rows (`codeword = G' s`)."
  def generator_dense(%__MODULE__{} = c) do
    for i <- 0..(c.n - 1) do
      g = elem(c.generator, i)
      for j <- 0..(c.k - 1), do: g >>> j &&& 1
    end
  end

  @doc "`H c = 0`?"
  def codeword?(%__MODULE__{h: h}, bits), do: Enum.all?(Matrix.syndrome(h, bits), &(&1 == 0))

  # -- Richardson-Urbanke -------------------------------------------------------------

  # gr-fec's ldpc_H_matrix::encode, on values that may be bit-sliced
  # (integers whose bits are independent messages):
  #   p2 = phi^-1 (D s + E T^-1 B s), p1 = T^-1 (A p2 + B s).
  defp ru_encode(ru, s) do
    bs = Matrix.mul_bits(ru.b, s)
    temp2 = back_solve(ru.tm, bs)
    temp3 = Matrix.mul_bits(ru.e, temp2)
    temp4 = Matrix.mul_bits(ru.d, s)
    temp5 = Enum.zip_with(temp4, temp3, &bxor/2)
    p2 = if ru.gap == 0, do: [], else: Matrix.mul_bits(ru.phi_inv, temp5)
    temp6 = if ru.gap == 0, do: List.duplicate(0, ru.t), else: Matrix.mul_bits(ru.a, p2)
    temp7 = Enum.zip_with(temp6, bs, &bxor/2)
    p1 = back_solve(ru.tm, temp7)
    p1 ++ p2 ++ s
  end

  # Solves T x = y for upper triangular T with unit diagonal (rows as bit
  # sets), from the last row up: x[i] = y[i] + sum_{j>i} T[i][j] x[j].
  defp back_solve(tm, y) do
    t = length(tm)
    tm = List.to_tuple(tm)
    y = List.to_tuple(y)

    Enum.reduce((t - 1)..0//-1, %{}, fn i, x ->
      row = elem(tm, i) >>> (i + 1)
      v = xor_upper(row, i + 1, x, elem(y, i))
      Map.put(x, i, v)
    end)
    |> then(fn x -> for i <- 0..(t - 1), do: Map.fetch!(x, i) end)
  end

  defp xor_upper(0, _j, _x, acc), do: acc

  defp xor_upper(row, j, x, acc) do
    acc = if (row &&& 1) == 1, do: bxor(acc, Map.fetch!(x, j)), else: acc
    xor_upper(row >>> 1, j + 1, x, acc)
  end

  # -- cldpc: GF2Mat::get_G ------------------------------------------------------------

  # Gaussian elimination with column swaps, as gr-fec does it. The column
  # order is tracked in `perm` (perm[p] = original column at position p)
  # instead of moving bits. Returns {rows, perm, rank}; rows keep original
  # column indexing.
  defp get_g(rows, m, n) do
    perm = :array.from_list(Enum.to_list(0..(n - 1)))
    rows = :array.from_list(rows)
    eliminate(rows, perm, 0, m, 0, n)
  end

  defp eliminate(rows, perm, diag, limit, rank, _n) when diag >= limit,
    do: {:array.to_list(rows), :array.to_list(perm), rank}

  defp eliminate(rows, perm, diag, limit, rank, n) do
    row = :array.get(diag, rows)

    case Enum.find(diag..(n - 1)//1, fn p -> (row >>> :array.get(p, perm) &&& 1) == 1 end) do
      nil ->
        # Move this (now zero) row to the end of the active rows.
        rows =
          Enum.reduce(diag..(limit - 2)//1, rows, fn r, acc ->
            :array.set(r, :array.get(r + 1, acc), acc)
          end)
          |> then(&:array.set(limit - 1, row, &1))

        eliminate(rows, perm, diag, limit - 1, rank, n)

      p ->
        a = :array.get(diag, perm)
        b = :array.get(p, perm)
        perm = :array.set(p, a, :array.set(diag, b, perm))
        col = b
        m = :array.size(rows)

        rows =
          Enum.reduce(0..(m - 1), rows, fn r, acc ->
            v = :array.get(r, acc)

            if r != diag and (v >>> col &&& 1) == 1,
              do: :array.set(r, bxor(v, row), acc),
              else: acc
          end)

        eliminate(rows, perm, diag + 1, limit, rank + 1, n)
    end
  end

  defp matrix(%Matrix{} = m), do: m
  defp matrix({:alist, text}), do: Matrix.parse_alist(text)
  defp matrix(path) when is_binary(path), do: Matrix.read_alist(path)
end
