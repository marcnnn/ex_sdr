defmodule ExSdr.FEC.LDPC.Matrix do
  @moduledoc """
  Sparse binary matrices (parity-check and generator matrices) and the
  GF(2) algebra the LDPC codes need on the host.

  A matrix keeps, for each row, the columns holding a 1 (`row_lists`) and,
  for each column, the rows (`col_lists`), 0-based. Read from an alist
  file the lists keep the file's order (GNU Radio's sum-product decoder
  iterates in that order); built otherwise they are ascending.

  For algebra, rows are Elixir integers used as bit sets (bit `j` is
  column `j`), see `row_bits/1`.
  """
  import Bitwise

  defstruct [:rows, :cols, :row_lists, :col_lists]

  @type t :: %__MODULE__{
          rows: pos_integer,
          cols: pos_integer,
          row_lists: tuple,
          col_lists: tuple
        }

  @doc """
  Parses an alist file's contents (MacKay's format, as used by gr-fec):
  `cols rows`, the maximum weights, the column weights, the row weights,
  then one line per column with its 1-based row indices and one line per
  row with its column indices. Zero entries (padding in some files) are
  ignored. The row section is optional; when present it must agree with
  the column section.
  """
  def parse_alist(text) when is_binary(text) do
    lines =
      text
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.map(fn l -> l |> String.split() |> Enum.map(&String.to_integer/1) end)

    [[cols, rows | _], _max, _col_weights, _row_weights | rest] = lines
    {col_lines, rest} = Enum.split(rest, cols)
    true = length(col_lines) == cols

    col_lists = Enum.map(col_lines, fn l -> for(r <- l, r > 0, do: r - 1) end)

    row_lines = Enum.take(rest, rows)

    row_lists =
      if length(row_lines) == rows and Enum.any?(row_lines, &(&1 != [])) do
        Enum.map(row_lines, fn l -> for(c <- l, c > 0, do: c - 1) end)
      else
        rows_from_cols(col_lists, rows)
      end

    m = %__MODULE__{
      rows: rows,
      cols: cols,
      row_lists: List.to_tuple(row_lists),
      col_lists: List.to_tuple(col_lists)
    }

    unless Enum.map(rows_from_cols(col_lists, rows), &Enum.sort/1) ==
             Enum.map(row_lists, &Enum.sort/1),
           do: raise(ArgumentError, "alist: row and column sections disagree")

    m
  end

  @doc "Reads an alist file, see `parse_alist/1`."
  def read_alist(path), do: path |> File.read!() |> parse_alist()

  @doc "The matrix in alist format (as gr-fec's `write_matrix_to_file`)."
  def to_alist(%__MODULE__{} = m) do
    cw = Enum.map(Tuple.to_list(m.col_lists), &length/1)
    rw = Enum.map(Tuple.to_list(m.row_lists), &length/1)
    line = fn l -> Enum.map_join(l, " ", &to_string/1) <> "\n" end

    IO.iodata_to_binary([
      line.([m.cols, m.rows]),
      line.([Enum.max(cw), Enum.max(rw)]),
      line.(cw),
      line.(rw),
      Enum.map(Tuple.to_list(m.col_lists), fn l -> line.(Enum.map(l, &(&1 + 1))) end),
      Enum.map(Tuple.to_list(m.row_lists), fn l -> line.(Enum.map(l, &(&1 + 1))) end)
    ])
  end

  @doc "From per-row column lists (`rows` x `cols`)."
  def from_row_lists(row_lists, cols) do
    rows = length(row_lists)
    row_lists = Enum.map(row_lists, &Enum.sort/1)

    %__MODULE__{
      rows: rows,
      cols: cols,
      row_lists: List.to_tuple(row_lists),
      col_lists: List.to_tuple(rows_from_cols(row_lists, cols))
    }
  end

  @doc "From a dense list of 0/1 rows."
  def from_dense([first | _] = dense) do
    row_lists = for row <- dense, do: for({1, j} <- Enum.with_index(row), do: j)
    from_row_lists(row_lists, length(first))
  end

  @doc "From bit-set rows (integers, bit `j` = column `j`)."
  def from_row_bits(bits, cols) do
    bits
    |> Enum.map(fn b -> for j <- 0..(cols - 1), (b >>> j &&& 1) == 1, do: j end)
    |> from_row_lists(cols)
  end

  @doc "Dense 0/1 rows."
  def to_dense(%__MODULE__{} = m) do
    for l <- Tuple.to_list(m.row_lists) do
      set = MapSet.new(l)
      for j <- 0..(m.cols - 1), do: if(MapSet.member?(set, j), do: 1, else: 0)
    end
  end

  @doc "Rows as bit sets."
  def row_bits(%__MODULE__{} = m),
    do: Enum.map(Tuple.to_list(m.row_lists), &bits_of/1)

  @doc "A list of positions as a bit set."
  def bits_of(list), do: Enum.reduce(list, 0, fn j, acc -> acc ||| 1 <<< j end)

  @doc "The transpose."
  def transpose(%__MODULE__{} = m) do
    %__MODULE__{
      rows: m.cols,
      cols: m.rows,
      row_lists: m.col_lists,
      col_lists: m.row_lists
    }
  end

  @doc "The submatrix of rows `r0..r0+nr-1` and columns `c0..c0+nc-1`, as bit-set rows."
  def sub_bits(%__MODULE__{} = m, r0, nr, c0, nc) do
    for r <- r0..(r0 + nr - 1)//1 do
      m.row_lists
      |> elem(r)
      |> Enum.filter(&(&1 >= c0 and &1 < c0 + nc))
      |> Enum.map(&(&1 - c0))
      |> bits_of()
    end
  end

  @doc "Parity of an integer's set bits."
  def parity(x), do: popcount(x) &&& 1

  @doc "Number of set bits."
  def popcount(0), do: 0
  def popcount(x) when x > 0, do: popcount(x, 0)

  defp popcount(0, acc), do: acc

  defp popcount(x, acc) do
    # 64 bits at a time
    w = x &&& 0xFFFFFFFFFFFFFFFF
    popcount(x >>> 64, acc + popcount64(w))
  end

  defp popcount64(w) do
    w = w - (w >>> 1 &&& 0x5555555555555555)
    w = (w &&& 0x3333333333333333) + (w >>> 2 &&& 0x3333333333333333)
    w = w + (w >>> 4) &&& 0x0F0F0F0F0F0F0F0F
    (w * 0x0101010101010101 &&& 0xFFFFFFFFFFFFFFFF) >>> 56
  end

  @doc """
  `H x` over GF(2) for a 0/1 list `x` of length `cols`: the syndrome, a
  0/1 list of length `rows`.
  """
  def syndrome(%__MODULE__{} = m, x) when is_list(x) do
    t = List.to_tuple(x)

    for l <- Tuple.to_list(m.row_lists),
        do: Enum.reduce(l, 0, fn j, acc -> bxor(acc, elem(t, j)) end)
  end

  @doc """
  Product of bit-set matrices over GF(2): rows of `a` (bits over the rows
  of `b`) times `b` (bit-set rows). Row `i` of the result is the XOR of
  the rows of `b` selected by row `i` of `a`. With bit-sliced vectors
  (each "row" an integer holding many independent values) this is also
  a matrix-vector product.
  """
  def mul_bits(a, b) do
    b = List.to_tuple(b)
    for row <- a, do: xor_selected(row, b, 0, 0)
  end

  defp xor_selected(0, _b, _j, acc), do: acc

  defp xor_selected(row, b, j, acc) do
    acc = if (row &&& 1) == 1, do: bxor(acc, elem(b, j)), else: acc
    xor_selected(row >>> 1, b, j + 1, acc)
  end

  @doc """
  Inverse of a square bit-set matrix (`size` x `size`) over GF(2) by
  Gauss-Jordan elimination; `{:ok, rows}` or `:singular`.
  """
  def inverse_bits(rows, size) do
    aug = rows |> Enum.with_index() |> Enum.map(fn {r, i} -> {r, 1 <<< i} end)

    Enum.reduce_while(0..(size - 1)//1, aug, fn col, aug ->
      {top, rest} = Enum.split(aug, col)

      case Enum.find_index(rest, fn {r, _} -> (r >>> col &&& 1) == 1 end) do
        nil ->
          {:halt, :singular}

        i ->
          {pr, pi} = pivot = Enum.at(rest, i)
          rest = List.delete_at(rest, i)

          eliminate = fn {r, ri} ->
            if (r >>> col &&& 1) == 1, do: {bxor(r, pr), bxor(ri, pi)}, else: {r, ri}
          end

          {:cont, Enum.map(top, eliminate) ++ [pivot | Enum.map(rest, eliminate)]}
      end
    end)
    |> case do
      :singular -> :singular
      aug -> {:ok, Enum.map(aug, &elem(&1, 1))}
    end
  end

  @doc "Rank over GF(2)."
  def rank(%__MODULE__{} = m) do
    m
    |> row_bits()
    |> Enum.reduce({0, []}, fn row, {rank, basis} ->
      row = Enum.reduce(basis, row, fn b, r -> if bxor(r, b) < r, do: bxor(r, b), else: r end)
      if row == 0, do: {rank, basis}, else: {rank + 1, Enum.sort([row | basis], :desc)}
    end)
    |> elem(0)
  end

  defp rows_from_cols(col_lists, rows) do
    acc = :array.new(rows, default: [])

    col_lists
    |> Enum.with_index()
    |> Enum.reduce(acc, fn {l, j}, acc ->
      Enum.reduce(l, acc, fn r, acc -> :array.set(r, [j | :array.get(r, acc)], acc) end)
    end)
    |> :array.to_list()
    |> Enum.map(&Enum.reverse/1)
  end
end
