defmodule ExSdr.FEC.LDPC.Graph do
  @moduledoc """
  Index tables of a code's Tanner graph, computed on the host, so the
  decoders are gathers over tensors (like `ExSdr.Comm.Permutation`).

  Edges are numbered row-major over a padded `m x dc` grid (`dc` the
  largest row weight): edge `(r, j)` is the `j`-th 1 of row `r`, flat
  index `r * dc + j`.

    * `row_vars` - `m` lists of `dc` variable indices; padding entries
      are `n` (a dummy variable).
    * `var_edges` - `n` lists of `dv` flat edge indices (the variable's
      edges in column order); padding entries are `m * dc` (a dummy edge
      whose message is 0).
    * `var_checks` - `n` lists of `dv` check indices; padding `m`.
    * `layers` - rows grouped so that no two rows of a group share a
      variable (greedy, in row order); per layer its rows, their padded
      variable lists and, for every variable, the flat index of its edge
      inside the layer (`pos`, 0 when it has none) and whether it has one
      (`in_layer`).
  """
  alias ExSdr.FEC.LDPC.Matrix

  defstruct [:n, :m, :dc, :dv, :row_vars, :var_edges, :var_checks, :layers]

  @doc "The tables of a parity-check matrix."
  def new(%Matrix{} = h) do
    n = h.cols
    m = h.rows
    rows = Tuple.to_list(h.row_lists)
    cols = Tuple.to_list(h.col_lists)
    dc = rows |> Enum.map(&length/1) |> Enum.max()
    dv = cols |> Enum.map(&length/1) |> Enum.max()

    row_vars = Enum.map(rows, &pad(&1, dc, n))

    # (row, variable) -> position j in the row
    pos =
      for {row, r} <- Enum.with_index(rows), {v, j} <- Enum.with_index(row), into: %{} do
        {{r, v}, j}
      end

    var_edges =
      for {col, v} <- Enum.with_index(cols) do
        col |> Enum.map(&(&1 * dc + Map.fetch!(pos, {&1, v}))) |> pad(dv, m * dc)
      end

    var_checks = Enum.map(cols, &pad(&1, dv, m))

    %__MODULE__{
      n: n,
      m: m,
      dc: dc,
      dv: dv,
      row_vars: row_vars,
      var_edges: var_edges,
      var_checks: var_checks,
      layers: layers(rows, n, dc)
    }
  end

  defp layers(rows, n, dc) do
    rows
    |> Enum.with_index()
    |> Enum.reduce([], fn {row, r}, layers ->
      set = MapSet.new(row)

      case Enum.find_index(layers, fn {_rows, used} -> MapSet.disjoint?(used, set) end) do
        nil -> layers ++ [{[r], set}]
        i -> List.update_at(layers, i, fn {rs, used} -> {rs ++ [r], MapSet.union(used, set)} end)
      end
    end)
    |> Enum.map(fn {layer_rows, _} ->
      vars = Enum.map(layer_rows, fn r -> rows |> Enum.at(r) |> pad(dc, n) end)

      local =
        for {row, i} <- Enum.with_index(vars), {v, j} <- Enum.with_index(row), v < n, into: %{} do
          {v, i * dc + j}
        end

      %{
        rows: layer_rows,
        row_vars: vars,
        pos: for(v <- 0..(n - 1), do: Map.get(local, v, 0)),
        in_layer: for(v <- 0..(n - 1), do: if(Map.has_key?(local, v), do: 1, else: 0))
      }
    end)
  end

  defp pad(list, size, value), do: list ++ List.duplicate(value, size - length(list))
end
