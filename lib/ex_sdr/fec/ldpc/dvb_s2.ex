defmodule ExSdr.FEC.LDPC.DVBS2 do
  @moduledoc """
  The DVB-S2 and DVB-S2X LDPC codes (normal frames, n = 64800, and short
  frames, n = 16200) as `ExSdr.FEC.LDPC.IRA` configurations. The address
  tables (`dvb_s2_tables.txt`) are those of ETSI EN 302 307-1 and -2, as
  used by GNU Radio's gr-dtv `dvb_ldpc_bb`. DVB-T2 uses the same codes
  except normal 2/3 and short 3/5 (`standard: :dvbt2`). The punctured and
  shortened VL-SNR codes of DVB-S2X are not included.
  """
  alias ExSdr.FEC.LDPC.IRA

  @external_resource tables_path = Path.join(__DIR__, "dvb_s2_tables.txt")

  @codes tables_path
         |> File.read!()
         |> String.split("\n", trim: true)
         |> Enum.reject(&String.starts_with?(&1, "#"))
         |> Enum.chunk_while(
           nil,
           fn
             "code " <> header, acc ->
               [frame, rate, std, n, k, q] = String.split(header)

               next = %{
                 frame: String.to_atom(frame),
                 rate: rate,
                 standard: String.to_atom(std),
                 n: String.to_integer(n),
                 k: String.to_integer(k),
                 q: String.to_integer(q),
                 table: []
               }

               if acc,
                 do: {:cont, %{acc | table: Enum.reverse(acc.table)}, next},
                 else: {:cont, next}

             line, acc ->
               row = line |> String.split() |> Enum.map(&String.to_integer/1)
               {:cont, %{acc | table: [row | acc.table]}}
           end,
           fn acc -> {:cont, %{acc | table: Enum.reverse(acc.table)}, nil} end
         )

  @doc """
  The codes available: `[%{frame:, rate:, standard:, n:, k:, q:}]`
  (`standard` is `:all`, or `:dvbs2`/`:dvbt2` where they differ).
  """
  def codes, do: Enum.map(@codes, &Map.delete(&1, :table))

  @doc """
  The parameters of one code: `frame` `:normal` or `:short`, `rate` as a
  string (`"1/2"`, `"9/10"`, `"132/180"`), `standard` `:dvbs2` (default)
  or `:dvbt2`. Returns `%{n:, k:, q:, table:, group: 360}`.
  """
  def params(frame, rate, standard \\ :dvbs2) do
    case Enum.filter(@codes, &(&1.frame == frame and &1.rate == rate)) do
      [] ->
        raise ArgumentError, "no DVB-S2 LDPC code #{frame} #{rate}"

      [c] ->
        Map.put(c, :group, 360)

      several ->
        several |> Enum.find(&(&1.standard == standard)) |> Map.put(:group, 360)
    end
  end

  @doc "The code (`ExSdr.FEC.LDPC.Code`), see `params/3`."
  def code(frame, rate, standard \\ :dvbs2) do
    p = params(frame, rate, standard)
    IRA.code(n: p.n, k: p.k, q: p.q, table: p.table, group: p.group)
  end
end
