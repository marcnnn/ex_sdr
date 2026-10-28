defmodule ExSdr.FEC.LDPCCodeTest do
  @moduledoc """
  Codes and encoders on any parity-check matrix: alist files, the three
  code constructions, H c = 0 for every codeword, the information bits at
  their positions, tensor encoder equal to the host reference.
  """
  use ExUnit.Case, async: true
  @moduletag :ldpc_alist
  use ExUnitProperties

  import Bitwise

  alias ExSdr.FEC.LDPC
  alias ExSdr.FEC.LDPC.{Code, Matrix}

  @alist_dir Path.expand("../fixtures/gnuradio/fec/ldpc_alist", __DIR__)

  defp alist(name), do: Path.join(@alist_dir, name)

  defp tensor_encode(code, kind, bits) do
    b = LDPC.block(kind, code: code)
    {out, _} = LDPC.run(b, Nx.tensor(List.flatten(bits), type: :u8))
    out |> Nx.to_flat_list() |> Enum.chunk_every(code.n)
  end

  defp check_code(code, kind, messages) do
    words = tensor_encode(code, kind, messages)

    for {s, c} <- Enum.zip(messages, words) do
      assert c == Code.encode(code, s)
      assert Code.codeword?(code, c)
      assert Enum.map(code.info, &Enum.at(c, &1)) == s
    end
  end

  describe "alist" do
    test "parse, write and parse again gives the same matrix" do
      for f <- File.ls!(@alist_dir) do
        m = Matrix.read_alist(alist(f))
        assert Matrix.parse_alist(Matrix.to_alist(m)) == m
      end
    end

    test "the simple G matrix" do
      m = Matrix.read_alist(alist("simple_g_matrix.alist"))
      assert {m.rows, m.cols} == {4, 8}

      assert Matrix.to_dense(m) == [
               [1, 0, 0, 0, 0, 1, 1, 1],
               [0, 1, 0, 0, 1, 0, 1, 1],
               [0, 0, 1, 0, 1, 1, 0, 1],
               [0, 0, 0, 1, 1, 1, 1, 0]
             ]
    end

    test "rejects inconsistent row and column sections" do
      bad = "2 1\n1 2\n1 1\n2\n1\n1\n1\n"
      assert_raise ArgumentError, fn -> Matrix.parse_alist(bad) end
    end
  end

  describe "GF(2)" do
    property "inverse_bits gives the inverse or reports singular" do
      check all size <- integer(1..10),
                rows <- list_of(integer(0..((1 <<< size) - 1)), length: size),
                max_runs: 200 do
        case Matrix.inverse_bits(rows, size) do
          {:ok, inv} ->
            identity = for i <- 0..(size - 1), do: 1 <<< i
            assert Matrix.mul_bits(inv, rows) == identity
            assert Matrix.mul_bits(rows, inv) == identity

          :singular ->
            assert Matrix.rank(Matrix.from_row_bits(rows, size)) < size
        end
      end
    end
  end

  describe "gr-fec's matrices" do
    test "H matrices with gap: every codeword satisfies H c = 0" do
      for {f, gap} <- [
            {"n_0100_k_0042_gap_02.alist", 2},
            {"n_0100_k_0023_gap_10.alist", 10},
            {"n_0300_k_0152_gap_03.alist", 3}
          ] do
        code = LDPC.h_matrix(alist(f), gap)
        check_code(code, :ldpc_par_mtrx_encoder, random_messages(code.k, 8, gap))
      end
    end

    test "G matrices" do
      for f <- ["n_0100_k_0058_gen_matrix.alist", "simple_g_matrix.alist"] do
        code = LDPC.g_matrix(alist(f))
        check_code(code, :ldpc_gen_mtrx_encoder, random_messages(code.k, 8, 1))
      end
    end

    test "cldpc (ldpc_encoder) on H and G-derived matrices" do
      for f <- ["n_0100_k_0042_gap_02.alist", "n_0300_k_0152_gap_03.alist"] do
        code = LDPC.cldpc(alist(f))
        assert code.k == code.n - Matrix.rank(code.h)
        check_code(code, :ldpc_encoder, random_messages(code.k, 8, 2))
      end
    end

    test "a gap too small for the matrix is rejected; a larger one encodes the same" do
      path = alist("n_0100_k_0042_gap_02.alist")
      assert_raise ArgumentError, fn -> LDPC.h_matrix(path, 1) end
      # the parity columns of H are invertible, so the parity bits are unique
      assert LDPC.h_matrix(path, 5).generator == LDPC.h_matrix(path, 2).generator
    end

    test "a matrix that is not [I P] is rejected as G" do
      assert_raise ArgumentError, fn -> LDPC.g_matrix(alist("n_0100_k_0042_gap_02.alist")) end
    end
  end

  describe "any parity-check matrix" do
    property "Richardson-Urbanke on random H = [T A B; E C D]" do
      check all t <- integer(3..12),
                gap <- integer(0..3),
                k <- integer(2..12),
                seed <- integer(0..1_000_000),
                max_runs: 60 do
        :rand.seed(:exsss, seed)
        h = random_ru_matrix(t, gap, k)

        case safe(fn -> LDPC.h_matrix(h, gap) end) do
          {:ok, code} ->
            assert code.k == k
            check_code(code, :ldpc_par_mtrx_encoder, random_messages(k, 4, seed))

          :rejected ->
            # only when phi is singular (T is valid by construction)
            :ok
        end
      end
    end

    property "G = [I P] with random P" do
      check all k <- integer(1..12),
                p <- integer(1..12),
                seed <- integer(0..1_000_000),
                max_runs: 40 do
        :rand.seed(:exsss, seed)

        rows =
          for i <- 0..(k - 1), do: [i | for(j <- 0..(p - 1), :rand.uniform(2) == 1, do: k + j)]

        code = LDPC.g_matrix(Matrix.from_row_lists(rows, k + p))
        check_code(code, :ldpc_gen_mtrx_encoder, random_messages(k, 4, seed))
      end
    end

    property "cldpc on random (possibly rank deficient) H" do
      check all m <- integer(2..10),
                extra <- integer(1..12),
                seed <- integer(0..1_000_000),
                max_runs: 40 do
        :rand.seed(:exsss, seed)
        n = m + extra
        rows = for _ <- 1..m, do: for(j <- 0..(n - 1), :rand.uniform(3) == 1, do: j)
        h = Matrix.from_row_lists(rows, n)
        code = LDPC.cldpc(h)
        assert code.k == n - Matrix.rank(h)
        check_code(code, :ldpc_encoder, random_messages(code.k, 4, seed))
      end
    end
  end

  defp safe(fun) do
    {:ok, fun.()}
  rescue
    ArgumentError -> :rejected
  end

  defp random_messages(k, count, seed) do
    :rand.seed(:exsss, seed + 17)
    for _ <- 1..count, do: for(_ <- 1..k, do: :rand.uniform(2) - 1)
  end

  # T (t x t) upper triangular with unit diagonal, other blocks random.
  defp random_ru_matrix(t, gap, k) do
    n = t + gap + k
    bit = fn -> :rand.uniform(4) == 1 end

    top =
      for i <- 0..(t - 1) do
        [i] ++ for(j <- (i + 1)..(n - 1)//1, bit.(), do: j)
      end

    bottom = for _ <- 1..gap//1, do: for(j <- 0..(n - 1), bit.(), do: j)
    Matrix.from_row_lists(top ++ bottom, n)
  end
end
