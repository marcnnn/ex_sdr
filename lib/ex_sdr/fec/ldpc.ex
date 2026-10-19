defmodule ExSdr.FEC.LDPC do
  @moduledoc """
  LDPC codes: gr-fec's LDPC encoders and decoders plus integer min-sum
  decoders for hardware, see `docs/blocks/fec_ldpc.md`.

  A code (`ExSdr.FEC.LDPC.Code`) comes from an alist file like in GNU
  Radio: `h_matrix/2` (`ldpc_H_matrix`, Richardson-Urbanke encoding),
  `g_matrix/1` (`ldpc_G_matrix`) or `cldpc/1` (the code of
  `ldpc_encoder`/`ldpc_decoder`). Any parity-check matrix works; the
  decoders only need H.

  `block(name, opts)` returns the `NxFixed.Block`; all are `:frame`
  blocks on whole codewords, batched over any number of frames.

    * `:ldpc_par_mtrx_encoder`, `:ldpc_gen_mtrx_encoder`, `:ldpc_encoder`
      - `code:` (or `alist:` and `gap:`); bits (`u8`) in, `k` -> `n`.
    * `:ldpc_bit_flip_decoder` - `code:`, `max_iter:`; LLRs in (only the
      signs are used), `n` -> `k` bits.
    * `:ldpc_decoder` - gr-fec's sum-product decoder (tanh rule, flooding,
      `f64`): `code:` (default `cldpc(alist)`), `max_iter:`.
    * `:ldpc_bp_decoder` - the same algorithm with `schedule: :flooding`
      or `:layered` and `float_type:`; the float reference.
    * `:ldpc_min_sum_decoder` - min-sum, `schedule:` (`:layered`
      default, `:flooding`), `scale:` (normalised min-sum, e.g. 0.75),
      `offset:` (offset min-sum, in LLR units), `format:`
      (`NxFixed.Num.float()` default, or `NxFixed.Num.q(bits, frac)`:
      channel LLRs and messages in `bits`-bit integers with `frac`
      fractional bits, saturated to +-(2^(bits-1) - 1)), `state_bits:`
      (posterior LLRs, default `bits + 2`), `max_iter:`.

  LLRs are positive for 1 (as GNU Radio's decoders and
  `ExSdr.Digital`'s soft decoder). Decoders output the information bits,
  or with `output: :codeword` all `n` hard decisions.
  """
  @behaviour NxFixed.Block

  import Bitwise

  alias ExSdr.FEC.LDPC.{Code, Graph, Kernels}
  alias NxFixed.{Block, Fixed, Num}

  @blocks ~w(ldpc_par_mtrx_encoder ldpc_gen_mtrx_encoder ldpc_encoder ldpc_bit_flip_decoder
             ldpc_decoder ldpc_bp_decoder ldpc_min_sum_decoder)a

  @doc "The names `block/2` knows."
  def blocks, do: @blocks

  @doc "`ldpc_H_matrix`: see `ExSdr.FEC.LDPC.Code.h_matrix/2`."
  defdelegate h_matrix(alist, gap), to: Code

  @doc "`ldpc_G_matrix`: see `ExSdr.FEC.LDPC.Code.g_matrix/1`."
  defdelegate g_matrix(alist), to: Code

  @doc "The code of `ldpc_encoder`/`ldpc_decoder`: see `ExSdr.FEC.LDPC.Code.cldpc/1`."
  defdelegate cldpc(alist), to: Code

  @impl NxFixed.Block
  def block(name, opts \\ [])

  def block(name, opts)
      when name in [:ldpc_par_mtrx_encoder, :ldpc_gen_mtrx_encoder, :ldpc_encoder] do
    code = code(name, opts)

    %Block{
      name: name,
      kind: :frame,
      fun: &Kernels.encode/3,
      opts: [n: code.n, k: code.k, generator: Code.generator_dense(code)],
      rate: {code.k, code.n},
      item: {:bit, :bit},
      hardware: :transform,
      gnuradio: "fec.#{name}"
    }
  end

  def block(:ldpc_bit_flip_decoder, opts) do
    code = code(:ldpc_bit_flip_decoder, opts)
    g = Graph.new(code.h)

    %Block{
      name: :ldpc_bit_flip_decoder,
      kind: :frame,
      fun: &Kernels.bit_flip/3,
      opts:
        [
          n: code.n,
          m: g.m,
          row_vars: g.row_vars,
          var_checks: g.var_checks,
          max_iter: Keyword.get(opts, :max_iter, 100)
        ] ++ output_opts(code, opts),
      rate: rate(code, opts),
      item: {:real, :bit},
      hardware: :transform,
      gnuradio: "fec.ldpc_bit_flip_decoder"
    }
  end

  def block(name, opts) when name in [:ldpc_decoder, :ldpc_bp_decoder] do
    code = code(name, opts)

    schedule =
      if name == :ldpc_decoder, do: :flooding, else: Keyword.get(opts, :schedule, :flooding)

    soft_block(name, code, opts,
      check: :bp,
      schedule: schedule,
      float_type: Keyword.get(opts, :float_type, :f64),
      gnuradio: if(name == :ldpc_decoder, do: "fec.ldpc_decoder")
    )
  end

  def block(:ldpc_min_sum_decoder, opts) do
    code = code(:ldpc_min_sum_decoder, opts)
    format = Keyword.get(opts, :format, Num.float())
    scale = opts[:scale]
    offset = opts[:offset]

    fixed =
      case format do
        %Num.Format{kind: :float} ->
          [fixed: false, float_type: :f32]

        %Num.Format{kind: :fixed, bits: bits, frac: frac} ->
          state_bits = Keyword.get(opts, :state_bits, bits + 2)

          if state_bits < bits or state_bits > 16,
            do: raise(ArgumentError, "state_bits #{state_bits}")

          [
            fixed: true,
            msg_max: (1 <<< (bits - 1)) - 1,
            state_max: (1 <<< (state_bits - 1)) - 1,
            msg_type: container(bits),
            state_type: container(state_bits),
            scale16: if(scale, do: round(scale * 16)),
            offset_q: if(offset, do: round(offset * (1 <<< frac)))
          ]
      end

    soft_block(
      :ldpc_min_sum_decoder,
      code,
      opts,
      [
        check: :min_sum,
        schedule: Keyword.get(opts, :schedule, :layered),
        scale: scale,
        offset: offset,
        format: format,
        gnuradio: nil
      ] ++ fixed
    )
  end

  defp soft_block(name, code, opts, kernel) do
    g = Graph.new(code.h)
    {gnuradio, kernel} = Keyword.pop(kernel, :gnuradio)

    layers =
      if kernel[:schedule] == :layered,
        do: Enum.map(g.layers, &Map.take(&1, [:rows, :row_vars, :pos, :in_layer])),
        else: []

    %Block{
      name: name,
      kind: :frame,
      fun: &Kernels.soft/3,
      opts:
        [
          n: code.n,
          m: g.m,
          dc: g.dc,
          row_vars: g.row_vars,
          var_edges: g.var_edges,
          layers: layers,
          max_iter: Keyword.get(opts, :max_iter, 50)
        ] ++ Keyword.put_new(kernel, :fixed, false) ++ output_opts(code, opts),
      rate: rate(code, opts),
      item: {:real, :bit},
      hardware: :transform,
      gnuradio: gnuradio
    }
  end

  defp output_opts(code, opts) do
    case Keyword.get(opts, :output, :info) do
      :info -> [info: code.info, output: :info]
      :codeword -> [info: code.info, output: :codeword]
    end
  end

  defp rate(code, opts) do
    case Keyword.get(opts, :output, :info) do
      :info -> {code.n, code.k}
      :codeword -> {code.n, code.n}
    end
  end

  defp code(name, opts) do
    case Keyword.fetch(opts, :code) do
      {:ok, %Code{} = c} ->
        c

      :error ->
        alist = Keyword.fetch!(opts, :alist)

        case name do
          :ldpc_par_mtrx_encoder -> Code.h_matrix(alist, Keyword.fetch!(opts, :gap))
          :ldpc_gen_mtrx_encoder -> Code.g_matrix(alist)
          _ -> if opts[:gap], do: Code.h_matrix(alist, opts[:gap]), else: Code.cldpc(alist)
        end
    end
  end

  defp container(bits) when bits <= 8, do: :s8
  defp container(_bits), do: :s16

  @doc """
  Quantises float LLRs (a tensor or list) to a decoder's fixed-point
  `format`: `round(llr * 2^frac)`, saturated to +-(2^(bits-1) - 1).
  """
  def quantize_llr(llr, %Num.Format{kind: :fixed, bits: bits, frac: frac}) do
    max = (1 <<< (bits - 1)) - 1

    t =
      llr
      |> Nx.tensor(type: :f32)
      |> Nx.multiply(1 <<< frac)
      |> Nx.round()
      |> Nx.clip(-max, max)
      |> Nx.as_type(container(bits))

    %Fixed{t: t, frac: frac}
  end

  @doc """
  Runs `block` on `input` (whole frames), compiled (EXLA when available)
  with the iterative decoders' early exit (`loop: :while`). Returns
  `{output, state}` like `NxFixed.Block.run/4`; same result as running
  `block.fun` directly.
  """
  def run(%Block{} = b, input, state \\ nil) do
    opts = Keyword.put(b.opts, :loop, :while)
    template = Nx.Defn.Composite.traverse(input, &Nx.to_template/1)
    key = {__MODULE__, b.name, :erlang.phash2(opts), template}

    f =
      case :persistent_term.get(key, nil) do
        nil ->
          compiler = if Elixir.Code.ensure_loaded?(EXLA), do: EXLA, else: Nx.Defn.Evaluator
          fun = fn x -> b.fun.(x, nil, opts) |> elem(0) end
          f = Nx.Defn.compile(fun, [template], compiler: compiler)
          :persistent_term.put(key, f)
          f

        f ->
          f
      end

    {f.(input) |> Nx.backend_transfer(Nx.BinaryBackend), state}
  end
end
