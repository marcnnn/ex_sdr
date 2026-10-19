defmodule ExSdr.FEC.Polar do
  @moduledoc """
  gr-fec's `polar_encoder` and `polar_encoder_systematic` as
  `ExSdr.FEC.Generic` objects, and the polar code constructions that choose
  the frozen bit positions (gr-fec's python helpers `fec.polar`).

  Code: `block_size` N (a power of 2), `num_info_bits` K, the N - K frozen
  positions (ascending, `frozen_bit_positions`) and their values
  (`frozen_bit_values`, default all 0). The info bits fill the other
  positions of the vector `u` in order; the codeword is the transform of
  `u` that GNU Radio's volk kernel `encodeframepolar` computes: log2 N stages,
  stage s turns every branch of N / 2^s items `t` into
  `[t0 xor t1, t2 xor t3, ..., t1, t3, ...]`. The decoders
  (`ExSdr.FEC.PolarDecoder`) decode `u` in natural order.

  `packed: true` (encoder only) takes K / 8 bytes (MSB first) and gives
  N / 8 bytes, GNU Radio's `is_packed`; it is the same code.

  The systematic encoder returns a codeword that contains the info bits (at
  the bit reversed info positions): it encodes, keeps the bits at the
  bit reversed info positions at the info positions and encodes again.
  """
  @behaviour ExSdr.FEC.Generic

  import Bitwise

  alias ExSdr.FEC.Generic

  @doc """
  Encoder object. Options: `:block_size`, `:num_info_bits`,
  `:frozen_bit_positions`, `:frozen_bit_values`, `:packed`.
  """
  def encoder(opts), do: make(:encoder, "polar_encoder", opts)

  @doc "Systematic encoder (no frozen values: all 0)."
  def encoder_systematic(opts), do: make(:encoder, "polar_encoder_systematic", opts)

  @doc false
  def make(kind, name, opts) do
    n = Keyword.fetch!(opts, :block_size)
    k = Keyword.fetch!(opts, :num_info_bits)
    positions = Keyword.fetch!(opts, :frozen_bit_positions)
    values = Keyword.get(opts, :frozen_bit_values, [])
    packed = Keyword.get(opts, :packed, false)
    power = log2(n)

    cond do
      power == nil ->
        raise ArgumentError, "block_size MUST be a power of 2"

      n - k != length(positions) ->
        raise ArgumentError,
              "number of frozen bit positions must equal block_size - num_info_bits"

      positions != Enum.sort(Enum.uniq(positions)) or
          Enum.any?(positions, &(&1 not in 0..(n - 1))) ->
        raise ArgumentError, "frozen bit positions must be sorted, unique and below block_size"

      packed and (rem(k, 8) != 0 or n < 8) ->
        raise ArgumentError, "packed mode needs num_info_bits a multiple of 8 and block_size >= 8"

      true ->
        :ok
    end

    values = Enum.take(values ++ List.duplicate(0, n - k), n - k)
    info = Enum.to_list(0..(n - 1)) -- positions

    conf = %{
      n: n,
      k: k,
      power: power,
      frozen: positions,
      values: values,
      info: info,
      info_reversed: Enum.map(info, &bit_reverse(&1, power)),
      packed: packed,
      systematic: name == "polar_encoder_systematic"
    }

    Generic.new(__MODULE__, name, kind, conf, k)
  end

  @doc "`log2 n` for a power of two, else `nil`."
  def log2(n) when n >= 1 and (n &&& n - 1) == 0, do: Enum.find(0..62, &(1 <<< &1 == n))
  def log2(_), do: nil

  @doc "The `bits` low bits of `v` in reverse order."
  def bit_reverse(v, bits) do
    Enum.reduce(0..(bits - 1)//1, 0, fn i, acc -> acc ||| (v >>> i &&& 1) <<< (bits - 1 - i) end)
  end

  # -- Generic ----------------------------------------------------------------------------------

  @impl true
  def input_size(%{conf: %{packed: true, k: k}}), do: div(k, 8)
  def input_size(%{conf: %{k: k}}), do: k

  @impl true
  def output_size(%{conf: %{packed: true, n: n}}), do: div(n, 8)
  def output_size(%{conf: %{n: n}}), do: n

  @impl true
  def rate(%{conf: %{k: k, n: n}}), do: k / n

  @impl true
  def input_conversion(%{conf: %{packed: true}}), do: "pack"
  def input_conversion(_), do: "none"

  @impl true
  def output_conversion(%{conf: %{packed: true}}), do: "packed_bits"
  def output_conversion(_), do: "none"

  @impl true
  def hardware(_), do: :transform

  # a polar code has a fixed block: GNU Radio's encoder accepts smaller frame
  # sizes without effect; the frame is always K bits
  @impl true
  def set_frame_size(g, _n), do: {false, g}

  @impl true
  def work(%{conf: c}, input, state) do
    info =
      if c.packed,
        do: input |> Nx.as_type(:u8) |> Nx.slice_along_axis(0, div(c.k, 8)) |> unpack(),
        else: input |> Nx.as_type(:u8) |> Nx.bitwise_and(1) |> Nx.slice_along_axis(0, c.k)

    u = interleave(c, info)
    x = transform(u)

    x =
      if c.systematic do
        sel = c.info_reversed |> Nx.tensor(type: :u32)
        kept = Nx.take(x, sel)
        mid = Nx.broadcast(Nx.tensor(0, type: :u8), {c.n})
        mid = Nx.indexed_put(mid, Nx.tensor(c.info) |> Nx.reshape({c.k, 1}), kept)
        transform(mid)
      else
        x
      end

    {if(c.packed, do: pack(x), else: x), state}
  end

  # frozen values at the frozen positions, info bits at the others
  defp interleave(c, info) do
    base = Nx.tensor(List.duplicate(0, c.n), type: :u8)

    base =
      if c.frozen == [],
        do: base,
        else:
          Nx.indexed_put(
            base,
            Nx.tensor(c.frozen) |> Nx.reshape({:auto, 1}),
            Nx.tensor(c.values, type: :u8)
          )

    if c.info == [],
      do: base,
      else: Nx.indexed_put(base, Nx.tensor(c.info) |> Nx.reshape({c.k, 1}), info)
  end

  @doc """
  The polar transform of a vector of 0/1 (length a power of two): GNU Radio's
  `volk_8u_x2_encodeframepolar_8u`.
  """
  def transform(bits) do
    n = Nx.axis_size(bits, 0)
    stages = log2(n)

    Enum.reduce(0..(stages - 1)//1, bits, fn s, t ->
      branches = 1 <<< s
      half = div(n, 2 * branches)
      pairs = Nx.reshape(t, {branches, half, 2})
      a = pairs[[.., .., 0]]
      b = pairs[[.., .., 1]]
      Nx.concatenate([Nx.bitwise_xor(a, b), b], axis: 1) |> Nx.reshape({n})
    end)
  end

  defp unpack(bytes) do
    bytes
    |> Nx.reshape({:auto, 1})
    |> Nx.right_shift(Nx.tensor([7, 6, 5, 4, 3, 2, 1, 0], type: :u8))
    |> Nx.bitwise_and(1)
    |> Nx.reshape({:auto})
  end

  defp pack(bits) do
    bits
    |> Nx.reshape({:auto, 8})
    |> Nx.left_shift(Nx.tensor([7, 6, 5, 4, 3, 2, 1, 0], type: :u8))
    |> Nx.sum(axes: [1])
    |> Nx.as_type(:u8)
  end

  # -- constructions ----------------------------------------------------------------------------

  @doc """
  GNU Radio's `design_snr_to_bec_eta`: the erasure probability of the BEC
  that stands in for the design SNR (dB).
  """
  def design_snr_to_bec_eta(design_snr), do: :math.exp(-:math.pow(10.0, design_snr / 10.0))

  @doc """
  Bhattacharyya (Z) parameters of the N synthetic channels for a BEC with
  erasure probability `eta`, natural index order (`calculate_bec_channel_z_parameters`):
  the smaller, the more reliable.
  """
  def z_parameters(eta, n) do
    Enum.reduce(1..log2(n)//1, [eta], fn _, zs ->
      Enum.flat_map(zs, fn z -> [2 * z - z * z, z * z] end)
    end)
  end

  @doc "Channel capacities `1 - Z` of the BEC synthetic channels (gr-fec `calculate_bec_channel_capacities`)."
  def bec_capacities(eta, n) do
    Enum.reduce(1..log2(n)//1, [1 - eta], fn _, iw ->
      Enum.flat_map(iw, fn w -> [w * w, 2 * w - w * w] end)
    end)
  end

  @doc """
  Frozen bit positions for an (N, K) code, ascending. Methods:

    * `:bec` (default) - GNU Radio's `frozen_bit_positions(block_size, info_size, design_snr)`:
      the N - K channels with the smallest BEC capacity (the first index on ties).
    * `:bhattacharyya` - the N - K largest Bhattacharyya parameters (Arikan's
      bounds, `bhattacharyya_bounds`); the same set as `:bec` up to rounding.
    * `:dega` - density evolution with a Gaussian approximation for the
      BI-AWGN channel at `design_snr` (dB, Es/N0 of the BPSK symbols): the N - K channels
      with the smallest mean LLR. Not part of gr-fec (which needs scipy for its
      Tal-Vardy construction); this one is checked against the BEC choice
      by overlap, not by equality.
  """
  def frozen_bit_positions(n, k, opts \\ []) do
    snr = Keyword.get(opts, :design_snr, 0.0)
    snr = if snr > -1.5917, do: snr, else: 0.0
    nfrozen = n - k

    scores =
      case Keyword.get(opts, :method, :bec) do
        :bec -> bec_capacities(design_snr_to_bec_eta(snr), n)
        :bhattacharyya -> z_parameters(design_snr_to_bec_eta(snr), n) |> Enum.map(&(-&1))
        :dega -> dega_means(snr, n)
      end

    scores
    |> Enum.with_index()
    |> Enum.sort()
    |> Enum.take(nfrozen)
    |> Enum.map(&elem(&1, 1))
    |> Enum.sort()
  end

  @doc """
  Mean LLR of each synthetic channel by Gaussian approximation density
  evolution at `design_snr` dB (Es/N0, BPSK); natural index order, the channel
  `2i` is the degraded one (check node), `2i + 1` the upgraded one (variable node).
  """
  def dega_means(design_snr, n) do
    m0 = 4.0 * :math.pow(10.0, design_snr / 10.0)

    Enum.reduce(1..log2(n)//1, [m0], fn _, ms ->
      Enum.flat_map(ms, fn m -> [phi_inv(1.0 - :math.pow(1.0 - phi(m), 2)), 2.0 * m] end)
    end)
  end

  # Chung's approximation of the density evolution function phi
  defp phi(x) when x < 10.0, do: :math.exp(-0.4527 * :math.pow(x, 0.86) + 0.0218)

  defp phi(x),
    do: :math.sqrt(:math.pi() / x) * :math.exp(-x / 4.0) * (1.0 - 10.0 / (7.0 * x))

  defp phi_inv(y) when y >= 1.0, do: 0.0

  defp phi_inv(y) do
    # phi is decreasing from 1: bisection on [0, hi]
    hi = Enum.find(Stream.iterate(1.0, &(&1 * 2)), &(phi(&1) <= y))
    bisect(0.0, hi, y, 60)
  end

  defp bisect(lo, hi, _y, 0), do: (lo + hi) / 2

  defp bisect(lo, hi, y, i) do
    mid = (lo + hi) / 2
    if phi(mid) > y, do: bisect(mid, hi, y, i - 1), else: bisect(lo, mid, y, i - 1)
  end
end
