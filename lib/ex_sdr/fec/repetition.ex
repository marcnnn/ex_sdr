defmodule ExSdr.FEC.Repetition do
  @moduledoc """
  gr-fec's `repetition_encoder` and `repetition_decoder` as
  `ExSdr.FEC.Generic` objects.

  The encoder repeats every bit `rep` times (`a a a b b b ...`). The decoder
  takes `f32` soft values, counts the positive ones (positive: bit 1) among
  the `rep` copies of a bit and decides 1 if their fraction is larger than
  `ap_prob` (default 0.5: majority, a tie decides 0). The fraction is
  computed in `f32` like GNU Radio.
  """
  @behaviour ExSdr.FEC.Generic

  alias ExSdr.FEC.Generic

  @doc "Encoder object: `frame_size` bits, each repeated `rep` times."
  def encoder(opts), do: make(:encoder, "repetition_encoder", opts)

  @doc "Decoder object; options `frame_size`, `rep`, `ap_prob` (0.5)."
  def decoder(opts), do: make(:decoder, "repetition_decoder", opts)

  defp make(kind, name, opts) do
    rep = Keyword.fetch!(opts, :rep)
    ap = Keyword.get(opts, :ap_prob, 0.5)
    if rep < 0, do: raise(ArgumentError, "repetition rate must be >= 0")
    if ap < 0 or ap > 1.0, do: raise(ArgumentError, "a priori probability must be in [0, 1]")
    if rep == 0, do: raise(ArgumentError, "repetition rate must be >= 1")

    Generic.new(
      __MODULE__,
      name,
      kind,
      %{rep: rep, ap_prob: ap},
      Keyword.fetch!(opts, :frame_size)
    )
  end

  @impl true
  def input_size(%{kind: :encoder, frame_size: f}), do: f
  def input_size(%{kind: :decoder, frame_size: f, conf: c}), do: f * c.rep

  @impl true
  def output_size(%{kind: :encoder, frame_size: f, conf: c}), do: f * c.rep
  def output_size(%{kind: :decoder, frame_size: f}), do: f

  @impl true
  def rate(%{kind: :encoder, conf: c}), do: c.rep * 1.0
  def rate(%{kind: :decoder, conf: c}), do: 1.0 / c.rep

  @impl true
  def input_item(%{kind: :encoder}), do: :u8
  def input_item(%{kind: :decoder}), do: :f32

  @impl true
  def hardware(_), do: :stateless

  @impl true
  def work(%{kind: :encoder} = g, input, state) do
    rep = g.conf.rep
    f = g.frame_size

    out =
      input
      |> Nx.as_type(:u8)
      |> Nx.slice_along_axis(0, f)
      |> Nx.reshape({f, 1})
      |> Nx.tile([1, rep])
      |> Nx.reshape({f * rep})

    {out, state}
  end

  def work(%{kind: :decoder} = g, input, state) do
    rep = g.conf.rep
    f = g.frame_size

    ones =
      input
      |> Nx.as_type(:f32)
      |> Nx.slice_along_axis(0, f * rep)
      |> Nx.reshape({f, rep})
      |> Nx.greater(0.0)
      |> Nx.sum(axes: [1])
      |> Nx.as_type(:f32)

    frac = Nx.divide(ones, Nx.tensor(rep, type: :f32))
    out = frac |> Nx.greater(Nx.tensor(g.conf.ap_prob, type: :f32)) |> Nx.as_type(:u8)
    {out, state}
  end
end
