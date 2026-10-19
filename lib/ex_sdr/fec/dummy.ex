defmodule ExSdr.FEC.Dummy do
  @moduledoc """
  gr-fec's `dummy_encoder` (copies the frame) and `dummy_decoder` (slices
  `f32` soft values: `>= 0` gives 1, `< 0` gives 0, so 0.0 is a 1 as in
  volk's `binary_slicer`). The encoder's `pack` / `packed_bits` options only
  change the conversions GNU Radio inserts around it (`input_conversion/1`,
  `output_conversion/1`); the frame function itself is a copy.
  """
  @behaviour ExSdr.FEC.Generic

  alias ExSdr.FEC.Generic

  def encoder(opts) do
    conf = %{
      pack: Keyword.get(opts, :pack, false),
      packed_bits: Keyword.get(opts, :packed_bits, false)
    }

    Generic.new(__MODULE__, "dummy_encoder", :encoder, conf, Keyword.fetch!(opts, :frame_size))
  end

  def decoder(opts),
    do: Generic.new(__MODULE__, "dummy_decoder", :decoder, %{}, Keyword.fetch!(opts, :frame_size))

  @impl true
  def input_size(%{frame_size: f}), do: f
  @impl true
  def output_size(%{frame_size: f}), do: f
  @impl true
  def rate(_), do: 1.0
  @impl true
  def hardware(_), do: :stateless
  @impl true
  def input_item(%{kind: :encoder}), do: :u8
  def input_item(%{kind: :decoder}), do: :f32

  @impl true
  def input_conversion(%{kind: :encoder, conf: %{pack: true}}), do: "pack"
  def input_conversion(_), do: "none"

  @impl true
  def output_conversion(%{kind: :encoder, conf: %{packed_bits: true}}), do: "packed_bits"
  def output_conversion(_), do: "none"

  @impl true
  def work(%{kind: :encoder} = g, input, state),
    do: {input |> Nx.as_type(:u8) |> Nx.slice_along_axis(0, g.frame_size), state}

  def work(%{kind: :decoder} = g, input, state) do
    out =
      input
      |> Nx.as_type(:f32)
      |> Nx.slice_along_axis(0, g.frame_size)
      |> Nx.greater_equal(0.0)
      |> Nx.as_type(:u8)

    {out, state}
  end
end
