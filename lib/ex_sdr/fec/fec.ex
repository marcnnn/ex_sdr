defmodule ExSdr.FEC do
  @moduledoc false
  def block(:cc_encoder, opts) do
    g = ExSdr.FEC.Conv.encoder(opts)
    %NxFixed.Block{name: :cc_encoder, kind: :frame, fun: &ExSdr.FEC.Generic.encode_frames/3, init: ExSdr.FEC.Generic.init(g), opts: [code: g], rate: {ExSdr.FEC.Generic.input_size(g), ExSdr.FEC.Generic.output_size(g)}}
  end
end
