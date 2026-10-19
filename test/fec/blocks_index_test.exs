defmodule ExSdr.FEC.BlocksIndexTest do
  use ExUnit.Case, async: true

  test "ExSdr.FEC.block/2 also builds the LDPC blocks" do
    assert :ldpc_bit_flip_decoder in ExSdr.FEC.blocks()
    assert :polar_encoder in ExSdr.FEC.blocks()
  end
end
