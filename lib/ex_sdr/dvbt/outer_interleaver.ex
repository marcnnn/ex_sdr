defmodule ExSdr.DVBT.OuterInterleaver do
  @moduledoc """
  DVB outer interleaver (ETSI EN 300 744, 4.3.3): `ExSdr.Comm.ConvInterleaver`
  with I = 12, M = 17 over the 204-byte packets, the sync byte of each
  packet always through branch 0. End-to-end delay 2244 bytes.
  """
  alias ExSdr.Comm.ConvInterleaver

  def interleaver, do: ConvInterleaver.new(branches: 12, depth: 17)

  def interleave(bytes), do: ConvInterleaver.interleave(interleaver(), bytes)
  def deinterleave(bytes), do: ConvInterleaver.deinterleave(interleaver(), bytes)

  @doc "Total delay in bytes of interleaver + deinterleaver."
  def delay, do: ConvInterleaver.delay(interleaver())
end
