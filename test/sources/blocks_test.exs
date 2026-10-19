defmodule ExSdr.Sources.BlocksTest do
  @moduledoc "Every block of the family declares itself honestly (NxFixed.Block contract)."
  use ExUnit.Case, async: true

  alias ExSdr.Sources
  alias NxFixed.Block
  alias NxFixed.Num

  @fixed Num.q(16, 15)

  # name, opts, item, hardware, gnuradio
  @blocks [
    {:sig_source_f, [], {:none, :real}, :feedback, "analog.sig_source_f"},
    {:sig_source_c, [], {:none, :complex}, :feedback, "analog.sig_source_c"},
    {:nco, [frequency: 0.1], {:none, :complex}, :feedback, nil},
    {:rotator_cc, [frequency: 0.1], {:complex, :complex}, :feedback, "blocks.rotator_cc"},
    {:vco_c, [sensitivity: 1.0], {:real, :complex}, :feedback, "blocks.vco_c"},
    {:noise_source_f, [], {:none, :real}, :feedback, "analog.noise_source_f"},
    {:noise_source_c, [], {:none, :complex}, :feedback, "analog.noise_source_c"},
    {:fastnoise_source_f, [samples: 64], {:none, :real}, :memory, "analog.fastnoise_source_f"},
    {:fastnoise_source_c, [samples: 64], {:none, :complex}, :memory, "analog.fastnoise_source_c"},
    {:vector_source_f, [data: [0.1, 0.2]], {:none, :real}, :memory, "blocks.vector_source_f"},
    {:vector_source_c, [data: [{0.1, 0.2}]], {:none, :complex}, :memory,
     "blocks.vector_source_c"},
    {:glfsr_source_b, [degree: 5], {:none, :bit}, :feedback, "digital.glfsr_source_b"},
    {:fir_ccc, [taps: [1.0, 0.5]], {:complex, :complex}, :feed_forward, "filter.fir_filter_ccf"},
    {:mmse_resampler_cc, [ratio: 1.01], {:complex, :complex}, :feedback,
     "filter.mmse_resampler_cc"},
    {:channel_model, [epsilon: 1.01, frequency_offset: 0.1], {:complex, :complex}, :feedback,
     "channels.channel_model"},
    {:fading_model, [], {:complex, :complex}, :feedback, "channels.fading_model"},
    {:iqbal_gen, [magnitude: -1.0], {:complex, :complex}, :stateless, "channels.iqbal_gen"},
    {:distortion_2_gen, [beta: 0.1], {:complex, :complex}, :stateless,
     "channels.distortion_2_gen"},
    {:distortion_3_gen, [beta: 0.1], {:complex, :complex}, :stateless,
     "channels.distortion_3_gen"},
    {:quantizer, [bits: 8], {:real, :real}, :stateless, "channels.quantizer"},
    {:cfo_model, [sample_rate: 1.0e6, std_dev_hz: 10.0, max_dev_hz: 1000.0], {:complex, :complex},
     :feedback, "channels.cfo_model"},
    {:selective_fading_model, [delays: [0.0, 2.0], mags: [1.0, 0.5]], {:complex, :complex},
     :feedback, "channels.selective_fading_model"}
  ]

  for {name, opts, item, hardware, gnuradio} <- @blocks, format <- [Num.float(), @fixed] do
    @args {name, opts, item, hardware, gnuradio, format}
    test "#{name} (#{format.kind}) declares kind, item types, hardware class and GNU Radio name" do
      {name, opts, item, hardware, gnuradio, f} = @args

      b =
        Sources.block(name, [format: f] ++ opts)

      assert %Block{} = b
      assert b.name == name
      assert b.kind in [:stream, :host]
      assert b.item == item
      assert b.hardware == hardware
      assert b.gnuradio == gnuradio
      assert {i, o} = b.rate
      assert is_integer(i) and is_integer(o) and i > 0 and o > 0
      assert b.latency >= 0
      assert b.warmup == :unbounded or (is_integer(b.warmup) and b.warmup >= 0)
    end
  end

  test "blocks with the same rate as input produce n items per n (sources and streams)" do
    for {name, opts, {:none, _}, _, _} <- @blocks, name != :glfsr_source_b do
      b = Sources.block(name, [format: @fixed] ++ opts)
      assert b.rate == {1, 1}
      {out, _} = Sources.generate(b, 5)

      n =
        case Num.to_float(out) do
          {re, _} -> length(re)
          xs -> length(xs)
        end

      assert n == Block.output_items(b, 5)
    end
  end

  test "blocks with an input refuse generate/3" do
    assert_raise ArgumentError, fn ->
      Sources.generate(Sources.block(:rotator_cc, frequency: 0.1), 4)
    end
  end

  test "unknown names and parameters are errors" do
    assert_raise ArgumentError, fn -> Sources.block(:nope, []) end
    assert_raise ArgumentError, fn -> Sources.block(:sig_source_f, waveform: :pulse) end
    assert_raise ArgumentError, fn -> Sources.block(:noise_source_f, type: :pink) end
  end

  test "a source and its split run use the NxFixed scan loop with the width check" do
    # fixed-point kernels reach the checked loop: a format that cannot hold the amplitude is
    # rejected up front instead of saturating silently
    assert_raise ArgumentError, fn ->
      Sources.block(:noise_source_f, format: @fixed, amplitude: 2.0)
    end
  end
end
