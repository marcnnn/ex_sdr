defmodule ExSdr.MathBlocksTest do
  @moduledoc """
  The math family beyond the GNU Radio cases: block descriptions, the
  two-input blocks (pairs), fixed-point division, constant quantization,
  bit packing with arbitrary splits, and the width check of every fixed
  kernel. Deterministic: seeded random data, no timing.
  """
  use ExUnit.Case, async: true
  import Bitwise

  alias ExSdr.Math
  alias ExSdr.Math.Bits
  alias NxFixed.Block
  alias NxFixed.Fixed
  alias NxFixed.Num

  @q15 Num.q(16, 15)
  @q17 Num.q(18, 17)
  @n 2048

  setup_all do
    :rand.seed(:exsss, {3, 1, 4})
    rand = fn lo, hi -> for _ <- 1..@n, do: Enum.random(lo..hi) end

    %{
      # |a| < 0.18 and 0.25 <= |b| < 0.9 keep a / b below 0.8
      a_div: rand.(-6000, 6000),
      b_div: Enum.map(rand.(8192, 29000), &if(:rand.uniform() < 0.5, do: -&1, else: &1)),
      # |a|, |b| < 0.43: sums stay inside Q15
      a: rand.(-14000, 14000),
      b: rand.(-14000, 14000)
    }
  end

  defp container(ints, format), do: Num.from_int(Nx.tensor(ints, type: :s16), 15, format)

  defp values(%Fixed{} = f), do: Num.to_float(f)
  defp values({re, im}), do: Enum.zip(values(re), values(im))
  defp values(t), do: Nx.to_flat_list(t)

  defp max_error(a, b) do
    a
    |> Enum.zip(b)
    |> Enum.map(fn
      {{x1, y1}, {x2, y2}} -> :math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
      {x, y} -> abs(x - y) * 1.0
    end)
    |> Enum.max()
  end

  describe "block descriptions" do
    @names [
      :multiply_const_ff,
      :multiply_const_cc,
      :add_const_ff,
      :add_const_cc,
      :conjugate_cc,
      :complex_to_mag_squared,
      :complex_to_real,
      :complex_to_imag,
      :abs_ff,
      :threshold_ff,
      :moving_average_ff,
      :moving_average_cc,
      :delay,
      :add_ff,
      :sub_ff,
      :mul_ff,
      :div_ff,
      :add_cc,
      :sub_cc,
      :mul_cc,
      :div_cc,
      :char_to_float,
      :float_to_char,
      :stream_to_vector,
      :integrate_ff,
      :integrate_cc,
      :repeat,
      :max_ff,
      :min_ff,
      :keep_one_in_n,
      :keep_m_in_n,
      :unpack_k_bits_bb,
      :pack_k_bits_bb,
      :repack_bits_bb,
      :packed_to_unpacked_bb,
      :unpacked_to_packed_bb,
      :float_to_complex,
      :magphase_to_complex
    ]

    defp opts_for(name) do
      case name do
        n when n in [:multiply_const_ff, :multiply_const_cc] -> [k: 0.5]
        n when n in [:add_const_ff, :add_const_cc] -> [c: 0.1]
        :threshold_ff -> [lo: -0.125, hi: 0.25]
        n when n in [:moving_average_ff, :moving_average_cc] -> [length: 4, scale: 0.25]
        :delay -> [delay: 3]
        :char_to_float -> [scale: 128.0]
        :float_to_char -> [scale: 127.0]
        n when n in [:integrate_ff, :integrate_cc] -> [decim: 4]
        :repeat -> [interp: 2]
        n when n in [:max_ff, :min_ff] -> [vlen: 4]
        :keep_one_in_n -> [n: 4]
        :keep_m_in_n -> [m: 3, n: 8]
        :unpack_k_bits_bb -> [k: 3]
        :pack_k_bits_bb -> [k: 8]
        :repack_bits_bb -> [k: 2, l: 4]
        n when n in [:packed_to_unpacked_bb, :unpacked_to_packed_bb] -> [k: 2]
        :stream_to_vector -> []
        _ -> []
      end
    end

    test "every name builds a block description with its contract fields" do
      for name <- @names do
        b = Math.block(name, [format: Num.float()] ++ opts_for(name))
        assert %Block{name: ^name} = b
        assert b.kind in [:stream, :frame, :host]
        assert {r_in, r_out} = b.rate
        assert r_in > 0 and r_out > 0
        assert b.hardware in [:stateless, :feed_forward, :feedback, :memory, :transform, :control]
        assert {_in, _out} = b.item
      end
    end

    test "rates and latency follow the GNU Radio sizes" do
      assert Math.block(:delay, delay: 5).latency == 5
      assert Math.block(:delay, delay: 5).warmup == 5
      assert Math.block(:integrate_ff, decim: 8).rate == {8, 1}
      assert Math.block(:repeat, interp: 3).rate == {1, 3}
      assert Math.block(:keep_m_in_n, m: 3, n: 8).rate == {8, 3}
      assert Math.block(:unpack_k_bits_bb, k: 3).rate == {1, 3}
      assert Math.block(:pack_k_bits_bb, k: 8).rate == {8, 1}
      assert Math.block(:repack_bits_bb, k: 2, l: 4).rate == {2, 4}
      assert Math.block(:max_ff, vlen: 4).rate == {4, 1}
    end

    test "moving averages warm up after length - 1 inputs" do
      assert Math.block(:moving_average_ff, length: 16, scale: 0.0625).warmup == 15
    end

    test "threshold has no bounded warm-up (hysteresis)" do
      assert Math.block(:threshold_ff, lo: -0.1, hi: 0.2).warmup == :unbounded
    end

    test "float-only blocks refuse a fixed-point format" do
      assert_raise ArgumentError, ~r/float instance only/, fn ->
        Math.block(:complex_to_mag, format: @q15)
      end

      assert_raise ArgumentError, ~r/float instance only/, fn ->
        Math.block(:nlog10_ff, format: @q15, n: 10, k: 0)
      end
    end

    test "unknown names and the delay of zero are errors" do
      assert_raise ArgumentError, ~r/no math block/, fn -> Math.block(:interleave) end
      assert_raise ArgumentError, fn -> Math.block(:delay, delay: 0) end
    end

    test "delay state is the delay line of the format" do
      b = Math.block(:delay, format: @q15, delay: 5)
      assert %Fixed{t: t, frac: 15} = b.init
      assert Nx.shape(t) == {5}
      assert Nx.type(t) == {:s, 16}
    end
  end

  describe "parameter quantization" do
    test "a constant keeps the format's fraction when it fits" do
      k = Math.param(0.7, @q15)
      assert %Fixed{frac: 15} = k
      assert Nx.to_number(k.t) == round(0.7 * 32768)
    end

    test "larger constants get fewer fractional bits instead of saturating" do
      k = Math.param(127.0, @q15)
      assert k.frac == 8
      assert Num.to_float(k) == 127.0
    end

    test "1.0 is representable (Q15 fraction drops to 14)" do
      assert Math.param(1.0, @q15).frac == 14
      assert Num.to_float(Math.param(1.0, @q15)) == 1.0
    end

    test "a constant wider than the word is an error" do
      assert_raise ArgumentError, ~r/does not fit/, fn -> Math.param(40_000, @q15) end
    end

    test "complex constants are pairs of quantized parts" do
      {re, im} = Math.param([-0.05, 0.02], @q15)
      assert_in_delta Num.to_float(re), -0.05, 1 / 32768
      assert_in_delta Num.to_float(im), 0.02, 1 / 32768
    end
  end

  describe "two-input blocks (pairs)" do
    test "real add, subtract and multiply: exact sums, one rounding in the product", %{a: a, b: b} do
      for {name, op} <- [add_ff: &+/2, sub_ff: &-/2, mul_ff: &*/2] do
        block = Math.block(name, format: @q15)
        {out, _} = Block.run(block, {container(a, @q15), container(b, @q15)}, :init, [])
        expected = Enum.zip_with(a, b, fn x, y -> op.(x / 32768, y / 32768) end)
        err = max_error(values(out), expected)
        bound = if name == :mul_ff, do: 1.6e-5, else: 0.0
        assert err <= bound, "#{name}: #{err} > #{bound}"
      end
    end

    test "real divide: one rounding, error at most half an LSB", %{a_div: a, b_div: b} do
      block = Math.block(:div_ff, format: @q15)
      {out, _} = Block.run(block, {container(a, @q15), container(b, @q15)}, :init, [])
      expected = Enum.zip_with(a, b, fn x, y -> x / y end)
      assert max_error(values(out), expected) <= 1.6e-5
    end

    test "float divide is the plain quotient", %{a_div: a, b_div: b} do
      block = Math.block(:div_ff, format: Num.float())
      af = Enum.map(a, &(&1 / 32768))
      bf = Enum.map(b, &(&1 / 32768))

      {out, _} =
        Block.run(block, {container(a, Num.float()), container(b, Num.float())}, :init, [])

      expected = Enum.zip_with(af, bf, &(&1 / &2))
      assert max_error(values(out), expected) < 1.0e-6
    end

    test "division by zero saturates by the sign of the numerator", _ do
      block = Math.block(:div_ff, format: @q15)
      a = [1000, -1000, 0]
      {out, _} = Block.run(block, {container(a, @q15), container([0, 0, 0], @q15)}, :init, [])
      assert values(out) == [32767 / 32768, -1.0, 0.0]
    end

    test "complex multiply and divide (one rounding per component)", %{a_div: a, b_div: b} do
      pa = Enum.zip(a, Enum.reverse(a))
      pb = Enum.zip(b, Enum.reverse(b))

      ca =
        {container(Enum.map(pa, &elem(&1, 0)), @q15), container(Enum.map(pa, &elem(&1, 1)), @q15)}

      cb =
        {container(Enum.map(pb, &elem(&1, 0)), @q15), container(Enum.map(pb, &elem(&1, 1)), @q15)}

      {mul, _} = Block.run(Math.block(:mul_cc, format: @q15), {ca, cb}, :init, [])

      expected_mul =
        Enum.zip_with(pa, pb, fn {ar, ai}, {br, bi} ->
          {(ar * br - ai * bi) / 32768 / 32768, (ar * bi + ai * br) / 32768 / 32768}
        end)

      # two components, each rounded once: up to sqrt(2) * 0.5 LSB
      assert max_error(values(mul), expected_mul) <= 2.2e-5

      {div, _} = Block.run(Math.block(:div_cc, format: @q15), {ca, cb}, :init, [])

      expected_div =
        Enum.zip_with(pa, pb, fn {ar, ai}, {br, bi} ->
          d = br * br + bi * bi
          {(ar * br + ai * bi) / d, (ai * br - ar * bi) / d}
        end)

      assert max_error(values(div), expected_div) <= 2.3e-5
    end

    test "pairs split in two give the same output as one call", %{a: a, b: b} do
      block = Math.block(:mul_ff, format: @q15)
      {whole, _} = Block.run(block, {container(a, @q15), container(b, @q15)}, :init, [])

      {p, q} = {Enum.split(a, 700), Enum.split(b, 700)}

      {o1, _} =
        Block.run(block, {container(elem(p, 0), @q15), container(elem(q, 0), @q15)}, :init, [])

      {o2, _} =
        Block.run(block, {container(elem(p, 1), @q15), container(elem(q, 1), @q15)}, :init, [])

      assert values(whole) == values(o1) ++ values(o2)
    end
  end

  describe "float to complex and magnitude-phase" do
    test "float_to_complex pairs two floats (quantized in the format)", %{a: a, b: b} do
      block = Math.block(:float_to_complex, format: @q15)
      {out, _} = Block.run(block, {container(a, @q15), container(b, @q15)}, :init, [])
      assert values(out) == Enum.zip_with(a, b, &{&1 / 32768, &2 / 32768})
    end

    test "magphase_to_complex: m * (cos p, sin p), float only" do
      block = Math.block(:magphase_to_complex, format: Num.float())
      m = Nx.tensor([0.5, 2.0], type: :f32)
      p = Nx.tensor([:math.pi() / 2, 0.0], type: :f32)
      {{re, im}, _} = Block.run(block, {m, p}, :init, [])
      assert_in_delta Nx.to_number(re[0]), 0.0, 1.0e-6
      assert_in_delta Nx.to_number(im[0]), 0.5, 1.0e-6
      assert_in_delta Nx.to_number(re[1]), 2.0, 1.0e-6
      assert_in_delta Nx.to_number(im[1]), 0.0, 1.0e-6
    end

    test "packed_to_unpacked_bb is the MSB-first unpack (GNU Radio default; not golden-checked)" do
      block = Math.block(:packed_to_unpacked_bb, k: 2)
      {out, _} = Block.run(block, Nx.tensor([2, 1], type: :u8), :init, [])
      assert Nx.to_flat_list(out) == [1, 0, 0, 1]
    end
  end

  describe "bit and byte blocks" do
    test "pack_k_bits undoes unpack_k_bits for k = 8" do
      bytes = Nx.tensor(Enum.map(1..64, &(&1 * 3 &&& 255)), type: :u8)
      {bits, _} = Bits.unpack_k_bits(bytes, nil, k: 8)
      {back, _} = Bits.pack_k_bits(bits, nil, k: 8)
      assert Nx.to_flat_list(back) == Nx.to_flat_list(bytes)
    end

    test "unpack takes the low k bits, most significant first" do
      {bits, _} = Bits.unpack_k_bits(Nx.tensor([13, 2], type: :u8), nil, k: 3)
      assert Nx.to_flat_list(bits) == [1, 0, 1, 0, 1, 0]
    end

    test "least significant first endianness reverses each group" do
      {bits, _} = Bits.unpack_k_bits(Nx.tensor([6], type: :u8), nil, k: 3, endianness: :lsb)
      assert Nx.to_flat_list(bits) == [0, 1, 1]
      {back, _} = Bits.pack_k_bits(Nx.tensor([0, 1, 1], type: :u8), nil, k: 3, endianness: :lsb)
      assert Nx.to_flat_list(back) == [6]
    end

    test "repacking bits to bytes is the same for any split of the input", _ do
      items = for _ <- 1..997, do: :rand.uniform(4) - 1
      input = Nx.tensor(items, type: :u8)
      repack = Math.block(:repack_bits_bb, k: 2, l: 4)

      {whole, _} = Block.run(repack, input, :init, [])

      split =
        for {chunk, _} <- Enum.chunk_every(items, 7) |> Enum.with_index(),
            reduce: {<<>>, :init} do
          {acc, state} ->
            {out, next} = Block.run(repack, Nx.tensor(chunk, type: :u8), state, [])
            {acc <> out, next}
        end
        |> elem(0)

      assert split == whole
      assert byte_size(whole) == div(length(items) * 2, 4)
    end

    test "repacking 1-bit items to bytes is LSB-first packing" do
      bits = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]

      {out, _} =
        Block.run(Math.block(:repack_bits_bb, k: 1, l: 8), Nx.tensor(bits, type: :u8), :init, [])

      assert out == <<1, 3>>
    end

    test "a chunk with no kept item is an error, not a silent empty output" do
      block = Math.block(:keep_one_in_n, n: 4)

      assert_raise ArgumentError, ~r/no item of this chunk is kept/, fn ->
        Block.run(block, container([1, 2], Num.float()), :init, [])
      end
    end

    test "a keep block gives the same items for any split of at least n items", %{a: a} do
      block = Math.block(:keep_m_in_n, m: 3, n: 8)
      {whole, _} = Block.run(block, container(a, Num.float()), :init, [])

      chunks = [Enum.take(a, 1000), Enum.drop(a, 1000) |> Enum.take(500), Enum.drop(a, 1500)]

      {parts, _} =
        Enum.reduce(chunks, {[], :init}, fn chunk, {acc, state} ->
          {out, next} = Block.run(block, container(chunk, Num.float()), state, [])
          {[values(out) | acc], next}
        end)

      assert parts |> Enum.reverse() |> Enum.concat() == values(whole)
    end
  end

  describe "width check of the fixed-point kernels" do
    # NxFixed.analyze with the container types each format produces. Inputs
    # are templates (the type is the only range assumed); the kernels quantize
    # their inputs, which is what keeps products and sums inside the format.
    defp template(%Fixed{t: t, frac: frac}), do: %Fixed{t: template(t), frac: frac}
    defp template({re, im}), do: {template(re), template(im)}
    defp template({}), do: {}
    defp template(%Nx.Tensor{} = t), do: Nx.template(Nx.shape(t), Nx.type(t))

    defp item(format, :real), do: template(Num.from_int(Nx.tensor(0, type: :s16), 15, format))
    defp item(format, :complex), do: {item(format, :real), item(format, :real)}
    defp item(format, :pair), do: {item(format, :real), item(format, :real)}
    defp item(format, :complex_pair), do: {item(format, :complex), item(format, :complex)}

    defp check(name, format, sample, opts) do
      b = Math.block(name, [format: format] ++ opts)
      args = [sample, template(b.init) | Enum.map(b.params, &template/1)] ++ [b.opts]
      assert {:ok, _graph} = NxFixed.analyze(b.fun, args), "#{name} #{inspect(format)}"
    end

    for {bits, format} <- [q15: @q15, q17: @q17] do
      @format format
      test "stream kernels, #{bits}" do
        f = @format
        check(:multiply_const_ff, f, item(f, :real), k: 0.7)
        check(:multiply_const_cc, f, item(f, :complex), k: 0.5)
        check(:add_const_ff, f, item(f, :real), c: 0.1)
        check(:add_const_cc, f, item(f, :complex), c: [-0.05, 0.02])
        check(:conjugate_cc, f, item(f, :complex), [])
        check(:complex_to_mag_squared, f, item(f, :complex), [])
        check(:complex_to_real, f, item(f, :complex), [])
        check(:complex_to_imag, f, item(f, :complex), [])
        check(:abs_ff, f, item(f, :real), [])
        check(:threshold_ff, f, item(f, :real), lo: -0.125, hi: 0.25)
        check(:float_to_char, f, item(f, :real), scale: 127.0)
        check(:add_ff, f, item(f, :pair), [])
        check(:sub_ff, f, item(f, :pair), [])
        check(:mul_ff, f, item(f, :pair), [])
        check(:div_ff, f, item(f, :pair), [])
        check(:add_cc, f, item(f, :complex_pair), [])
        check(:mul_cc, f, item(f, :complex_pair), [])
        check(:div_cc, f, item(f, :complex_pair), [])
        check(:moving_average_ff, f, item(f, :real), length: 4, scale: 0.25)
        check(:moving_average_cc, f, item(f, :complex), length: 4, scale: 0.25)
        check(:delay, f, item(f, :real), delay: 3)
      end
    end

    test "char_to_float reads the byte input" do
      check(:char_to_float, @q15, Nx.template({}, {:u, 8}), scale: 128.0)
    end

    # Frame kernels take a whole group (the item axis first), not one sample.
    test "frame kernels: integrate sums in 64 bits and saturates once" do
      for {format, group} <- [{@q15, 8}, {Num.q(20, 15), 8}] do
        b = Math.block(:integrate_ff, format: format, decim: group)
        input = Num.from_int(Nx.tensor(0, type: :s16), 15, format) |> template()
        input = %Fixed{t: Nx.template({group}, Nx.type(input.t)), frac: input.frac}
        assert {:ok, _} = NxFixed.analyze(b.fun, [input, {}, b.opts])
      end
    end
  end
end
