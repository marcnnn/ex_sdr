defmodule ExSdr.Test.GnuRadioAlist do
  @moduledoc false
  # GNU Radio's LDPC alist files (gr-fec/ldpc_alist) are GPL-3.0-or-later, so
  # they are not part of this repository. test_helper.exs downloads the ones
  # the tests use from a pinned GNU Radio release into the git-ignored
  # test/fixtures/gnuradio/fec/ldpc_alist/ and checks their SHA-256.

  @release "v3.10.12.0"
  @base "https://raw.githubusercontent.com/gnuradio/gnuradio/#{@release}/gr-fec/ldpc_alist/"
  @dir Path.expand("../fixtures/gnuradio/fec/ldpc_alist", __DIR__)

  @files %{
    "n_0100_k_0023_gap_10.alist" =>
      "3b433a51eb90a541f8807fca9b6dd5744c6bcdb9faa76e7ebec2633eeba894b1",
    "n_0100_k_0042_gap_02.alist" =>
      "d742e881cabb3af491a4ce768c20cfd000508451a6392b412ac2f47409cfbdfb",
    "n_0100_k_0058_gen_matrix.alist" =>
      "b265861a1d8edcf6544c2ab1ba30a66c49c9dc8ae47ba52066963965d614cf96",
    "n_0300_k_0152_gap_03.alist" =>
      "0338efb43d9653c0394e75ed25beb10203feab93a01a8403b16d3f5017eba3b7",
    "n_1800_k_0902_gap_28.alist" =>
      "8aa5f87db3a4e05d5137cf49cb3604ae9e6eaea1ca53868afe37f6d328329172",
    "simple_g_matrix.alist" => "91d7e86a961d34029845e62d051ff2f455862a18a4fcbece68ad9bf3710cde23"
  }

  def dir, do: @dir

  @doc "Downloads the missing files; `:ok` or `{:error, reason}`."
  def ensure do
    File.mkdir_p!(@dir)

    Enum.reduce_while(@files, :ok, fn {name, sha}, :ok ->
      path = Path.join(@dir, name)

      if File.exists?(path) and hash(File.read!(path)) == sha,
        do: {:cont, :ok},
        else: fetch(name, sha, path)
    end)
  end

  defp fetch(name, sha, path) do
    {:ok, _} = Application.ensure_all_started([:inets, :ssl])

    ssl = [
      verify: :verify_peer,
      cacerts: :public_key.cacerts_get(),
      depth: 3,
      customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]
    ]

    request = {~c"#{@base}#{name}", []}

    case :httpc.request(:get, request, [ssl: ssl, timeout: 30_000], body_format: :binary) do
      {:ok, {{_, 200, _}, _, body}} ->
        if hash(body) == sha do
          File.write!(path, body)
          {:cont, :ok}
        else
          {:halt, {:error, "#{name}: checksum mismatch"}}
        end

      other ->
        {:halt, {:error, "#{name}: #{inspect(other)}"}}
    end
  end

  defp hash(data), do: :crypto.hash(:sha256, data) |> Base.encode16(case: :lower)
end
