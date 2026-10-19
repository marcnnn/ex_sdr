defmodule ExSdr.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_sdr,
      version: "0.1.0",
      elixir: "~> 1.17",
      description:
        "Software-defined radio building blocks for Nx: DSP, channel coding, OFDM; DVB-T as an example",
      elixirc_paths: elixirc_paths(Mix.env()),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_env: fn -> %{"FINE_INCLUDE_DIR" => Fine.include_dir()} end,
      make_clean: ["clean"],
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger, :crypto]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 1.0"},
      {:nx_fixed, path: "../nx_fixed"},
      {:fine, "~> 0.1.6", runtime: false},
      {:elixir_make, "~> 0.9", runtime: false},
      {:exla, "~> 1.0", optional: true},
      {:stream_data, "~> 1.1", only: [:dev, :test]}
    ]
  end
end
