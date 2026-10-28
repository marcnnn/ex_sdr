# :gnuradio_live - runs GNU Radio during the test (native or podman)
#                  (run with `mix test --only gnuradio_live`)
# :dvbt - DVB-T comparison against gr-dtv reference data (tens of seconds)
# :dvbt_live - DVB-T end to end against gr-dtv's receiver (podman)
# :ldpc_alist - needs GNU Radio's LDPC alist files, downloaded here on the
#               first run (ExSdr.Test.GnuRadioAlist); skipped when offline
# The suite runs about 1500 tests in parallel; on a loaded machine single
# property runs (LDPC, SIMD) can pass ExUnit's 60 s default.
exclude = [gnuradio_live: true, dvbt: true, dvbt_live: true]

exclude =
  case ExSdr.Test.GnuRadioAlist.ensure() do
    :ok ->
      exclude

    {:error, reason} ->
      IO.puts("Skipping :ldpc_alist tests, GNU Radio's alist files are missing (#{reason})")
      [{:ldpc_alist, true} | exclude]
  end

ExUnit.start(exclude: exclude, timeout: 300_000)
