# :gnuradio_live - runs GNU Radio during the test (native or podman)
#                  (run with `mix test --only gnuradio_live`)
# :dvbt - DVB-T comparison against gr-dtv reference data (tens of seconds)
# :dvbt_live - DVB-T end to end against gr-dtv's receiver (podman)
# The suite runs about 1500 tests in parallel; on a loaded machine single
# property runs (LDPC, SIMD) can pass ExUnit's 60 s default.
ExUnit.start(exclude: [gnuradio_live: true, dvbt: true, dvbt_live: true], timeout: 300_000)
