# GNMT (Google Neural Machine Translation) Model

This directory contains an implementation of GNMT that was adapted from the
code found in the [MLPerf training repository](https://github.com/mlperf/training/tree/master/rnn_translator).

To launch an interactive Skyline profiling session for GNMT, run
```
skyline interactive entry_point.py
```


## License

This code, with the exception of the `skyline_` prefixed functions in
`entry_point.py`, was adapted from the MLPerf training benchmarks and therefore
shares the same license. The unmodified license can be found in the `LICENSE`
file in the `seq2seq` directory.
