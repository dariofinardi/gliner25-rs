# Benchmarks

Measured 2026-08-25 on a **shared development machine** — other users on some
cores, a training job on the other GPU. The figures below are an indication of
magnitude, not a clean measurement; see the caveats.

Measured with `cargo run --release --example bench -p gliner25-core`.
Read the caveats before quoting any of it — for this repository they are larger
than the numbers.

## Hardware

| | |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA RTX 3090, 24 GB (device 1) |
| ONNX Runtime | 1.25.1 CPU-only wheel; 1.23.2 GPU wheel for CUDA |
| CUDA | 12.8, cuDNN 9 |
| crate | `gliner25-core`, `ort` 2.0.0-rc.13, `load-dynamic` |

Device 0, an RTX 4090, was carrying an unrelated multi-day training job and was
deliberately left alone; `GLINER2_DEVICE=cuda:1` pins the benchmark to the 3090,
which was idle. The host itself is shared, so the CPU cores were not exclusively
available — which is why the CPU figures are the unreliable ones here, and the
GPU figures the more usable.

### About the card

An RTX 3090 is a 2020 consumer part, not a datacenter accelerator, and it is the
slower end of what you would deploy on. Its performance class sits close to an
**NVIDIA L4**, which is what a good deal of cloud inference actually runs on
today — GCP G2, AWS G6 and similar. The two are within the same range on FP32
throughput; the 3090 has substantially more memory bandwidth, the L4 draws a
fraction of the power.

For this workload the distinction matters less than it looks. The span pipeline
is bound by kernel launches and host round-trips across eight ONNX sessions, not
by arithmetic or bandwidth, so a faster card moves these numbers less than
implementing `IoBinding` would. Read the GPU figures as *what a realistic
inference instance gives you*, not as a ceiling — and not as a best case either.

## Method

One paragraph of Italian text, ~90 words, five entity labels, 14 mentions found.
The text falls in the **128-word bucket**, so the boundary head runs at L=128.
One warm-up run, then 20 timed runs on GPU and 12 on CPU. Median reported
alongside the minimum, which is the cleanest estimate of uncontended time.

## Correctness across devices

This part is solid, and it is the reason the file is worth reading. The
end-to-end suite was run on both devices and every precision against the same
PyTorch reference:

| device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|
| CPU | 43/43 (**0.0000**) | 43/43 (0.0004) | 43/43 (0.0004) |
| RTX 3090 | 43/43 (**0.0000**) | 43/43 (0.0007) | 43/43 (0.0007) |

Identical spans in all six; brackets give the largest score delta. In `fp32` the
agreement with PyTorch is **exact** at the precision the harness records, on both
devices — which puts a floor under the FP16 rows: their deviation is
quantisation, not a defect in the graphs. Choosing a device is a performance
decision, not an accuracy one.

## Timings — and why you should not use them

Two runs of the same matrix, an hour apart, same machine:

| device | precision | run 1 median | run 2 median | ratio |
|---|---|---|---|---|
| RTX 3090 | `fp32` | 446 ms | 1922 ms | **4.3×** |
| RTX 3090 | `fp16` | 2245 ms | 1233 ms | 1.8× |
| RTX 3090 | `fp16_iobinding` | 765 ms | 642 ms | 1.2× |
| Ryzen 5900XT | `fp32` | 703 ms | 375 ms | 1.9× |
| Ryzen 5900XT | `fp16` | 900 ms | 573 ms | 1.6× |
| Ryzen 5900XT | `fp16_iobinding` | 673 ms | 527 ms | 1.3× |

The same configuration moved by a factor of four. **Nothing in this table is a
measurement of the engine**, and no ordering in it should be quoted. The host is
a shared development machine and held load average 15–19 throughout; that is
what is being measured.

Recorded anyway because the *instability itself* is the finding: on a contended
host this workload is not reproducible, and anyone benchmarking it needs to know
that before drawing conclusions.

## What the investigation ruled out

CPU came out faster than GPU in both runs, which for a 190 M-parameter encoder
should not happen. Two hypotheses were tested rather than assumed:

**Execution-provider fallback — ruled out.** Profiling the boundary head under
the CUDA provider shows **97.1% of nodes on CUDA and 2.9% on CPU**. The graph is
not being chopped into partitions with transfers between them.

**Graph size — ruled out.** The two architectures are comparable, and share an
identical encoder:

| | encoder | rest | total per extraction |
|---|---|---|---|
| boundary (GLiNER2.5) | 4588 | 480 | 5068 nodes |
| span (GLiNER2) | 4588 | 1227 | 5815 nodes |

If anything the boundary pipeline is the smaller one, yet the span engine
returns in ~26 ms on the same card while this one takes hundreds. **That gap is
unexplained.** The candidate-pool builder does heavier work per node — pairwise
scoring over a 32×32 endpoint grid, attention over the full bucket, a scorer
across 192 candidates per query — but nothing measured here attributes the
difference, and a contended host cannot settle it.

Settling it needs an idle machine and per-fragment profiling. Until then, treat
GLiNER2.5 GPU performance as **unknown**, not as slow.

## Reproducing

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
GLINER2_DEVICE=cuda:1 GLINER2_PRECISION=fp32 \
cargo run --release --example bench -p gliner25-core -- models/gliner2.5-multi-v1-onnx 20
```

The GPU run needs a runtime that ships the CUDA provider — the plain
`onnxruntime` wheel does not. `pip install onnxruntime-gpu` provides
`libonnxruntime_providers_cuda.so`; point `LD_LIBRARY_PATH` at its directory
along with the CUDA 12 libraries.
