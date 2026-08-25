# Benchmarks

Measured 2026-08-25 with `cargo run --release --example bench -p gliner25-core`.
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

Device 0, an RTX 4090, was running an unrelated multi-day training job and was
deliberately left alone. `GLINER2_DEVICE=cuda:1` pins the benchmark to the idle
card.

## Method

One paragraph of Italian text, ~90 words, five entity labels, 14 mentions found.
The text falls in the **128-word bucket**, so the boundary head runs at L=128.
One warm-up run, then 20 timed runs on GPU and 12 on CPU. Median reported
alongside the minimum, which is the cleanest estimate of uncontended time.

## Results

| device | precision | median | min | p95 | per mention |
|---|---|---|---|---|---|
| RTX 3090 | `fp32` | **446 ms** | 329 ms | 2033 ms | 31.9 ms |
| RTX 3090 | `fp16` | 2245 ms | 775 ms | 3849 ms | 160 ms |
| RTX 3090 | `fp16_iobinding` | 765 ms | 692 ms | 2787 ms | 54.6 ms |
| Ryzen 5900XT | `fp32` | 703 ms | 384 ms | 2645 ms | 50.2 ms |
| Ryzen 5900XT | `fp16` | 900 ms | 637 ms | 3631 ms | 64.3 ms |
| Ryzen 5900XT | `fp16_iobinding` | 673 ms | 605 ms | 2229 ms | 48.1 ms |

Load time is 8–13 s in every configuration.

## These numbers are not trustworthy yet

**The machine was contended throughout.** Load average held at 18 on 32 threads
for the whole run — dozens of PyTorch dataloader workers from the training job
on device 0. Two things in the table say plainly that the measurement is
polluted:

- **p95 is 3–6× the median in every row.** That spread is scheduling, not the
  engine.
- **GPU is slower than CPU in two of three precisions.** For a 190 M-parameter
  encoder that cannot be true, and it is not a result — it is noise.

**Re-run on an idle machine before drawing any conclusion from this table.** It
is recorded here because a measurement with its caveats is worth more than no
measurement, not because it is publishable.

The one comparison that survives is the shape of the FP16 penalty, which is
consistent with what the span engine shows on the same host: FP16 graph I/O
means `float_tensor` and `take_float` convert element by element in a scalar
Rust loop at every fragment boundary. See
[`gliner2-rs/BENCHMARKS.md`](https://github.com/dariofinardi/gliner2-rs/blob/main/BENCHMARKS.md)
for that measurement.

## What is worth noting anyway

**Boundary is roughly an order of magnitude slower than span on the same host.**
446 ms against 26 ms for `gliner2-core` on the same GPU with the same text. Part
of that is real: the boundary head runs a candidate-pool builder with attention
layers, a shared-pool scorer and two auxiliary heads, against a fixed 128-word
bucket regardless of the text being ~90 words. Part is likely contention. The
split between the two is exactly what an idle-machine re-run would settle.

**Bucket choice matters and is visible.** A 90-word text pays for 128, and a
91-word text would pay for 128 while an 65-word one pays for 64. If your
documents cluster just above a bucket boundary, adding a bucket at that size is
cheap — a head is a few MB.

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
