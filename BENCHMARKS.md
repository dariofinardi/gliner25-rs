# Benchmarks

Measured 2026-08-25 on a **shared development machine** — other users on some
cores, a training job on the other GPU, load average 17–19 throughout. Treat
every figure as an indication of magnitude, not a measurement.

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 9 5900XT, 16 cores / 32 threads |
| GPU | NVIDIA RTX 3090, 24 GB, device 1, idle |
| ONNX Runtime | 1.25.1 CPU wheel; 1.23.2 GPU wheel for CUDA |
| CUDA | 12.8, cuDNN 9 |
| crate | `gliner25-core`, `ort` 2.0.0-rc.13, `load-dynamic` |

An RTX 3090 is a 2020 consumer part whose performance class sits close to an
**NVIDIA L4** — what a good deal of cloud inference runs on today. Comparable
FP32 throughput; the 3090 has substantially more memory bandwidth and draws
several times the power. Read the GPU figures as a realistic inference instance,
not a ceiling.

**Method.** One Italian paragraph, ~90 words, five entity labels, 14 mentions
found. The text falls in the **128-word bucket**, so the head runs at L=128.
Five warm-up runs — the boundary head needs several before it settles — then 25
timed on GPU and 15 on CPU, with a 2 ms yield between iterations. See *The
harness was wrong* below for why that is not cosmetic.

## Results

| device | precision | median | min | p95 |
|---|---|---|---|---|
| RTX 3090 | `fp32` | **17.9 ms** | 16.1 ms | 27.5 ms |
| RTX 3090 | `fp16` | 18.6 ms | 16.2 ms | 20.6 ms |
| RTX 3090 | `fp16_iobinding` | 20.3 ms | 17.0 ms | 21.5 ms |
| Ryzen 5900XT | `fp32` | 1179 ms | 364 ms | 2694 ms |
| Ryzen 5900XT | `fp16` | **630 ms** | 555 ms | 1469 ms |
| Ryzen 5900XT | `fp16_iobinding` | 635 ms | 487 ms | 2127 ms |

The three GPU precisions sit within 13% of each other, which is inside what this
host moves on its own. **Do not choose a precision from this table.**

## Against the span architecture, at parity

Same paragraph, same five labels, same precision, same host, same harness, same
card — compared with `gliner2-core` running
`GLiNER2-Guardrails-PII-Multi`, which like this one is a flat export:

| device | precision | span (GLiNER2) | boundary (GLiNER2.5) | ratio |
|---|---|---|---|---|
| RTX 3090 | `fp32` | 26.5 ms | **17.9 ms** | 1.5× |
| RTX 3090 | `fp16` | 55.4 ms | **18.6 ms** | 3.0× |
| RTX 3090 | `fp16_iobinding` | 20.6 ms | 20.3 ms | 1.0× |
| Ryzen 5900XT | `fp32` | 2608 ms | **1179 ms** | 2.2× |
| Ryzen 5900XT | `fp16` | 2641 ms | **630 ms** | 4.2× |
| Ryzen 5900XT | `fp16_iobinding` | 3165 ms | **635 ms** | 5.0× |

Boundary is the faster architecture in every configuration but one, and the gap
widens on CPU. That is consistent with what the two pipelines have to do: span
enumerates every span up to eight words wide and scores each against every
label, then runs a GRU over twenty occurrence slots — 1227 nodes of work around
the encoder. Boundary proposes 192 candidates once, shared across all queries,
and scores each pair — 480 nodes.

**What this comparison is not.** The two models find different entities — 13
against 14 — because they are different checkpoints with different training, so
this measures the *architectures on one paragraph*, not their quality. It says
nothing about which extracts better. And it is one text at one length: span cost
grows with the number of schema tasks, boundary cost is dominated by the bucket
it lands in, so the ordering can move with a different workload.

## The harness was wrong

Worth recording, because the failure was invisible and produced a table that had
GPU slower than CPU — which for a 190 M-parameter encoder cannot be true.

The first harness ran one warm-up and then timed back-to-back iterations in a
tight loop. On this contended host that produced numbers up to **100× too
large**, reproducibly: the same model, in the same minute, measured 13 ms with a
variant that printed each run and 1700 ms with the one that did not. The only
difference was a `println!` inside the loop.

The explanation that fits: the CUDA synchronisation spins, and a process spinning
in a tight loop on an oversubscribed machine gets descheduled while it waits. A
syscall per iteration yields the CPU and breaks that pattern. Pinning to eight
dedicated cores halved the damage but did not remove it — 915 ms against 1696 ms
— which points at scheduling rather than at the engine.

The harness now does five warm-up runs and sleeps 2 ms between iterations.
`--example warmup` prints every run so the curve is visible: on this pipeline
the first run costs ~565 ms against a ~14 ms steady state.

Two conclusions in earlier revisions of this file were artefacts of that bug and
are withdrawn: that CPU beat GPU, and that the span/boundary gap was
unexplained and unfavourable to boundary. Both reversed once the harness was
fixed.

## Correctness across devices

Unlike the timings, this part is solid.

| device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|
| CPU | 43/43 (**0.0000**) | 43/43 (0.0004) | 43/43 (0.0004) |
| RTX 3090 | 43/43 (**0.0000**) | 43/43 (0.0007) | 43/43 (0.0007) |

Identical spans in all six configurations; brackets give the largest score delta.
In `fp32` the agreement with PyTorch is exact at the precision the harness
records, on both devices — which puts a floor under the FP16 rows: their
deviation is quantisation, not a defect in the graphs. Choosing a device is a
performance decision, not an accuracy one.

## Two hypotheses that turned out to be wrong

Recorded so nobody spends the afternoon on them again, from when the broken
harness made GPU look slower than CPU.

**The GPU is not falling back to CPU.** Profiling the boundary head under the
CUDA provider puts **97.1% of nodes on CUDA** and 2.9% on CPU. The graph is not
being partitioned with a device transfer between each piece.

**The pipeline is not unusually large.** 5068 nodes per extraction, of which
4588 are the mDeBERTa encoder.

Both were correct rulings; the anomaly they were chasing simply did not exist.

## Not measured

Long documents, wide schemas, and buckets other than 128. Boundary cost is
dominated by the bucket the text lands in, so a 60-word text and a 260-word one
will differ by more than their length suggests.

## Reproducing

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
GLINER2_DEVICE=cuda:1 GLINER2_PRECISION=fp32 \
cargo run --release --example bench -p gliner25-core -- models/gliner2.5-multi-v1-onnx 25
```

The GPU run needs a runtime shipping the CUDA provider — the plain `onnxruntime`
wheel does not. `pip install onnxruntime-gpu` provides
`libonnxruntime_providers_cuda.so`; point `LD_LIBRARY_PATH` at its directory
along with the CUDA 12 libraries.
