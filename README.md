# gliner25-rs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/dariofinardi/gliner25-rs)

**Native Rust inference for GLiNER2.5 on ONNX Runtime**

Entity extraction with no Python at inference time. This repository is a Cargo
workspace: a shared foundation, the boundary engine, and a thin extension on top.

Written by **Dario Finardi**. Published by **Jugaad s.r.l.**, which uses it in
production inside **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).

For GLiNER2 `span` checkpoints use
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs) instead.

---

## The crates

| crate | what it is | docs |
|---|---|---|
| [`gliner25-core`](crates/gliner25-core) | the engine: prompt construction, ONNX Runtime helpers, overlap policies, boundary inference | [README](crates/gliner25-core/README.md) |
| [`gliner25`](crates/gliner25) | schema families and merging | [README](crates/gliner25/README.md) |

One model, two crates. `gliner25-core` is self-contained: prompt construction,
`ort` helpers and overlap policies live in it rather than in a crate shared with
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs).

That is a deliberate trade. A shared crate would have to be published under a
single name and would tie the two repositories' release cadences together, for
four modules that change rarely. The cost is real and worth stating: **a fix to
those modules has to land in both repositories.** It has happened twice already —
the cross-label suppression rule and the multi-label argmax fallback — so treat
them as a pair when changing anything below the engine.

---

## Model

Weights are **not** ours. GLiNER2.5 is developed by
[Fastino](https://fastino.ai) (arXiv:2507.18546); the GLiNER line it descends
from is the work of Urchade Zaratiana et al. Converting a model changes neither
its licence nor its ownership — see [`NOTICE`](NOTICE).

| upstream | ONNX export |
|---|---|
| [`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1) | [`jugaadsrl/gliner2.5-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx) |

mDeBERTa-v3-base, multilingual, 512-position encoder.

### Precision

| suffix | I/O | use for |
|---|---|---|
| `_fp32` | FP32 | universal fallback, OpenVINO, CPU |
| `_fp16` | FP32 (`keep_io_types=True`) | CoreML, which demands FP32 I/O |
| `_fp16_iobinding` | FP16 | CUDA, ROCm, QNN — see the note below |

Selected automatically per platform, overridable with
`GLINER2_PRECISION=fp32|fp16|fp16_iobinding`. A full FP16 set is about 540 MB.

### A note on `_fp16_iobinding`

The suffix names what the variant was *exported for*, not what this engine does
with it. `keep_io_types=False` leaves the graph inputs and outputs in FP16 as
well as the weights, which is what ORT's zero-copy `IoBinding` needs to keep
tensors in device memory across the fragment chain.

**This engine does not implement `IoBinding`.** It loads those graphs and runs
them normally, so the variant still saves the FP32↔FP16 conversions at each
boundary, but intermediate tensors round-trip through host memory between
fragments. On CPU that costs nothing; on a discrete GPU it is the PCIe traffic
the variant exists to avoid.

If you need real zero-copy binding today, use the V2 pipeline in
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs)'s `gliner2-inference`
crate, which does — though only for the span architecture. Implementing it here
is tracked work, not a claim.


---

## Quick start

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run --release --example extract -p gliner25-core -- models/gliner2.5-multi-v1-onnx
```

```rust
use gliner25_core::{BoundaryConfig, BoundaryEngine, SchemaTask};

gliner25_core::init("my-app");
let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-multi-v1-onnx"))?;

let tasks = vec![SchemaTask::Entities(vec![
    "person".into(), "organization".into(), "location".into(),
])];

for m in engine.extract("Mario Rossi works at Apple.", &tasks)?.mentions {
    println!("{} -> {} ({:.1}%)", m.text, m.field, m.score * 100.0);
}
```

Spans are **half-open** `[start, end)` — unlike GLiNER2's inclusive ranges.
Byte offsets index the original text, so extracted spans keep their casing.

Passing many labels at once makes them compete; see
[`gliner25`](crates/gliner25/README.md) for schema families, the documented
remedy.

---

## Requirements

- Rust **edition 2024**, MSRV **1.88**
- `ort` **≥ 2.0.0-rc.13, < 3.0**, with `default-features = false` — nothing is
  downloaded at build time and no execution-provider libraries are copied next
  to your binary.
- ONNX Runtime shared library, resolved at run time from `ORT_DYLIB_PATH`.
  Verified against ONNX Runtime 1.25.1 at API level 17, and against the
  `onnxruntime-gpu` 1.23.2 build for CUDA.

  **The rc.13 floor is not arbitrary.** Release candidates 10 through 12 were
  tried and rejected: on **ARM CPU** some models hung during session
  initialisation or inference, reproducibly enough that this project stayed on
  rc.9 for months rather than move to them. rc.13 is the first candidate since
  rc.9 that runs those models on ARM, which is why the migration skipped three
  releases. Do not lower the floor.

  Upwards, the requirement is a caret rather than an exact pin, so these crates
  can be combined with anything else depending on `ort`. That is a calculated
  risk while `ort` is still in release candidates: between rc.9 and rc.13,
  `commit()` changed its return type, `Session::run` started taking `&mut self`,
  `try_extract_tensor` began returning a `Shape`, and `Outlet`'s fields went
  private. A later rc can break the build the same way. **Pin exactly in your
  own application** if you need that guarantee — a library should not impose it
  on its dependents.

---

## Performance

On an RTX 3090, one ~90-word paragraph, five entity labels, 14 mentions:
**17.9 ms** in `fp32`, 18.6 ms in `fp16`, 20.3 ms in `fp16_iobinding`. The three
sit within 13% of each other, which is inside what a shared host moves on its
own — do not pick a precision from that.

### Against the span architecture, at parity

Same paragraph, same five labels, same precision, same host, same card, compared
with [gliner2-rs](https://github.com/dariofinardi/gliner2-rs) running
`GLiNER2-Guardrails-PII-Multi`, which is likewise a flat export:

| device | precision | span (GLiNER2) | boundary (GLiNER2.5) | ratio |
|---|---|---|---|---|
| RTX 3090 | `fp32` | 26.5 ms | **17.9 ms** | 1.5× |
| RTX 3090 | `fp16` | 55.4 ms | **18.6 ms** | 3.0× |
| RTX 3090 | `fp16_iobinding` | 20.6 ms | 20.3 ms | 1.0× |
| Ryzen 5900XT | `fp32` | 2608 ms | **1179 ms** | 2.2× |
| Ryzen 5900XT | `fp16` | 2641 ms | **630 ms** | 4.2× |
| Ryzen 5900XT | `fp16_iobinding` | 3165 ms | **635 ms** | 5.0× |

Boundary is faster in every configuration but one, and the gap widens on CPU.
That follows from what the two pipelines do: span enumerates every span up to
eight words wide, scores each against every label and runs a GRU over twenty
occurrence slots — 1227 nodes around the encoder. Boundary proposes 192
candidates once, shared across all queries, and scores each pair — 480 nodes.

**This is not a quality comparison.** The two models find different entities, 13
against 14, because they are different checkpoints with different training. It
measures the architectures on one paragraph, and says nothing about which
extracts better. It is also one text at one length: span cost grows with the
number of schema tasks, boundary cost is dominated by the bucket it lands in, so
a different workload can move the ordering.

The host was shared and under load throughout; see [`BENCHMARKS.md`](BENCHMARKS.md)
for the caveats, and for the harness bug that made an earlier revision of this
table say the opposite.

## Verification

Per-fragment parity proves the ONNX graphs are faithful. It is not enough:
prompt construction, routing, candidate decoding, abstention and overlap
resolution all live in Rust, outside the graphs.

Against the PyTorch reference, 12 cases across 6 languages:

Run on both devices and every precision, because a CUDA kernel producing
different numbers from its CPU counterpart is exactly the kind of thing that
goes unnoticed otherwise:

| device | `fp32` | `fp16` | `fp16_iobinding` |
|---|---|---|---|
| CPU | 43/43 (**0.0000**) | 43/43 (0.0004) | 43/43 (0.0004) |
| RTX 3090 | 43/43 (**0.0000**) | 43/43 (0.0007) | 43/43 (0.0007) |

Identical spans in all six; brackets give the largest score delta. In `fp32` the
agreement with PyTorch is exact at the precision the harness records, which puts
a floor under the FP16 rows: their deviation is quantisation, not a defect in
the graphs.

```sh
python onnx_conversion_scripts/compare_with_pytorch.py reference \
    --model_path fastino/gliner2.5-multi-v1 --cases tests/cases.json --out /tmp/pytorch.json

ORT_DYLIB_PATH=… cargo run --release --example dump_json -p gliner25-core -- \
    models/gliner2.5-multi-v1-onnx tests/cases.json > /tmp/rust.json

python onnx_conversion_scripts/compare_with_pytorch.py diff \
    --reference /tmp/pytorch.json --candidate /tmp/rust.json
```

`verify_parity.py` covers the fragments individually, including the assumption
the length bucketing rests on: padding to a larger bucket, even with random
noise in the padded rows, yields the same candidate set and probabilities to
within 5e-07.

---

## Exporting

```sh
python onnx_conversion_scripts/export_boundary_v1.py \
    --model_path fastino/gliner2.5-multi-v1 \
    --out_dir models/gliner2.5-multi-v1-onnx
```

Needs `torch`, `transformers`, `gliner2>=2.0.0`, `onnx`, `onnxruntime`,
`onnxscript`. It refuses `span` checkpoints rather than producing a silently
wrong export.

Three things the exporter has to work around, each documented in its module
docstring: `torch.export` specialises `num_words`, so heads are emitted per
length bucket; ONNX has no stable Sort, so `stable=True` is stripped and pool
order stops being meaningful; and `convert_float_to_float16` leaves
`ConstantOfShape` untyped, which makes every FP16 head unloadable until it is
repaired.

---

## License and attribution

Licensed under the [Apache License, Version 2.0](LICENSE).

The model is the work of the [Fastino](https://fastino.ai) team and is not
distributed from this repository. See [`NOTICE`](NOTICE) for the full
attribution.

Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
