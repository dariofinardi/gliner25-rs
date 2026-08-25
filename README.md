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
| [`gliner-core`](crates/gliner-core) | prompt construction, ONNX Runtime helpers, overlap policies | [README](crates/gliner-core/README.md) |
| [`gliner25-core`](crates/gliner25-core) | the boundary inference engine | [README](crates/gliner25-core/README.md) |
| [`gliner25`](crates/gliner25) | schema families and merging | [README](crates/gliner25/README.md) |

One model, three crates — deliberately. The layering is not for this repository's
sake: `gliner-core` is shared with `gliner2-rs`, and keeping the seam visible is
what will let this workspace depend on it from crates.io once it is published,
rather than carrying a copy. **Today it is a copy**, and that is the one piece of
duplication the split does not remove. A second copy is exactly how the two
drifted apart before.

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
- ONNX Runtime shared library, resolved at run time from `ORT_DYLIB_PATH`. The
  workspace pins `ort = 2.0.0-rc.13` with `default-features = false`, so nothing
  is downloaded at build time and no EP libraries are copied next to your
  binary. Verified against ONNX Runtime 1.25.1 at API level 17.

---

## Performance

Measured on an RTX 3090 and a Ryzen 9 5900XT, with the caveats that matter — the
host was under load 18 throughout, so the CPU figures are an upper bound and
some GPU rows are noise. See [`BENCHMARKS.md`](BENCHMARKS.md) for the full table
and what can and cannot be concluded from it.

The short version for this repository: **the timing figures are not usable.**
Two runs of the same matrix an hour apart differ by up to 4× on identical
configurations, and CPU comes out ahead of GPU — which for a 190 M-parameter
encoder is not a result, it is contention. Profiling ruled out provider fallback
(97% of nodes do run on CUDA) and graph size (the two architectures are within
15% of each other), so the span/boundary gap remains unexplained and needs an
idle machine to settle.

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
