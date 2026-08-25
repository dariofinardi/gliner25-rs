# gliner25-rs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen.svg)](https://github.com/dariofinardi/gliner25-rs)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/dariofinardi/gliner25-rs)

**Native Rust inference engine for GLiNER2.5**

`gliner25-rs` runs **GLiNER2.5** models on ONNX Runtime through `ort`
2.0.0-rc.13, with no Python at inference time. It ships both the ONNX exporter
and the Rust runtime: entity extraction, relations and classification from a
single encoder pass.

Written by **Dario Finardi**. Published by **Jugaad s.r.l.**, which uses it in
production inside **Edito** and **Omissis** — [edito-pdf.com](https://edito-pdf.com).

For GLiNER2 `span` checkpoints (such as `gliner2-multi-v1` or
`GLiNER2-Guardrails-PII-Multi`) use the separate
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs) crate instead. The two
architectures share nothing but the encoder and the prompt format, so they get
one crate each.

---

## GLiNER2.5 is a different model, not a newer GLiNER2

The `span` architecture enumerates every span up to `max_width` words,
represents each with `span_rep`, predicts how many occurrences to expect
(`count_pred` plus a `count_lstm` GRU) and scores every span × label pair.

The `boundary` architecture (`BoundaryExtractor`) enumerates nothing. It
proposes a pool of `(start, end)` candidates **shared across all queries**, of
constant size `pool_size` (typically 192), and assigns a logit to each
query × candidate pair. Two extra heads give per-query abstention
(`null_logits`) and an expected mention count (`count_log_rates`). There is no
`span_rep` and no `count_lstm`.

Boundary spans are **half-open**, `[start, end)`. Span-architecture spans are
**inclusive**, `[w, w+k]`. Mixing the conventions is an easy way to introduce
off-by-one errors.

---

## Quick start

### 1. Export the model

```sh
python onnx_conversion_scripts/export_boundary_v1.py \
    --model_path fastino/gliner2.5-multi-v1 \
    --out_dir models/gliner2.5-multi-v1-onnx
```

Requires `torch`, `transformers`, `gliner2>=2.0.0`, `onnx`, `onnxruntime`,
`onnxscript`. The exporter refuses `span` checkpoints rather than producing a
silently wrong export.

Pre-converted weights are published at
[`jugaadsrl/gliner2.5-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx).

### 2. Check parity against PyTorch

```sh
python onnx_conversion_scripts/verify_parity.py \
    --model_path fastino/gliner2.5-multi-v1 \
    --onnx_dir models/gliner2.5-multi-v1-onnx
```

Every fragment is compared with its PyTorch counterpart across all three
precision variants, and the assumption the bucketing rests on is checked
explicitly.

### 3. Run it

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run --release --example extract -- models/gliner2.5-multi-v1-onnx
```

```rust
use gliner25::{BoundaryConfig, BoundaryEngine, SchemaTask};

gliner25::init("my-app");

let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-multi-v1-onnx"))?;

let tasks = vec![SchemaTask::Entities(vec![
    "person".into(), "organization".into(), "location".into(),
])];

for m in engine.extract("Mario Rossi works at Apple in Cupertino.", &tasks)?.mentions {
    println!("{} -> {} ({:.1}%)  bytes [{}..{})",
             m.text, m.field, m.score * 100.0, m.char_start, m.char_end);
}
```

`BoundaryParams` exposes the per-call knobs: `threshold`, `overlap_policy`,
`use_abstention`, `classification_temperature`, `multi_label`.

---

## Precision variants

| Variant | I/O | Use for |
|---|---|---|
| `_fp32` | FP32 | universal fallback, OpenVINO, CPU |
| `_fp16` | FP32 (`keep_io_types=True`) | CoreML, which demands FP32 I/O |
| `_fp16_iobinding` | FP16 | CUDA, ROCm, QNN with IOBinding |

Selection is automatic — `_fp16_iobinding` on Linux and Windows, `_fp16` on
macOS — and can be forced with
`GLINER2_PRECISION=fp32|fp16|fp16_iobinding`.

---

## Design notes

### Length buckets

`torch.export` specialises `num_words` to a constant: the candidate-pool
builder contains a Python loop over a symbolic dimension, and no `export_mode`
removes it on this branch. One head is therefore exported per length bucket
(64/128/256/512 by default) and the runtime picks the smallest that fits,
padding with `text_mask = 0`.

This is not a compromise. A head weighs under 5 MB against 1.1 GB of encoder,
and static shapes are exactly what TensorRT, QNN and IOBinding want.

The assumption it rests on — masked padding is transparent — is verified by
`verify_parity.py`: for the same real words, padding to a larger bucket **even
with random noise in the padded rows** yields the same candidate set and
probabilities to within 5e-07.

The smallest bucket is 32 because `select_top_boundaries` computes
`k = min(pool_boundary_top_k, n_boundaries)`; below that the graph would be
traced with a reduced `k` and would no longer hold for longer texts.

### Stable sort

The pool uses `torch.sort(..., stable=True)`. ONNX has no stable Sort and
`aten.sort.stable` has no translation, so `stable=True` is stripped at export.

The consequence worth remembering: **pool order carries no meaning**. Comparing
`cand_indices` positionally will fail a perfectly correct graph — under FP16,
rounding permutes near-ties while still selecting the same 192 candidates.
`verify_parity.py` compares them as a set of `(start, end)` pairs with their
associated probabilities. The Rust decoder is order-independent.

### ConstantOfShape after FP16 conversion

`convert_float_to_float16` rewrites a `ConstantOfShape` output's `value_info`
as `float16` but leaves the `value` attribute alone; when the attribute is
absent the ONNX spec mandates `float32`, and the graph will not load:

```
Type Error: Type (tensor(float16)) of output arg (val_684) of node
(node_ConstantOfShape_676) does not match expected type (tensor(float))
```

`export_boundary_v1.py::_fix_constant_of_shape` materialises the attribute in
the declared type. Without it every FP16 boundary head is unusable.

### Multi-label classification falls back to the argmax

gliner2's multi-label decoding never returns an empty list: **when no label
clears the threshold, the top-scoring one is returned anyway**. Threshold the
scores yourself and you will silently disagree with the reference
implementation. Use [`BoundaryOutput::verdict`], which implements the rule.

Relatedly, `multi_label` belongs to the task rather than the request — a single
call routinely mixes single-label and multi-label tasks — hence
`SchemaTask::classification` and `SchemaTask::multi_label_classification`.

### Overlap policies

`flat` is **not** a greedy NMS: it is weighted interval scheduling, the
non-overlapping subset of maximum total score. With `A=[0,4) 0.6`,
`B=[0,2) 0.5`, `C=[2,4) 0.5`, greedy keeps `A` while the correct answer is
`B+C`. `allow`, `nested` and `longest` are also implemented, matching
`gliner2/inference/overlap.py`.

### Prompt construction

`processor.rs` reproduces `gliner2/processor.py::SchemaTransformer` exactly,
including three details that are easy to get wrong: the prompt is **not**
wrapped in `[CLS]`/`[SEP]`; field indices point at the `[E]`/`[R]`/`[L]` marker
rather than the label name that follows it; and text words are lower-cased
before sub-word tokenization while character offsets keep indexing the original
string, so extracted spans preserve their casing. The layout is pinned by
`processor::tests::ground_truth_layout` against ground truth generated by
gliner2 2.0.0.

---

## Requirements

- Rust **edition 2024**, MSRV **1.88**
- ONNX Runtime shared library, resolved at run time via `ORT_DYLIB_PATH`
  (the crate enables `ort`'s `load-dynamic`). Enable `ort`'s
  `download-binaries` feature instead to fetch it at build time.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
