# gliner25-core

The GLiNER2.5 **boundary**-architecture inference engine, on ONNX Runtime via
`ort` 2.0.0-rc.13.

Model: [`jugaadsrl/gliner2.5-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx),
converted from [`fastino/gliner2.5-multi-v1`](https://huggingface.co/fastino/gliner2.5-multi-v1).

Schema families and merging live in [`gliner25`](../gliner25) on top.

## Usage

```rust
use gliner25_core::{BoundaryConfig, BoundaryEngine, BoundaryParams, SchemaTask};

gliner25_core::init("my-app");
let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-multi-v1-onnx"))?;

let tasks = vec![SchemaTask::Entities(vec![
    "person".into(), "organization".into(), "location".into(),
])];

let out = engine.extract_with(
    text,
    &tasks,
    &BoundaryParams { threshold: 0.5, ..Default::default() },
)?;

for m in &out.mentions {
    println!("{} -> {} ({:.1}%)  words [{}..{})",
             m.text, m.field, m.score * 100.0, m.word_start, m.word_end);
}
```

`BoundaryConfig::new` reads `boundary_manifest.json` for the pool size, the
length buckets, the overlap policy and whether the abstention and count heads
are present, then picks the best precision for the platform. Override with
`GLINER2_PRECISION=fp32|fp16|fp16_iobinding`.

## Spans are half-open

`[start, end)`, unlike the **inclusive** `[w, w + k]` of the span architecture in
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs). If you consume both
engines, convert deliberately at the boundary between them — mixing the
conventions is a silent off-by-one that survives every type check.

## How the architecture works

It enumerates nothing. A pool of `(start, end)` candidates is proposed **once,
shared across all queries**, at constant size `pool_size` (192), and each
query/candidate pair gets a logit:

```text
encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
  +- routed_gather(lhs, idx, mask) -> text / query / choice states
  +- boundary_head_L{bucket}(text_states, text_mask, query_states, query_mask)
  |     -> cand_indices    [1,Q,C,2]   half-open (start, end)
  |     -> pair_logits     [1,Q,C]
  |     -> cand_valid      [1,Q,C]
  |     -> null_logits     [1,Q]       per-query abstention
  |     -> count_log_rates [1,Q]       expected mentions per query
  +- classifier(choice_states) -> logits [K]
```

`BoundaryParams::use_abstention` honours `null_logits`: a query whose null logit
beats its best candidate produces nothing at all. `BoundaryOutput::expected_counts`
exposes the count head.

## Length buckets

The heads have a **static** `num_words`: `torch.export` specialises it, because
the candidate-pool builder loops in Python over a symbolic dimension. One head
is exported per bucket — 64, 128, 256, 512 — and the engine picks the smallest
that fits, padding with `text_mask = 0`.

This costs nothing: a head is a few MB against 530 MB of encoder, and static
shapes are what TensorRT and QNN want, and what zero-copy binding would need.
Masked padding is verified
transparent to 5e-07, even with random noise in the padded rows. Texts beyond
the largest bucket raise `E_GLI_007` rather than silently truncating; the
encoder caps at 512 positions anyway.

## Pool order carries no meaning

It comes from an `argsort` over frequently near-tied scores, and sort stability
is exactly what the ONNX export removes — ONNX has no stable Sort and
`aten.sort.stable` has no translation. Under FP16, rounding permutes the ties
while still selecting the same candidates. If you compare two runs, compare
`cand_indices` as a **set** of `(start, end)` pairs, never positionally. The
decoder here is order-independent.

## Multi-label classification falls back to the argmax

gliner2's multi-label decoding never returns an empty list: when no label clears
the threshold, the top-scoring one comes back anyway. Use
`BoundaryOutput::verdict`, which implements the rule. Thresholding the raw
scores yourself silently disagrees with the reference.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
