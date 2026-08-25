# gliner25

Schema families for GLiNER2.5. Thin layer over
[`gliner25-core`](../gliner25-core): the engine is there, this crate carries the
schema hygiene a boundary model needs in practice.

## Why families

Labels within one schema compete. The queries share the encoder context, so a
wide schema makes them interfere: on `gliner2.5-multi-v1` this shows up as
date-like entities being lost when many unrelated labels are present — a
regression against the span models that only appears at width.

The remedy is to split into families of related labels, run each separately and
merge.

```rust
use gliner25::{Family, run_families};
use gliner25_core::{BoundaryConfig, BoundaryEngine, BoundaryParams};

gliner25_core::init("my-app");
let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-multi-v1-onnx"))?;

let families = vec![
    Family::new("people",  ["person", "job_title", "organization"]),
    Family::new("places",  ["location", "address", "country"]),
    Family::new("temporal", ["date", "time", "duration"]),
];

let out = run_families(&mut engine, text, &families, &BoundaryParams::default())?;
```

Merging is by `(word_start, word_end, field)`: a label belongs to exactly one
family, so the same triple appearing twice is a genuine duplicate and the
higher-scoring one wins. Mentions from **different** families are kept side by
side even when they cover the same text — labels are independent, and two
families disagreeing about a stretch is information, not a conflict.

## The cost

One encoder pass per family. That is the price of not letting the labels
compete, and whether it is worth paying depends on how wide your schema was
going to be. Measure before assuming.

If you have no semantic grouping to offer, `chunk_into_families` splits a flat
list into fixed-size groups. It is a fallback, not an equivalent: the
interference is between *unrelated* labels, so real families — dates together,
identifiers together — work better than arbitrary chunks.

Apache-2.0. Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
