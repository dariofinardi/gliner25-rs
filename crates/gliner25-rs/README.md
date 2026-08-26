# gliner25-rs

Native Rust inference for GLiNER2.5 **boundary** checkpoints on ONNX Runtime.
No Python at inference time, and none at build time either.

One crate: the boundary engine, with the schema families behind a default-on
feature.

```toml
[dependencies]
gliner25-rs = "0.5"
```

For GLiNER2 `span` checkpoints use
[gliner2-rs](https://crates.io/crates/gliner2-rs) instead. The two share a
lineage but not a graph — the architectures differ in what they enumerate, how
they score, and even in whether spans are inclusive — and mixing them produces
nonsense rather than an error.

---

## The five-minute version

```rust
use gliner25_rs::{BoundaryConfig, BoundaryEngine, SchemaTask};

gliner25_rs::init("my-app");                     // ort::init(), once per process

let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-onnx"))?;

let tasks = vec![SchemaTask::Entities(vec!["person".into(), "location".into()])];
let out = engine.extract("Mario Rossi lavora a Milano.", &tasks)?;

for m in &out.mentions {
    println!("{:?} {} {:.1}%", m.text, m.field, m.score * 100.0);
}
```

`ORT_DYLIB_PATH` must point at an ONNX Runtime **1.23 or newer** shared library.
Older ones run correctly and then segfault at process exit — see the root README
for the measurements and the upstream issue.

---

## What the engine gives you

| type | what it is |
|---|---|
| `BoundaryEngine` | the loaded model. `extract` takes `&mut self` |
| `BoundaryConfig` | where the model is, which variant, how it runs. Builder-style |
| `BoundaryManifest` | what the export declares: buckets, pool size, abstention, count head |
| `SchemaTask` | one group: entities, relations, or a classification |
| `BoundaryOutput` | `mentions`, `classifications`, `expected_counts` |
| `Mention` | text, task, field, score, byte range, **half-open** word range, query id |
| `BoundaryParams` | `threshold`, `overlap_policy`, `use_abstention`, temperature, multi-label override |
| `OverlapPolicy` | `Allow`, `Nested`, `Flat`, `Longest` |
| `GlinerError` | eight diagnosable failures, `E_GLI_001`…`E_GLI_008` |

```rust
let m = engine.manifest();
println!("{:?} buckets, pool {}", m.length_buckets, m.pool_size);
```

`engine.manifest()` is worth reading before anything else: it tells you the
largest text the model can see in one call, whether abstention is available,
and which overlap policy the export was trained for.

---

## Schema families

A wide schema makes labels compete for the same candidate pool, and spans
fragment. Families are one pass per group, so they do not:

```rust
use gliner25_rs::families::{Family, chunk_into_families, run_families};

let families = vec![
    Family::new("people",  ["person", "organization"]),
    Family::new("places",  ["location", "address"]),
    Family::new("contact", ["email", "phone number"]),
];
let out = run_families(&mut engine, text, &families, &BoundaryParams::default())?;
```

`chunk_into_families(&labels, 8)` splits a flat label list mechanically when you
have no natural grouping. Merging is by `(word_start, word_end, field)`, so
mentions from different families coexist over the same text — two families
disagreeing about a stretch is information, not a conflict.

The cost is one encoder pass per family. Measure before deciding it is too much:
on a wide schema it is often cheaper than the fragmentation it prevents.

---

## Long documents

The export declares length buckets, and the largest is a hard ceiling — 512
words for `gliner2.5-multi-v1`. `extract` does not truncate past it, it returns
`E_GLI_007 NO_LENGTH_BUCKET`, which is right for one call and useless for a
document.

```rust
let out = engine.extract_long(&document, &tasks)?;    // 384-word windows, 64 overlap
```

Measured on a 1 200-word document: `extract` fails, `extract_long` returns 51
mentions in 716 ms with every offset indexing the original text.

`extract_long_with(text, tasks, params, Chunker::new(256, 48)?)` sets the
geometry. What no merge can recover is a mention longer than the overlap, or a
relation whose ends fall in different windows — see the [`chunker`] module docs.

---

## Execution modes

Between any two fragments an intermediate tensor either returns to host memory
or stays where the provider produced it. Same maths, different transport.

```rust
use gliner25_rs::chain::ExecutionMode;
let cfg = BoundaryConfig::new("models/g25").with_execution(ExecutionMode::IoBinding);
```

| mode | |
|---|---|
| `Auto` *(default)* | bound on a device provider, standard on CPU |
| `IoBinding` | intermediates stay in device memory — **1.7× on an RTX 3090** |
| `Standard` | every output returns to the host first; works everywhere |

Smaller than the span engine's 2.4×, and structurally so: all five head outputs
are decoded on the host, so binding cannot keep them anywhere. What it saves is
the encoder output, which `routed_gather` consumes three times per sentence.

A device allocation failure drops the engine to the standard path for the rest
of its life rather than failing the call. `engine.execution()` reports what is
actually in force.

---

## Getting the weights

```rust
use gliner25_rs::hub;
let cfg = BoundaryConfig::new("models/g25").or_download(hub::GLINER25_MULTI_V1);
```

**Only the variant you will run is downloaded** — the execution mode picks it,
so a bound engine fetches the FP16-I/O graphs and a standard one the FP32,
rather than pulling every copy of every fragment. If a repository does not
publish the preferred variant the engine falls back rather than failing.
`jugaadsrl/gliner2.5-multi-v1-onnx` publishes all three variants since
2026-08-26, so a bound engine fetches 590 MB instead of 1.1 GB.

The manifest is fetched and parsed *before* the heads, because which
`boundary_head_L*` exist is a property of the export — guessing would either
miss a bucket or request files that were never written. Sidecar `.onnx.data`
files travel with their graph.

Both layouts load: flat, and grouped into `fp32_25/` + `fp16_25/` (or GLiNER2's
`fp32_v2/` + `fp16_v2/`). Point the engine at the parent and it picks the
subfolder; point it at one folder and it loads that.

---

## Four things that will surprise you

**Spans are half-open.** `Mention::word_start` and `word_end` are `[start, end)`
— a one-word mention has `word_end == word_start + 1`. The `span` architecture
in `gliner2-rs` uses inclusive ranges. Byte offsets are half-open in both.

**Length buckets are not a detail.** `torch.export` specialises `num_words`, so
each bucket is its own exported head. The engine pads the text to the smallest
bucket that fits and loads that head lazily. Padding is transparent to 5e-07,
verified against the unpadded run.

**Pool order carries no meaning.** The candidate pool is produced by a sort that
had `stable=True` stripped at export — ONNX has no stable `Sort`. Compare
candidate *sets*, never positions: a parity check that lines up `cand_indices`
by index will report differences that are not there.

**Multi-label classification falls back to the argmax.** When no label clears
the threshold, `verdict` returns the single highest rather than nothing, which
is what `gliner2` does. Read `out.classifications` directly if you want the raw
probabilities.

---

## Weights are not ours

GLiNER2.5 is developed by [Fastino](https://fastino.ai); the GLiNER line it
descends from is the work of Urchade Zaratiana et al. Converting a model to ONNX
changes neither its licence nor its ownership. See [`NOTICE`](../../NOTICE).

Engine code: Apache-2.0. Written by **Dario Finardi**, published by **Jugaad
s.r.l.**, used in production in **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).
