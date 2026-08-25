# gliner25-rs

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/Status-Beta-blue.svg)](https://github.com/dariofinardi/gliner25-rs)

**Native Rust inference for GLiNER2.5 on ONNX Runtime**

Entity extraction with no Python at inference time. One crate: the boundary
engine, with the schema families behind a default-on feature.

Written by **Dario Finardi**. Published by **Jugaad s.r.l.**, which uses it in
production inside **Edito** and **Omissis** —
[edito-pdf.com](https://edito-pdf.com).

For GLiNER2 `span` checkpoints use
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs) instead.

---

## The crates

| crate | what it is | docs |
|---|---|---|
| [`gliner25-rs`](crates/gliner25-rs) | the engine, plus schema families behind a default-on feature | [README](crates/gliner25-rs/README.md) |

```toml
[dependencies]
gliner25-rs = "0.4"
```

`gliner25-rs` loads a model from a local directory, and fetches it from the Hub
if that directory is empty — see [Getting the weights](#getting-the-weights).
Switch the `hub` feature off and the crate has no HTTP client, no TLS stack and
no Hub client in its dependency tree at all.

One model, one crate. Prompt construction, `ort` helpers, overlap policies and
schema families all live in it rather than in separate packages — 0.1 split them
and produced pieces that were only ever installed as a set.

That is a deliberate trade. A shared crate would have to be published under a
single name and would tie the two repositories' release cadences together, for
four modules that change rarely. The cost is real and worth stating: **a fix to
those modules has to land in both repositories.** It has happened twice already —
the cross-label suppression rule and the multi-label argmax fallback — so treat
them as a pair when changing anything below the engine.

---

## Getting the weights

Point the engine at a directory. If it holds an export, it is used untouched.
If it does not, the export is fetched from the Hub before the engine starts:

```rust
use gliner25_rs::{BoundaryConfig, BoundaryEngine, hub};

let cfg = BoundaryConfig::new("models/gliner2.5-onnx").or_download(hub::GLINER25_MULTI_V1);
let mut engine = BoundaryEngine::new(cfg)?;   // downloads only if the directory is empty
```

Skip the local path entirely and work straight out of the cache:

```rust
let mut engine = BoundaryEngine::new(BoundaryConfig::from_hub(hub::GLINER25_MULTI_V1))?;
```

**The local directory always wins.** A checkout already on disk is never
re-fetched, and the network is reached only on a miss. Files land in the shared
Hub cache (`HF_HOME`, else `~/.cache/huggingface`), so a model already pulled by
the Python library is not pulled again.

`hub::GLINER25_MULTI_V1` is
[`jugaadsrl/gliner2.5-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx).
Any other repository works — `hub::Model::new(repo_id)` takes a private
fine-tune as readily as a published one.

Try it:

```sh
ORT_DYLIB_PATH=… cargo run --release --example download -p gliner25-rs -- models/gliner2.5-onnx
```

### Length buckets are read, not guessed

A boundary export ships one head per length bucket, and which buckets exist is a
property of the export. `boundary_manifest.json` is therefore fetched and parsed
first, and only the heads it declares are requested. Fragments whose weights sit
past the 2 GB protobuf limit carry a sidecar `.onnx.data`, which is fetched with
them — without it the download succeeds and the *session build* fails, with a
filesystem error naming a file nobody asked for.

### If you would rather it never touched the network

```toml
gliner25-rs = { version = "0.3", default-features = false, features = ["families"] }
```

That removes `hf-hub`, `ureq` and `rustls` and leaves the crate with no network
stack whatsoever.

### On the TLS backend

`hf-hub` arrives with `default-features = false, features = ["ureq"]`, which
resolves TLS through **`rustls`**. Its default feature set pulls `native-tls`
and with it `openssl` — a C library and the CVE stream that comes with it — for
no benefit here.

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

### Execution modes

The engine is a chain of ONNX fragments. Between any two of them the
intermediate tensor either goes back to host memory and is rebuilt for the next
fragment, or stays where the provider produced it and is bound straight into the
next fragment's input.

Same fragments, same order, same arithmetic — only the transport differs. One
pipeline and a switch:

```rust
use gliner25_rs::{BoundaryConfig, BoundaryEngine, chain::ExecutionMode};

let cfg = BoundaryConfig::new("models/gliner2.5-onnx")
    .with_execution(ExecutionMode::IoBinding);
```

| mode | what it does |
|---|---|
| `Auto` *(default)* | bound on a device provider, standard on CPU |
| `IoBinding` | intermediates stay in device memory across the chain |
| `Standard` | every output returns to host memory first — works everywhere |

`GLINER2_EXECUTION=standard|binding|auto` sets it from the environment, and
`GLINER2_NO_IOBINDING=1` forces the standard path.
`BoundaryEngine::execution()` reports the mode actually in force, after `Auto`
resolves and after any fallback. A device allocation failure during binding
drops the engine to the standard path for the rest of its life rather than
failing the call.

### What it is worth

RTX 3090, `gliner2.5-multi-v1`, fp32, 25 runs, median:

| | `Standard` | `IoBinding` | |
|---|---|---|---|
| boundary | 20.4 ms | **11.8 ms** | 1.7× |

The gain is real but smaller than the span engine's 2.4× in
[gliner2-rs](https://github.com/dariofinardi/gliner2-rs), and the reason is
structural: every one of the head's five outputs — `cand_indices`,
`pair_logits`, `cand_valid`, `null_logits`, `count_log_rates` — is decoded on
the host, so binding cannot keep them anywhere. What it does save is the
encoder output, which `routed_gather` consumes **three times** per sentence and
which is the largest tensor in the pipeline.

**No CPU figures here, deliberately.** The measurements were taken on a shared
development machine at load average 17–19, where the same mode varied by 13×
between consecutive runs. Anything quoted from that would be invented. What can
be said is what the design implies: on CPU "device memory" *is* host memory, so
binding saves no copy and only its bookkeeping remains — which is why `Auto`
does not use it there.

### A note on `_fp16_iobinding`

The suffix names what a variant was exported for: `keep_io_types=False` leaves
graph inputs and outputs in FP16 as well as the weights, which is what zero-copy
binding wants. The published `gliner2.5-multi-v1` export ships **fp32 only**, so
the question does not arise for it — `Precision::autodetect` finds no FP16
variant and settles on fp32. It matters if you export your own.

---

## Quick start

```sh
ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
cargo run --release --example extract -p gliner25-rs -- models/gliner2.5-multi-v1-onnx
```

```rust
use gliner25_rs::{BoundaryConfig, BoundaryEngine, SchemaTask};

gliner25_rs::init("my-app");
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

Passing many labels at once makes them compete; see the `families` module for
schema families, the documented remedy.

---

## Requirements

- Rust **edition 2024**, MSRV **1.88**
- `ort` **≥ 2.0.0-rc.13, < 3.0**, with `default-features = false` — nothing is
  downloaded at build time and no execution-provider libraries are copied next
  to your binary.
- ONNX Runtime shared library **1.23 or newer**, resolved at run time from
  `ORT_DYLIB_PATH`. Verified against ONNX Runtime 1.25.1 at API level 17, and
  against the `onnxruntime-gpu` 1.23.2 build for CUDA.

  **Older runtimes segfault when the process exits.** The API level `ort`
  requires is 17, which runtimes well below 1.23 satisfy — so an old shared
  library loads, runs correctly, and then crashes on the way out. One binary,
  three runtimes:

  | ONNX Runtime | inference | exit code |
  |---|---|---|
  | 1.20.0 | correct | **139 — SIGSEGV** |
  | 1.22.0 | correct | **139 — SIGSEGV** |
  | 1.23.2 | correct | 0 |

  The scores are identical in all three; only the exit differs. The crash lands
  after `main` returns, so output is complete and nothing is corrupted — but the
  exit code breaks CI, shell `&&` chains and process supervisors, and in a
  long-running server it surfaces at shutdown.

  It is not this crate, and it is not the CUDA EP either — this is CPU-only.
  The same fault reproduces in twenty lines of `ort` with no GLiNER code
  involved: create a session, drop it, return from `main`. It goes away if the
  session is leaked instead of dropped, and it reproduces with one intra-op
  thread as readily as with four, so it is not the thread pool.

  The root cause is `ort`'s global `Environment` being released at process exit
  after the session state it refers to is gone. It is fixed upstream by
  [pykeio/ort#610](https://github.com/pykeio/ort/pull/610) (details and
  measurements in [pykeio/ort#614](https://github.com/pykeio/ort/issues/614)),
  which makes the
  environment manual instead of global — verified here: `ort` from git exits
  cleanly even against ONNX Runtime 1.20.0. That change is **not in rc.13**, the
  newest release, so until it ships the runtime version is what decides it.

  If you are pinned to an older runtime, `std::process::exit(0)` at the end of
  `main` sidesteps the crash, at the cost of skipping every other destructor
  too.

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

ORT_DYLIB_PATH=… cargo run --release --example dump_json -p gliner25-rs -- \
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

### Deriving FP16 from an FP32 export

The exporter emits all three precisions, but it needs the PyTorch checkpoint.
When you have the ONNX and not the checkpoint — the published
[`jugaadsrl/gliner2.5-multi-v1-onnx`](https://huggingface.co/jugaadsrl/gliner2.5-multi-v1-onnx)
ships FP32 only — derive them from what you have:

```sh
python onnx_conversion_scripts/downcast_fp16.py <export_dir> --out models/g25-fp16
```

Needs only `onnx`, `onnxruntime` and `numpy`. It writes both variants beside
each fragment, repairs the `ConstantOfShape` nodes the FP16 converter leaves
inconsistent, and copies `tokenizer.json` and `boundary_manifest.json` so the
output directory loads as it stands.

Measured on the published export: **1141 MB → 574 MB**, exactly half, the
encoder accounting for nearly all of it (1111 → 556 MB).

#### Layout

The result is grouped by precision, the way the published GLiNER2 exports are,
with names that say which lineage it belongs to:

```
gliner2.5-multi-v1-onnx/
├── fp32_25/                       13 files, 1157 MB
│   ├── encoder_fp32.onnx  (+ .data)
│   ├── boundary_head_L{64,128,256,512}_fp32.onnx  (+ .data)
│   ├── routed_gather_fp32.onnx, classifier_fp32.onnx
│   ├── tokenizer.json
│   └── boundary_manifest.json
└── fp16_25/                       16 files, 1163 MB
    ├── *_fp16.onnx                keep_io_types — FP32 in/out
    ├── *_fp16_iobinding.onnx      FP16 throughout
    ├── tokenizer.json
    └── boundary_manifest.json
```

`fp16_25/` is not smaller than `fp32_25/` because it holds *two* variants at
574 MB each. Either folder is a complete export: `tokenizer.json` and
`boundary_manifest.json` are in both, so one can be downloaded without the
other.

The engine reads either shape. Point it at the parent and it picks the
subfolder matching the precision; point it at `fp16_25/` alone and it loads
that. A flat directory still works, and so does GLiNER2's `fp32_v2/`,
`fp16_v2/` naming — nothing needs renaming to be loadable.

#### Uploading to the Hub

```sh
pip install -U "huggingface_hub[cli]"
hf auth login                       # or HF_TOKEN=…

hf upload jugaadsrl/gliner2.5-multi-v1-onnx models/g25-fp16/fp16_25 fp16_25 \
    --repo-type model \
    --commit-message "Add FP16 and FP16-IOBinding variants"
```

Upload the folders separately rather than the parent: `fp32_25/` is already on
the Hub, and re-uploading a gigabyte that has not changed wastes everyone's
bandwidth. Add `--include "*.onnx" --include "*.data" --include "*.json"` if the
directory has picked up anything else.

The `.data` sidecars must go up with their `.onnx` — a fragment whose weights
sit outside the graph loads to a filesystem error naming a file the downloader
never asked for. `hf upload` on a directory takes them along; a hand-picked file
list is where they get missed.

Then say so in the model card, since `Precision::autodetect` will start choosing
FP16 on Linux and Windows the moment those files exist.

Accuracy against the FP32 reference over the 12-case suite, on `cuda:1`:

| | |
|---|---|
| spans | **41/41**, none lost, none added |
| largest score delta | 0.0017 |
| mean score delta | 0.0002 |

`_fp16` and `_fp16_iobinding` agreed with each other exactly.

**No speed claim, deliberately.** Repeated interleaved runs on this host put
FP16 anywhere between 9.6 and 16.5 ms while FP32 sat at 13–15 ms: the spread
*within* one precision covered the differences *between* them. The GPU was idle
at the time, so what moves the numbers is host-side contention at load average
16–17, and any ranking taken from that would be an artefact. What the script
buys you for certain is half the disk and half the device memory.

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
