## [v0.5.5] - 2026-08-26
### 📦 The published export now ships all three precision variants
- The FP16 and FP16-IOBinding variants produced by `downcast_fp16.py` (41/41
  spans against FP32, largest score delta 0.0017) are on
  `jugaadsrl/gliner2.5-multi-v1-onnx` as of 2026-08-26, flat beside the FP32
  files. A bound engine now fetches **590 MB instead of 1.1 GB** — verified
  against a cold cache — and `Precision::autodetect` picks `_fp16_iobinding` on
  Linux and Windows. Every "ships FP32 only" claim in the docs is updated;
  the fallback chain stays, for repositories that publish less.

## [v0.5.4] - 2026-08-26
### 🛡️ Architecture gating
- **A GLiNER2 span export is refused by name.** The manifest check already
  stopped it, but the message said "boundary_manifest.json not found" — a
  missing-file report for what is really a wrong-crate situation. If `span_rep`
  is present the error now says so: "this is a GLiNER2 span export — use the
  gliner2-rs crate". Same probe on the Hub path, before anything downloads.

## [v0.5.3] - 2026-08-26
### 🐛 Fixes
- **`OverlapPolicy::parse` accepts `all`**, matching Python's alias table — a
  manifest using that spelling failed to load.
- The score-quantisation note in the resolver, as in gliner2-rs 0.9.3.

The per-query scoping this crate already had matches Python's boundary engine
exactly (`_resolve_spans` per `query_id`), confirmed in the same comparison —
no change needed there.

## [v0.5.2] - 2026-08-26
### 🐛 Fixes — from the re-audit of gliner2-rs, applied symmetrically
- **Chunk merging now removes seam artefacts.** Within each `(task, field)`,
  overlapping mentions from neighbouring windows are resolved greedily by
  score, half-open semantics — adjacent spans touch without clashing. Fields
  still never interact.
- **A standard-path OOM is `E_GLI_002` now**, not `E_GLI_001` with a
  contradictory message.
- `Chunker` and `ExecutionMode` re-exported at the crate root.

## [v0.5.1] - 2026-08-26
### 📚 Diagnostics
- `E_GLI_007 NO_LENGTH_BUCKET` said "split the text" without saying how. It now
  names `extract_long()`, which windows the text and merges the results, and
  keeps the re-export hint as the second option.

## [v0.5.0] - 2026-08-26
### ✨ Features
- **Long documents.** The export's largest length bucket is a hard ceiling —
  512 words — and `extract` returns `E_GLI_007` past it rather than truncating.
  `extract_long` splits into overlapping word windows, remaps the offsets and
  merges, mirroring `gliner2.inference.chunking` with the same 384/64 defaults.
  On a 1 208-word document `extract` fails and `extract_long` returns 51
  mentions in 716 ms.
- **Only the variant you will run is downloaded.** The execution mode picks the
  precision when the model has to be fetched. A repository that does not publish
  the preferred variant is handled by falling back rather than failing, which is
  the case today: `jugaadsrl/gliner2.5-multi-v1-onnx` ships FP32 only.

### 🐛 Fixes
- **The device-OOM fallback never fired.** `Chain::fall_back` was called by
  nobody, and the classifier meant to reach it matched "out of memory" while ORT
  writes "Failed to allocate memory for requested buffer of size N".

### 📚 Documentation
- The crate README now covers the whole surface — every public type, the
  manifest, schema families, the chunker, execution modes and the Hub — with a
  `readme_check` example that compiles every sample in it.

