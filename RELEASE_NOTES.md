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

