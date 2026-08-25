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

