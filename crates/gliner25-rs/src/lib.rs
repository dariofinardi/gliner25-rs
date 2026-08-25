// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! GLiNER2.5 **boundary**-architecture inference on ONNX Runtime.
//!
//! The boundary architecture enumerates nothing. It proposes a pool of
//! `(start, end)` candidates **shared across all queries**, of constant size
//! (`pool_size`, typically 192), and scores each query/candidate pair. Two
//! auxiliary heads give per-query abstention and an expected mention count.
//!
//! Its spans are **half-open** `[start, end)`, unlike the inclusive ranges of
//! the span architecture in
//! [gliner2-rs](https://github.com/dariofinardi/gliner2-rs). Mixing the two
//! conventions is a silent off-by-one.
//!
//! Schema families live in [`families`], behind a default-on feature.

//! ## Layout
//!
//! | module | what it holds |
//! |---|---|
//! | [`processor`] | prompt construction and tokenization |
//! | [`runtime`] | `ort` helpers, precision selection, export-layout resolution |
//! | [`overlap`] | span overlap policies |
//! | [`boundary`] | the inference engine |
//! | [`error`] | diagnosable engine errors |
//!
//! The first four are architecture-agnostic — identical to what the span engine
//! in [gliner2-rs](https://github.com/dariofinardi/gliner2-rs) needs, because
//! gliner2 itself shares them. They live here rather than in a crate of their
//! own: one shared crate would have to be published under a single name and
//! would tie the two repositories' release cadences together, for four modules
//! that change rarely. The cost is that a fix to them lands in two places; the
//! alternative was worse.
//!
//! Note that [`overlap`] works on half-open intervals, which is what this
//! architecture produces natively — the span engine converts its inclusive
//! ranges on the way in.

pub mod boundary;
pub mod error;
/// Fetching a published export from the Hub when it is not on disk.
#[cfg(feature = "hub")]
pub mod hub;
pub mod overlap;
pub mod processor;
pub mod runtime;

/// Schema families: splitting a wide schema into groups of related labels and
/// merging the results, the documented remedy for labels interfering with each
/// other when many are passed at once.
#[cfg(feature = "families")]
pub mod families;

pub use boundary::{
    BoundaryConfig, BoundaryEngine, BoundaryManifest, BoundaryOutput, BoundaryParams,
    Classification, Mention, pair_relations,
};
pub use error::GlinerError;
pub use overlap::{OverlapPolicy, Spanned, resolve_overlaps};
pub use processor::{ProcessedRecord, SchemaTask, SchemaTransformer, TaskMapping, TaskType};
pub use runtime::Precision;

/// Initialises the ONNX Runtime environment. Call once per process; later calls
/// are ignored.
///
/// In `ort` rc.13 `commit()` returns `bool` rather than `Result`: `false` means
/// an environment already existed, not that anything failed.
pub fn init(name: &str) -> bool {
    ort::init().with_name(name.to_string()).commit()
}

#[cfg(feature = "test-support")]
pub mod test_support {
    use std::path::PathBuf;

    /// Locates a gliner2 `tokenizer.json` for tests.
    ///
    /// Model directories are gitignored, so a test that cannot find one should
    /// skip rather than fail. All GLiNER2 checkpoints share the mDeBERTa-v3
    /// vocabulary plus the same added markers, so any of them pins the prompt
    /// layout equally well.
    pub fn find_tokenizer() -> Option<PathBuf> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .to_path_buf();
        [
            "models/tokenizer.json",
            "models/pii-onnx/tokenizer.json",
            "models/pii-legacy/fp16_v2/tokenizer.json",
            "models/guardrails-pii-multi-onnx/tokenizer.json",
            "models/gliner2.5-multi-v1-onnx/tokenizer.json",
        ]
        .iter()
        .map(|p| root.join(p))
        .find(|p| p.exists())
    }
}

