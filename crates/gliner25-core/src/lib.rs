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
//! Schema families and merging live in the [`gliner25`] extension crate.

pub mod boundary;

pub use boundary::{
    BoundaryConfig, BoundaryEngine, BoundaryManifest, BoundaryOutput, BoundaryParams,
    Classification, Mention, pair_relations,
};
pub use gliner_core::{GlinerError, OverlapPolicy, SchemaTask, TaskType, init};
