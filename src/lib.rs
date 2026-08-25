// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # gliner25
//!
//! Native Rust inference engine for **GLiNER2.5** models, running on ONNX
//! Runtime through `ort` 2.0.0-rc.13. Zero Python at inference time.
//!
//! GLiNER2.5 uses the `boundary` architecture (`BoundaryExtractor`), which
//! shares nothing with the `span` architecture of GLiNER2 beyond the encoder
//! and the prompt format. For `span` checkpoints use the separate
//! [`gliner2-rs`](https://github.com/dariofinardi/gliner2-rs) crate.
//!
//! ## How the boundary architecture works
//!
//! It does not enumerate spans. It proposes a pool of `(start, end)` candidates
//! **shared across every query**, of constant size (`pool_size`, typically
//! 192), then scores each query/candidate pair. Two auxiliary heads provide
//! per-query abstention (`null_logits`) and an expected mention count
//! (`count_log_rates`).
//!
//! Boundary spans are **half-open**, `[start, end)` — unlike the inclusive
//! spans of the span architecture. Mixing the two conventions is an easy
//! source of off-by-one errors.
//!
//! ```no_run
//! use gliner25::{BoundaryEngine, BoundaryConfig, SchemaTask};
//!
//! # fn main() -> anyhow::Result<()> {
//! gliner25::init("my-app");
//!
//! let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-multi-v1-onnx"))?;
//! let tasks = vec![SchemaTask::Entities(vec!["person".into(), "organization".into()])];
//!
//! for m in engine.extract("Mario Rossi works at Apple.", &tasks)?.mentions {
//!     println!("{} -> {} ({:.1}%)", m.text, m.field, m.score * 100.0);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Native runtime
//!
//! The crate enables `ort`'s `load-dynamic` feature, so the ONNX Runtime shared
//! library is resolved at run time via `ORT_DYLIB_PATH`. Alternatively, enable
//! `ort`'s `download-binaries` feature to fetch it at build time.

pub mod boundary;
pub mod error;
pub mod overlap;
pub mod processor;
pub mod runtime;

pub use boundary::{
    BoundaryConfig, BoundaryEngine, BoundaryManifest, BoundaryOutput, BoundaryParams,
    Classification, Mention, pair_relations,
};
pub use error::GlinerError;
pub use overlap::OverlapPolicy;
pub use processor::{SchemaTask, TaskType};
pub use runtime::Precision;

/// Initialises the ONNX Runtime environment. Call once per process; later
/// calls are ignored.
///
/// In `ort` rc.13 `commit()` returns `bool` rather than `Result`: `false` means
/// an environment already existed, not that anything failed.
pub fn init(name: &str) -> bool {
    ort::init().with_name(name.to_string()).commit()
}
