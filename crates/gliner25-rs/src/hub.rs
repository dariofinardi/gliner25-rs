//! Fetching a published export from the Hub when it is not on disk.
//!
//! The engine loads ONNX fragments from a directory. If that directory is not
//! there — first run, fresh container, a machine where nobody has fetched the
//! weights yet — [`BoundaryConfig::or_download`](crate::BoundaryConfig::or_download)
//! names the repository to pull it from, and the fetch happens inside
//! [`BoundaryEngine::new`](crate::BoundaryEngine::new) rather than as a separate
//! step the caller has to remember.
//!
//! ```no_run
//! use gliner25_rs::{BoundaryConfig, BoundaryEngine, hub};
//!
//! // Uses ./models/g25 if it holds an export, downloads it if it does not.
//! let cfg = BoundaryConfig::new("models/g25").or_download(hub::GLINER25_MULTI_V1);
//! let mut engine = BoundaryEngine::new(cfg)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! Files land in the Hub cache (`HF_HOME`, else `~/.cache/huggingface`), shared
//! with every other tool on the machine, so a model already fetched by the
//! Python library is not fetched again.
//!
//! ## Length buckets are read before the heads are fetched
//!
//! A boundary export ships one head per length bucket, and which buckets exist
//! is a property of the export rather than of this crate. `boundary_manifest.json`
//! is therefore downloaded first and parsed, and only the heads it declares are
//! requested — fetching `boundary_head_L*` by guesswork would either miss a
//! bucket or ask for files that were never exported.
//!
//! ## Transport
//!
//! `hf-hub` is pulled with `default-features = false, features = ["ureq"]`,
//! which resolves TLS through `rustls` rather than `native-tls`. There is no
//! `openssl` in the tree, and so no OpenSSL C library to keep patched.
//!
//! Turn the feature off (`default-features = false`) and the crate goes back to
//! having no network stack whatsoever.

use crate::boundary::BoundaryManifest;
use crate::error::GlinerError;
use crate::runtime::Precision;
use anyhow::{Context, Result};
use std::path::PathBuf;

/// A published ONNX export.
///
/// The constant below names the export Jugaad publishes. Any other repository
/// works too — build a `Model` with [`Model::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub repo_id: &'static str,
}

impl Model {
    pub const fn new(repo_id: &'static str) -> Self {
        Self { repo_id }
    }
}

/// `fastino/gliner2.5-multi-v1`, the boundary checkpoint.
pub const GLINER25_MULTI_V1: Model = Model::new("jugaadsrl/gliner2.5-multi-v1-onnx");

/// Fragments every boundary export carries, whatever its buckets.
const FRAGMENTS: [&str; 3] = ["encoder", "routed_gather", "classifier"];

/// Downloads `model` at `precision` and returns the directory to load from.
pub fn download(model: Model, precision: Precision) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_user_agent(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        .build()
        .map_err(|e| GlinerError::Hub(format!("could not initialise the Hub client: {e}")))?;
    let repo = api.model(model.repo_id.to_string());

    let fetch = |file: &str| -> Result<PathBuf> {
        repo.get(file).map_err(|e| {
            GlinerError::Hub(format!("{}: could not fetch {file} ({e})", model.repo_id)).into()
        })
    };
    // A fragment past the 2 GB protobuf limit keeps its weights in a sidecar
    // `.onnx.data`, which ONNX Runtime opens by relative name at session build
    // time. Most fragments have none, so a miss is not an error - but a fragment
    // that has one and does not get it fails at load, not at download, with a
    // filesystem error naming a file nobody asked for.
    let fetch_onnx = |file: &str| -> Result<PathBuf> {
        let path = fetch(file)?;
        let _ = repo.get(&format!("{file}.data"));
        Ok(path)
    };

    // The manifest first: it is what says which heads exist.
    let manifest_path = fetch("boundary_manifest.json")?;
    let manifest: BoundaryManifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .with_context(|| {
            format!("{}: boundary_manifest.json is not a boundary manifest", model.repo_id)
        })?;

    let sfx = precision.suffix();
    for stem in FRAGMENTS {
        fetch_onnx(&format!("{stem}{sfx}.onnx"))?;
    }
    for bucket in &manifest.length_buckets {
        fetch_onnx(&format!("boundary_head_L{bucket}{sfx}.onnx"))?;
    }
    fetch("tokenizer.json")?;

    Ok(manifest_path
        .parent()
        .context("downloaded manifest has no parent directory")?
        .to_path_buf())
}
