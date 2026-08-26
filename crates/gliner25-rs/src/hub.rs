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

/// How an export arranges its files in the repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Everything at the repository root, precision in the file name.
    Flat,
    /// One folder per precision — `fp32_25/`, `fp16_25/` — each self-contained
    /// with its own `tokenizer.json` and `boundary_manifest.json`, so fetching
    /// one variant downloads nothing of the others. The published export is
    /// laid out this way.
    Grouped,
}

/// A published ONNX export.
///
/// The constant below names the export Jugaad publishes. Any other repository
/// works too — [`Model::new`] for a flat one, [`Model::grouped`] for one
/// organised into precision folders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub repo_id: &'static str,
    pub layout: Layout,
}

impl Model {
    pub const fn new(repo_id: &'static str) -> Self {
        Self { repo_id, layout: Layout::Flat }
    }

    pub const fn grouped(repo_id: &'static str) -> Self {
        Self { repo_id, layout: Layout::Grouped }
    }
}

/// `fastino/gliner2.5-multi-v1`, the boundary checkpoint.
pub const GLINER25_MULTI_V1: Model = Model::grouped("jugaadsrl/gliner2.5-multi-v1-onnx");

/// Fragments every boundary export carries, whatever its buckets.
const FRAGMENTS: [&str; 3] = ["encoder", "routed_gather", "classifier"];

/// Downloads `model` and returns the directory to load from, with the variant
/// actually obtained.
///
/// Only one variant is fetched — the one asked for, or the first of its
/// fallbacks the repository publishes. An export carries up to three copies of
/// every fragment and the encoder alone is half a gigabyte, so fetching all of
/// them to use one is most of a download wasted.
pub fn download(model: Model, precision: Precision) -> Result<(PathBuf, Precision)> {
    let mut last_err = None;
    for candidate in precision.fallback_chain() {
        match download_exact(model, *candidate) {
            Ok(dir) => {
                if *candidate != precision {
                    eprintln!(
                        "[gliner25] {} does not publish the {} variant; using {} instead",
                        model.repo_id,
                        precision.suffix().trim_start_matches('_'),
                        candidate.suffix().trim_start_matches('_'),
                    );
                }
                return Ok((dir, *candidate));
            }
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no precision variant could be fetched")))
}

/// Fetches exactly one variant, failing if the repository does not carry it.
fn download_exact(model: Model, precision: Precision) -> Result<PathBuf> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_user_agent(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"))
        .build()
        .map_err(|e| GlinerError::Hub(format!("could not initialise the Hub client: {e}")))?;
    let repo = api.model(model.repo_id.to_string());

    // The folder this variant lives in, per the declared layout. Grouped
    // folders are self-contained, so every file — manifest and tokenizer
    // included — comes from the same place, and nothing of the other variants
    // is touched.
    let prefix = match model.layout {
        Layout::Flat => String::new(),
        Layout::Grouped => format!("{}/", precision.legacy_subdir()),
    };
    let fetch = |file: &str| -> Result<PathBuf> {
        let path = format!("{prefix}{file}");
        repo.get(&path).map_err(|e| {
            GlinerError::Hub(format!("{}: could not fetch {path} ({e})", model.repo_id)).into()
        })
    };
    // A fragment past the 2 GB protobuf limit keeps its weights in a sidecar
    // `.onnx.data`, which ONNX Runtime opens by relative name at session build
    // time. Most fragments have none, so a miss is not an error - but a fragment
    // that has one and does not get it fails at load, not at download, with a
    // filesystem error naming a file nobody asked for.
    let fetch_onnx = |file: &str| -> Result<PathBuf> {
        let path = fetch(file)?;
        // Through `fetch`'s prefix, not `repo.get` directly: the sidecar sits
        // in the same folder as its graph, and asking the root for it finds
        // nothing in a grouped layout — the exact bug that made the FP32 heads
        // download cleanly and then fail at session build.
        let _ = fetch(&format!("{file}.data"));
        Ok(path)
    };

    // The manifest first: it is what says which heads exist — and it is the
    // architecture's signature, so its absence with span_rep present means the
    // caller pointed this crate at a GLiNER2 span export.
    let manifest_path = match fetch("boundary_manifest.json") {
        Ok(p) => p,
        Err(e) => {
            let is_span = repo.get("span_rep_fp32.onnx").is_ok()
                || repo.get("fp32_v2/span_rep_fp32.onnx").is_ok();
            if is_span {
                return Err(GlinerError::Hub(format!(
                    "{} is a GLiNER2 **span** export (it publishes span_rep and \
                     no boundary_manifest.json). This crate runs the boundary \
                     architecture only — use gliner2-rs for this model.",
                    model.repo_id
                ))
                .into());
            }
            return Err(e);
        }
    };
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

    // Hand back the snapshot ROOT for a grouped layout: `resolve_fragment`
    // and `resolve_aux` look in the root and then in the precision subfolders,
    // so the root serves both layouts — and lets an engine later ask for a
    // different variant of the same snapshot.
    let leaf = manifest_path
        .parent()
        .context("downloaded manifest has no parent directory")?
        .to_path_buf();
    Ok(match model.layout {
        Layout::Flat => leaf,
        Layout::Grouped => leaf
            .parent()
            .context("grouped snapshot has no root")?
            .to_path_buf(),
    })
}
