// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Engine for the **boundary** architecture (GLiNER2.5).
//!
//! It shares nothing with the span architecture: no exhaustive span
//! enumeration, no `count_lstm`. The model proposes a pool of `(start, end)`
//! candidates **shared across all queries**, of constant size (`pool_size`,
//! typically 192), then assigns a logit to each query/candidate pair.
//!
//! ```text
//! encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
//!   +- routed_gather(lhs, word_idx,  word_mask)  -> text_states  [1,L,H]
//!   +- routed_gather(lhs, query_idx, ones)       -> query_states [1,Q,H]
//!   +- routed_gather(lhs, cls_idx,   ones)       -> cls_states   [1,K,H]
//!
//! boundary_head_L{bucket}(text_states, text_mask, query_states, query_mask)
//!   -> cand_indices    [1,Q,C,2]   HALF-OPEN (start, end) pairs
//!   -> pair_logits     [1,Q,C]
//!   -> cand_valid      [1,Q,C]
//!   -> null_logits     [1,Q]       per-query abstention
//!   -> count_log_rates [1,Q]       expected mention count per query
//!
//! classifier(cls_states) -> logits [K]
//! ```
//!
//! ## Length buckets
//!
//! The boundary heads have a **static** `num_words`: `torch.export` specialises
//! it because the candidate-pool builder contains a Python loop over a symbolic
//! dimension. The engine therefore picks the smallest bucket that fits the text
//! and pads with `text_mask = 0`.
//!
//! Masked padding is verified to be transparent: for the same real words,
//! padding to a larger bucket yields the same candidate set and probabilities
//! to within ~5e-06, even with random noise in the padded rows.
//!
//! This costs nothing on disk — a head is under 5 MB against 1.1 GB of encoder —
//! and is an advantage at run time: static shapes are what TensorRT, QNN and
//! IOBinding prefer.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::path::PathBuf;

use ort::session::Session;

use crate::error::GlinerError;
use crate::overlap::{OverlapPolicy, Spanned, resolve_overlaps};
use crate::processor::{SchemaTask, SchemaTransformer, TaskType};
use crate::runtime::{
    IoDType, Precision, build_session, float_tensor, i64_tensor, sigmoid, softmax, take_bool,
    take_float, take_i64,
};

/// `boundary_manifest.json`, written by `export_boundary_v1.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct BoundaryManifest {
    pub architecture: String,
    pub hidden_size: usize,
    pub pool_size: usize,
    pub length_buckets: Vec<usize>,
    pub min_bucket: usize,
    pub enable_abstention: bool,
    pub enable_count_head: bool,
    pub overlap_policy: String,
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub struct BoundaryParams {
    /// Threshold on each query/candidate pair probability.
    pub threshold: f32,
    /// Overlap policy; `None` uses the one recorded in the manifest.
    pub overlap_policy: Option<OverlapPolicy>,
    /// When `true` and the model exposes abstention, a query whose
    /// `null_logit` beats its best candidate logit yields no mention.
    pub use_abstention: bool,
    pub classification_temperature: f32,
    /// Overrides the per-task `multi_label` flag carried by
    /// [`SchemaTask::Classifications`]. Leave it `None` — the schema is the
    /// right place for that decision, since a single request routinely mixes
    /// single-label and multi-label tasks.
    pub multi_label_override: Option<bool>,
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            overlap_policy: None,
            use_abstention: true,
            classification_temperature: 1.0,
            multi_label_override: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mention {
    pub text: String,
    /// Schema group name (e.g. `entities`, or the relation name).
    pub task: String,
    /// Field that produced the query (the label, or the `head`/`tail` role).
    pub field: String,
    pub score: f32,
    /// Byte range `[start, end)` in the original text.
    pub char_start: usize,
    pub char_end: usize,
    /// Half-open word range `[start, end)`.
    pub word_start: usize,
    pub word_end: usize,
    /// Query index, useful when reassembling relations.
    pub query_id: usize,
}

impl Spanned for Mention {
    fn start(&self) -> usize {
        self.word_start
    }
    fn end(&self) -> usize {
        self.word_end
    }
    fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub task: String,
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct BoundaryOutput {
    pub mentions: Vec<Mention>,
    /// Every label of every classification task, with its probability. Use
    /// [`BoundaryOutput::verdict`] to turn one task into the answer gliner2
    /// would give.
    pub classifications: Vec<Classification>,
    /// Expected mention count per query, when the model exposes the count head.
    pub expected_counts: Vec<f32>,
}

impl BoundaryOutput {
    /// The labels gliner2 would report for a classification task.
    ///
    /// Reproduces `_extract_classification_result`, including the detail that
    /// is easy to miss: in multi-label mode, **when no label clears the
    /// threshold the top-scoring one is returned anyway**. The list is never
    /// empty. Thresholding the scores yourself and keeping the empty result
    /// will silently disagree with the reference implementation.
    ///
    /// In single-label mode the argmax is returned, which is the same thing.
    pub fn verdict(&self, task: &str, threshold: f32) -> Vec<&Classification> {
        let mut rows: Vec<&Classification> =
            self.classifications.iter().filter(|c| c.task == task).collect();
        if rows.is_empty() {
            return rows;
        }
        rows.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let over: Vec<&Classification> =
            rows.iter().copied().filter(|c| c.score >= threshold).collect();
        if over.is_empty() { vec![rows[0]] } else { over }
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    pub models_dir: PathBuf,
    pub precision: Precision,
    pub intra_threads: usize,
    /// Where to fetch the export from if `models_dir` does not hold one.
    #[cfg(feature = "hub")]
    pub hub: Option<crate::hub::Model>,
    /// Loads only the heads actually needed instead of every bucket, which
    /// shortens start-up when there are many buckets.
    pub lazy_heads: bool,
}

impl BoundaryConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        let precision = Precision::autodetect(&models_dir, "encoder");
        Self {
            models_dir,
            precision,
            intra_threads: 4,
            lazy_heads: true,
            #[cfg(feature = "hub")]
            hub: None,
        }
    }

    /// Fetches the export straight from the Hub, into the shared cache.
    ///
    /// Nothing is downloaded until [`BoundaryEngine::new`] runs, so
    /// `with_precision` still applies to what gets fetched.
    #[cfg(feature = "hub")]
    pub fn from_hub(model: crate::hub::Model) -> Self {
        Self::new(PathBuf::new()).or_download(model)
    }

    /// Names the repository to fall back to when `models_dir` holds no export.
    ///
    /// The local directory always wins: a checkout already on disk is used as
    /// it is, and the network is touched only when the export is missing.
    #[cfg(feature = "hub")]
    pub fn or_download(mut self, model: crate::hub::Model) -> Self {
        self.hub = Some(model);
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    pub fn eager_heads(mut self) -> Self {
        self.lazy_heads = false;
        self
    }
}

pub struct BoundaryEngine {
    encoder: Session,
    routed_gather: Session,
    classifier: Session,
    /// Heads by bucket, ascending. `None` until first needed.
    heads: Vec<(usize, Option<Session>)>,

    transformer: SchemaTransformer,
    manifest: BoundaryManifest,
    dtype: IoDType,
    dir: PathBuf,
    suffix: &'static str,
    intra_threads: usize,
    default_policy: OverlapPolicy,
}

impl BoundaryEngine {
    pub fn new(config: BoundaryConfig) -> Result<Self> {
        #[allow(unused_mut)]
        let mut config = config;

        // A directory that already holds the export is used untouched; only a
        // missing one reaches the network.
        #[cfg(feature = "hub")]
        if let Some(model) = config.hub {
            if !config.models_dir.join("boundary_manifest.json").exists() {
                config.models_dir = crate::hub::download(model, config.precision)?;
            }
        }

        let dir = config.models_dir.clone();
        let sfx = config.precision.suffix();

        let manifest_path = dir.join("boundary_manifest.json");
        if !manifest_path.exists() {
            return Err(GlinerError::IncompleteModelDir(format!(
                "boundary_manifest.json missing in {}; the directory does not hold \
                 a boundary export",
                dir.display()
            ))
            .into());
        }
        let manifest: BoundaryManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        if manifest.architecture != "boundary" {
            return Err(GlinerError::IncompleteModelDir(format!(
                "the manifest declares architecture '{}', expected 'boundary'",
                manifest.architecture
            ))
            .into());
        }

        let default_policy = OverlapPolicy::parse(&manifest.overlap_policy).ok_or_else(|| {
            anyhow!("overlap_policy '{}' not recognised", manifest.overlap_policy)
        })?;

        let tok_path = dir.join("tokenizer.json");
        let transformer = SchemaTransformer::from_tokenizer_file(&tok_path)?;

        let load = |stem: &str| -> Result<Session> {
            build_session(&dir.join(format!("{stem}{sfx}.onnx")), config.intra_threads)
        };

        let mut buckets: Vec<usize> = manifest.length_buckets.clone();
        buckets.sort_unstable();

        let mut heads: Vec<(usize, Option<Session>)> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let session = if config.lazy_heads {
                None
            } else {
                Some(build_session(
                    &dir.join(format!("boundary_head_L{b}{sfx}.onnx")),
                    config.intra_threads,
                )?)
            };
            heads.push((*b, session));
        }

        Ok(Self {
            encoder: load("encoder")?,
            routed_gather: load("routed_gather")?,
            classifier: load("classifier")?,
            heads,
            transformer,
            manifest,
            dtype: config.precision.io_dtype(),
            dir,
            suffix: sfx,
            intra_threads: config.intra_threads,
            default_policy,
        })
    }

    pub fn manifest(&self) -> &BoundaryManifest {
        &self.manifest
    }

    pub fn extract(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<BoundaryOutput> {
        self.extract_with(text, tasks, &BoundaryParams::default())
    }

    pub fn extract_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &BoundaryParams,
    ) -> Result<BoundaryOutput> {
        let record = self.transformer.transform(text, tasks)?;
        let num_words = record.num_words();
        if num_words == 0 {
            return Ok(BoundaryOutput::default());
        }

        let bucket = self.pick_bucket(num_words)?;
        let hidden_size = self.manifest.hidden_size;
        let seq = record.input_ids.len() as i64;

        // ── 1. encoder ────────────────────────────────────────────────────
        let hidden = {
            let ids = i64_tensor(vec![1, seq], record.input_ids.clone())?;
            let mask = i64_tensor(vec![1, seq], record.attention_mask.clone())?;
            let out = self.encoder.run(ort::inputs![ids, mask])?;
            take_float(&out["last_hidden_state"], self.dtype)?.1
        };

        // ── 2. routing: text padded to the bucket, queries, choices ───────
        let mut word_idx = record.word_first_positions();
        let mut word_mask = vec![1i64; num_words];
        word_idx.resize(bucket, 0);
        word_mask.resize(bucket, 0);

        let text_states = self.gather(&hidden, seq, hidden_size, &word_idx, &word_mask)?;

        let (query_idx, query_specs) = record.query_markers();
        let num_queries = query_idx.len();
        if num_queries == 0 {
            // nothing to extract: only classifications remain
            let mut out = BoundaryOutput::default();
            self.run_classifications(&record, &hidden, seq, hidden_size, params, &mut out)?;
            return Ok(out);
        }
        let query_mask = vec![1i64; num_queries];
        let query_states = self.gather(&hidden, seq, hidden_size, &query_idx, &query_mask)?;

        // ── 3. boundary head ──────────────────────────────────────────────
        // `head_for` takes `&mut self`, so fields needed inside the block must
        // be copied out first or they stay locked by the borrow.
        let dtype = self.dtype;
        let head = self.head_for(bucket)?;
        let (cand_indices, pair_logits, cand_valid, null_logits, count_log_rates) = {
            let ts = float_tensor(
                dtype,
                vec![1, bucket as i64, hidden_size as i64],
                text_states,
            )?;
            let tm = i64_tensor(vec![1, bucket as i64], word_mask.clone())?;
            let qs = float_tensor(
                dtype,
                vec![1, num_queries as i64, hidden_size as i64],
                query_states,
            )?;
            let qm = i64_tensor(vec![1, num_queries as i64], query_mask)?;
            let out = head.run(ort::inputs![ts, tm, qs, qm])?;
            (
                take_i64(&out["cand_indices"])?.1,
                take_float(&out["pair_logits"], dtype)?.1,
                take_bool(&out["cand_valid"])?.1,
                take_float(&out["null_logits"], dtype)?.1,
                take_float(&out["count_log_rates"], dtype)?.1,
            )
        };

        // ── 4. decoding ───────────────────────────────────────────────────
        let policy = params.overlap_policy.unwrap_or(self.default_policy);
        let c = self.manifest.pool_size;
        let mut output = BoundaryOutput {
            expected_counts: count_log_rates.iter().map(|r| r.exp()).collect(),
            ..Default::default()
        };

        for (q, &(group, role)) in query_specs.iter().enumerate() {
            let task = &record.tasks[group];

            // abstention: if the null logit beats the best one, the query stays silent
            if params.use_abstention && self.manifest.enable_abstention {
                let best = (0..c)
                    .filter(|&i| cand_valid[q * c + i])
                    .map(|i| pair_logits[q * c + i])
                    .fold(f32::NEG_INFINITY, f32::max);
                if null_logits[q] > best {
                    continue;
                }
            }

            let mut candidates: Vec<Mention> = Vec::new();
            for i in 0..c {
                if !cand_valid[q * c + i] {
                    continue;
                }
                let score = sigmoid(pair_logits[q * c + i]);
                if score < params.threshold {
                    continue;
                }
                // indices are half-open: [start, end)
                let start = cand_indices[(q * c + i) * 2] as usize;
                let end = cand_indices[(q * c + i) * 2 + 1] as usize;
                if end <= start || end > num_words {
                    // drop candidates falling inside the padding
                    continue;
                }
                let (cs, _) = record.word_to_char_maps[start];
                let (_, ce) = record.word_to_char_maps[end - 1];
                candidates.push(Mention {
                    text: text[cs..ce].to_string(),
                    task: task.task_name.clone(),
                    field: task.labels[role].clone(),
                    score,
                    char_start: cs,
                    char_end: ce,
                    word_start: start,
                    word_end: end,
                    query_id: q,
                });
            }

            output.mentions.extend(resolve_overlaps(&candidates, policy));
        }

        // stable global ranking: descending confidence, then start, end, field
        output.mentions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.word_start.cmp(&b.word_start))
                .then(a.word_end.cmp(&b.word_end))
                .then(a.field.cmp(&b.field))
        });

        self.run_classifications(&record, &hidden, seq, hidden_size, params, &mut output)?;
        Ok(output)
    }

    fn run_classifications(
        &mut self,
        record: &crate::processor::ProcessedRecord,
        hidden: &[f32],
        seq: i64,
        hidden_size: usize,
        params: &BoundaryParams,
        output: &mut BoundaryOutput,
    ) -> Result<()> {
        let (cls_idx, cls_specs) = record.cls_markers();
        if cls_idx.is_empty() {
            return Ok(());
        }
        let mask = vec![1i64; cls_idx.len()];
        let cls_states = self.gather(hidden, seq, hidden_size, &cls_idx, &mask)?;

        let logits = {
            let cs = float_tensor(
                self.dtype,
                vec![cls_idx.len() as i64, hidden_size as i64],
                cls_states,
            )?;
            let out = self.classifier.run(ort::inputs![cs])?;
            take_float(&out["logits"], self.dtype)?.1
        };

        // logits must be normalised per group, not across every choice at once
        let mut by_group: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
            Default::default();
        for (flat, &(group, choice)) in cls_specs.iter().enumerate() {
            by_group.entry(group).or_default().push((flat, choice));
        }

        for (group, entries) in by_group {
            let task = &record.tasks[group];
            let scaled: Vec<f32> = entries
                .iter()
                .map(|&(flat, _)| logits[flat] / params.classification_temperature)
                .collect();
            let multi_label = params.multi_label_override.unwrap_or(task.multi_label);
            let probs = if multi_label {
                scaled.iter().copied().map(sigmoid).collect::<Vec<_>>()
            } else {
                softmax(&scaled)
            };
            for (&(_, choice), score) in entries.iter().zip(probs) {
                output.classifications.push(Classification {
                    task: task.task_name.clone(),
                    label: task.labels[choice].clone(),
                    score,
                });
            }
        }
        Ok(())
    }

    fn gather(
        &mut self,
        hidden: &[f32],
        seq: i64,
        hidden_size: usize,
        indices: &[i64],
        mask: &[i64],
    ) -> Result<Vec<f32>> {
        let n = indices.len() as i64;
        let h = float_tensor(
            self.dtype,
            vec![1, seq, hidden_size as i64],
            hidden.to_vec(),
        )?;
        let idx = i64_tensor(vec![1, n], indices.to_vec())?;
        let msk = i64_tensor(vec![1, n], mask.to_vec())?;
        let out = self.routed_gather.run(ort::inputs![h, idx, msk])?;
        Ok(take_float(&out["states"], self.dtype)?.1)
    }

    /// Smallest bucket that fits `num_words`.
    fn pick_bucket(&self, num_words: usize) -> Result<usize> {
        let max_bucket = self.heads.last().map(|(b, _)| *b).unwrap_or(0);
        self.heads
            .iter()
            .map(|(b, _)| *b)
            .find(|&b| b >= num_words)
            .ok_or_else(|| GlinerError::NoLengthBucket { words: num_words, max_bucket }.into())
    }

    /// Returns the bucket's head, loading it on demand when `lazy_heads`.
    fn head_for(&mut self, bucket: usize) -> Result<&mut Session> {
        let slot = self
            .heads
            .iter()
            .position(|(b, _)| *b == bucket)
            .ok_or_else(|| anyhow!("bucket {bucket} not present"))?;
        if self.heads[slot].1.is_none() {
            let path = self
                .dir
                .join(format!("boundary_head_L{bucket}{}.onnx", self.suffix));
            self.heads[slot].1 = Some(build_session(&path, self.intra_threads)?);
        }
        Ok(self.heads[slot].1.as_mut().unwrap())
    }
}

/// Relations are reassembled by grouping mentions per schema group: the first
/// role is the head, the second the tail.
pub fn pair_relations(mentions: &[Mention], tasks: &[SchemaTask]) -> Vec<(Mention, Mention, String)> {
    let mut pairs = Vec::new();
    for task in tasks {
        let SchemaTask::Relations(name, roles) = task else { continue };
        if roles.len() < 2 {
            continue;
        }
        let heads: Vec<&Mention> = mentions
            .iter()
            .filter(|m| m.task == *name && m.field == roles[0])
            .collect();
        let tails: Vec<&Mention> = mentions
            .iter()
            .filter(|m| m.task == *name && m.field == roles[1])
            .collect();
        for h in &heads {
            for t in &tails {
                if h.word_start == t.word_start && h.word_end == t.word_end {
                    continue;
                }
                pairs.push(((*h).clone(), (*t).clone(), name.clone()));
            }
        }
    }
    pairs
}

#[allow(unused)]
fn _assert_task_type_used(t: TaskType) -> bool {
    matches!(t, TaskType::Relations)
}
