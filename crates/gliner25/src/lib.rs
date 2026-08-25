// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Schema families and relation reassembly for GLiNER2.5.
//!
//! Thin layer over [`gliner25_core`]: the engine is there, this crate carries
//! the schema hygiene that a boundary model needs in practice.
//!
//! ## Why families
//!
//! Labels within one schema compete. The queries share the encoder context, so
//! a wide schema makes them interfere: on `gliner2.5-multi-v1` this shows up as
//! date-like entities being lost when many unrelated labels are present, a
//! regression against the span models that only appears at width.
//!
//! The remedy is to split the schema into families of related labels, run each
//! separately and merge. [`Family`] holds a group, [`run_families`] does the
//! passes and merges the results.

use std::collections::HashMap;

use gliner25_core::{BoundaryEngine, BoundaryOutput, BoundaryParams, Mention, SchemaTask};

/// A named group of related labels, run as one pass.
#[derive(Debug, Clone)]
pub struct Family {
    pub name: String,
    pub labels: Vec<String>,
}

impl Family {
    pub fn new(name: impl Into<String>, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            labels: labels.into_iter().map(Into::into).collect(),
        }
    }

    pub fn task(&self) -> SchemaTask {
        SchemaTask::Entities(self.labels.clone())
    }
}

/// Runs one pass per family and merges the mentions.
///
/// Merging is by `(word_start, word_end, field)`: a label belongs to exactly one
/// family, so the only way the same triple appears twice is a genuine duplicate,
/// and the higher-scoring one wins. Mentions from different families are kept
/// side by side even when they cover the same text — labels are independent, and
/// two families disagreeing about a stretch is information, not a conflict.
///
/// Cost is one encoder pass per family. That is the price of not letting the
/// labels compete; measure before assuming it is too much.
pub fn run_families(
    engine: &mut BoundaryEngine,
    text: &str,
    families: &[Family],
    params: &BoundaryParams,
) -> anyhow::Result<BoundaryOutput> {
    let mut merged: HashMap<(usize, usize, String), Mention> = HashMap::new();
    let mut classifications = Vec::new();
    let mut expected_counts = Vec::new();

    for family in families {
        let out = engine.extract_with(text, &[family.task()], params)?;
        for mention in out.mentions {
            let key = (mention.word_start, mention.word_end, mention.field.clone());
            merged
                .entry(key)
                .and_modify(|kept| {
                    if mention.score > kept.score {
                        *kept = mention.clone();
                    }
                })
                .or_insert(mention);
        }
        classifications.extend(out.classifications);
        expected_counts.extend(out.expected_counts);
    }

    let mut mentions: Vec<Mention> = merged.into_values().collect();
    mentions.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
            .then(a.field.cmp(&b.field))
    });

    Ok(BoundaryOutput { mentions, classifications, expected_counts })
}

/// Splits a flat label list into families of at most `max_per_family`.
///
/// A fallback for callers who have no semantic grouping to offer. Real families
/// — dates together, identifiers together — work better than arbitrary chunks,
/// because the interference is between *unrelated* labels.
pub fn chunk_into_families(labels: &[String], max_per_family: usize) -> Vec<Family> {
    assert!(max_per_family > 0, "max_per_family must be positive");
    labels
        .chunks(max_per_family)
        .enumerate()
        .map(|(i, chunk)| Family::new(format!("family_{i}"), chunk.to_vec()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_becomes_an_entity_task() {
        let f = Family::new("dates", ["date", "date_of_birth"]);
        match f.task() {
            SchemaTask::Entities(labels) => assert_eq!(labels, vec!["date", "date_of_birth"]),
            other => panic!("expected entities, got {other:?}"),
        }
    }

    #[test]
    fn chunking_covers_every_label_exactly_once() {
        let labels: Vec<String> = (0..7).map(|i| format!("l{i}")).collect();
        let families = chunk_into_families(&labels, 3);
        assert_eq!(families.len(), 3);
        let flat: Vec<String> = families.iter().flat_map(|f| f.labels.clone()).collect();
        assert_eq!(flat, labels);
    }
}
