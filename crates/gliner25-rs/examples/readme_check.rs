//! Compiles every code sample in the crate README, so a signature that drifts
//! breaks the build instead of misleading a reader.
#![allow(unused_variables, dead_code, unreachable_code)]

use gliner25_rs::chain::ExecutionMode;
use gliner25_rs::chunker::Chunker;
use gliner25_rs::families::{Family, chunk_into_families, run_families};
use gliner25_rs::{BoundaryConfig, BoundaryEngine, BoundaryParams, SchemaTask, hub};

fn main() -> anyhow::Result<()> {
    return Ok(()); // compile-only

    gliner25_rs::init("my-app");
    let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/gliner2.5-onnx"))?;

    let tasks = vec![SchemaTask::Entities(vec!["person".into(), "location".into()])];
    let out = engine.extract("Mario Rossi lavora a Milano.", &tasks)?;
    for m in &out.mentions {
        println!("{:?} {} {:.1}%", m.text, m.field, m.score * 100.0);
    }

    let m = engine.manifest();
    println!("{:?} buckets, pool {}", m.length_buckets, m.pool_size);

    let text = "…";
    let families = vec![
        Family::new("people", ["person", "organization"]),
        Family::new("places", ["location", "address"]),
        Family::new("contact", ["email", "phone number"]),
    ];
    let out = run_families(&mut engine, text, &families, &BoundaryParams::default())?;
    let labels: Vec<String> = vec!["a".into(), "b".into()];
    let _ = chunk_into_families(&labels, 8);

    let document = String::new();
    let out = engine.extract_long(&document, &tasks)?;
    let params = BoundaryParams::default();
    let out = engine.extract_long_with(&document, &tasks, &params, Chunker::new(256, 48)?)?;

    let cfg = BoundaryConfig::new("models/g25").with_execution(ExecutionMode::IoBinding);
    let _ = engine.execution();

    let cfg = BoundaryConfig::new("models/g25").or_download(hub::GLINER25_MULTI_V1);
    let _ = BoundaryConfig::from_hub(hub::GLINER25_MULTI_V1);
    let _ = hub::Model::new("acme/my-export");
    let _ = out.verdict("tone", 0.5);
    Ok(())
}
