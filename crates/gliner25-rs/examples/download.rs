//! Loading a model without having downloaded it first.
//!
//! ```sh
//! ORT_DYLIB_PATH=/path/libonnxruntime.so \
//! cargo run --release --example download -p gliner25-rs -- models/gliner2.5-onnx
//! ```
//!
//! Pass the directory you want the export in. If it already holds one it is
//! used untouched and nothing is fetched; if it does not, the export is pulled
//! from the Hub before the engine starts.

use gliner25_rs::{BoundaryConfig, BoundaryEngine, SchemaTask, hub};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models/gliner2.5-onnx".to_string());
    gliner25_rs::init("gliner25-download-example");

    let present = std::path::Path::new(&dir).join("boundary_manifest.json").exists();
    println!("{dir}: {}", if present { "present, no fetch" } else { "absent, fetching" });

    let t0 = Instant::now();
    let mut engine =
        BoundaryEngine::new(BoundaryConfig::new(&dir).or_download(hub::GLINER25_MULTI_V1))?;
    println!("ready in {:.1}s", t0.elapsed().as_secs_f32());

    let text = "Giuseppe Verdi, g.verdi@example.it, Milano.";
    let tasks = vec![SchemaTask::Entities(vec![
        "person".into(),
        "email".into(),
        "location".into(),
    ])];
    let out = engine.extract(text, &tasks)?;
    for m in &out.mentions {
        println!("  {:<20} {:<12} {:>6.2}%", m.text, m.field, m.score * 100.0);
    }
    Ok(())
}
