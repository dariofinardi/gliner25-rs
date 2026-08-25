//! Entity extraction with a GLiNER2.5 boundary model.
//!
//! ```sh
//! ORT_DYLIB_PATH=/path/libonnxruntime.so \
//! GLINER2_PRECISION=fp32 \
//! cargo run --release --example extract -- models/gliner2.5-multi-v1-onnx
//! ```

use gliner25_core::{BoundaryConfig, BoundaryEngine, SchemaTask};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gliner2.5-multi-v1-onnx".to_string());

    gliner25_core::init("gliner25-example");

    let t0 = Instant::now();
    let mut engine = BoundaryEngine::new(BoundaryConfig::new(&dir))?;
    let m = engine.manifest();
    println!(
        "loaded in {:.2}s — pool_size {}, buckets {:?}, overlap {}",
        t0.elapsed().as_secs_f32(),
        m.pool_size,
        m.length_buckets,
        m.overlap_policy
    );

    let text = "Mario Rossi works at Apple in Cupertino, his email is mario.rossi@example.com.";
    let tasks = vec![SchemaTask::Entities(vec![
        "person".into(),
        "organization".into(),
        "location".into(),
        "email".into(),
    ])];

    let t1 = Instant::now();
    let out = engine.extract(text, &tasks)?;
    println!(
        "\n{}\n{} mentions in {:.1} ms\n",
        text,
        out.mentions.len(),
        t1.elapsed().as_secs_f32() * 1000.0
    );

    for m in &out.mentions {
        println!(
            "  {:<28} {:<14} {:>6.2}%   words [{}..{})  bytes [{}..{})",
            format!("{:?}", m.text),
            m.field,
            m.score * 100.0,
            m.word_start,
            m.word_end,
            m.char_start,
            m.char_end
        );
    }

    for c in &out.classifications {
        println!("  [{}] {} {:.2}%", c.task, c.label, c.score * 100.0);
    }
    Ok(())
}
