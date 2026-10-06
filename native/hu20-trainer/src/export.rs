//! Inference exports from a `jsonl-v2` training checkpoint (native or Python):
//! the current policy in `src.blueprint.artifact.export_policy` format and the
//! stored average in `src.diagnostics.cfr_average.extract` format. Probabilities
//! use the same `fsum` normalization, so they equal Python's bit for bit.

use crate::cfr::{fsum, regret_match};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn sha256_file(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer).unwrap();
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn write_gz(path: &Path, write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>) -> std::io::Result<()> {
    let temporary = path.with_extension("tmp");
    {
        let file = std::fs::File::create(&temporary)?;
        let mut out = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());
        write(&mut out)?;
        out.finish()?.flush()?;
    }
    std::fs::rename(temporary, path)
}

/// How a key whose stored average has no mass plays, as `cfr_average.ZERO_MASS_RULES` names it.
/// `current` plays its regret-matched policy: traverser-reach averaging leaves many trained keys
/// without mass (subtrees reached only through zero-probability own actions), and uniform play
/// discards their regrets.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ZeroMass {
    Uniform,
    Current,
}

impl ZeroMass {
    pub fn parse(name: &str) -> ZeroMass {
        match name {
            "uniform" => ZeroMass::Uniform,
            "current" => ZeroMass::Current,
            other => panic!("unknown zero-mass rule {other}"),
        }
    }
    fn label(self) -> &'static str {
        match self {
            ZeroMass::Uniform => "uniform in retained menu; reported separately from missing keys",
            ZeroMass::Current => "current regret-matched policy in retained menu; reported separately from missing keys",
        }
    }
}

pub fn export(checkpoint: &Path, current: Option<&Path>, average: Option<&Path>, zero_mass: ZeroMass) -> std::io::Result<u64> {
    let source = BufReader::new(GzDecoder::new(std::fs::File::open(checkpoint)?));
    let mut lines = source.lines();
    let header: Value = serde_json::from_str(&lines.next().unwrap()?).unwrap();
    assert_eq!(header["kind"], "training");
    assert_eq!(header["checkpoint_format"], "jsonl-v2");
    let iteration = header["iteration"].as_u64().unwrap() as f64;
    // The traverser-visit bound holds only for the production average; see `cfr::AverageRule`.
    let rule = header.get("average_rule").and_then(Value::as_str).unwrap_or("traverser-reach");
    let (bounded, extraction) = match rule {
        "traverser-reach" => (true, "normalize-lifetime-iteration-own-reach-accumulator-v1"),
        "opponent-sampled" => (false, "normalize-lifetime-iteration-opponent-sampled-accumulator-v1"),
        other => panic!("unknown stored average rule {other}"),
    };
    let mut entries = Map::new();
    let mut averages: Vec<Value> = Vec::new();
    for line in lines {
        let row: Value = serde_json::from_str(&line?).unwrap();
        let key = row[0].as_str().unwrap().to_string();
        let names = row[1].clone();
        let (regrets, accumulated, visits) = (floats(&row[2]), floats(&row[3]), row[4].as_u64().unwrap());
        let n = regrets.len();
        if current.is_some() {
            let p = regret_match(&regrets);
            entries.insert(key.clone(), json!([names, &p[..n]]));
        }
        if average.is_some() {
            let total = fsum(accumulated.iter().copied());
            let bound = iteration * visits as f64;
            assert!(total.is_finite() && (!bounded || total <= bound + 1e-9 * bound.max(1.0)),
                    "stored average violates the iteration/reach bound");
            let p: Vec<f64> = if total != 0.0 {
                accumulated.iter().map(|x| x / total).collect()
            } else if zero_mass == ZeroMass::Current {
                regret_match(&regrets)[..n].to_vec()
            } else {
                vec![1.0 / n as f64; n]
            };
            averages.push(json!([key, names, p, total, visits]));
        }
    }
    let count = entries.len().max(averages.len()) as u64;
    if let Some(path) = current {
        let mut document = Map::new();
        for field in ["format", "abstraction", "table", "config", "iteration", "identity"] {
            document.insert(field.into(), header[field].clone());
        }
        // Keeps a non-production run's label with its policy, e.g. CFR+'s "regret-floor-0".
        if let Some(label) = header.get("training_options") {
            document.insert("training_options".into(), label.clone());
        }
        document.insert("kind".into(), json!("inference"));
        document.insert("strategy".into(), json!("current"));
        document.insert("entries".into(), Value::Object(entries));
        write_gz(path, |out| out.write_all(serde_json::to_string(&Value::Object(document)).unwrap().as_bytes()))?;
    }
    if let Some(path) = average {
        let metadata = json!({
            "format": "holdem-hu20-stored-cfr-average-diagnostic-v1", "kind": "diagnostic-inference",
            "extraction": extraction,
            "source_checkpoint_sha256": sha256_file(checkpoint), "checkpoint_header": header,
            "zero_mass_rule": zero_mass.label(),
        });
        write_gz(path, |out| {
            writeln!(out, "{}", serde_json::to_string(&metadata).unwrap())?;
            for row in &averages {
                writeln!(out, "{}", serde_json::to_string(row).unwrap())?;
            }
            Ok(())
        })?;
    }
    Ok(count)
}
