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
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::path::Path;

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

pub fn sha256_file(path: &Path) -> String {
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
    let game = crate::checkpoint::header_game(&header).expect("checkpoint identity");
    let iteration = header["iteration"].as_u64().unwrap() as f64;
    // The traverser-visit bound holds only for the production average; see `cfr::AverageRule`.
    let rule = header.get("average_rule").and_then(Value::as_str).unwrap_or("traverser-reach");
    let (bounded, extraction) = match rule {
        "traverser-reach" => (true, "normalize-lifetime-iteration-own-reach-accumulator-v1"),
        "opponent-sampled" => (false, "normalize-lifetime-iteration-opponent-sampled-accumulator-v1"),
        other => panic!("unknown stored average rule {other}"),
    };
    // Disk runs bound retained JSON and file descriptors independently of table size.
    // Sorting also detects duplicates in unordered checkpoints without a key HashSet.
    let destination = current.or(average).unwrap_or(checkpoint);
    let scratch = Scratch::new(destination)?;
    let mut sorter = Rows::new(&scratch.0);
    let average_scratch = average.map(Scratch::new).transpose()?;
    let average_stage = average_scratch.as_ref().map(|s| s.0.join("average.gz"));
    let metadata = json!({
        "format": game.average_format(), "kind": "diagnostic-inference",
        "extraction": extraction,
        "source_checkpoint_sha256": sha256_file(checkpoint), "checkpoint_header": header,
        "zero_mass_rule": zero_mass.label(),
    });
    let mut count = 0u64;
    let consume = |out: &mut dyn Write| -> std::io::Result<()> {
        if average.is_some() { writeln!(out, "{}", metadata)?; }
        for line in lines {
            let row: Value = serde_json::from_str(&line?).unwrap();
            crate::checkpoint::row_node(&row).expect("invalid checkpoint node");
            let key = row[0].as_str().unwrap();
            count += 1;
            assert!(count <= header["config"]["max_entries"].as_u64().unwrap(), "entry cap");
            let names = &row[1];
            let (regrets, accumulated, visits) = (floats(&row[2]), floats(&row[3]), row[4].as_u64().unwrap());
            let n = regrets.len();
            let encoded = if current.is_some() {
                let p = regret_match(&regrets);
                serde_json::to_string(&json!([names, &p[..n]])).unwrap()
            } else { String::new() };
            sorter.add(format!("{key}\t{encoded}"))?;
            if average.is_some() {
                let total = fsum(accumulated.iter().copied());
                let bound = iteration * visits as f64;
                assert!(total.is_finite() && (!bounded || total <= bound + 1e-9 * bound.max(1.0)),
                        "stored average violates the iteration/reach bound");
                let p: Vec<f64> = if total != 0.0 {
                    accumulated.iter().map(|x| x / total).collect()
                } else if zero_mass == ZeroMass::Current {
                    regret_match(&regrets)[..n].to_vec()
                } else { vec![1.0 / n as f64; n] };
                writeln!(out, "{}", json!([key, names, p, total, visits]))?;
            }
        }
        Ok(())
    };
    if average.is_some() { write_gz(average_stage.as_ref().unwrap(), consume)?; }
    else { consume(&mut std::io::sink())?; }
    let sorted = sorter.finish()?;
    if let Some(path) = current {
        let mut document = Map::new();
        for field in ["format", "abstraction", "table", "config", "iteration", "identity"] {
            document.insert(field.into(), header[field].clone());
        }
        if let Some(label) = header.get("training_options") {
            document.insert("training_options".into(), label.clone());
        }
        document.insert("kind".into(), json!("inference"));
        document.insert("strategy".into(), json!("current"));
        document.insert("entries".into(), Value::Null);
        write_gz(path, |out| {
            out.write_all(b"{")?;
            for (index, (name, value)) in document.iter().enumerate() {
                if index != 0 { out.write_all(b",")?; }
                write!(out, "{}:", serde_json::to_string(name).unwrap())?;
                if name == "entries" {
                    out.write_all(b"{")?;
                    for (i, line) in BufReader::new(std::fs::File::open(&sorted)?).lines().enumerate() {
                        let line = line?;
                        let (key, row) = line.split_once('\t').unwrap();
                        if i != 0 { out.write_all(b",")?; }
                        write!(out, "\"{key}\":{row}")?;
                    }
                    out.write_all(b"}")?;
                } else { write!(out, "{value}")?; }
            }
            out.write_all(b"}")
        })?;
    }
    if let Some(path) = average { std::fs::rename(average_stage.unwrap(), path)?; }
    Ok(if current.is_some() || average.is_some() { count } else { 0 })
}

static SCRATCH_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new(path: &Path) -> std::io::Result<Self> {
        let root = path.with_extension(format!("export-scratch-{}-{}", std::process::id(), SCRATCH_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        std::fs::create_dir(&root)?;
        Ok(Self(root))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

const RUN_ROWS: usize = 65_536;
const FAN_IN: usize = 32;
struct Rows<'a> {
    root: &'a Path,
    buffer: Vec<String>,
    runs: Vec<PathBuf>,
    serial: usize,
}
impl<'a> Rows<'a> {
    fn new(root: &'a Path) -> Self { Self { root, buffer: Vec::new(), runs: Vec::new(), serial: 0 } }
    fn path(&mut self) -> PathBuf {
        let path = self.root.join(format!("run-{}", self.serial));
        self.serial += 1;
        path
    }
    fn add(&mut self, row: String) -> std::io::Result<()> {
        self.buffer.push(row);
        if self.buffer.len() == RUN_ROWS { self.flush()?; }
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if self.buffer.is_empty() { return Ok(()); }
        self.buffer.sort_unstable();
        let path = self.path();
        let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for row in self.buffer.drain(..) { writeln!(out, "{row}")?; }
        out.flush()?;
        self.runs.push(path);
        // Compact as we go: the run list and merge heap have a fixed upper bound.
        if self.runs.len() == FAN_IN {
            let merged = self.path();
            merge(&self.runs, &merged)?;
            for path in self.runs.drain(..) { std::fs::remove_file(path)?; }
            self.runs.push(merged);
        }
        Ok(())
    }
    fn finish(mut self) -> std::io::Result<PathBuf> {
        self.flush()?;
        let result = self.path();
        merge(&self.runs, &result)?;
        Ok(result)
    }
}
fn merge(paths: &[PathBuf], output: &Path) -> std::io::Result<()> {
    let mut readers: Vec<_> = paths.iter().map(|p| std::fs::File::open(p).map(BufReader::new)).collect::<Result<_, _>>()?;
    let mut heap = BinaryHeap::new();
    for (i, reader) in readers.iter_mut().enumerate() {
        let mut line = String::new();
        if reader.read_line(&mut line)? != 0 { heap.push(Reverse((line, i))); }
    }
    let mut previous = String::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(output)?);
    while let Some(Reverse((line, i))) = heap.pop() {
        let key = line.split_once('\t').unwrap().0;
        assert_ne!(key, previous, "duplicate checkpoint key");
        previous.clear(); previous.push_str(key);
        out.write_all(line.as_bytes())?;
        let mut next = String::new();
        if readers[i].read_line(&mut next)? != 0 { heap.push(Reverse((next, i))); }
    }
    out.flush()
}
