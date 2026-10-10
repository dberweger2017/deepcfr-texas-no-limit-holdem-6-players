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
    // Checkpoints the trainers write are in key order, so the current policy streams straight
    // into place. Any other order sorts on disk instead; both give the same bytes.
    match export_in(checkpoint, current, average, zero_mass, Order::Streamed)? {
        Some(count) => Ok(count),
        None => Ok(export_in(checkpoint, current, average, zero_mass, Order::Sorted)?.expect("sorted export")),
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Order { Streamed, Sorted }

/// The current-policy entries: written directly while keys strictly increase, or sorted on disk.
enum Entries<'a> {
    Streamed { out: GzEncoder<std::io::BufWriter<std::fs::File>>, previous: Option<String>, first: bool },
    Sorted(Rows<'a>),
}

/// None when `Order::Streamed` meets a key out of order; nothing is left behind.
fn export_in(checkpoint: &Path, current: Option<&Path>, average: Option<&Path>, zero_mass: ZeroMass, order: Order)
    -> std::io::Result<Option<u64>> {
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
    let current_stage = scratch.0.join("current.gz");
    let average_scratch = average.map(Scratch::new).transpose()?;
    let average_stage = average_scratch.as_ref().map(|s| s.0.join("average.gz"));
    // The current document is its header fields in key order, with the entries streamed in place.
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
    let field = |out: &mut dyn Write, index: usize, name: &String, value: &Value| -> std::io::Result<()> {
        if index != 0 { out.write_all(b",")?; }
        write!(out, "{}:", serde_json::to_string(name).unwrap())?;
        if name == "entries" { out.write_all(b"{") } else { write!(out, "{value}") }
    };
    let split = document.keys().position(|name| name == "entries").unwrap();
    let mut entries = match order {
        Order::Sorted => Entries::Sorted(Rows::new(&scratch.0)),
        Order::Streamed => {
            let mut out = GzEncoder::new(std::io::BufWriter::new(std::fs::File::create(&current_stage)?), Compression::default());
            out.write_all(b"{")?;
            for (index, (name, value)) in document.iter().enumerate().take(split + 1) {
                field(&mut out, index, name, value)?;
            }
            Entries::Streamed { out, previous: None, first: true }
        }
    };
    let metadata = json!({
        "format": game.average_format(), "kind": "diagnostic-inference",
        "extraction": extraction,
        "source_checkpoint_sha256": sha256_file(checkpoint), "checkpoint_header": header,
        "zero_mass_rule": zero_mass.label(),
    });
    let mut count = 0u64;
    let mut ordered = true;
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
            match &mut entries {
                Entries::Sorted(sorter) => sorter.add(format!("{key}\t{encoded}"))?,
                Entries::Streamed { out, previous, first } => {
                    // Strictly increasing keys also rule out duplicates.
                    if previous.as_deref().map_or(false, |p| p >= key) {
                        ordered = false;
                        return Ok(());
                    }
                    if current.is_some() {
                        if !*first { out.write_all(b",")?; }
                        write!(out, "\"{key}\":{encoded}")?;
                    }
                    *first = false;
                    *previous = Some(key.to_string());
                }
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
                } else { vec![1.0 / n as f64; n] };
                writeln!(out, "{}", json!([key, names, p, total, visits]))?;
            }
        }
        Ok(())
    };
    if average.is_some() { write_gz(average_stage.as_ref().unwrap(), consume)?; }
    else { consume(&mut std::io::sink())?; }
    if !ordered {
        return Ok(None); // both scratch directories, with every staged file, are removed on drop
    }
    match entries {
        Entries::Streamed { mut out, .. } => {
            if let Some(path) = current {
                out.write_all(b"}")?;
                for (index, (name, value)) in document.iter().enumerate().skip(split + 1) {
                    field(&mut out, index, name, value)?;
                }
                out.write_all(b"}")?;
                out.finish()?.flush()?;
                std::fs::rename(&current_stage, path)?;
            }
        }
        Entries::Sorted(sorter) => {
            let sorted = sorter.finish()?;
            if let Some(path) = current {
                write_gz(path, |out| {
                    out.write_all(b"{")?;
                    for (index, (name, value)) in document.iter().enumerate() {
                        field(out, index, name, value)?;
                        if name == "entries" {
                            for (i, line) in BufReader::new(std::fs::File::open(&sorted)?).lines().enumerate() {
                                let line = line?;
                                let (key, row) = line.split_once('\t').unwrap();
                                if i != 0 { out.write_all(b",")?; }
                                write!(out, "\"{key}\":{row}")?;
                            }
                            out.write_all(b"}")?;
                        }
                    }
                    out.write_all(b"}")
                })?;
            }
        }
    }
    if let Some(path) = average { std::fs::rename(average_stage.unwrap(), path)?; }
    Ok(Some(if current.is_some() || average.is_some() { count } else { 0 }))
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
    levels: Vec<Vec<PathBuf>>,
    serial: usize,
}
impl<'a> Rows<'a> {
    fn new(root: &'a Path) -> Self { Self { root, buffer: Vec::new(), levels: vec![Vec::new(); 64], serial: 0 } }
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
        self.retain_run(path)
    }
    fn retain_run(&mut self, mut path: PathBuf) -> std::io::Result<()> {
        // Merge only equal-sized runs. A growing accumulated run would make
        // total scratch I/O quadratic; this gives logarithmic merge depth.
        // 64 levels exceed the depth possible with a u64 entry count.
        for level in 0..self.levels.len() {
            self.levels[level].push(path);
            if self.levels[level].len() < FAN_IN { return Ok(()); }
            let merged = self.path();
            merge(&self.levels[level], &merged)?;
            for old in self.levels[level].drain(..) { std::fs::remove_file(old)?; }
            path = merged;
        }
        unreachable!("u64 entry count cannot exhaust merge levels")
    }
    fn finish(mut self) -> std::io::Result<PathBuf> {
        self.flush()?;
        let mut paths: Vec<_> = self.levels.iter_mut().flat_map(|runs| runs.drain(..)).collect();
        while paths.len() > FAN_IN {
            let mut next = Vec::new();
            for group in paths.chunks(FAN_IN) {
                let path = self.path();
                merge(group, &path)?;
                for old in group { std::fs::remove_file(old)?; }
                next.push(path);
            }
            paths = next;
        }
        let result = self.path();
        merge(&paths, &result)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A small real checkpoint, and the same rows in another order.
    fn checkpoints(dir: &Path) -> (PathBuf, PathBuf) {
        let mut trainer = crate::trainer::Trainer::new(3, 1);
        trainer.average = crate::cfr::AverageRule::OpponentSampled;
        while trainer.nodes < 200_000 { trainer.step(); }
        let sorted = dir.join("sorted.json.gz");
        trainer.save(&sorted, 1_000_000_000, 1_000_000).unwrap();
        let text = std::io::read_to_string(GzDecoder::new(std::fs::File::open(&sorted).unwrap())).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines[1..].reverse();
        let shuffled = dir.join("shuffled.json.gz");
        write_gz(&shuffled, |out| { for l in &lines { writeln!(out, "{l}")?; } Ok(()) }).unwrap();
        (sorted, shuffled)
    }

    fn read(path: &Path) -> String {
        std::io::read_to_string(GzDecoder::new(std::fs::File::open(path).unwrap())).unwrap()
    }

    #[test]
    fn streamed_exports_equal_sorted_ones_and_unordered_checkpoints_fall_back() {
        let dir = std::env::temp_dir().join(format!("hu-export-order-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (sorted, shuffled) = checkpoints(&dir);
        let outputs = |name: &str| (dir.join(format!("{name}-current.json.gz")), dir.join(format!("{name}-average.jsonl.gz")));
        let (c1, a1) = outputs("streamed");
        assert!(export_in(&sorted, Some(&c1), Some(&a1), ZeroMass::Uniform, Order::Streamed).unwrap().is_some());
        let (c2, a2) = outputs("sorted");
        export_in(&sorted, Some(&c2), Some(&a2), ZeroMass::Uniform, Order::Sorted).unwrap().unwrap();
        assert_eq!(read(&c1), read(&c2));
        assert_eq!(read(&a1), read(&a2));
        // Out of order: the streamed attempt leaves nothing behind, and export() sorts instead.
        let (c3, a3) = outputs("fallback");
        assert!(export_in(&shuffled, Some(&c3), Some(&a3), ZeroMass::Uniform, Order::Streamed).unwrap().is_none());
        assert!(!c3.exists() && !a3.exists());
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 6, "no scratch remains");
        export(&shuffled, Some(&c3), Some(&a3), ZeroMass::Uniform).unwrap();
        assert_eq!(read(&c3), read(&c1), "entries come out in key order either way");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn duplicate_keys_are_rejected_on_either_path() {
        let dir = std::env::temp_dir().join(format!("hu-export-duplicate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (sorted, _) = checkpoints(&dir);
        let text = read(&sorted);
        let mut lines: Vec<&str> = text.lines().collect();
        lines.insert(2, lines[1]);
        let duplicated = dir.join("duplicated.json.gz");
        write_gz(&duplicated, |out| { for l in &lines { writeln!(out, "{l}")?; } Ok(()) }).unwrap();
        let out = dir.join("current.json.gz");
        assert!(std::panic::catch_unwind(|| export(&duplicated, Some(&out), None, ZeroMass::Uniform)).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn disk_runs_merge_across_compaction_with_bounded_fan_in() {
        let scratch = Scratch::new(&std::env::temp_dir().join("hu-export-merge-test")).unwrap();
        let mut rows = Rows::new(&scratch.0);
        for key in (0..FAN_IN * 2 + 3).rev() {
            rows.add(format!("{key:032x}\t[[],[]]")).unwrap();
            rows.flush().unwrap();
            assert!(rows.levels.iter().all(|runs| runs.len() < FAN_IN));
        }
        let path = rows.finish().unwrap();
        let lines: Vec<_> = BufReader::new(std::fs::File::open(path).unwrap()).lines().map(Result::unwrap).collect();
        assert_eq!(lines.len(), FAN_IN * 2 + 3);
        assert!(lines.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn duplicates_in_separate_disk_runs_are_rejected() {
        let scratch = Scratch::new(&std::env::temp_dir().join("hu-export-duplicate-test")).unwrap();
        let mut rows = Rows::new(&scratch.0);
        rows.add(format!("{}\t[]", "f".repeat(32))).unwrap();
        rows.flush().unwrap();
        rows.add(format!("{}\t[1]", "f".repeat(32))).unwrap();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rows.finish())).is_err());
    }
}
