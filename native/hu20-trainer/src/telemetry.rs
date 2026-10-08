//! Optional campaign receipts; checkpoint bytes and training updates remain unchanged.
use crate::trainer::Trainer;
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn unix_seconds() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64()
}

/// A checkpoint-time process sample complements the external family's five-second
/// history. `ps` reports KiB on the supported macOS worker (and Linux CI).
pub fn process_rss_bytes() -> std::io::Result<u64> {
    let result = std::process::Command::new("ps").args([
        "-o", "rss=", "-p", &std::process::id().to_string()]).output()?;
    if !result.status.success() {
        return Err(std::io::Error::new(std::io::ErrorKind::Other, "checkpoint RSS sample failed"));
    }
    let kib: u64 = String::from_utf8_lossy(&result.stdout).trim().parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid checkpoint RSS sample"))?;
    kib.checked_mul(1024).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "checkpoint RSS overflow"))
}

/// Visit diagnostics describe stored keys, never the set of possible information sets.
pub fn diagnostics(trainer: &Trainer) -> Value {
    let mut histogram = [0u64; 5];
    let mut positive = 0u64;
    let mut visits = 0u64;
    for (_, node) in trainer.table.iter() {
        let bucket = match node.visits { 0 => 0, 1 => 1, 2..=9 => 2, 10..=99 => 3, _ => 4 };
        histogram[bucket] += 1;
        visits = visits.checked_add(node.visits).expect("visit counter overflow");
        positive += u64::from(node.average[..node.len as usize].iter().any(|&v| v > 0.0));
    }
    json!({"entries": trainer.table.len(), "traverser_visits": visits,
        "visits_histogram": {"0": histogram[0], "1": histogram[1], "2-9": histogram[2],
            "10-99": histogram[3], "100+": histogram[4]},
        "positive_average_mass_keys": positive,
        "zero_average_mass_keys": trainer.table.len() as u64 - positive,
        "coverage_start": trainer.coverage_start,
        "decisions_by_street": trainer.decisions_by_street,
        "traverser_visits_by_street": trainer.traverser_visits_by_street})
}

/// Publish only after the atomic save succeeded. A crashed or failed save has no receipt.
pub fn append(path: &Path, receipt: &Value) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
    writeln!(file, "{}", receipt)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::Node;
    #[test]
    fn diagnostics_include_opponent_only_mass_and_zero_visit_keys() {
        let mut trainer = Trainer::new(1, 1);
        let mut a = Node::empty(1, 1);
        a.average[0] = 2.0;
        trainer.table.0[0].insert([0;16], a);
        let mut b = Node::empty(1, 1);
        b.visits = 10;
        trainer.table.0[1].insert([1;16], b);
        let d = diagnostics(&trainer);
        assert_eq!(d["entries"], 2);
        assert_eq!(d["positive_average_mass_keys"], 1);
        assert_eq!(d["zero_average_mass_keys"], 1);
        assert_eq!(d["traverser_visits"], 10);
        assert_eq!(d["visits_histogram"]["0"], 1);
        assert_eq!(d["visits_histogram"]["10-99"], 1);
    }
}
