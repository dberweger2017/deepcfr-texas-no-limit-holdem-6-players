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
        trainer.table.insert([0;16], a);
        let mut b = Node::empty(1, 1);
        b.visits = 10;
        trainer.table.insert([1;16], b);
        let d = diagnostics(&trainer);
        assert_eq!(d["entries"], 2);
        assert_eq!(d["positive_average_mass_keys"], 1);
        assert_eq!(d["zero_average_mass_keys"], 1);
        assert_eq!(d["traverser_visits"], 10);
        assert_eq!(d["visits_histogram"]["0"], 1);
        assert_eq!(d["visits_histogram"]["10-99"], 1);
    }
}

/// A receipt interval starts at process admission or the previous successful save.
/// Miss counters restart on recovery; cumulative discard counters live in the checkpoint.
pub struct FreezeSnapshot {
    iteration: u64,
    discarded: [u64; 2],
    decisions: [u64; 4],
    traverser: [u64; 4],
    missing_traverser: [u64; 4],
    missing_opponent: [u64; 4],
}

impl FreezeSnapshot {
    pub fn new(trainer: &Trainer) -> Self {
        let discarded = trainer.frozen_table.map_or([0; 2], |s| [s.discarded_traverser_visits, s.discarded_updates]);
        Self { iteration: trainer.iteration, discarded, decisions: trainer.decisions_by_street,
            traverser: trainer.traverser_visits_by_street, missing_traverser: trainer.missing_traverser_by_street,
            missing_opponent: trainer.missing_opponent_by_street }
    }
}

pub fn freeze_receipt(trainer: &Trainer, previous: &FreezeSnapshot) -> Value {
    let state = trainer.frozen_table;
    let counts = state.map_or([0; 2], |s| [s.discarded_traverser_visits, s.discarded_updates]);
    let decisions = std::array::from_fn::<_, 4, _>(|i| trainer.decisions_by_street[i] - previous.decisions[i]);
    let traverser = std::array::from_fn::<_, 4, _>(|i| trainer.traverser_visits_by_street[i] - previous.traverser[i]);
    let opponent = std::array::from_fn::<_, 4, _>(|i| decisions[i] - traverser[i]);
    let missing_traverser = std::array::from_fn::<_, 4, _>(|i| trainer.missing_traverser_by_street[i] - previous.missing_traverser[i]);
    let missing_opponent = std::array::from_fn::<_, 4, _>(|i| trainer.missing_opponent_by_street[i] - previous.missing_opponent[i]);
    let share = |missing: [u64; 4], total: [u64; 4]| {
        (0..4).map(|i| if total[i] == 0 { Value::Null } else { json!(missing[i] as f64 / total[i] as f64) }).collect::<Vec<_>>()
    };
    json!({"state": state.map(|s| s.header()), "interval_start_iteration": previous.iteration,
        "discarded_traverser_visits_since_previous_save": counts[0] - previous.discarded[0],
        "discarded_updates_since_previous_save": counts[1] - previous.discarded[1],
        "streets": ["preflop", "flop", "turn", "river"],
        "traverser_decisions_by_street": traverser, "opponent_decisions_by_street": opponent,
        "missing_traverser_by_street": missing_traverser, "missing_opponent_by_street": missing_opponent,
        "missing_traverser_share_by_street": share(missing_traverser, traverser),
        "missing_opponent_share_by_street": share(missing_opponent, opponent)})
}
