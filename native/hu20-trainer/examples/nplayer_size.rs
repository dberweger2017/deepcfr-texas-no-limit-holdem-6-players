//! Bounded traversal-key census. No changes to the production traversal or keys.
#[path = "support/memory.rs"]
mod memory;
#[path = "support/survey.rs"]
mod survey;
#[global_allocator]
static METER: memory::Meter = memory::Meter;
use hu20_trainer::{
    cards::EquityTables,
    multi_cfr::{Game, Trainer},
    multi_holdem::CashGame,
    nplayer::Config,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{cell::RefCell, path::Path, rc::Rc, time::Instant};
fn fingerprint(trainer: &Trainer, survey: &survey::Survey) -> serde_json::Value {
    let mut keys: Vec<_> = survey.keys.keys().collect();
    keys.sort_unstable();
    let mut reached = Sha256::new();
    for key in keys {
        reached.update(key);
    }
    let mut regrets = Sha256::new();
    for (key, e) in trainer.table.sorted().filter(|(_, e)| e.visits > 0) {
        regrets.update(key);
        regrets.update(e.code.to_le_bytes());
        regrets.update(e.visits.to_le_bytes());
        for r in e.regrets {
            regrets.update(r.to_bits().to_le_bytes());
        }
    }
    json!({"reached_keys_sha256":format!("{:x}",reached.finalize()),"visited_regrets_sha256":format!("{:x}",regrets.finalize())})
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert_eq!(
        a.len(),
        9,
        "nplayer_size N STACK_BB SCOPE SEED NODES MAX_ENTRIES MAX_SECONDS K50_DIRECTORY"
    );
    let n: usize = a[1].parse().unwrap();
    let bb: u64 = a[2].parse().unwrap();
    let scope: usize = a[3].parse().unwrap();
    let seed: u64 = a[4].parse().unwrap();
    let nodes: u64 = a[5].parse().unwrap();
    let cap: usize = a[6].parse().unwrap();
    let seconds: f64 = a[7].parse().unwrap();
    assert!(nodes > 0 && seconds.is_finite() && seconds > 0.);
    let config = Config::new(&vec![bb.checked_mul(100).unwrap(); n], 50, 100).unwrap();
    let tables = EquityTables::load(Path::new(&a[8]), false).unwrap();
    let survey = Rc::new(RefCell::new(survey::Survey::default()));
    let game = survey::Observed {
        game: CashGame { config, tables },
        survey: survey.clone(),
    };
    let mut trainer = Trainer::new(&game, seed, scope, cap, false);
    println!(
        "{}",
        json!({"identity":game.identity(),"scope":scope,"seed":seed,"target_nodes":nodes,"max_entries":cap,"baseline_requested_bytes":memory::live(),"entry_record_bytes":hu20_trainer::store::ENTRY_BYTES,"shards":hu20_trainer::store::SHARDS})
    );
    memory::reset_peak();
    let start = Instant::now();
    let mut training_seconds = 0.;
    let mut table_bytes: i128 = 0;
    let mut next = 0;
    let milestones = [
        100_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000,
    ];
    let reason = loop {
        if trainer.counts.nodes >= nodes {
            break "nodes";
        }
        if trainer.table.len() >= cap {
            break "capacity";
        }
        if start.elapsed().as_secs_f64() >= seconds {
            break "time";
        }
        let observer_before = survey.borrow().allocated;
        let before = memory::live();
        let step_start = Instant::now();
        trainer.step(&game);
        training_seconds += step_start.elapsed().as_secs_f64();
        // Only table and observer allocations survive a single-threaded step;
        // all simulator, identity, root, and delta-map temporaries have dropped.
        table_bytes += memory::live() as i128
            - before as i128
            - (survey.borrow().allocated - observer_before) as i128;
        assert!(table_bytes >= 0);
        if (next < milestones.len() && trainer.counts.nodes >= milestones[next])
            || trainer.counts.nodes >= nodes
            || trainer.table.len() >= cap
        {
            let s = survey.borrow();
            assert_eq!(s.decisions, trainer.counts.decisions);
            let mut entries = [0u64; 4];
            let mut averages = [0u64; 4];
            for e in trainer.table.values() {
                entries[e.street as usize] += 1;
                if e.average.is_some() {
                    averages[e.street as usize] += 1;
                }
            }
            let (regret_bytes, average_bytes) = trainer.table.payload_bytes();
            let a = trainer.table.allocation();
            println!(
                "{}",
                json!({"nodes":trainer.counts.nodes,"iteration":trainer.iteration,"entries":trainer.table.len(),"index_capacity":trainer.table.index_capacity(),"index_bytes":a.index,"entry_bytes":a.entries,"value_bytes":a.values,"entries_by_street":entries,"reached_keys":s.keys.len(),"reached_by_street":s.by_street,"observer_requested_bytes":s.allocated,"table_requested_bytes":table_bytes,"regret_payload_bytes":regret_bytes,"average_payload_bytes":average_bytes,"average_entries_by_street":averages,"requested_live_bytes":memory::live(),"requested_peak_bytes":memory::peak(),"rss_bytes":hu20_trainer::telemetry::process_rss_bytes().unwrap(),"training_seconds":training_seconds,"elapsed_seconds":start.elapsed().as_secs_f64()})
            );
            while next < milestones.len() && milestones[next] <= trainer.counts.nodes {
                next += 1;
            }
        }
    };
    println!(
        "{}",
        json!({"stop_reason":reason,"nodes":trainer.counts.nodes,"entries":trainer.table.len(),"fingerprints":fingerprint(&trainer,&survey.borrow()),"training_seconds":training_seconds,"elapsed_seconds":start.elapsed().as_secs_f64()})
    );
    if reason != "nodes" {
        std::process::exit(3);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use hu20_trainer::small_games::{Kind, SmallGame};
    #[test]
    fn instrumentation_cannot_change_updates_or_checkpoint_bytes() {
        for n in 2..=6 {
            for scope in [0, 3] {
                let game = SmallGame::new(Kind::Leduc, n);
                let watched = survey::Observed {
                    game,
                    survey: Rc::new(RefCell::new(survey::Survey::default())),
                };
                let mut plain = Trainer::new(&game, 19, scope, usize::MAX, false);
                let mut observed = Trainer::new(&watched, 19, scope, usize::MAX, false);
                let mut measured = 0i128;
                for _ in 0..30 {
                    plain.step(&game);
                    let old_observer = watched.survey.borrow().allocated;
                    let before = memory::accounted_live();
                    observed.step(&watched);
                    measured += memory::accounted_live()
                        - before
                        - (watched.survey.borrow().allocated - old_observer) as i128;
                }
                assert_eq!(observed.table.allocation().total() as i128, measured);
                // A deep clone owns exactly the same index capacity and chunks, plus its shard array.
                let shards = (hu20_trainer::store::SHARDS * std::mem::size_of::<hu20_trainer::store::Shard>()) as i128;
                let before = memory::accounted_live();
                let copy = observed.table.clone();
                assert_eq!(memory::accounted_live() - before, measured + shards);
                assert_eq!(copy.allocation(), observed.table.allocation());
                drop(copy);
                let (regret_bytes, average_bytes) = observed.table.payload_bytes();
                let logical: usize = observed.table.values().map(|e| e.regrets.len() * 8).sum();
                assert!(regret_bytes < logical, "opponent-only keys keep no regrets");
                assert_eq!(average_bytes, observed.table.values().filter_map(|e| e.average).map(|a| a.len() * 8).sum::<usize>());
                assert_eq!(plain.table, observed.table);
                assert_eq!(plain.counts, observed.counts);
                let s = watched.survey.borrow();
                assert_eq!(s.decisions, observed.counts.decisions);
                assert!(observed.table.keys().all(|k| s.keys.contains_key(&k)));
                if scope == 3 {
                    assert_eq!(observed.table.len(), s.keys.len());
                } else {
                    assert!(observed.table.len() < s.keys.len());
                }
                let path = std::env::temp_dir().join(format!(
                    "nplayer-size-{}-{n}-{scope}.gz",
                    std::process::id()
                ));
                let gzip = hu20_trainer::gz::Gzip::Compatible;
                hu20_trainer::multi_checkpoint::save(&plain, &path, gzip).unwrap();
                let bytes = std::fs::read(&path).unwrap();
                hu20_trainer::multi_checkpoint::save(&observed, &path, gzip).unwrap();
                assert_eq!(bytes, std::fs::read(&path).unwrap());
                std::fs::remove_file(path).unwrap();
            }
        }
    }
}
