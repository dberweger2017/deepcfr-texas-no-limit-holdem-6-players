//! Iterations with the Python trainer's semantics, and checkpoints it can load.
//!
//! Each iteration runs `roots_per_seat` traversals per seat against the table as
//! it stood when the iteration began, merges their deltas in task order and only
//! then applies them, exactly like `BlueprintTrainer.step`. Production HU20 runs
//! use one root per seat; more roots per seat run in parallel.

use crate::cfr::{Node, Stream, Table, Traversal};
use crate::game::Hand;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub const GAME: &str = "hu20-native-reopening-20bb-52card-no-ante-rake-v1";
pub const FORMAT: &str = "holdem-hu20-native-reopening-blueprint-v1";

pub struct Trainer {
    pub table: Table,
    /// Per-task delta maps, emptied and reused every iteration.
    scratch: Vec<Table>,
    pub iteration: u64,
    pub seed: u64,
    pub roots_per_seat: usize,
    pub nodes: u64,
}

fn mix(parts: &[u64]) -> u64 {
    let mut stream = Stream(0x5EED_0F_4C_0FFEE);
    for &part in parts {
        stream.0 ^= part;
        stream.next_u64();
    }
    stream.next_u64()
}

pub fn deal(seed: u64) -> Vec<u8> {
    let mut deck: Vec<u8> = (0..52).collect();
    let mut stream = Stream(seed);
    for i in (1..52).rev() {
        let j = stream.below(i + 1);
        deck.swap(i, j);
    }
    deck
}

impl Trainer {
    pub fn new(seed: u64, roots_per_seat: usize) -> Trainer {
        Trainer { table: Table::default(), scratch: Vec::new(), iteration: 0, seed, roots_per_seat, nodes: 0 }
    }

    pub fn step(&mut self) -> u64 {
        let iteration = self.iteration + 1;
        let tasks: Vec<(usize, usize)> =
            (0..2).flat_map(|seat| (0..self.roots_per_seat).map(move |sample| (seat, sample))).collect();
        let table = &self.table;
        let seed = self.seed;
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize_with(tasks.len(), Table::default);
        let run = |(&(seat, sample), deltas): (&(usize, usize), Table)| {
            let deck = deal(mix(&[seed, iteration, seat as u64, sample as u64, 1]));
            let sampler = Stream(mix(&[seed, iteration, seat as u64, sample as u64, 2]));
            let mut traversal = Traversal::with_deltas(table, iteration, seat, sampler, deltas);
            traversal.run(&mut Hand::from_deck(0, &deck));
            (traversal.deltas, traversal.nodes)
        };
        let results: Vec<(Table, u64)> = if tasks.len() > 2 {
            tasks.par_iter().zip(scratch.into_par_iter()).map(run).collect()
        } else {
            tasks.iter().zip(scratch).map(run).collect()
        };
        // Merge in task order, then apply: identical floating-point order to Python.
        let mut merged = Table::default();
        let mut nodes = 0;
        for (deltas, count) in &results {
            nodes += count;
            for (key, delta) in deltas {
                let target = merged.entry(*key).or_insert_with(|| Node::empty(delta.code, delta.len as usize));
                assert!(target.code == delta.code, "an abstract infoset changed its action labels");
                for i in 0..delta.len as usize {
                    target.regrets[i] += delta.regrets[i];
                    target.average[i] += delta.average[i];
                }
                target.visits += delta.visits;
            }
        }
        self.scratch = results.into_iter().map(|(deltas, _)| deltas).collect();
        for (key, delta) in merged {
            let node = self.table.entry(key).or_insert_with(|| Node::empty(delta.code, delta.len as usize));
            assert!(node.code == delta.code, "an abstract infoset changed its action labels");
            for i in 0..delta.len as usize {
                node.regrets[i] += delta.regrets[i];
                node.average[i] += delta.average[i];
                assert!(node.regrets[i].is_finite() && node.average[i].is_finite(), "non-finite blueprint update");
            }
            node.visits += delta.visits;
        }
        self.iteration = iteration;
        self.nodes += nodes;
        nodes
    }

    /// A `jsonl-v2` training checkpoint that `src.blueprint.artifact.load_training` accepts.
    pub fn save(&self, path: &Path, max_nodes: u64, max_entries: u64) -> std::io::Result<()> {
        let config = json!({
            "seed": self.seed, "raise_cap": null, "roots_per_seat": self.roots_per_seat,
            "max_nodes": max_nodes, "max_entries": max_entries, "max_seconds": 900.0,
            "abstraction": crate::key::SCHEMA, "game": GAME,
        });
        let header = json!({
            "format": FORMAT, "abstraction": crate::key::SCHEMA, "kind": "training",
            "checkpoint_format": "jsonl-v2", "iteration": self.iteration, "config": config,
            "table": {"player_ids": ["player-0", "player-1"], "stacks": [2000, 2000], "button": 0,
                      "small_blind": 50, "big_blind": 100, "chip_unit": "0.01"},
            "identity": {"game": GAME, "players": 2, "stacks": [2000, 2000], "small_blind": 50, "big_blind": 100,
                         "action_menu": "hu20-min-pot-conditional-jam-native-reopening-v1",
                         "card_descriptor": "legacy-postflop-descriptor-v1",
                         "raise_cap_semantics": "none; native minimum-raise/reopening/stack bounds"},
        });
        let temporary = path.with_extension("tmp");
        {
            let file = std::fs::File::create(&temporary)?;
            let mut out = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());
            writeln!(out, "{}", serde_json::to_string(&header).unwrap())?;
            let mut keys: Vec<&crate::cfr::Key> = self.table.keys().collect();
            keys.sort();
            for key in keys {
                let node = &self.table[key];
                let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
                let n = node.len as usize;
                let row = json!([hex, node.names(), &node.regrets[..n], &node.average[..n], node.visits]);
                writeln!(out, "{}", serde_json::to_string(&row).unwrap())?;
            }
            out.finish()?.flush()?;
        }
        std::fs::rename(temporary, path)
    }
}
