//! Iterations with the Python trainer's semantics, and checkpoints it can load.
//!
//! Each iteration runs `roots_per_seat` traversals per seat against the table as
//! it stood when the iteration began, merges their deltas in task order and only
//! then applies them, exactly like `BlueprintTrainer.step`. Production HU20 runs
//! use one root per seat; more roots per seat run in parallel.

use crate::cfr::{Key, Lookup, Node, Stream, Table, Traversal};
use crate::game::Hand;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::prelude::*;
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub const GAME: &str = "hu20-native-reopening-20bb-52card-no-ante-rake-v1";
pub const FORMAT: &str = "holdem-hu20-native-reopening-blueprint-v1";

pub const SHARDS: usize = 64;

/// The strategy table split by key so deltas merge and apply in parallel.
#[derive(Default)]
pub struct Sharded(pub Vec<Table>);

impl Sharded {
    fn new() -> Sharded {
        Sharded((0..SHARDS).map(|_| Table::default()).collect())
    }
    pub fn shard(key: &Key) -> usize {
        key[15] as usize % SHARDS
    }
    pub fn len(&self) -> usize {
        self.0.iter().map(|t| t.len()).sum()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Key, &Node)> {
        self.0.iter().flat_map(|t| t.iter())
    }
}

impl Lookup for Sharded {
    fn lookup(&self, key: &Key) -> Option<&Node> {
        self.0[Self::shard(key)].get(key)
    }
}

pub struct Trainer {
    pub table: Sharded,
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
        Trainer { table: Sharded::new(), scratch: Vec::new(), iteration: 0, seed, roots_per_seat, nodes: 0 }
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
        let nodes: u64 = results.iter().map(|r| r.1).sum();
        // Group each task's deltas by shard, keeping task order inside every shard.
        let group = |(deltas, _): &(Table, u64)| {
                let mut by_shard: Vec<Vec<(Key, Node)>> = vec![Vec::new(); SHARDS];
                for (key, delta) in deltas {
                    by_shard[Sharded::shard(key)].push((*key, *delta));
                }
                by_shard
        };
        let grouped: Vec<Vec<Vec<(Key, Node)>>> =
            if tasks.len() > 2 { results.par_iter().map(group).collect() } else { results.iter().map(group).collect() };
        // Per shard: merge tasks in order, then apply. Each key sees Python's addition order.
        let apply = |(shard, table): (usize, &mut Table)| {
            let mut merged = Table::default();
            for task in &grouped {
                for (key, delta) in &task[shard] {
                    let target = merged.entry(*key).or_insert_with(|| Node::empty(delta.code, delta.len as usize));
                    assert!(target.code == delta.code, "an abstract infoset changed its action labels");
                    for i in 0..delta.len as usize {
                        target.regrets[i] += delta.regrets[i];
                        target.average[i] += delta.average[i];
                    }
                    target.visits += delta.visits;
                }
            }
            for (key, delta) in merged {
                let node = table.entry(key).or_insert_with(|| Node::empty(delta.code, delta.len as usize));
                assert!(node.code == delta.code, "an abstract infoset changed its action labels");
                for i in 0..delta.len as usize {
                    node.regrets[i] += delta.regrets[i];
                    node.average[i] += delta.average[i];
                    assert!(node.regrets[i].is_finite() && node.average[i].is_finite(), "non-finite blueprint update");
                }
                node.visits += delta.visits;
            }
        };
        // Two traversals per iteration (the production setting) are cheaper to apply serially.
        if tasks.len() > 2 {
            self.table.0.par_iter_mut().enumerate().for_each(apply);
        } else {
            self.table.0.iter_mut().enumerate().for_each(apply);
        }
        self.scratch = results.into_iter().map(|(deltas, _)| deltas).collect();
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
            let mut rows: Vec<(&Key, &Node)> = self.table.iter().collect();
            rows.sort_by(|a, b| a.0.cmp(b.0));
            for (key, node) in rows {
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
