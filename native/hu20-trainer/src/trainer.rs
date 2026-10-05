//! Iterations with the Python trainer's semantics, and checkpoints it can load.
//!
//! Each iteration runs `roots_per_seat` traversals per seat against the table as
//! it stood when the iteration began, merges their deltas in task order and only
//! then applies them, exactly like `BlueprintTrainer.step`. Roots draw the Python
//! trainer's own random streams, so a native run equals the Python run with the
//! same seed and roots per seat. Production HU20 runs use one root per seat; more
//! roots per seat run in parallel.

use crate::cfr::{AverageRule, Discounts, Key, Lookup, Node, Options, Sampler, Table, Traversal};
use crate::streams::{engine_deck, python_seed, Mt};
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
    /// The production rule unless chosen otherwise; regrets and play are the same under both.
    pub average: AverageRule,
    /// Training changes beyond the production rule; none by default.
    pub options: Options,
    discounts: Discounts,
}

impl Trainer {
    pub fn new(seed: u64, roots_per_seat: usize) -> Trainer {
        Trainer { table: Sharded::new(), scratch: Vec::new(), iteration: 0, seed, roots_per_seat, nodes: 0,
                  average: AverageRule::TraverserReach, options: Options::default(), discounts: Discounts::default() }
    }

    pub fn step(&mut self) -> u64 {
        let seed = self.seed;
        self.step_with(|iteration, seat, sample| {
            (Hand::from_deck(0, &engine_deck(python_seed(seed, iteration, seat, sample, "deal"))),
             Mt::new(python_seed(seed, iteration, seat, sample, "actions")))
        })
    }

    /// One iteration whose roots take their starting hand and opponent sampler from `root(iteration, seat, sample)`.
    pub fn step_with<S, F>(&mut self, root: F) -> u64
    where
        S: Sampler,
        F: Fn(u64, usize, usize) -> (Hand, S) + Sync,
    {
        let iteration = self.iteration + 1;
        let tasks: Vec<(usize, usize)> =
            (0..2).flat_map(|seat| (0..self.roots_per_seat).map(move |sample| (seat, sample))).collect();
        let table = &self.table;
        let average = self.average;
        let options = self.options;
        let (regret_weight, average_weight) = options.weights(iteration);
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize_with(tasks.len(), Table::default);
        let run = |(&(seat, sample), deltas): (&(usize, usize), Table)| {
            let (mut hand, sampler) = root(iteration, seat, sample);
            let mut traversal = Traversal::with_deltas(table, iteration, seat, sampler, deltas);
            traversal.average = average;
            traversal.regret_weight = regret_weight;
            traversal.average_weight = average_weight;
            traversal.run(&mut hand);
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
        if let Some([alpha, beta, _]) = options.dcfr {
            self.discounts.extend([alpha, beta], iteration);
        }
        let discounts = &self.discounts;
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
                let node = table.entry(key).or_insert_with(|| Node {
                    stamp: (iteration - 1).min(u32::MAX as u64) as u32,
                    ..Node::empty(delta.code, delta.len as usize)
                });
                assert!(node.code == delta.code, "an abstract infoset changed its action labels");
                // Only traverser visits carry regrets; an opponent-only delta adds to the average alone,
                // so discounting and flooring happen on the same updates under either average rule.
                let regrets = delta.visits > 0;
                if regrets && options.dcfr.is_some() {
                    discounts.catch_up(node, iteration - 1);
                }
                for i in 0..delta.len as usize {
                    node.regrets[i] += delta.regrets[i];
                    node.average[i] += delta.average[i];
                    assert!(node.regrets[i].is_finite() && node.average[i].is_finite(), "non-finite blueprint update");
                }
                if regrets && options.dcfr.is_some() {
                    discounts.catch_up(node, iteration);
                }
                if let (true, Some(floor)) = (regrets, options.regret_floor) {
                    for r in &mut node.regrets[..delta.len as usize] {
                        *r = r.max(floor);
                    }
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

    /// Brings every regret to the current iteration's discount, before reading or saving them.
    pub fn settle(&mut self) {
        if self.options.dcfr.is_some() {
            let (discounts, iteration) = (&self.discounts, self.iteration);
            self.table.0.par_iter_mut().for_each(|shard| {
                shard.values_mut().for_each(|node| discounts.catch_up(node, iteration));
            });
        }
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
        let mut header = header;
        // Only a non-production average is named, so production checkpoints stay identical to Python's.
        if self.average != AverageRule::TraverserReach {
            header["average_rule"] = json!(self.average.name());
        }
        if let Some(label) = self.options.label() {
            assert!(self.options.dcfr.is_none() || self.table.iter().all(|(_, n)| n.stamp as u64 == self.iteration),
                    "settle discounted regrets before saving");
            header["training_options"] = json!(label);
        }
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
