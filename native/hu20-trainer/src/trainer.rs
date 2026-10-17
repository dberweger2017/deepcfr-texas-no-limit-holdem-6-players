//! Iterations with the Python trainer's semantics, and checkpoints it can load.
//!
//! Each iteration runs `roots_per_seat` traversals per seat against the table as
//! it stood when the iteration began, merges their deltas in task order and only
//! then applies them, exactly like `BlueprintTrainer.step`. Roots draw the Python
//! trainer's own random streams, so a native run equals the Python run with the
//! same seed and roots per seat. Production HU20 runs use one root per seat; more
//! roots per seat run in parallel.

use crate::cards::Cards;
use crate::key::MenuVersion;
use crate::cfr::{AverageRule, Discounts, Key, Node, Options, Sampler, Table, Traversal};
use crate::store::{Shard, Store, SHARDS};
use crate::streams::{engine_deck, python_seed, Mt};
use crate::game::{Game, Hand};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::path::Path;

/// Two workers for the production setting's two traversals: the global pool would wake every
/// idle worker on each ~0.3 ms iteration.
static PAIR: std::sync::LazyLock<rayon::ThreadPool> =
    std::sync::LazyLock::new(|| rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());

pub const GAME: &str = "hu20-native-reopening-20bb-52card-no-ante-rake-v1";
pub const FORMAT: &str = "holdem-hu20-native-reopening-blueprint-v1";

/// Permanent table boundary and cumulative accounting; absent until freezing actually happens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrozenTable {
    pub since_iteration: u64,
    pub entries: u64,
    pub discarded_traverser_visits: u64,
    /// One update is one missing key's merged delta in one complete iteration, including opponent-only deltas.
    pub discarded_updates: u64,
}

impl FrozenTable {
    pub fn header(self) -> Value {
        json!({"version": 1, "since_iteration": self.since_iteration, "entries": self.entries,
            "discarded_traverser_visits": self.discarded_traverser_visits, "discarded_updates": self.discarded_updates})
    }
}

type TraversalResult = (Table, u64, [u64; 4], [u64; 4], [u64; 4], [u64; 4]);

pub struct Trainer {
    pub game: Game,
    pub table: Store,
    /// v1 cards unless chosen otherwise; set on every root hand.
    pub cards: Cards,
    /// The v1 menu unless chosen otherwise; set on every root hand.
    pub menu: MenuVersion,
    /// Per-task delta maps, emptied and reused every iteration.
    scratch: Vec<Table>,
    /// Per-shard merged deltas, drained and reused every iteration.
    merged: Vec<Table>,
    /// Process-local; content is the same either way.
    pub gzip: crate::gz::Gzip,
    pub iteration: u64,
    pub seed: u64,
    pub roots_per_seat: usize,
    pub nodes: u64,
    pub coverage_start: [u64; 3], // iteration, completed nodes, traverser visits before telemetry
    pub decisions_by_street: [u64; 4],
    pub traverser_visits_by_street: [u64; 4],
    pub frozen_table: Option<FrozenTable>,
    /// Process-local telemetry; deliberately absent from checkpoints so enabling receipts cannot change bytes.
    pub missing_traverser_by_street: [u64; 4],
    pub missing_opponent_by_street: [u64; 4],
    /// The production rule unless chosen otherwise; regrets and play are the same under both.
    pub average: AverageRule,
    /// Training changes beyond the production rule; none by default.
    pub options: Options,
    discounts: Discounts,
}

impl Trainer {
    pub fn new(seed: u64, roots_per_seat: usize) -> Trainer {
        Trainer { game: Game::Hu20, table: Store::new(), cards: Cards::V1, menu: MenuVersion::V1, scratch: Vec::new(), merged: Vec::new(), gzip: Default::default(), iteration: 0, seed, roots_per_seat, nodes: 0, coverage_start: [0; 3], decisions_by_street: [0; 4], traverser_visits_by_street: [0; 4],
                  frozen_table: None, missing_traverser_by_street: [0; 4], missing_opponent_by_street: [0; 4],
                  average: AverageRule::TraverserReach, options: Options::default(), discounts: Discounts::default() }
    }

    /// Called before the first resumed step and after each complete iteration. Never unfreezes.
    pub fn freeze_at_cap(&mut self, max_entries: u64) {
        if self.frozen_table.is_none() && self.table.len() as u64 >= max_entries {
            self.frozen_table = Some(FrozenTable { since_iteration: self.iteration,
                entries: self.table.len() as u64, discarded_traverser_visits: 0, discarded_updates: 0 });
        }
    }

    pub fn step(&mut self) -> u64 {
        let seed = self.seed;
        let game = self.game;
        self.step_with(|iteration, seat, sample| {
            (Hand::from_deck_for(game, 0, &engine_deck(python_seed(seed, iteration, seat, sample, "deal"))),
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
        let cards = self.cards;
        let menu = self.menu;
        let (regret_weight, average_weight) = options.weights(iteration);
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize_with(tasks.len(), Table::default);
        let run = |(&(seat, sample), deltas): (&(usize, usize), Table)| {
            let (mut hand, sampler) = root(iteration, seat, sample);
            assert_eq!(hand.game, self.game, "root belongs to another game");
            hand.core.cards = cards;
            hand.core.menu = menu;
            let mut traversal = Traversal::with_deltas(table, iteration, seat, sampler, deltas);
            traversal.average = average;
            traversal.regret_weight = regret_weight;
            traversal.average_weight = average_weight;
            traversal.run(&mut hand);
            (traversal.deltas, traversal.nodes, traversal.decisions_by_street, traversal.traverser_visits_by_street,
             traversal.missing_traverser_by_street, traversal.missing_opponent_by_street)
        };
        // Every task reads the same table and draws from its own streams, and deltas merge in task
        // order below, so running the production setting's two traversals at once changes nothing.
        let results: Vec<TraversalResult> = if tasks.len() == 2 {
            let mut scratch = scratch.into_iter();
            let (a, b) = (scratch.next().unwrap(), scratch.next().unwrap());
            let (x, y) = PAIR.install(|| rayon::join(|| run((&tasks[0], a)), || run((&tasks[1], b))));
            vec![x, y]
        } else {
            tasks.par_iter().zip(scratch.into_par_iter()).map(run).collect()
        };
        let nodes: u64 = results.iter().map(|r| r.1).sum();
        for (_, _, decisions, visits, missing_traverser, missing_opponent) in &results {
            for i in 0..4 {
                self.decisions_by_street[i] += decisions[i];
                self.traverser_visits_by_street[i] += visits[i];
                self.missing_traverser_by_street[i] += missing_traverser[i];
                self.missing_opponent_by_street[i] += missing_opponent[i];
            }
        }
        // Python's addition order per key: +0.0, then each task's delta in task order.
        let mut merged = std::mem::take(&mut self.merged);
        merged.resize_with(SHARDS, Table::default);
        let add_task = |merged: &mut Table, key: &Key, delta: &Node| {
            let target = merged.entry(*key).or_insert_with(|| Node::empty(delta.code, delta.len as usize));
            assert!(target.code == delta.code, "an abstract infoset changed its action labels");
            for i in 0..delta.len as usize {
                target.regrets[i] += delta.regrets[i];
                target.average[i] += delta.average[i];
            }
            target.visits += delta.visits;
        };
        if tasks.len() > 2 {
            let grouped: Vec<Vec<Vec<(&Key, &Node)>>> = results.par_iter().map(|result| {
                let mut by_shard: Vec<Vec<(&Key, &Node)>> = vec![Vec::new(); SHARDS];
                for (key, delta) in &result.0 {
                    by_shard[Store::shard(key)].push((key, delta));
                }
                by_shard
            }).collect();
            merged.par_iter_mut().enumerate().for_each(|(shard, target)| {
                for task in &grouped {
                    for &(key, delta) in &task[shard] {
                        add_task(target, key, delta);
                    }
                }
            });
        } else {
            for result in &results {
                for (key, delta) in &result.0 {
                    add_task(&mut merged[Store::shard(key)], key, delta);
                }
            }
        }
        if let Some([alpha, beta, _]) = options.dcfr {
            self.discounts.extend([alpha, beta], iteration);
        }
        let discounts = &self.discounts;
        let frozen = self.frozen_table.is_some();
        let apply = |((table, merged), _): ((&mut Shard, &mut Table), usize)| {
            let (mut discarded_visits, mut discarded_updates) = (0u64, 0u64);
            for (key, delta) in merged.drain() {
                // Filter only by membership, never visits: existing opponent-only keys still learn their average.
                if frozen && !table.contains(&key) {
                    discarded_visits += delta.visits;
                    discarded_updates += 1;
                    continue;
                }
                let fresh = || Node { stamp: (iteration - 1).min(u32::MAX as u64) as u32, ..Node::empty(delta.code, delta.len as usize) };
                table.update(key, fresh, |node| {
                    assert!(node.code == delta.code, "an abstract infoset changed its action labels");
                    // Only traverser visits carry regrets; an opponent-only delta adds to the average alone,
                    // so discounting and flooring happen on the same updates under either average rule.
                    let regrets = delta.visits > 0;
                    if regrets && options.dcfr.is_some() {
                        discounts.catch_up(node, iteration - 1);
                    }
                    for i in 0..delta.len as usize {
                        // An opponent-only delta's regrets are +0.0; adding them would still turn a -0.0 regret
                        // into +0.0 in this trainer alone.
                        if regrets {
                            node.regrets[i] += delta.regrets[i];
                        }
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
                });
            }
            (discarded_visits, discarded_updates)
        };
        // Two traversals per iteration (the production setting) are cheaper to apply serially.
        let sum = |a: (u64, u64), b: (u64, u64)| (a.0 + b.0, a.1 + b.1);
        let discarded = if tasks.len() > 2 {
            self.table.0.par_iter_mut().zip(merged.par_iter_mut()).zip(0..SHARDS).map(apply).reduce(|| (0, 0), sum)
        } else {
            self.table.0.iter_mut().zip(merged.iter_mut()).zip(0..SHARDS).map(apply).fold((0, 0), sum)
        };
        self.merged = merged;
        if let Some(state) = &mut self.frozen_table {
            state.discarded_traverser_visits += discarded.0;
            state.discarded_updates += discarded.1;
            assert_eq!(self.table.len() as u64, state.entries, "frozen key set grew");
        }
        self.scratch = results.into_iter().map(|(deltas, _, _, _, _, _)| deltas).collect();
        self.iteration = iteration;
        self.nodes += nodes;
        nodes
    }

    pub fn schema(&self) -> &'static str {
        crate::key::schema(self.game, matches!(self.cards, Cards::Equity(_)), self.menu)
    }

    /// `node` with its regrets brought to the current iteration's discount. Reads never write the
    /// table, so how often a run exports or saves cannot change its training.
    pub fn caught_up(&self, node: &Node) -> Node {
        let mut node = *node;
        if self.options.dcfr.is_some() {
            self.discounts.catch_up(&mut node, self.iteration);
        }
        node
    }

    /// A `jsonl-v2` training checkpoint that `src.blueprint.artifact.load_training` accepts.
    pub fn save(&self, path: &Path, max_nodes: u64, max_entries: u64) -> std::io::Result<()> {
        self.save_impl(path, max_nodes, max_entries, false)
    }

    pub fn save_recoverable(&self, path: &Path, max_nodes: u64, max_entries: u64) -> std::io::Result<()> {
        self.save_impl(path, max_nodes, max_entries, true)
    }

    fn save_impl(&self, path: &Path, max_nodes: u64, max_entries: u64, recovery: bool) -> std::io::Result<()> {
        let config = json!({
            "seed": self.seed, "raise_cap": null, "roots_per_seat": self.roots_per_seat,
            "max_nodes": max_nodes, "max_entries": max_entries, "max_seconds": 900.0,
            "abstraction": self.schema(), "game": self.game.id(),
        });
        let header = json!({
            "format": self.game.format(), "abstraction": self.schema(), "kind": "training",
            "checkpoint_format": "jsonl-v2", "iteration": self.iteration, "config": config,
            "table": {"player_ids": ["player-0", "player-1"], "stacks": [self.game.stack(), self.game.stack()], "button": 0,
                      "small_blind": 50, "big_blind": 100, "chip_unit": "0.01"},
            "identity": crate::checkpoint::identity(self.game, self.cards.descriptor(), self.cards.tables(), self.menu),
        });
        let mut header = header;
        if let Some(state) = self.frozen_table {
            header["frozen_table"] = state.header();
        }
        if recovery || self.game != Game::Hu20 || self.frozen_table.is_some() {
            header["native_state"] = json!({"version": 1, "completed_nodes": self.nodes,
                "coverage_start": self.coverage_start, "decisions_by_street": self.decisions_by_street, "traverser_visits_by_street": self.traverser_visits_by_street});
        }
        // Only a non-production average is named, so production checkpoints stay identical to Python's.
        if self.average != AverageRule::TraverserReach {
            header["average_rule"] = json!(self.average.name());
        }
        if let Some(label) = self.options.label() {
            header["training_options"] = json!(label);
        }
        let temporary = path.with_extension("tmp");
        let head = serde_json::to_string(&header).unwrap() + "\n";
        self.table.write_sorted(std::fs::File::create(&temporary)?, self.gzip, head.as_bytes(), |shard, i, out| {
            let (key, node) = shard.row(i);
            let node = &self.caught_up(&node);
            let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
            let n = node.len as usize;
            serde_json::to_writer(&mut *out, &(hex, node.names(), &node.regrets[..n], &node.average[..n], node.visits)).unwrap();
            out.push(b'\n');
        })?;
        std::fs::rename(temporary, path)
    }
}

#[cfg(test)]
mod frozen_tests {
    use super::*;

    #[test]
    fn frozen_merge_keeps_every_existing_update_including_opponent_only() {
        for roots in [1, 4] {
            let make = || {
                let mut t = Trainer::new(5, roots);
                t.average = AverageRule::OpponentSampled;
                for _ in 0..12 { t.step(); }
                t
            };
            let mut frozen = make();
            let before: Table = frozen.table.iter().collect();
            frozen.freeze_at_cap(before.len() as u64);
            let mut reference = make();
            let mut changed_opponent_only = 0;
            let mut changed_regrets = 0;
            let mut acquired_regrets = 0;
            for _ in 0..8 {
                // Rebuild the unfrozen reference's input with exactly the frozen key set each time.
                reference.table = Store::new();
                for (key, node) in frozen.table.iter() { reference.table.insert(key, node); }
                reference.iteration = frozen.iteration;
                reference.nodes = frozen.nodes;
                let previous: Table = frozen.table.iter().collect();
                let previous_discards = frozen.frozen_table.unwrap().discarded_traverser_visits;
                let previous_misses: u64 = frozen.missing_traverser_by_street.iter().sum();
                let nodes = frozen.step();
                assert_eq!(frozen.frozen_table.unwrap().discarded_traverser_visits - previous_discards,
                    frozen.missing_traverser_by_street.iter().sum::<u64>() - previous_misses);
                assert_eq!(reference.step(), nodes);
                assert_eq!(frozen.table.len(), before.len());
                for (key, node) in frozen.table.iter() {
                    let expected = reference.table.get(&key).unwrap();
                    let old = previous.get(&key).unwrap();
                    assert_eq!(node.regrets.map(f64::to_bits), expected.regrets.map(f64::to_bits));
                    assert_eq!(node.average.map(f64::to_bits), expected.average.map(f64::to_bits));
                    assert_eq!(node.visits, expected.visits);
                    changed_regrets += usize::from(node.regrets != old.regrets);
                    changed_opponent_only += usize::from(old.visits == 0 && node.visits == 0 && node.average != old.average);
                    acquired_regrets += usize::from(old.visits == 0 && node.visits > 0 && node.regrets != old.regrets);
                }
            }
            assert!(changed_regrets > 0 && changed_opponent_only > 0 && acquired_regrets > 0);
            let state = frozen.frozen_table.unwrap();
            assert!(state.discarded_traverser_visits > 0 && state.discarded_updates > 0);
            let stored: u64 = frozen.table.iter().map(|(_, n)| n.visits).sum();
            assert_eq!(stored + state.discarded_traverser_visits, frozen.traverser_visits_by_street.iter().sum::<u64>());
        }
    }

    #[test]
    fn frozen_recovery_preserves_bytes_and_validates_accounting() {
        let mut full = Trainer::new(7, 1);
        full.average = AverageRule::OpponentSampled;
        for _ in 0..5 { full.step(); }
        full.freeze_at_cap(1);
        for _ in 0..5 { full.step(); }
        let dir = std::env::temp_dir().join(format!("frozen-recovery-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let checkpoint = dir.join("half.gz");
        full.save_recoverable(&checkpoint, 1000000000, 1000000).unwrap();
        let mut resumed = crate::checkpoint::load(&checkpoint, full.game, full.cards, full.menu, None, 1000000).unwrap();
        assert_eq!(resumed.frozen_table, full.frozen_table);
        for _ in 0..10 { full.step(); resumed.step(); }
        for (t, name) in [(&full, "full.gz"), (&resumed, "resumed.gz")] {
            t.save_recoverable(&dir.join(name), 1000000000, 1000000).unwrap();
        }
        assert_eq!(std::fs::read(dir.join("full.gz")).unwrap(), std::fs::read(dir.join("resumed.gz")).unwrap());
        resumed.frozen_table.as_mut().unwrap().discarded_traverser_visits += 1;
        resumed.save_recoverable(&checkpoint, 1000000000, 1000000).unwrap();
        assert!(crate::checkpoint::load(&checkpoint, full.game, full.cards, full.menu, None, 1000000).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
