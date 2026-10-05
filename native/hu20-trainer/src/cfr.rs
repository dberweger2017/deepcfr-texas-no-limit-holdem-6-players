//! External-sampling CFR with `src/blueprint/solver.py`'s update rules.
//!
//! The traverser branches on every menu action; the opponent and chance are
//! sampled. Regret increments are `t * (v(a) - v)` and average increments
//! `t * own_reach * policy(a)` at the traverser's nodes, accumulated per
//! traversal in depth-first post-order and applied after the iteration, as
//! the Python trainer does. Sums use Python's correctly rounded `math.fsum`,
//! so a traversal reproduces Python's deltas bit for bit.

use crate::game::{Hand, BIG_BLIND, STACK};
use crate::key::{key_bytes, menu, names_of};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub type Key = [u8; 16];
pub const MAX_ACTIONS: usize = 5;

/// Keys are already uniform BLAKE2b digests, so their first eight bytes are the hash.
#[derive(Default)]
pub struct KeyHasher(u64);
impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 ^= u64::from_le_bytes(word);
        }
    }
    fn write_usize(&mut self, _: usize) {}
    fn finish(&self) -> u64 {
        self.0
    }
}
pub type Table = HashMap<Key, Node, BuildHasherDefault<KeyHasher>>;

/// Read access to a strategy table during traversals.
pub trait Lookup: Sync {
    fn lookup(&self, key: &Key) -> Option<&Node>;
}
impl Lookup for Table {
    fn lookup(&self, key: &Key) -> Option<&Node> {
        self.get(key)
    }
}

/// CPython's `math.fsum` (Shewchuk partials with a final rounding correction).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials = [0f64; 16];
    let mut count = 0;
    for value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..count {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials[i] = x;
        count = i + 1;
    }
    if count == 0 {
        return 0.0;
    }
    let mut n = count - 1;
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

pub fn regret_match(regrets: &[f64]) -> [f64; MAX_ACTIONS] {
    let mut positive = [0f64; MAX_ACTIONS];
    for (p, &r) in positive.iter_mut().zip(regrets) {
        *p = if r > 0.0 { r } else { 0.0 };
    }
    let total = fsum(positive[..regrets.len()].iter().copied());
    let mut out = [0f64; MAX_ACTIONS];
    for i in 0..regrets.len() {
        out[i] = if total > 0.0 { positive[i] / total } else { 1.0 / regrets.len() as f64 };
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Menu bitmask (`key::NAMES`); stored to catch any key whose menu changes.
    pub code: u8,
    pub len: u8,
    pub regrets: [f64; MAX_ACTIONS],
    pub average: [f64; MAX_ACTIONS],
    pub visits: u64,
}

impl Node {
    pub fn empty(code: u8, len: usize) -> Node {
        Node { code, len: len as u8, regrets: [0.0; MAX_ACTIONS], average: [0.0; MAX_ACTIONS], visits: 0 }
    }
    pub fn names(&self) -> Vec<&'static str> {
        names_of(self.code)
    }
}

/// How opponent actions are drawn from the current policy.
pub trait Sampler {
    fn sample(&mut self, policy: &[f64]) -> usize;
}

/// Replays recorded draws, for parity with Python traversals.
pub struct Forced<'a>(pub std::slice::Iter<'a, usize>);
impl Sampler for Forced<'_> {
    fn sample(&mut self, _policy: &[f64]) -> usize {
        *self.0.next().expect("recorded draws exhausted")
    }
}

/// Splitmix64 stream; draws by cumulative weights like `random.choices`.
pub struct Stream(pub u64);
impl Stream {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.unit() * n as f64) as usize % n
    }
}
impl Sampler for Stream {
    fn sample(&mut self, policy: &[f64]) -> usize {
        let mut cumulative = [0f64; MAX_ACTIONS];
        let mut running = 0.0;
        for (c, &p) in cumulative.iter_mut().zip(policy) {
            running += p;
            *c = running;
        }
        let target = self.unit() * running;
        cumulative[..policy.len() - 1].partition_point(|&c| c <= target)
    }
}

pub struct Traversal<'a, S: Sampler, L: Lookup = Table> {
    pub table: &'a L,
    pub iteration: u64,
    pub traverser: usize,
    pub sampler: S,
    pub deltas: Table,
    pub nodes: u64,
    pub terminals: u64,
}

impl<'a, S: Sampler, L: Lookup> Traversal<'a, S, L> {
    pub fn new(table: &'a L, iteration: u64, traverser: usize, sampler: S) -> Self {
        Self::with_deltas(table, iteration, traverser, sampler, Table::default())
    }

    /// Reuses an emptied delta map's allocation across iterations.
    pub fn with_deltas(table: &'a L, iteration: u64, traverser: usize, sampler: S, mut deltas: Table) -> Self {
        deltas.clear();
        Traversal { table, iteration, traverser, sampler, deltas, nodes: 0, terminals: 0 }
    }

    pub fn run(&mut self, hand: &mut Hand) -> f64 {
        self.visit(hand, 1.0)
    }

    fn visit(&mut self, hand: &mut Hand, own_reach: f64) -> f64 {
        self.nodes += 1;
        if hand.finished() {
            self.terminals += 1;
            let stack = hand.final_stacks()[self.traverser];
            return (stack as f64 - STACK as f64) / BIG_BLIND as f64;
        }
        let options = menu(hand);
        let choices = options.as_slice();
        let n = choices.len();
        let key = key_bytes(hand, choices);
        let policy = match self.table.lookup(&key) {
            Some(node) => {
                assert!(node.code == options.code, "a v1 key changed its action menu");
                regret_match(&node.regrets[..n])
            }
            None => [1.0 / n as f64; MAX_ACTIONS],
        };
        if hand.actor.unwrap() as usize != self.traverser {
            let index = self.sampler.sample(&policy[..n]);
            let saved = hand.save();
            hand.apply_mut(choices[index].action);
            let value = self.visit(hand, own_reach);
            hand.restore(saved);
            return value;
        }
        let mut values = [0f64; MAX_ACTIONS];
        for index in 0..n {
            let saved = hand.save();
            hand.apply_mut(choices[index].action);
            values[index] = self.visit(hand, own_reach * policy[index]);
            hand.restore(saved);
        }
        let value = fsum((0..n).map(|i| policy[i] * values[i]));
        let t = self.iteration as f64;
        let delta = self.deltas.entry(key).or_insert_with(|| Node::empty(options.code, n));
        assert!(delta.code == options.code, "a v1 key changed its action menu");
        for index in 0..n {
            delta.regrets[index] += t * (values[index] - value);
            delta.average[index] += t * own_reach * policy[index];
        }
        delta.visits += 1;
        value
    }
}
