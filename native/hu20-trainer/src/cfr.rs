//! External-sampling CFR with `src/blueprint/solver.py`'s update rules.
//!
//! The traverser branches on every menu action; the opponent and chance are
//! sampled. Regret increments are `t * (v(a) - v)` and average increments
//! `t * own_reach * policy(a)` at the traverser's nodes, accumulated per
//! traversal in depth-first post-order and applied after the iteration, as
//! the Python trainer does. Sums use Python's correctly rounded `math.fsum`,
//! so a traversal reproduces Python's deltas bit for bit.

use crate::game::{Hand, STACK, BIG_BLIND};
use crate::key::{key_bytes, menu};
use std::collections::HashMap;

pub type Key = [u8; 16];

/// CPython's `math.fsum` (Shewchuk partials with a final rounding correction).
pub fn fsum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
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
        partials.truncate(i);
        partials.push(x);
    }
    let mut n = partials.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
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

pub fn regret_match(regrets: &[f64]) -> Vec<f64> {
    let positive: Vec<f64> = regrets.iter().map(|&r| if r > 0.0 { r } else { 0.0 }).collect();
    let total = fsum(positive.iter().copied());
    if total > 0.0 {
        positive.iter().map(|&p| p / total).collect()
    } else {
        vec![1.0 / regrets.len() as f64; regrets.len()]
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub names: Vec<&'static str>,
    pub regrets: Vec<f64>,
    pub average: Vec<f64>,
    pub visits: u64,
}

impl Node {
    pub fn empty(names: Vec<&'static str>) -> Node {
        let n = names.len();
        Node { names, regrets: vec![0.0; n], average: vec![0.0; n], visits: 0 }
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
        let mut cumulative = Vec::with_capacity(policy.len());
        let mut running = 0.0;
        for &p in policy {
            running += p;
            cumulative.push(running);
        }
        let target = self.unit() * running;
        cumulative[..policy.len() - 1].partition_point(|&c| c <= target)
    }
}

pub struct Traversal<'a, S: Sampler> {
    pub table: &'a HashMap<Key, Node>,
    pub iteration: u64,
    pub traverser: usize,
    pub sampler: S,
    pub deltas: HashMap<Key, Node>,
    pub nodes: u64,
    pub terminals: u64,
}

impl<'a, S: Sampler> Traversal<'a, S> {
    pub fn new(table: &'a HashMap<Key, Node>, iteration: u64, traverser: usize, sampler: S) -> Self {
        Traversal { table, iteration, traverser, sampler, deltas: HashMap::new(), nodes: 0, terminals: 0 }
    }

    pub fn run(&mut self, hand: &Hand) -> f64 {
        self.visit(hand, 1.0)
    }

    fn visit(&mut self, hand: &Hand, own_reach: f64) -> f64 {
        self.nodes += 1;
        if hand.finished() {
            self.terminals += 1;
            let stack = hand.final_stacks()[self.traverser];
            return (stack as f64 - STACK as f64) / BIG_BLIND as f64;
        }
        let choices = menu(hand);
        let names: Vec<&'static str> = choices.iter().map(|c| c.name).collect();
        let key = key_bytes(hand, &choices);
        let policy = match self.table.get(&key) {
            Some(node) => {
                assert!(node.names == names, "a v1 key changed its action menu");
                regret_match(&node.regrets)
            }
            None => vec![1.0 / names.len() as f64; names.len()],
        };
        if hand.actor.unwrap() as usize != self.traverser {
            let index = self.sampler.sample(&policy);
            return self.visit(&hand.apply(choices[index].action), own_reach);
        }
        let mut values = Vec::with_capacity(choices.len());
        for (index, choice) in choices.iter().enumerate() {
            values.push(self.visit(&hand.apply(choice.action), own_reach * policy[index]));
        }
        let value = fsum(policy.iter().zip(&values).map(|(p, v)| p * v));
        let t = self.iteration as f64;
        let delta = self.deltas.entry(key).or_insert_with(|| Node::empty(names.clone()));
        assert!(delta.names == names, "a v1 key changed its action menu");
        for index in 0..choices.len() {
            delta.regrets[index] += t * (values[index] - value);
            delta.average[index] += t * own_reach * policy[index];
        }
        delta.visits += 1;
        value
    }
}
