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
    /// With discounting, the iteration whose discount the regrets last received (fits in padding).
    pub stamp: u32,
}

impl Node {
    pub fn empty(code: u8, len: usize) -> Node {
        Node { code, len: len as u8, regrets: [0.0; MAX_ACTIONS], average: [0.0; MAX_ACTIONS], visits: 0, stamp: 0 }
    }
    pub fn names(&self) -> Vec<&'static str> {
        names_of(self.code)
    }
}

/// Which visits accumulate the average strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AverageRule {
    /// `solver.py`'s rule: `t * own reach * policy` at the traverser's nodes. Those nodes are
    /// reached through sampled opponent and chance actions, so the sum is also weighted by
    /// that sampled reach, which is not the average CFR's guarantee is about.
    TraverserReach,
    /// The standard external-sampling average: `t * policy` at each sampled opponent node.
    /// Opponent actions are sampled from that player's own policy, so the sum is weighted
    /// by the player's own reach only.
    OpponentSampled,
}

impl AverageRule {
    pub fn name(self) -> &'static str {
        match self {
            AverageRule::TraverserReach => "traverser-reach",
            AverageRule::OpponentSampled => "opponent-sampled",
        }
    }
    pub fn parse(name: &str) -> AverageRule {
        match name {
            "traverser-reach" => AverageRule::TraverserReach,
            "opponent-sampled" => AverageRule::OpponentSampled,
            other => panic!("unknown average rule {other}"),
        }
    }
}

/// Training changes beyond the production rule, each off by default and named when used.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Options {
    /// Accumulated regrets never fall below this after an update; 0 is CFR+'s floor.
    pub regret_floor: Option<f64>,
    /// Discounted CFR (alpha, beta, gamma): unweighted regret deltas; after iteration t positive
    /// regrets are multiplied by t^alpha / (t^alpha + 1) and negative ones by t^beta / (t^beta + 1);
    /// average contributions weigh t^gamma, which normalizes to DCFR's (t / (t + 1))^gamma discounting.
    pub dcfr: Option<[f64; 3]>,
}

impl Options {
    /// None for the production rule, so production outputs stay unlabeled.
    pub fn label(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(floor) = self.regret_floor {
            parts.push(format!("regret-floor-{floor}"));
        }
        if let Some([a, b, g]) = self.dcfr {
            parts.push(format!("dcfr-{a}-{b}-{g}"));
        }
        (!parts.is_empty()).then(|| parts.join("+"))
    }

    /// Rejects settings whose arithmetic would leave f64 within `iterations`, before any training.
    pub fn check(&self, iterations: u64) -> Result<(), String> {
        if let Some(floor) = self.regret_floor {
            if !floor.is_finite() {
                return Err("--regret-floor must be finite".into());
            }
        }
        let Some([alpha, beta, gamma]) = self.dcfr else { return Ok(()) };
        if ![alpha, beta, gamma].iter().all(|x| x.is_finite()) {
            return Err("--dcfr needs finite exponents".into());
        }
        // Only for alpha > 1 is the product of positive discounts over any gap bounded below, so lazy
        // catch-up cannot underflow a node's positive regrets to zero where eager discounting would
        // still have them (or the reverse).
        if alpha <= 1.0 {
            return Err("--dcfr needs alpha > 1".into());
        }
        // Stamps are u32 iterations.
        if iterations > u32::MAX as u64 {
            return Err("--dcfr runs stop before 2^32 iterations".into());
        }
        // Catch-up subtracts prefix endpoints, so every prefix through the run must be finite.
        let mut discounts = Discounts::default();
        discounts.extend([alpha, beta], iterations);
        if !discounts.0.iter().all(|p| p.iter().all(|v| v.is_finite())) {
            return Err("--dcfr exponents overflow the discount prefixes within --iterations".into());
        }
        // Average weights t^gamma for gamma in [0, 8] are at least 1, so they never underflow, and below
        // (2^32)^8 = 2^256, so even 2^64 contributions of them sum far inside f64. Larger or negative gamma
        // would need rescaled accumulators; DCFR's recommended gamma is 2.
        if !(0.0..=8.0).contains(&gamma) {
            return Err("--dcfr needs gamma in [0, 8] so iterations^gamma weights and their sums stay in range".into());
        }
        Ok(())
    }

    /// The regret and average weights of iteration t's deltas.
    pub fn weights(&self, iteration: u64) -> (f64, f64) {
        let t = iteration as f64;
        match self.dcfr {
            Some([_, _, gamma]) => (1.0, t.powf(gamma)),
            None => (t, t),
        }
    }
}

/// Running sums of ln(s^e / (s^e + 1)) for the positive and negative DCFR exponents; index 0 is 0.
#[derive(Default)]
pub struct Discounts(pub Vec<[f64; 2]>);

/// ln(1 + e^x), finite for every finite x.
fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

impl Discounts {
    /// Extends the sums through `iteration`.
    pub fn extend(&mut self, exponents: [f64; 2], iteration: u64) {
        if self.0.is_empty() {
            self.0.push([0.0; 2]);
        }
        while (self.0.len() as u64) <= iteration {
            let s = self.0.len() as f64;
            let last = *self.0.last().unwrap();
            // ln(s^e / (s^e + 1)) = -softplus(-e ln s), without forming s^-e, which overflows for
            // strongly negative e although the discount itself is finite.
            self.0.push([0, 1].map(|i| last[i] - softplus(-exponents[i] * s.ln())));
        }
    }

    /// Applies the discounts of iterations `node.stamp + 1 ..= to`. A positive factor never
    /// flips a sign, so skipped iterations compose into one factor per sign; regret matching
    /// scales every positive regret of a node alike, so play does not wait for this. With
    /// alpha > 1 the positive factor over any gap stays above exp(-sum ln(1 + s^-alpha)) > 0.
    pub fn catch_up(&self, node: &mut Node, to: u64) {
        let from = node.stamp as usize;
        let to = to as usize;
        if to > from {
            let factor = [0, 1].map(|i| (self.0[to][i] - self.0[from][i]).exp());
            for r in &mut node.regrets[..node.len as usize] {
                *r *= if *r > 0.0 { factor[0] } else { factor[1] };
            }
            node.stamp = u32::try_from(to).expect("discounted runs stop before 2^32 iterations");
        }
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

pub struct Traversal<'a, S: Sampler, L: Lookup = Table> {
    pub table: &'a L,
    pub iteration: u64,
    pub traverser: usize,
    pub sampler: S,
    pub average: AverageRule,
    /// Weights of regret and average deltas: `t` and `t` in production (`Options::weights`).
    pub regret_weight: f64,
    pub average_weight: f64,
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
        let t = iteration as f64;
        Traversal { table, iteration, traverser, sampler, average: AverageRule::TraverserReach, regret_weight: t,
                    average_weight: t, deltas, nodes: 0, terminals: 0 }
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
            if self.average == AverageRule::OpponentSampled {
                let t = self.average_weight;
                let delta = self.deltas.entry(key).or_insert_with(|| Node::empty(options.code, n));
                assert!(delta.code == options.code, "a v1 key changed its action menu");
                for index in 0..n {
                    delta.average[index] += t * policy[index];
                }
            }
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
        let (t, w) = (self.regret_weight, self.average_weight);
        let delta = self.deltas.entry(key).or_insert_with(|| Node::empty(options.code, n));
        assert!(delta.code == options.code, "a v1 key changed its action menu");
        for index in 0..n {
            delta.regrets[index] += t * (values[index] - value);
        }
        if self.average == AverageRule::TraverserReach {
            for index in 0..n {
                delta.average[index] += w * own_reach * policy[index];
            }
        }
        delta.visits += 1;
        value
    }
}

#[cfg(test)]
mod tests {
    use super::{Discounts, Node, Options};

    #[test]
    fn checks_accept_the_planned_run_and_reject_settings_that_leave_f64() {
        let dcfr = |v: [f64; 3]| Options { regret_floor: None, dcfr: Some(v) };
        // Planned: gamma 2 at 3M iterations; also extreme but representable betas and a negative gamma.
        for v in [[1.5, 0.0, 2.0], [1.5, f64::MAX, 2.0], [1.5, -200.0, 0.0], [1.0 + 1e-9, 0.0, 2.0], [1.5, 0.0, 8.0]] {
            assert_eq!(dcfr(v).check(3_000_000), Ok(()), "{v:?}");
        }
        assert!(Options::default().check(3_000_000).is_ok());
        for (v, iterations) in [([1.0, 0.0, 2.0], 10), ([1.5, 0.0, 88.5], 3000), ([1.5, -1e307, 2.0], 100),
                                ([1.5, -f64::MAX, 2.0], 100), ([1.5, f64::NAN, 2.0], 10), ([1.5, 0.0, -400.0], 10),
                                ([1.5, 0.0, -107.8], 1000), ([1.5, 0.0, -0.5], 10), ([1.5, 0.0, 8.5], 10),
                                ([1.5, 0.0, 2.0], 1 << 32)] {
            assert!(dcfr(v).check(iterations).is_err(), "{v:?} at {iterations}");
        }
        assert!(Options { regret_floor: Some(f64::NEG_INFINITY), dcfr: None }.check(10).is_err());
    }

    /// Lazy catch-up equals discounting every node after every iteration, as DCFR is defined,
    /// including strongly negative beta, whose s^-beta overflows (#169 review).
    #[test]
    fn lazy_discounts_equal_eager_ones() {
        for exponents in [[1.5, 0.0], [1.5, -200.0], [3.0, 2.0], [1.0 + 1e-9, -2.0], [1000.0, 200.0]] {
            let iterations = 2000u64;
            let mut discounts = Discounts::default();
            discounts.extend(exponents, iterations);
            assert!(discounts.0.iter().all(|p| p.iter().all(|v| v.is_finite())), "{exponents:?}");
            let mut state = 12345u64;
            let mut next = || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 53) as f64
            };
            // s^e / (s^e + 1) = 1 / (1 + e^(-e ln s)): 0 or 1 at the extremes, never NaN.
            let factor = |e: f64, t: f64| 1.0 / (1.0 + (-e * t.ln()).exp());
            let (mut lazy, mut eager) = (Node::empty(0, 3), Node::empty(0, 3));
            for t in 1..=iterations {
                let touched = next() < 0.1;
                let delta: Vec<f64> = (0..3).map(|_| next() * 2.0 - 1.0).collect();
                if touched {
                    discounts.catch_up(&mut lazy, t - 1);
                    for i in 0..3 {
                        lazy.regrets[i] += delta[i];
                        eager.regrets[i] += delta[i];
                    }
                    discounts.catch_up(&mut lazy, t);
                }
                let tf = t as f64;
                for r in &mut eager.regrets[..3] {
                    *r *= if *r > 0.0 { factor(exponents[0], tf) } else { factor(exponents[1], tf) };
                }
            }
            discounts.catch_up(&mut lazy, iterations);
            for i in 0..3 {
                let (a, b) = (lazy.regrets[i], eager.regrets[i]);
                assert!(a.is_finite() && (a - b).abs() <= 1e-9 * b.abs() + 1e-300, "{exponents:?}: {a} vs {b}");
            }
        }
    }
}
