//! N-player external sampling. The legacy HU traversal is intentionally separate.
use crate::cfr::{fsum, regret_match, Key, KeyHasher, MAX_ACTIONS};
use crate::store::{Allocation, Store, View};
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub key: Key,
    pub actor: usize,
    /// Stable identity inside the key (button-relative in cash games).
    pub owner: usize,
    pub street: usize,
    /// Ordered actions have ascending bit positions; identity is checked on every hit.
    pub code: u16,
}
impl Decision {
    pub fn len(self) -> usize {
        self.code.count_ones() as usize
    }
}

/// Privileged simulator interface. Only `decision` constructs policy inputs;
/// root sampling and terminal payoffs may use the hidden simulated deal.
pub trait State {
    type Saved;
    fn players(&self) -> usize;
    fn decision(&self) -> Option<Decision>;
    fn payoff(&self, player: usize) -> f64;
    fn save(&self) -> Self::Saved;
    fn restore(&mut self, saved: Self::Saved);
    fn apply(&mut self, action_index: usize);
}

pub trait Game {
    type State: State;
    fn players(&self) -> usize;
    fn identity(&self) -> serde_json::Value;
    fn sample(&self, rng: &mut rand::rngs::StdRng) -> Self::State;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub actor: u8,
    pub street: u8,
    pub code: u16,
    pub regrets: Box<[f64]>,
    /// No allocation for excluded streets, including opponent-only visits.
    pub average: Option<Box<[f64]>>,
    pub visits: u64,
}
impl Entry {
    pub fn new(d: Decision, last_average_street: usize) -> Self {
        assert!((1..=MAX_ACTIONS).contains(&d.len()) && d.street < 4 && d.actor < 6 && d.owner < 6);
        Self {
            actor: d.owner as u8,
            street: d.street as u8,
            code: d.code,
            regrets: vec![0.; d.len()].into_boxed_slice(),
            average: (d.street <= last_average_street)
                .then(|| vec![0.; d.len()].into_boxed_slice()),
            visits: 0,
        }
    }
    pub fn stored(&self) -> Stored<'_> {
        Stored {
            actor: self.actor,
            street: self.street,
            code: self.code,
            visits: self.visits,
            regrets: &self.regrets,
            average: self.average.as_deref(),
        }
    }
    pub fn check(&self, d: Decision) {
        self.stored().check(d)
    }
}

/// An entry as the table holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stored<'a> {
    pub actor: u8,
    pub street: u8,
    pub code: u16,
    pub visits: u64,
    pub regrets: &'a [f64],
    pub average: Option<&'a [f64]>,
}
impl<'a> Stored<'a> {
    /// The store's `meta`: owner in the low byte, street in the next.
    fn meta(&self) -> u32 {
        self.actor as u32 | (self.street as u32) << 8
    }
    fn view(&self) -> View<'a> {
        View { code: self.code, meta: self.meta(), visits: self.visits, regrets: self.regrets, average: self.average }
    }
    fn of(v: View<'a>) -> Self {
        Stored {
            actor: v.meta as u8,
            street: (v.meta >> 8) as u8,
            code: v.code,
            visits: v.visits,
            regrets: v.regrets,
            average: v.average,
        }
    }
    pub fn check(&self, d: Decision) {
        assert_eq!(
            (
                self.actor as usize,
                self.street as usize,
                self.code,
                self.regrets.len()
            ),
            (d.owner, d.street, d.code, d.len()),
            "information key changed owner, street or actions"
        );
    }
    pub fn policy(&self, average: bool) -> [f64; MAX_ACTIONS] {
        if average {
            regret_match(self.average.expect("average not stored for this street"))
        } else {
            regret_match(self.regrets)
        }
    }
}

/// One iteration's updates, in deterministic DFS order per key.
pub type Deltas = HashMap<Key, Entry, BuildHasherDefault<KeyHasher>>;

/// The training table: the HU trainer's compact sharded store.
#[derive(Clone, Default)]
pub struct Table(Store);
impl Table {
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, key: &Key) -> Option<Stored<'_>> {
        self.0.view(key).map(Stored::of)
    }
    pub fn contains_key(&self, key: &Key) -> bool {
        self.0.contains(key)
    }
    /// Adds a new key's entry; false if the key is already stored.
    pub fn insert(&mut self, key: Key, entry: Entry) -> bool {
        self.0.insert_view(key, entry.stored().view())
    }
    /// Adds a delta to its key's entry, or stores it as a new entry.
    pub fn merge(&mut self, key: Key, delta: &Entry) {
        self.0.merge(key, delta.stored().view())
    }
    /// Every entry, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (Key, Stored<'_>)> + '_ {
        self.0.views().map(|(k, v)| (k, Stored::of(v)))
    }
    /// Every entry in key order.
    pub fn sorted(&self) -> impl Iterator<Item = (Key, Stored<'_>)> + '_ {
        self.0.sorted_views().map(|(k, v)| (k, Stored::of(v)))
    }
    pub fn keys(&self) -> impl Iterator<Item = Key> + '_ {
        self.iter().map(|(k, _)| k)
    }
    pub fn values(&self) -> impl Iterator<Item = Stored<'_>> + '_ {
        self.iter().map(|(_, v)| v)
    }
    pub fn allocation(&self) -> Allocation {
        self.0.allocation()
    }
    /// Bytes of stored regrets and stored averages.
    pub fn payload_bytes(&self) -> (usize, usize) {
        self.0.payload_bytes()
    }
    pub fn index_capacity(&self) -> usize {
        self.0.index_capacity()
    }
}
impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.sorted().eq(other.sorted())
    }
}
impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.sorted()).finish()
    }
}

fn sample_policy<R: Rng>(rng: &mut R, policy: &[f64]) -> usize {
    let draw: f64 = rng.gen();
    let mut cumulative = 0.;
    for (i, &p) in policy.iter().enumerate() {
        cumulative += p;
        if draw < cumulative {
            return i;
        }
    }
    // Rounded cumulative probabilities can end just below one. The fallback
    // must still have positive mass; the last legal action may have zero regret.
    policy
        .iter()
        .rposition(|&p| p > 0.)
        .expect("empty policy support")
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub nodes: u64,
    pub terminals: u64,
    pub decisions: [u64; 4],
    pub missing: [u64; 4],
    pub traverser: [u64; 6],
    pub discarded_keys: u64,
    pub discarded_visits: u64,
}

/// Immutable snapshot until *all* traversers finish: no branch or seat observes
/// an earlier branch's update. Deltas are accumulated in deterministic DFS order.
pub struct Traversal<'a, R: Rng> {
    pub table: &'a Table,
    pub deltas: &'a mut Deltas,
    pub rng: &'a mut R,
    pub traverser: usize,
    pub weight: f64,
    pub last_average_street: usize,
    pub counts: &'a mut Counts,
}
impl<R: Rng> Traversal<'_, R> {
    pub fn run<S: State>(&mut self, state: &mut S) -> f64 {
        self.counts.nodes += 1;
        let Some(d) = state.decision() else {
            self.counts.terminals += 1;
            let payoff = state.payoff(self.traverser);
            assert!(payoff.is_finite(), "non-finite terminal utility");
            return payoff;
        };
        assert!(d.actor < state.players() && self.traverser < state.players());
        self.counts.decisions[d.street] += 1;
        let n = d.len();
        let policy = if let Some(entry) = self.table.get(&d.key) {
            entry.check(d);
            entry.policy(false)
        } else {
            self.counts.missing[d.street] += 1;
            [1. / n as f64; MAX_ACTIONS]
        };
        if d.actor != self.traverser {
            // For N>2 this retains the requested opponent-sampled estimator,
            // whose visitation also includes other sampled players' reach.
            if d.street <= self.last_average_street {
                let delta = self
                    .deltas
                    .entry(d.key)
                    .or_insert_with(|| Entry::new(d, self.last_average_street));
                delta.check(d);
                for (sum, p) in delta.average.as_mut().unwrap().iter_mut().zip(policy) {
                    *sum += self.weight * p;
                }
            }
            let action = sample_policy(self.rng, &policy[..n]);
            let saved = state.save();
            state.apply(action);
            let value = self.run(state);
            state.restore(saved);
            return value;
        }
        self.counts.traverser[d.actor] += 1;
        let mut values = [0.; MAX_ACTIONS];
        for (a, value) in values[..n].iter_mut().enumerate() {
            let saved = state.save();
            state.apply(a);
            *value = self.run(state);
            state.restore(saved);
        }
        let value = fsum((0..n).map(|a| policy[a] * values[a]));
        let delta = self
            .deltas
            .entry(d.key)
            .or_insert_with(|| Entry::new(d, self.last_average_street));
        delta.check(d);
        for (a, r) in delta.regrets.iter_mut().enumerate() {
            *r += self.weight * (values[a] - value);
        }
        delta.visits += 1;
        value
    }
}

#[cfg(test)]
mod sampling_tests {
    #[test]
    fn rounding_cannot_sample_a_zero_probability_action() {
        let mut rng = rand::rngs::mock::StepRng::new(u64::MAX, 0);
        let mut policy = [1. / 6.; 7];
        policy[6] = 0.;
        assert_eq!(super::sample_policy(&mut rng, &policy), 5);
    }
}

#[derive(Clone, Debug)]
pub struct Trainer {
    pub identity: serde_json::Value,
    pub players: usize,
    pub seed: u64,
    pub iteration: u64,
    pub last_average_street: usize,
    pub max_entries: usize,
    pub freeze_at_cap: bool,
    pub frozen_at: Option<u64>,
    pub table: Table,
    pub counts: Counts,
}
impl Trainer {
    pub fn new<G: Game>(
        game: &G,
        seed: u64,
        last_average_street: usize,
        max_entries: usize,
        freeze_at_cap: bool,
    ) -> Self {
        assert!((2..=6).contains(&game.players()) && last_average_street < 4 && max_entries > 0);
        Self {
            identity: game.identity(),
            players: game.players(),
            seed,
            iteration: 0,
            last_average_street,
            max_entries,
            freeze_at_cap,
            frozen_at: None,
            table: Table::default(),
            counts: Counts::default(),
        }
    }
    pub fn step<G: Game>(&mut self, game: &G) {
        assert_eq!(game.identity(), self.identity);
        assert_eq!(game.players(), self.players);
        assert!(
            self.freeze_at_cap || self.table.len() < self.max_entries,
            "table cap reached"
        );
        self.iteration = self.iteration.checked_add(1).expect("iteration overflow");
        // Exact integer linear weights; checkpoint boundaries never split a sweep.
        assert!(
            self.iteration <= 1 << 53,
            "linear weight no longer exactly representable"
        );
        let mut deltas = Deltas::default();
        for player in 0..self.players {
            // Domain-separated deterministic root streams need no opaque RNG state
            // in checkpoints and do not depend on thread scheduling or hash order.
            use sha2::{Digest, Sha256};
            let mut hash = Sha256::new();
            hash.update(b"nplayer-external-sampling-v1");
            hash.update(self.seed.to_le_bytes());
            hash.update(self.iteration.to_le_bytes());
            hash.update((player as u64).to_le_bytes());
            let mut rng = rand::rngs::StdRng::from_seed(hash.finalize().into());
            let mut state = game.sample(&mut rng);
            assert_eq!(state.players(), self.players);
            Traversal {
                table: &self.table,
                deltas: &mut deltas,
                rng: &mut rng,
                traverser: player,
                weight: self.iteration as f64,
                last_average_street: self.last_average_street,
                counts: &mut self.counts,
            }
            .run(&mut state);
        }
        for (key, delta) in deltas {
            if self.frozen_at.is_some() && !self.table.contains_key(&key) {
                self.counts.discarded_keys += 1;
                self.counts.discarded_visits += delta.visits;
                continue;
            }
            self.table.merge(key, &delta);
        }
        if self.freeze_at_cap && self.frozen_at.is_none() && self.table.len() >= self.max_entries {
            self.frozen_at = Some(self.iteration);
        }
    }
}
