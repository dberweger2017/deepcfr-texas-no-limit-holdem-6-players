//! Observation-only instrumentation around the reviewed traversal's State API.
#[cfg(test)]
use super::memory::accounted_live;
#[cfg(not(test))]
use super::memory::live as accounted_live;
use hu20_trainer::{
    cfr::{Key, KeyHasher},
    multi_cfr::{Decision, Game, State},
};
use std::{cell::RefCell, collections::HashMap, hash::BuildHasherDefault, rc::Rc};
#[derive(Default)]
pub struct Survey {
    pub keys: HashMap<Key, (u8, u8, u16), BuildHasherDefault<KeyHasher>>,
    pub by_street: [u64; 4],
    pub decisions: [u64; 4],
    pub allocated: usize,
}
impl Survey {
    fn record(&mut self, d: Decision) {
        let before = accounted_live();
        let meta = (d.owner as u8, d.street as u8, d.code);
        let old = self.keys.insert(d.key, meta);
        self.allocated += usize::try_from(accounted_live() - before).unwrap();
        self.decisions[d.street] += 1;
        if let Some(old) = old {
            assert_eq!(old, meta);
        } else {
            self.by_street[d.street] += 1;
        }
    }
}
pub struct Observed<G: Game> {
    pub game: G,
    pub survey: Rc<RefCell<Survey>>,
}
pub struct ObservedState<S: State> {
    state: S,
    survey: Rc<RefCell<Survey>>,
}
impl<G: Game> Game for Observed<G> {
    type State = ObservedState<G::State>;
    fn players(&self) -> usize {
        self.game.players()
    }
    fn identity(&self) -> serde_json::Value {
        self.game.identity()
    }
    fn sample(&self, rng: &mut rand::rngs::StdRng) -> Self::State {
        ObservedState {
            state: self.game.sample(rng),
            survey: self.survey.clone(),
        }
    }
}
impl<S: State> State for ObservedState<S> {
    type Saved = S::Saved;
    fn players(&self) -> usize {
        self.state.players()
    }
    fn decision(&self) -> Option<Decision> {
        let d = self.state.decision();
        if let Some(d) = d {
            self.survey.borrow_mut().record(d);
        }
        d
    }
    fn payoff(&self, p: usize) -> f64 {
        self.state.payoff(p)
    }
    fn save(&self) -> Self::Saved {
        self.state.save()
    }
    fn restore(&mut self, s: Self::Saved) {
        self.state.restore(s);
    }
    fn apply(&mut self, a: usize) {
        self.state.apply(a);
    }
}
