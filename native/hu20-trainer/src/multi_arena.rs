//! Three-seat duplicate schedule and observation-only scripted controls.
use crate::{
    game::Kind,
    multi_holdem::menu,
    nplayer::{Action, Observation},
};
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};

#[cfg(test)]
#[path = "multi_arena_tests.rs"]
mod tests;

/// Each labeled player receives every fixed physical hand/position twice.
/// The two copies of B have distinct identities and independent action streams.
pub const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
pub fn stream_seed(root: u64, domain: &[u8], block: u64, identity: u64) -> u64 {
    let mut hash = Sha256::new();
    hash.update(b"nplayer-duplicate-v1");
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    hash.update(root.to_le_bytes());
    hash.update(block.to_le_bytes());
    hash.update(identity.to_le_bytes());
    u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap())
}

pub struct Bot {
    name: String,
    rng: rand::rngs::StdRng,
}
impl Bot {
    pub fn new(name: &str, seed: u64) -> Result<Self, String> {
        if !["random", "check-call", "tight-aggressive", "pot-pressure"].contains(&name) {
            return Err("unknown scripted bot".into());
        }
        Ok(Self {
            name: name.into(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        })
    }
    pub fn choose(&mut self, view: &Observation<'_>) -> Result<Action, String> {
        if view.actor != Some(view.seat) || view.legal.is_none() {
            return Err("bot needs its acting-seat observation".into());
        }
        let choices = menu(view);
        if self.name == "random" {
            return Ok(choices[self.rng.gen_range(0..choices.len())].1);
        }
        let legal = view.legal.unwrap();
        let passive = Action {
            kind: if legal.check { Kind::Check } else { Kind::Call },
            raise_to: 0,
        };
        if self.name == "check-call" {
            return Ok(passive);
        }
        let (continue_at, raise_at, raise_frequency, bluff_frequency) =
            if self.name == "tight-aggressive" {
                (0.51, 0.67, 0.85, 0.04)
            } else {
                (0.40, 0.60, 0.95, 0.20)
            };
        let score = hand_score(view);
        let pot: u64 = view.players.iter().map(|p| p.contributed).sum();
        let price = legal.call_amount as f64 / (pot + legal.call_amount).max(1) as f64;
        let aggressive: f64 = self.rng.gen();
        let bluff: f64 = self.rng.gen();
        if let Some((low, high)) = legal.raise_to {
            if (score >= raise_at && aggressive < raise_frequency)
                || (price < 0.25 && bluff < bluff_frequency)
            {
                // Matched-pot pressure keeps the first campaign on the training menu.
                let matched = view.players[view.seat].street_bet + legal.call_amount;
                return Ok(Action {
                    kind: Kind::Raise,
                    raise_to: (matched + pot + legal.call_amount).clamp(low, high),
                });
            }
        }
        if legal.check || score >= continue_at + 0.35 * price {
            Ok(passive)
        } else {
            Ok(Action {
                kind: Kind::Fold,
                raise_to: 0,
            })
        }
    }
}

/// Port of arena/heuristics.py's public card score, not an equity estimate.
fn hand_score(view: &Observation<'_>) -> f64 {
    let hi = (view.hole[0] / 4).max(view.hole[1] / 4) as f64 + 2.;
    let lo = (view.hole[0] / 4).min(view.hole[1] / 4) as f64 + 2.;
    if view.board.is_empty() {
        if hi == lo {
            return 0.55 + hi / 35.;
        }
        return (0.95f64).min(
            (hi + lo - 4.) / 40.
                + if view.hole[0] % 4 == view.hole[1] % 4 {
                    0.08
                } else {
                    0.
                }
                + if hi - lo <= 2. { 0.06 } else { 0. },
        );
    }
    let mut cards = view.board.to_vec();
    cards.extend_from_slice(&view.hole);
    let value = hu20_buckets::evaluate(&cards);
    if view.board.len() == 5 && value == hu20_buckets::evaluate(view.board) {
        return 0.20;
    }
    let category = (value >> 20) as usize;
    let shift = [16, 12, 8, 8, 0, 16, 4, 4, 0][category];
    let high_rank = ((value >> shift) & 15) + 2;
    (1.0f64).min(
        [0.12, 0.40, 0.62, 0.74, 0.82, 0.88, 0.93, 0.97, 1.0][category] + high_rank as f64 / 140.,
    )
}
