//! #162's fixed-turn-root bench (`src/diagnostics/subgame_bench.py`), reproduced exactly.
//!
//! Both players start at a #149 limped turn root holding cards drawn from that root's frozen
//! ranges, and the production traversal trains the v1 keys from there. Each traversal seeds
//! CPython's `Random(f"{seed}/{iteration}/{traverser}")`, which picks the root, draws the
//! holdings, shuffles the river and then samples the opponent, so a native run equals
//! `SubgameTrainer` with the same seed and roots.

use crate::cfr::{fsum, regret_match, AverageRule, Node};
use crate::game::{Action, Hand, Kind};
use crate::streams::Mt;
use crate::trainer::Trainer;
use serde_json::{json, Value};

/// One public turn root with the exact ranges its #149 equilibrium used.
pub struct FrozenRoot {
    pub spot: String,
    button: u8,
    board: [u8; 4],
    actions: Vec<(usize, Action)>,
    hands: [Vec<[u8; 2]>; 2],
    /// Running weight sums as Python's `accumulate` made them, passed through unchanged.
    cumulative: [Vec<f64>; 2],
}

fn card(value: &Value) -> u8 {
    hu20_buckets::parse_card(value.as_str().unwrap())
}

impl FrozenRoot {
    /// A root as `FrozenRoot.to_native()` writes it.
    pub fn from_json(root: &Value) -> FrozenRoot {
        let kind = |name: &str| match name {
            "fold" => Kind::Fold,
            "check" => Kind::Check,
            "call" => Kind::Call,
            "raise" => Kind::Raise,
            other => panic!("unknown action kind {other}"),
        };
        let actions = root["actions"].as_array().unwrap().iter().map(|a| {
            let raise_to = a[2].as_u64().unwrap_or(0) as u32;
            (a[0].as_u64().unwrap() as usize, Action { kind: kind(a[1].as_str().unwrap()), raise_to })
        }).collect();
        let seat = |field: &str, s: usize| root[field][s].as_array().unwrap().clone();
        let hands = [0, 1].map(|s| seat("hands", s).iter().map(|h| [card(&h[0]), card(&h[1])]).collect());
        let cumulative = [0, 1].map(|s| seat("cumulative", s).iter().map(|c| c.as_f64().unwrap()).collect::<Vec<_>>());
        let board: Vec<u8> = root["board"].as_array().unwrap().iter().map(card).collect();
        for s in 0..2 {
            let (h, c): (&Vec<[u8; 2]>, &Vec<f64>) = (&hands[s], &cumulative[s]);
            assert!(!h.is_empty() && h.len() == c.len() && *c.last().unwrap() > 0.0, "invalid frozen range");
        }
        FrozenRoot {
            spot: root["spot"].as_str().unwrap().into(),
            button: root["button"].as_u64().unwrap() as u8,
            board: board.try_into().expect("a turn root has four board cards"),
            actions,
            hands,
            cumulative,
        }
    }

    /// Joint law proportional to w0(h0) w1(h1) for card-disjoint pairs.
    pub fn sample_holdings(&self, random: &mut Mt) -> [[u8; 2]; 2] {
        loop {
            let mut pair = [[0u8; 2]; 2];
            for seat in 0..2 {
                let cumulative = &self.cumulative[seat];
                let target = random.random() * cumulative.last().unwrap();
                let index = cumulative.partition_point(|&c| c <= target);
                pair[seat] = self.hands[seat][index.min(cumulative.len() - 1)];
            }
            if !pair[0].iter().any(|c| pair[1].contains(c)) {
                return pair;
            }
        }
    }

    /// The engine at this turn root with the given hole cards and a fresh river.
    pub fn hand(&self, holdings: [[u8; 2]; 2], random: &mut Mt) -> Hand {
        let order = [(self.button as usize + 1) % 2, self.button as usize];
        let mut deck: Vec<u8> = (0..2).flat_map(|i| order.map(|seat| holdings[seat][i])).collect();
        deck.extend_from_slice(&self.board);
        // Python's deck is rank-major clubs to spades: card ids in ascending order.
        let mut rest: Vec<u8> = (0..52).filter(|c| !deck.contains(c)).collect();
        random.shuffle(&mut rest);
        deck.extend(rest);
        let mut hand = Hand::from_deck(self.button, &deck);
        for &(seat, action) in &self.actions {
            assert_eq!(hand.actor.map(usize::from), Some(seat), "turn root actor mismatch");
            hand.apply_mut(action);
        }
        hand
    }
}

/// One `SubgameTrainer.step`: a traversal per seat, each from its own seeded stream.
pub fn step(trainer: &mut Trainer, roots: &[FrozenRoot]) -> u64 {
    let seed = trainer.seed;
    trainer.step_with(|iteration, seat, _sample| {
        let mut random = Mt::from_text(&format!("{seed}/{iteration}/{seat}"));
        let root = &roots[random.below(roots.len() as u32) as usize];
        let holdings = root.sample_holdings(&mut random);
        (root.hand(holdings, &mut random), random)
    })
}

/// The strategy names `SubgameTrainer.export` takes.
pub fn strategy(rule: AverageRule) -> &'static str {
    match rule {
        AverageRule::TraverserReach => "average-traverser-reach",
        AverageRule::OpponentSampled => "average-opponent-sampled",
    }
}

/// Python keeps both averages in one table; natively they come from two lockstep trainers.
/// Opponent nodes enter Python's table, and the opponent-sampled trainer's, with zero regrets
/// and visits, so that trainer's keys are the full key set. Panics unless the traverser-reach
/// trainer's regrets and visits are bit-identical to its partner's.
pub fn check_lockstep(reach: &Trainer, sampled: &Trainer) {
    assert_eq!((reach.average, sampled.average), (AverageRule::TraverserReach, AverageRule::OpponentSampled));
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (key, node) in sampled.table.iter() {
        let node = &sampled.caught_up(&node);
        let n = node.len as usize;
        match reach.table.get(&key).map(|r| reach.caught_up(&r)) {
            Some(r) => assert!(r.code == node.code && r.visits == node.visits && bits(&r.regrets[..n]) == bits(&node.regrets[..n]),
                               "the averaging rule changed the regrets"),
            None => assert!(node.visits == 0 && node.regrets[..n].iter().all(|&v| v == 0.0), "a traverser key is missing"),
        }
    }
    assert!(reach.table.iter().all(|(key, _)| sampled.table.get(&key).is_some()), "a traverser key is missing");
}

/// `SubgameTrainer.export`: groups in the #149 pooled-policy format read by the native lock pass.
/// Keys come from the opponent-sampled trainer; values from `source`. `average` exports
/// `source`'s average, otherwise the current policy.
pub fn export(sampled: &Trainer, source: &Trainer, lineage: &Value, average: bool) -> Value {
    let groups: Vec<Value> = sampled.table.sorted().map(|(key, keyed)| {
        let node = source.table.get(&key).map(|n| source.caught_up(&n))
            .unwrap_or_else(|| Node::empty(keyed.code, keyed.len as usize));
        let n = node.len as usize;
        let (p, mass): (Vec<f64>, f64) = if average {
            let mass = fsum(node.average[..n].iter().copied());
            let p = if mass > 0.0 { node.average[..n].iter().map(|v| v / mass).collect() } else { vec![1.0 / n as f64; n] };
            (p, mass)
        } else {
            (regret_match(&node.regrets[..n])[..n].to_vec(), node.visits as f64)
        };
        assert!(p.iter().all(|v| v.is_finite()) && (fsum(p.iter().copied()) - 1.0).abs() <= 2e-5,
                "non-finite or unnormalized exported policy");
        let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
        json!({"lineage": lineage, "metric": "v1", "key": hex, "names": node.names(), "probabilities": p,
               "mass": mass, "roots": node.visits})
    }).collect();
    let mut document = json!({"format": "hu20-board-pooling-policy-v1", "groups": groups, "lineage": lineage,
           "strategy": if average { strategy(source.average) } else { "current" }, "iteration": source.iteration,
           "zero_mass_rule": "uniform within the actual menu"});
    if let Some(label) = source.options.label() {
        document["training_options"] = json!(label);
    }
    document
}
