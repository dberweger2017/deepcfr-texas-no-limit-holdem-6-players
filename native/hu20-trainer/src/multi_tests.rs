use crate::multi_cfr::*;
use crate::small_exact::Tree;
use crate::small_games::{published_kuhn_policy, Kind, SmallGame};
use rand::RngCore;

#[test]
fn published_kuhn_family_and_an_exploitable_control() {
    let game = SmallGame::new(Kind::Kuhn, 3);
    let tree = Tree::new(game, 1000);
    assert_eq!((tree.nodes(), tree.infos.len()), (600, 48));
    for beta in [0., 0.0625, 0.125, 0.25] {
        let profile = tree
            .infos
            .iter()
            .map(|i| published_kuhn_policy(&i.example, beta))
            .collect::<Vec<_>>();
        let e = tree.evaluate(&profile);
        assert!(e.gains.iter().all(|g| g.abs() < 1e-12));
        for (a, b) in e
            .values
            .iter()
            .zip([-(0.5 + beta) / 24., -0.5 / 24., (1. + beta) / 24.])
        {
            assert!((a - b).abs() < 1e-12);
        }
    }
    let e = tree.evaluate(&tree.profile(&Table::default(), false));
    assert!(e.gains.iter().all(|g| *g > 0.1));
}

#[test]
fn all_player_counts_restore_and_conserve() {
    for n in 2..=6 {
        for kind in [Kind::Kuhn, Kind::Leduc] {
            let game = SmallGame::new(kind, n);
            for seed in 0..20 {
                use rand::SeedableRng;
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut hand = game.sample(&mut rng);
                while let Some(d) = hand.decision() {
                    for a in 0..d.len() {
                        let saved = hand.save();
                        hand.apply(a);
                        let child = hand;
                        hand.restore(saved);
                        assert_eq!(hand, saved);
                        hand.apply(a);
                        assert_eq!(hand, child);
                        hand.restore(saved);
                    }
                    use rand::Rng;
                    hand.apply(rng.gen_range(0..d.len()));
                }
                assert!((0..n).map(|p| hand.payoff(p)).sum::<f64>().abs() < 1e-12);
            }
        }
    }
}

#[test]
fn hidden_private_and_future_public_cards_never_enter_small_keys() {
    let game = SmallGame::new(Kind::Leduc, 3);
    let mut a = game.from_cards(&[0, 1, 2, 3]);
    let mut b = game.from_cards(&[0, 3, 1, 2]);
    assert_eq!(a.decision(), b.decision());
    for _ in 0..3 {
        a.play(1);
        b.play(1);
    }
    assert_ne!(a.decision(), b.decision(), "board is now public");
    let mut c = game.from_cards(&[0, 2, 1, 3]);
    for _ in 0..3 {
        c.play(1);
    }
    assert_eq!(
        a.decision(),
        c.decision(),
        "opponents remain private after the reveal"
    );
}

#[test]
fn resume_matches_complete_bytes_and_frozen_membership() {
    let game = SmallGame::new(Kind::Leduc, 3);
    for freeze in [false, true] {
        for street in [0, 3] {
            let cap = if freeze { 30 } else { usize::MAX };
            let mut fresh = Trainer::new(&game, 71, street, cap, freeze);
            let mut prefix = fresh.clone();
            for _ in 0..20 {
                prefix.step(&game);
            }
            let path = std::env::temp_dir()
                .join(format!("multi-{}-{freeze}-{street}.gz", std::process::id()));
            crate::multi_checkpoint::save(&prefix, &path).unwrap();
            let mut resumed = crate::multi_checkpoint::load(&path).unwrap();
            let frozen_keys: std::collections::HashSet<_> = resumed.table.keys().collect();
            for _ in 0..40 {
                fresh.step(&game);
            }
            for _ in 20..40 {
                resumed.step(&game);
            }
            assert_eq!(fresh.table, resumed.table);
            assert_eq!(fresh.counts, resumed.counts);
            crate::multi_checkpoint::save(&fresh, &path).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            crate::multi_checkpoint::save(&resumed, &path).unwrap();
            assert_eq!(bytes, std::fs::read(&path).unwrap());
            std::fs::remove_file(&path).unwrap();
            assert!(fresh.counts.traverser[..3].iter().all(|&n| n > 0));
            if freeze {
                assert_eq!(frozen_keys, fresh.table.keys().collect());
                assert!(fresh.counts.discarded_keys > 0 && fresh.counts.discarded_visits > 0);
            }
            for e in fresh.table.values() {
                assert_eq!(e.average.is_some(), e.street as usize <= street);
            }
        }
    }
}

struct Tape {
    bits: usize,
    used: usize,
}
impl RngCore for Tape {
    fn next_u64(&mut self) -> u64 {
        assert!(self.used < 8, "increase exact sampling tape support");
        let bit = (self.bits >> self.used) & 1;
        self.used += 1;
        if bit == 0 {
            0
        } else {
            u64::MAX
        }
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn fill_bytes(&mut self, _: &mut [u8]) {
        unreachable!()
    }
    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), rand::Error> {
        unreachable!()
    }
}

#[test]
fn exhaustive_sample_expectation_equals_full_tree_counterfactual_updates() {
    fn full(
        hand: crate::small_games::Hand,
        player: usize,
        reach: f64,
        expected: &mut Deltas,
    ) -> f64 {
        let Some(d) = hand.decision() else {
            return hand.payoff(player);
        };
        let mut values = Vec::new();
        for a in 0..d.len() {
            let mut child = hand;
            child.apply(a);
            values.push(full(
                child,
                player,
                reach * if d.actor == player { 1. } else { 0.5 },
                expected,
            ));
        }
        let v = values.iter().sum::<f64>() / 2.;
        let e = expected.entry(d.key).or_insert_with(|| Entry::new(d, 3));
        if d.actor == player {
            for (r, value) in e.regrets.iter_mut().zip(values) {
                *r += 3. * reach * (value - v);
            }
        } else {
            for a in e.average.as_mut().unwrap().iter_mut() {
                *a += 3. * reach * 0.5;
            }
        }
        v
    }
    let game = SmallGame::new(Kind::Kuhn, 3);
    for player in 0..3 {
        let mut expected = Deltas::default();
        let mut actual: HashMap<[u8; 16], ([f64; 2], [f64; 2])> = HashMap::new();
        for (hand, p) in game.deals() {
            full(hand, player, p, &mut expected);
            for bits in 0..256 {
                let mut state = hand;
                let mut deltas = Deltas::default();
                let mut counts = Counts::default();
                let mut rng = Tape { bits, used: 0 };
                Traversal {
                    table: &Table::default(),
                    deltas: &mut deltas,
                    rng: &mut rng,
                    traverser: player,
                    weight: 3.,
                    last_average_street: 3,
                    counts: &mut counts,
                }
                .run(&mut state);
                assert_eq!(state, hand);
                for (key, e) in deltas {
                    let a = actual.entry(key).or_insert(([0.; 2], [0.; 2]));
                    for i in 0..2 {
                        a.0[i] += p * e.regrets[i] / 256.;
                        a.1[i] += p * e.average.as_ref().unwrap()[i] / 256.;
                    }
                }
            }
        }
        for (key, e) in expected {
            let (regrets, average) = actual[&key];
            for i in 0..2 {
                assert!((regrets[i] - e.regrets[i]).abs() < 1e-11);
                assert!((average[i] - e.average.as_ref().unwrap()[i]).abs() < 1e-11);
            }
        }
    }
}
use std::collections::HashMap;

#[test]
fn corrupt_checkpoint_metadata_and_accumulators_are_rejected() {
    use std::io::{Read, Write};
    let game = SmallGame::new(Kind::Kuhn, 3);
    let mut trainer = Trainer::new(&game, 23, 3, usize::MAX, false);
    for _ in 0..5 {
        trainer.step(&game);
    }
    let path = std::env::temp_dir().join(format!("multi-invalid-{}.gz", std::process::id()));
    crate::multi_checkpoint::save(&trainer, &path).unwrap();
    let mut plain = String::new();
    flate2::read::MultiGzDecoder::new(std::fs::File::open(&path).unwrap())
        .read_to_string(&mut plain)
        .unwrap();
    let original: Vec<serde_json::Value> = plain
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for case in 0..8 {
        let mut records = original.clone();
        match case {
            0 => records[0]["players"] = serde_json::json!(2),
            1 => records[0]["counts"]["nodes"] = serde_json::json!(0),
            2 => records[1][1] = serde_json::json!(6),
            3 => records[1][3] = serde_json::json!(0),
            4 => records[1][5] = serde_json::json!([null, null]),
            5 => records[1][6] = serde_json::json!([-1., 1.]),
            6 => records.push(records[1].clone()),
            7 => records[0]["frozen_at"] = serde_json::json!(1),
            _ => unreachable!(),
        }
        let mut out = flate2::write::GzEncoder::new(
            std::fs::File::create(&path).unwrap(),
            flate2::Compression::default(),
        );
        for record in records {
            writeln!(out, "{record}").unwrap();
        }
        out.finish().unwrap();
        assert!(
            crate::multi_checkpoint::load(&path).is_err(),
            "corruption case {case}"
        );
    }
    std::fs::remove_file(path).unwrap();
}
