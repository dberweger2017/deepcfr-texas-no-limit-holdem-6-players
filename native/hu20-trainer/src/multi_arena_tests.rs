use super::*;
use crate::nplayer::{Config, Hand};
use rand::seq::SliceRandom;

#[test]
fn duplicate_rotates_every_identity_into_every_seat_twice() {
    let mut counts = [[0; 3]; 3];
    let mut unique = std::collections::HashSet::new();
    for permutation in PERMUTATIONS {
        assert!(unique.insert(permutation));
        for (seat, identity) in permutation.iter().enumerate() {
            counts[*identity][seat] += 1;
        }
    }
    assert_eq!(counts, [[2; 3]; 3]);
}

#[test]
fn streams_are_domain_separated_and_repeatable() {
    assert_eq!(stream_seed(41, b"deck", 9, 0), stream_seed(41, b"deck", 9, 0));
    let mut seeds = std::collections::HashSet::new();
    for block in 0..10 {
        assert!(seeds.insert(stream_seed(41, b"deck", block, 0)));
        for identity in 0..3 {
            assert!(seeds.insert(stream_seed(41, b"action", block, identity)));
        }
    }
}

#[test]
fn all_scripts_settle_and_ignore_hidden_cards() {
    let config = Config::new(&[2000; 3], 50, 100).unwrap();
    for name in ["random", "check-call", "tight-aggressive", "pot-pressure"] {
        for root in 0..200 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(root);
            let mut deck: Vec<_> = (0..52).collect();
            deck.shuffle(&mut rng);
            let mut hand = Hand::from_deck(config, 0, &deck).unwrap();
            let actor = hand.actor().unwrap();
            let mut alternative = deck.clone();
            // With button zero, seat zero's cards occupy indices 2 and 5.
            assert_eq!(actor, 0);
            alternative.swap(0, 20);
            alternative.swap(6, 21);
            let other = Hand::from_deck(config, 0, &alternative).unwrap();
            let mut a = Bot::new(name, root).unwrap();
            let mut b = Bot::new(name, root).unwrap();
            assert_eq!(a.choose(&hand.observe(actor).unwrap()).unwrap(),
                       b.choose(&other.observe(actor).unwrap()).unwrap());
            let mut bots: Vec<_> = (0..3).map(|i| Bot::new(name, root + i).unwrap()).collect();
            let mut decisions = 0;
            while let Some(actor) = hand.actor() {
                let action = bots[actor].choose(&hand.observe(actor).unwrap()).unwrap();
                hand.apply_mut(action).unwrap();
                decisions += 1;
                assert!(decisions < 200);
            }
            assert_eq!(hand.settlement().unwrap().rewards.iter().sum::<i64>(), 0);
        }
    }
}
