use super::*;
use hu20_buckets::parse_card;

fn hand(stacks: &[u64], button: usize) -> Hand {
    Hand::from_deck(
        Config::new(stacks, 1, 2).unwrap(),
        button,
        &(0..52).collect::<Vec<_>>(),
    )
    .unwrap()
}
fn act(h: &mut Hand, kind: Kind, raise_to: u64) {
    h.apply_mut(Action { kind, raise_to }).unwrap();
}
fn jam(h: &mut Hand) {
    while let Some(l) = h.legal() {
        match l.raise_to {
            Some((_, hi)) => act(h, Kind::Raise, hi),
            None => act(h, if l.check { Kind::Check } else { Kind::Call }, 0),
        }
    }
}
fn rigged(holes: &[[&str; 2]], board: &[&str; 5], stacks: &[u64], button: usize) -> Hand {
    let n = stacks.len();
    let mut deck = Vec::new();
    for round in 0..2 {
        for offset in 1..=n {
            deck.push(parse_card(holes[(button + offset) % n][round]));
        }
    }
    deck.extend(board.iter().map(|c| parse_card(c)));
    for c in 0..52 {
        if !deck.contains(&c) {
            deck.push(c);
        }
    }
    Hand::from_deck(Config::new(stacks, 1, 2).unwrap(), button, &deck).unwrap()
}

#[test]
fn order_blinds_and_big_blind_option_for_every_size_and_button() {
    for n in 2..=MAX_PLAYERS {
        for button in 0..n {
            let mut h = hand(&vec![100; n], button);
            let sb = if n == 2 { button } else { (button + 1) % n };
            let bb = (sb + 1) % n;
            assert_eq!(h.actor(), Some((bb + 1) % n));
            assert_eq!(h.players()[sb].street_bet, 1);
            for _ in 0..n - 1 {
                act(&mut h, Kind::Call, 0);
            }
            assert_eq!(h.actor(), Some(bb));
            assert!(h.legal().unwrap().check);
            act(&mut h, Kind::Check, 0);
            assert_eq!(h.street(), Street::Flop);
            assert_eq!(h.actor(), Some((button + 1) % n));
            assert_eq!(h.pot(), 2 * n as u64);
        }
    }
}

#[test]
fn reopening_is_per_player_and_cumulative_not_the_pending_bit() {
    for (stack0, reopened) in [(200, false), (18, true)] {
        let mut h = hand(&[stack0, 200, 200, 200, 200, 14], 0);
        act(&mut h, Kind::Raise, 10);
        act(&mut h, Kind::Call, 0);
        act(&mut h, Kind::Raise, 14);
        if reopened {
            act(&mut h, Kind::Raise, 18);
        } else {
            act(&mut h, Kind::Call, 0);
        }
        for _ in 0..2 {
            act(&mut h, Kind::Call, 0);
        }
        assert_eq!(h.actor(), Some(3));
        assert_eq!(h.legal().unwrap().raise_to.is_some(), reopened);
        assert_eq!(h.min_raise(), 8);
    }
    let mut h = hand(&[79, 192, 172, 11, 100], 2);
    act(&mut h, Kind::Raise, 79);
    for _ in 0..3 {
        act(&mut h, Kind::Call, 0);
    }
    act(&mut h, Kind::Raise, 100);
    assert_eq!(h.actor(), Some(1));
    assert_eq!(h.legal().unwrap().raise_to, None);
}

#[test]
fn checking_preserves_raise_right_against_a_short_opening_all_in() {
    let mut h = hand(&[48, 12, 5], 2);
    act(&mut h, Kind::Raise, 4);
    for _ in 0..2 {
        act(&mut h, Kind::Call, 0);
    }
    for _ in 0..2 {
        act(&mut h, Kind::Check, 0);
    }
    act(&mut h, Kind::Raise, 1);
    assert_eq!(h.actor(), Some(0));
    assert_eq!(h.legal().unwrap().raise_to, Some((3, 44)));
}

#[test]
fn short_blinds_preserve_bring_in_only_while_it_is_contestable() {
    let mut h = hand(&[70, 1, 186], 2);
    assert_eq!(h.wager(), 2);
    act(&mut h, Kind::Call, 0);
    act(&mut h, Kind::Call, 0);
    assert_eq!(h.street(), Street::Flop);
    assert_eq!(h.pot(), 5);
    let mut h = hand(&[70, 1, 186], 2);
    act(&mut h, Kind::Fold, 0);
    assert!(h.finished());
    assert_eq!(h.board().len(), 5);
    let h = hand(&[1, 1, 1], 0);
    assert!(!h.finished());
    assert_eq!(h.legal().unwrap().call_amount, 1);
    let h = hand(&[1, 1], 0);
    assert!(h.finished());
    assert_eq!(h.board().len(), 5);
}

#[test]
fn side_pots_refunds_and_odd_chips_have_fixed_expected_payouts() {
    let board = ["2c", "3d", "7h", "8s", "9c"];
    let mut h = rigged(
        &[["Ac", "Ad"], ["Kc", "Kd"], ["Qc", "Qd"]],
        &board,
        &[50, 100, 200],
        2,
    );
    jam(&mut h);
    assert_eq!(
        h.players().iter().map(|p| p.stack).collect::<Vec<_>>(),
        [150, 100, 100]
    );
    assert_eq!(h.settlement().unwrap().refunds[..3], [0, 0, 100]);
    assert_eq!(
        h.settlement()
            .unwrap()
            .pots()
            .iter()
            .map(|p| p.amount)
            .collect::<Vec<_>>(),
        [150, 100]
    );
    let mut h = rigged(
        &[["Ac", "Ad"], ["Ah", "As"], ["Kc", "Kd"]],
        &board,
        &[5, 5, 5],
        2,
    );
    jam(&mut h);
    assert_eq!(
        h.players().iter().map(|p| p.stack).collect::<Vec<_>>(),
        [8, 7, 0]
    );
    assert_eq!(h.pot(), 0);
    assert!(h
        .players()
        .iter()
        .all(|p| p.street_bet == 0 && p.contributed == 0));
}

#[test]
fn folded_contribution_levels_do_not_create_extra_rounding_events() {
    // Two blind folds leave three tied winners. One 18-chip pot pays 6/6/6;
    // independently rounding the 5/4/9-chip contribution layers would pay 7/6/5.
    let mut h = rigged(
        &[
            ["2c", "3c"],
            ["2d", "3d"],
            ["2h", "3h"],
            ["2s", "3s"],
            ["4c", "5c"],
        ],
        &["Tc", "Jc", "Qc", "Kc", "Ac"],
        &[5; 5],
        0,
    );
    act(&mut h, Kind::Raise, 5); // seat 3
    act(&mut h, Kind::Call, 0); // seat 4
    act(&mut h, Kind::Call, 0); // seat 0
    act(&mut h, Kind::Fold, 0); // seat 1, dead SB 1
    act(&mut h, Kind::Fold, 0); // seat 2, dead BB 2
    assert_eq!(h.settlement().unwrap().pots().len(), 1);
    assert_eq!(h.settlement().unwrap().pots()[0].amount, 18);
    assert_eq!(
        h.players().iter().map(|p| p.stack).collect::<Vec<_>>(),
        [6, 4, 3, 6, 6]
    );
}

#[test]
fn fold_win_does_not_reveal_a_board_and_invalid_actions_do_not_mutate() {
    let mut h = hand(&[100; 3], 0);
    for action in [
        Action {
            kind: Kind::Check,
            raise_to: 0,
        },
        Action {
            kind: Kind::Raise,
            raise_to: 3,
        },
        Action {
            kind: Kind::Raise,
            raise_to: 101,
        },
        Action {
            kind: Kind::Call,
            raise_to: 1,
        },
    ] {
        let before = h.clone();
        assert!(h.apply_mut(action).is_err());
        assert_eq!(h, before);
    }
    act(&mut h, Kind::Fold, 0);
    act(&mut h, Kind::Fold, 0);
    assert!(h.finished());
    assert!(h.board().is_empty());
    assert_eq!(
        h.players().iter().map(|p| p.stack).collect::<Vec<_>>(),
        [100, 99, 101]
    );
    let before = h.clone();
    assert!(h
        .apply_mut(Action {
            kind: Kind::Fold,
            raise_to: 0
        })
        .is_err());
    assert_eq!(h, before);
}

#[test]
fn every_branch_restores_accounting_and_history_for_all_table_sizes() {
    for n in 2..=MAX_PLAYERS {
        let mut h = hand(&vec![100; n], n - 1);
        for _ in 0..100 {
            let Some(l) = h.legal() else { break };
            let parent = h.clone();
            let mut actions = vec![
                Action {
                    kind: Kind::Fold,
                    raise_to: 0,
                },
                Action {
                    kind: if l.check { Kind::Check } else { Kind::Call },
                    raise_to: 0,
                },
            ];
            if let Some((lo, hi)) = l.raise_to {
                actions.extend([
                    Action {
                        kind: Kind::Raise,
                        raise_to: lo,
                    },
                    Action {
                        kind: Kind::Raise,
                        raise_to: hi,
                    },
                ]);
            }
            for action in actions {
                let saved = h.save();
                h.apply_mut(action).unwrap();
                assert_eq!(
                    h.players()
                        .iter()
                        .map(|p| p.stack + p.contributed)
                        .sum::<u64>(),
                    100 * n as u64
                );
                h.restore(saved);
                assert_eq!(h, parent);
            }
            act(&mut h, if l.check { Kind::Check } else { Kind::Call }, 0);
        }
        assert!(h.finished());
    }
}

#[test]
fn observations_hide_opponents_and_future_cards_at_each_street() {
    for n in 2..=MAX_PLAYERS {
        let deck: Vec<_> = (0..52).collect();
        let cfg = Config::new(&vec![100; n], 1, 2).unwrap();
        for owner in 0..n {
            for dealt in [0, 3, 4, 5] {
                let mut other = deck.clone();
                let opponent = (owner + 1) % n;
                other.swap((opponent + n - 1) % n, 51);
                if dealt < 5 {
                    other.swap(2 * n + dealt, 50);
                }
                let mut a = Hand::from_deck(cfg, 0, &deck).unwrap();
                let mut b = Hand::from_deck(cfg, 0, &other).unwrap();
                while a.board().len() < dealt {
                    let kind = if a.legal().unwrap().check {
                        Kind::Check
                    } else {
                        Kind::Call
                    };
                    act(&mut a, kind, 0);
                    act(&mut b, kind, 0);
                }
                assert_eq!(a.observe(owner).unwrap(), b.observe(owner).unwrap());
                if a.actor() != Some(owner) {
                    assert!(a.observe(owner).unwrap().legal.is_none());
                }
            }
        }
    }
}

#[test]
fn configuration_and_deck_validation_cover_engine_integer_limits() {
    for stacks in [vec![1], vec![1; 7], vec![0, 1], vec![MAX_CHIPS + 1, 1]] {
        assert!(Config::new(&stacks, 1, 2).is_err());
    }
    assert!(Config::new(&[100; 3], 0, 2).is_err());
    assert!(Config::new(&[100; 3], 3, 2).is_err());
    assert!(Config::new(&[100; 3], 1, MAX_CHIPS + 1).is_err());
    let cfg = Config::new(&[MAX_CHIPS; 6], 1, MAX_CHIPS).unwrap();
    let mut deck: Vec<_> = (0..52).collect();
    assert!(Hand::from_deck(cfg, 6, &deck).is_err());
    assert!(Hand::from_deck(cfg, 0, &deck[..51]).is_err());
    let mut h = Hand::from_deck(cfg, 0, &deck).unwrap();
    jam(&mut h);
    assert_eq!(
        h.players().iter().map(|p| p.stack).sum::<u64>(),
        6 * MAX_CHIPS
    );
    deck[0] = deck[1];
    assert!(Hand::from_deck(cfg, 0, &deck).is_err());
}
