//! Finite perfect-recall poker games for validation of the production traversal.
//! Kuhn follows Szafron/Gibson/Sturtevant (2013); multiplayer Leduc follows
//! OpenSpiel's 2*(N+1)-card, two-raises-per-round convention (see protocol).
use crate::multi_cfr::{Decision, Game, State};
use rand::seq::SliceRandom;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Kuhn,
    Leduc,
}

#[derive(Clone, Copy, Debug)]
pub struct SmallGame {
    pub kind: Kind,
    pub players: usize,
}
impl SmallGame {
    pub fn new(kind: Kind, players: usize) -> Self {
        assert!((2..=6).contains(&players));
        Self { kind, players }
    }
    pub fn from_cards(self, cards: &[u8]) -> Hand {
        let needed = self.players + usize::from(self.kind == Kind::Leduc);
        assert_eq!(cards.len(), needed);
        let ranks = self.players + 1;
        let mut counts = [0; 7];
        for &c in cards {
            assert!((c as usize) < ranks);
            counts[c as usize] += 1;
        }
        assert!(counts
            .iter()
            .all(|&n| n <= if self.kind == Kind::Kuhn { 1 } else { 2 }));
        let mut hand = Hand {
            kind: self.kind,
            n: self.players,
            cards: [0; 7],
            contributed: [0; 6],
            folded: 0,
            pending: (1 << self.players) - 1,
            actor: Some(0),
            round: 0,
            raises: 0,
            wager: 1,
            history: [0; 64],
            history_len: 0,
        };
        hand.cards[..needed].copy_from_slice(cards);
        hand.contributed[..self.players].fill(1);
        hand
    }
    /// Aggregate physically identical rank deals, retaining exact multiplicities.
    pub fn deals(self) -> Vec<(Hand, f64)> {
        fn visit(
            game: SmallGame,
            left: &mut [u8],
            cards: &mut Vec<u8>,
            probability: f64,
            out: &mut Vec<(Hand, f64)>,
        ) {
            if cards.len() == game.players + usize::from(game.kind == Kind::Leduc) {
                out.push((game.from_cards(cards), probability));
                return;
            }
            let total: u8 = left.iter().sum();
            for rank in 0..left.len() {
                let count = left[rank];
                if count == 0 {
                    continue;
                }
                left[rank] -= 1;
                cards.push(rank as u8);
                visit(
                    game,
                    left,
                    cards,
                    probability * f64::from(count) / f64::from(total),
                    out,
                );
                cards.pop();
                left[rank] += 1;
            }
        }
        let mut out = Vec::new();
        visit(
            self,
            &mut vec![if self.kind == Kind::Kuhn { 1 } else { 2 }; self.players + 1],
            &mut Vec::new(),
            1.,
            &mut out,
        );
        out
    }
}
impl Game for SmallGame {
    type State = Hand;
    fn players(&self) -> usize {
        self.players
    }
    fn identity(&self) -> Value {
        json!({"schema":"small-poker-perfect-recall-v1", "players":self.players,
            "game":if self.kind == Kind::Kuhn {"kuhn"} else {"leduc"}})
    }
    fn sample(&self, rng: &mut rand::rngs::StdRng) -> Hand {
        let mut cards: Vec<u8> = (0..=self.players as u8)
            .flat_map(|r| std::iter::repeat(r).take(if self.kind == Kind::Kuhn { 1 } else { 2 }))
            .collect();
        cards.shuffle(rng);
        self.from_cards(&cards[..self.players + usize::from(self.kind == Kind::Leduc)])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hand {
    pub kind: Kind,
    pub n: usize,
    cards: [u8; 7],
    pub contributed: [u8; 6],
    pub folded: u8,
    pending: u8,
    pub actor: Option<usize>,
    pub round: usize,
    raises: u8,
    wager: u8,
    history: [u8; 64],
    history_len: usize,
}
impl Hand {
    pub fn history(&self) -> &[u8] {
        &self.history[..self.history_len]
    }
    pub fn hole(&self, owner: usize) -> u8 {
        assert!(owner < self.n);
        self.cards[owner]
    }
    pub fn board(&self) -> Option<u8> {
        (self.round == 1).then_some(self.cards[self.n])
    }
    pub fn actions(&self) -> Vec<u8> {
        let Some(a) = self.actor else {
            return Vec::new();
        };
        let mut actions = Vec::with_capacity(3);
        if self.contributed[a] < self.wager {
            actions.push(0);
        }
        actions.push(1);
        if self.raises < if self.kind == Kind::Kuhn { 1 } else { 2 } {
            actions.push(2);
        }
        actions
    }
    pub fn payload(&self) -> Vec<u8> {
        let a = self.actor.expect("terminal has no information key");
        let mut out = vec![
            if self.kind == Kind::Kuhn { b'K' } else { b'L' },
            self.n as u8,
            a as u8,
            self.cards[a],
            self.board().unwrap_or(255),
            self.round as u8,
        ];
        out.extend_from_slice(self.history());
        out
    }
    fn push(&mut self, event: u8) {
        self.history[self.history_len] = event;
        self.history_len += 1;
    }
    pub fn play(&mut self, action: u8) {
        assert!(self.actions().contains(&action));
        let a = self.actor.unwrap();
        self.push((a as u8) * 4 + action);
        self.pending &= !(1 << a);
        match action {
            0 => self.folded |= 1 << a,
            1 => self.contributed[a] = self.wager,
            2 => {
                self.wager += if self.kind == Kind::Kuhn {
                    1
                } else if self.round == 0 {
                    2
                } else {
                    4
                };
                self.contributed[a] = self.wager;
                self.raises += 1;
                self.pending = ((1 << self.n) - 1) & !self.folded & !(1 << a);
            }
            _ => unreachable!(),
        }
        let live = ((1u8 << self.n) - 1) & !self.folded;
        if live.count_ones() == 1 {
            self.actor = None;
            return;
        }
        if self.pending == 0 {
            if self.kind == Kind::Kuhn || self.round == 1 {
                self.actor = None;
                return;
            }
            self.round = 1;
            self.raises = 0;
            self.pending = live;
            self.push(255);
            self.actor = (0..self.n).find(|&p| live & (1 << p) != 0);
        } else {
            self.actor = (1..=self.n)
                .map(|i| (a + i) % self.n)
                .find(|&p| self.pending & (1 << p) != 0);
        }
    }
    /// Table 1's four situations: initial, facing bet, bet+fold, bet+call.
    pub fn kuhn_situation(&self) -> usize {
        assert_eq!(self.kind, Kind::Kuhn);
        assert_eq!(self.n, 3);
        let actions: Vec<u8> = self.history().iter().map(|e| e % 4).collect();
        match (self.actor.unwrap(), actions.as_slice()) {
            (0, []) | (1, [1]) | (2, [1, 1]) => 0,
            (0, [1, 1, 2]) | (1, [2]) | (2, [1, 2]) => 1,
            (0, [1, 2, 0]) | (1, [1, 1, 2, 0]) | (2, [2, 0]) => 2,
            (0, [1, 2, 1]) | (1, [1, 1, 2, 1]) | (2, [2, 1]) => 3,
            other => panic!("unexpected Kuhn situation {other:?}"),
        }
    }
}
impl State for Hand {
    type Saved = Self;
    fn players(&self) -> usize {
        self.n
    }
    fn decision(&self) -> Option<Decision> {
        let a = self.actor?;
        let key = crate::key::digest16(&self.payload());
        Some(Decision {
            key,
            actor: a,
            owner: a,
            street: self.round,
            code: self.actions().iter().fold(0, |c, &a| c | 1 << a),
        })
    }
    fn payoff(&self, player: usize) -> f64 {
        assert!(self.actor.is_none() && player < self.n);
        let mut winners = 0u8;
        let mut best = 0;
        for p in 0..self.n {
            if self.folded & (1 << p) != 0 {
                continue;
            }
            let rank = self.cards[p]
                + if self.board() == Some(self.cards[p]) {
                    16
                } else {
                    0
                };
            if winners == 0 || rank > best {
                winners = 1 << p;
                best = rank;
            } else if rank == best {
                winners |= 1 << p;
            }
        }
        let pot: u8 = self.contributed[..self.n].iter().sum();
        -f64::from(self.contributed[player])
            + if winners & (1 << player) != 0 {
                f64::from(pot) / f64::from(winners.count_ones())
            } else {
                0.
            }
    }
    fn save(&self) -> Self {
        *self
    }
    fn restore(&mut self, saved: Self) {
        *self = saved;
    }
    fn apply(&mut self, action_index: usize) {
        self.play(self.actions()[action_index]);
    }
}

/// A convenient subfamily of the published equilibria (Tables 2/3, c11=0).
/// beta in [0,1/4]; choose b11=b21=beta, b32=1/2, c33=c34=0.
pub fn published_kuhn_policy(hand: &Hand, beta: f64) -> Vec<f64> {
    assert!((0. ..=0.25).contains(&beta));
    let p = hand.actor.unwrap();
    let card = hand.hole(p);
    let situation = hand.kuhn_situation();
    let aggressive = match (card, situation) {
        (0, 1..=3) | (1, 3) => 0.,
        (3, 1..=3) => 1.,
        _ => match p {
            0 => {
                if (card, situation) == (2, 2) {
                    0.5
                } else {
                    0.
                }
            }
            1 => match (card, situation) {
                (0 | 1, 0) => beta,
                (2, 1) => 0.5,
                (2, 2) => (1. + 3. * beta) / 2.,
                (3, 0) => 4. * beta,
                _ => 0.,
            },
            2 => match (card, situation) {
                (1, 0) => 0.5,
                (3, 0) => 1.,
                _ => 0.,
            },
            _ => unreachable!(),
        },
    };
    vec![1. - aggressive, aggressive]
}
