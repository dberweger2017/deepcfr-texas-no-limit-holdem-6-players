//! Heads-up 20 BB no-limit hold'em with the pinned engine's betting rules.
//!
//! Both seats start with 2,000 chips; blinds are 50/100. The button posts the
//! small blind and acts first preflop; the big blind acts first afterwards.
//! A full raise increases the wager by at least the last full increment; a
//! smaller increase is legal only as an exact all-in. Raising is unavailable
//! when the opponent cannot contest chips above the current wager. With equal
//! stacks heads-up, that also covers the per-player reopening rule: a short
//! all-in leaves the raiser's opponent nothing to raise against.

use hu20_buckets::evaluate;

pub const STACK: u32 = 2000;
pub const SMALL_BLIND: u32 = 50;
pub const BIG_BLIND: u32 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Street {
    Preflop,
    Flop,
    Turn,
    River,
}

impl Street {
    pub fn name(self) -> &'static str {
        match self {
            Street::Preflop => "preflop",
            Street::Flop => "flop",
            Street::Turn => "turn",
            Street::River => "river",
        }
    }
    fn next(self) -> Street {
        match self {
            Street::Preflop => Street::Flop,
            Street::Flop => Street::Turn,
            _ => Street::River,
        }
    }
    pub fn board_cards(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Fold,
    Check,
    Call,
    Raise,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Fold => "fold",
            Kind::Check => "check",
            Kind::Call => "call",
            Kind::Raise => "raise",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub kind: Kind,
    /// Raise-to target for raises, otherwise 0.
    pub raise_to: u32,
}

/// Public events in the order the key's history walks them.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Blind { seat: u8, amount: u32 },
    Board { street: Street },
    Act { street: Street, seat: u8, kind: Kind, paid: u32 },
}

#[derive(Clone, Debug)]
pub struct Legal {
    pub fold: bool,
    pub check: bool,
    pub call: bool,
    pub raise: bool,
    pub call_amount: u32,
    pub min_raise_to: u32,
    pub max_raise_to: u32,
}

#[derive(Clone, Debug)]
pub struct Hand {
    pub button: u8,
    pub holes: [[u8; 2]; 2],
    pub board: [u8; 5],
    pub street: Street,
    pub dealt: usize,
    pub stack: [u32; 2],
    pub street_bet: [u32; 2],
    pub contributed: [u32; 2],
    pub folded: [bool; 2],
    pub actor: Option<u8>,
    pub events: Vec<Event>,
    max_bet: u32,
    min_raise: u32,
    acted: [bool; 2],
}

impl Hand {
    /// Deal as the engine does: two rounds starting left of the button, then the board.
    pub fn from_deck(button: u8, deck: &[u8]) -> Hand {
        let bb = 1 - button;
        let mut holes = [[0u8; 2]; 2];
        holes[bb as usize] = [deck[0], deck[2]];
        holes[button as usize] = [deck[1], deck[3]];
        let mut board = [0u8; 5];
        board.copy_from_slice(&deck[4..9]);
        Hand::new(button, holes, board)
    }

    pub fn new(button: u8, holes: [[u8; 2]; 2], board: [u8; 5]) -> Hand {
        let (sb, bb) = (button as usize, 1 - button as usize);
        let mut hand = Hand {
            button,
            holes,
            board,
            street: Street::Preflop,
            dealt: 0,
            stack: [STACK; 2],
            street_bet: [0; 2],
            contributed: [0; 2],
            folded: [false; 2],
            actor: Some(sb as u8),
            events: Vec::with_capacity(24),
            max_bet: BIG_BLIND,
            min_raise: BIG_BLIND,
            acted: [false; 2],
        };
        for (seat, amount) in [(sb, SMALL_BLIND), (bb, BIG_BLIND)] {
            hand.stack[seat] -= amount;
            hand.street_bet[seat] += amount;
            hand.contributed[seat] += amount;
            hand.events.push(Event::Blind { seat: seat as u8, amount });
        }
        hand
    }

    pub fn finished(&self) -> bool {
        self.actor.is_none()
    }

    pub fn pot(&self) -> u32 {
        self.contributed[0] + self.contributed[1]
    }

    pub fn all_in(&self, seat: usize) -> bool {
        !self.folded[seat] && self.stack[seat] == 0
    }

    pub fn visible_board(&self) -> &[u8] {
        &self.board[..self.dealt]
    }

    pub fn legal(&self) -> Legal {
        let a = self.actor.expect("legal actions need a live decision") as usize;
        let o = 1 - a;
        let owe = self.max_bet - self.street_bet[a];
        let call_amount = owe.min(self.stack[a]);
        let max_raise_to = self.street_bet[a] + self.stack[a];
        let raise = self.stack[a] > owe && self.street_bet[o] + self.stack[o] > self.max_bet;
        Legal {
            fold: true,
            check: owe == 0,
            call: owe > 0,
            raise,
            call_amount,
            min_raise_to: if raise { (self.max_bet + self.min_raise).min(max_raise_to) } else { 0 },
            max_raise_to: if raise { max_raise_to } else { 0 },
        }
    }

    pub fn apply(&self, action: Action) -> Hand {
        let mut next = self.clone();
        next.apply_mut(action);
        next
    }

    pub fn apply_mut(&mut self, action: Action) {
        let a = self.actor.expect("cannot act in a finished hand") as usize;
        let o = 1 - a;
        let legal = self.legal();
        let paid = match action.kind {
            Kind::Fold => {
                assert!(legal.fold);
                self.folded[a] = true;
                0
            }
            Kind::Check => {
                assert!(legal.check);
                0
            }
            Kind::Call => {
                assert!(legal.call);
                legal.call_amount
            }
            Kind::Raise => {
                assert!(legal.raise && legal.min_raise_to <= action.raise_to && action.raise_to <= legal.max_raise_to,
                        "illegal raise {action:?} {legal:?}");
                let increment = action.raise_to - self.max_bet;
                if increment >= self.min_raise {
                    self.min_raise = increment;
                }
                self.max_bet = action.raise_to;
                self.acted[o] = false;
                action.raise_to - self.street_bet[a]
            }
        };
        self.stack[a] -= paid;
        self.street_bet[a] += paid;
        self.contributed[a] += paid;
        self.acted[a] = true;
        self.events.push(Event::Act { street: self.street, seat: a as u8, kind: action.kind, paid });
        if action.kind == Kind::Fold {
            self.actor = None;
            return;
        }
        let matched = self.street_bet[a] == self.street_bet[o] || self.all_in(a) && self.street_bet[a] <= self.street_bet[o];
        let round_over = (self.acted[o] || self.all_in(o)) && matched;
        if !round_over {
            self.actor = Some(o as u8);
            return;
        }
        // Run out the board once at most one player can still bet.
        let runout = self.all_in(0) || self.all_in(1);
        loop {
            if self.street == Street::River {
                self.actor = None;
                return;
            }
            self.street = self.street.next();
            self.dealt = self.street.board_cards();
            self.events.push(Event::Board { street: self.street });
            self.street_bet = [0; 2];
            self.max_bet = 0;
            self.min_raise = BIG_BLIND;
            self.acted = [false; 2];
            if !runout {
                self.actor = Some(1 - self.button);
                return;
            }
        }
    }

    /// Final stacks: the pot to the non-folder, or a showdown split.
    pub fn final_stacks(&self) -> [u32; 2] {
        assert!(self.finished());
        let mut stacks = self.stack;
        // An unmatched contribution returns to its owner.
        let matched = self.contributed[0].min(self.contributed[1]);
        for seat in 0..2 {
            stacks[seat] += self.contributed[seat] - matched;
        }
        let pot = 2 * matched;
        if self.folded[0] || self.folded[1] {
            let winner = if self.folded[0] { 1 } else { 0 };
            stacks[winner] += pot;
            return stacks;
        }
        let value = |seat: usize| {
            let h = self.holes[seat];
            let b = self.board;
            evaluate(&[h[0], h[1], b[0], b[1], b[2], b[3], b[4]])
        };
        let (v0, v1) = (value(0), value(1));
        if v0 > v1 {
            stacks[0] += pot;
        } else if v1 > v0 {
            stacks[1] += pot;
        } else {
            stacks[0] += pot / 2;
            stacks[1] += pot / 2;
        }
        stacks
    }
}
