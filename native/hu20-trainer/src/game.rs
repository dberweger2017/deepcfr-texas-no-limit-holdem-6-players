//! Heads-up 20 BB no-limit hold'em with the pinned engine's betting rules.
//!
//! Both seats start with 2,000 chips; blinds are 50/100. The button posts the
//! small blind and acts first preflop; the big blind acts first afterwards.
//! A full raise increases the wager by at least the last full increment; a
//! smaller increase is legal only as an exact all-in. Raising is unavailable
//! when the opponent cannot contest chips above the current wager. With equal
//! stacks heads-up, that also covers the per-player reopening rule: a short
//! all-in leaves the raiser's opponent nothing to raise against.
//!
//! For speed the hand is mutated in place and restored with `save`/`restore`,
//! and the key's public-history JSON is appended as events happen.

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

#[derive(Clone, Copy, Debug)]
pub struct Legal {
    pub fold: bool,
    pub check: bool,
    pub call: bool,
    pub raise: bool,
    pub call_amount: u32,
    pub min_raise_to: u32,
    pub max_raise_to: u32,
}

/// Every scalar of a hand; copying it is a complete snapshot apart from the history bytes.
#[derive(Clone, Copy, Debug)]
pub struct Core {
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
    max_bet: u32,
    min_raise: u32,
    acted: [bool; 2],
}

#[derive(Clone, Debug)]
pub struct Hand {
    pub core: Core,
    /// Comma-joined JSON tokens of the key's ordered public history.
    pub history: Vec<u8>,
}

pub struct Saved(Core, usize);

impl std::ops::Deref for Hand {
    type Target = Core;
    fn deref(&self) -> &Core {
        &self.core
    }
}

fn push_token(history: &mut Vec<u8>, parts: &[&[u8]]) {
    if !history.is_empty() {
        history.push(b',');
    }
    for part in parts {
        history.extend_from_slice(part);
    }
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
        let mut core = Core {
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
            max_bet: BIG_BLIND,
            min_raise: BIG_BLIND,
            acted: [false; 2],
        };
        for (seat, amount) in [(sb, SMALL_BLIND), (bb, BIG_BLIND)] {
            core.stack[seat] -= amount;
            core.street_bet[seat] += amount;
            core.contributed[seat] += amount;
        }
        Hand { core, history: Vec::with_capacity(256) }
    }

    pub fn save(&self) -> Saved {
        Saved(self.core, self.history.len())
    }

    pub fn restore(&mut self, saved: Saved) {
        self.core = saved.0;
        self.history.truncate(saved.1);
    }
}

impl Core {
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

impl Hand {
    pub fn apply(&self, action: Action) -> Hand {
        let mut next = self.clone();
        next.apply_mut(action);
        next
    }

    pub fn apply_mut(&mut self, action: Action) {
        let c = &mut self.core;
        let a = c.actor.expect("cannot act in a finished hand") as usize;
        let o = 1 - a;
        let legal = c.legal();
        let pot_before = c.contributed[0] + c.contributed[1];
        let stack_before = c.stack[a];
        let paid = match action.kind {
            Kind::Fold => {
                assert!(legal.fold);
                c.folded[a] = true;
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
                let increment = action.raise_to - c.max_bet;
                if increment >= c.min_raise {
                    c.min_raise = increment;
                }
                c.max_bet = action.raise_to;
                c.acted[o] = false;
                action.raise_to - c.street_bet[a]
            }
        };
        c.stack[a] -= paid;
        c.street_bet[a] += paid;
        c.contributed[a] += paid;
        c.acted[a] = true;
        // Key token: street, actor relative to the button, label with the raise-size bucket.
        let relative: &[u8] = if (a as u8 + 2 - c.button) % 2 == 0 { b"0" } else { b"1" };
        let label: &[u8] = match action.kind {
            Kind::Raise => {
                let base = pot_before.max(BIG_BLIND);
                let all_in = paid == stack_before;
                match (if 2 * paid < base { 0 } else if 2 * paid < 3 * base { 1 } else if paid < 3 * base { 2 } else { 3 }, all_in) {
                    (0, false) => b"raise-0",
                    (1, false) => b"raise-1",
                    (2, false) => b"raise-2",
                    (_, false) => b"raise-3",
                    (0, true) => b"raise-0-all-in",
                    (1, true) => b"raise-1-all-in",
                    (2, true) => b"raise-2-all-in",
                    (_, true) => b"raise-3-all-in",
                }
            }
            other => other.name().as_bytes(),
        };
        push_token(&mut self.history, &[b"[\"", c.street.name().as_bytes(), b"\",", relative, b",\"", label, b"\"]"]);
        if action.kind == Kind::Fold {
            c.actor = None;
            return;
        }
        let matched = c.street_bet[a] == c.street_bet[o] || c.all_in(a) && c.street_bet[a] <= c.street_bet[o];
        let round_over = (c.acted[o] || c.all_in(o)) && matched;
        if !round_over {
            c.actor = Some(o as u8);
            return;
        }
        // Run out the board once at most one player can still bet.
        let runout = c.all_in(0) || c.all_in(1);
        loop {
            if c.street == Street::River {
                c.actor = None;
                return;
            }
            c.street = c.street.next();
            c.dealt = c.street.board_cards();
            push_token(&mut self.history, &[b"[\"", c.street.name().as_bytes(), b"\",\"board\"]"]);
            c.street_bet = [0; 2];
            c.max_bet = 0;
            c.min_raise = BIG_BLIND;
            c.acted = [false; 2];
            if !runout {
                c.actor = Some(1 - c.button);
                return;
            }
        }
    }
}
