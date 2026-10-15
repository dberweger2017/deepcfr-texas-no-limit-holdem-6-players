//! Integer-chip cash Hold'em for 2–6 seats, independent of the byte-pinned HU path.
//!
//! `Hand` is privileged simulator state. Policy/key consumers must use `observe`,
//! which exposes only one seat's cards and the public betting record. Settlement
//! is the engine contract; live showdown disclosure belongs to `src/game/hand.py`.

use crate::game::{Kind, Street};
use hu20_buckets::evaluate;

#[cfg(test)]
#[path = "nplayer_tests.rs"]
mod tests;

pub const MAX_PLAYERS: usize = 6;
pub const MAX_CHIPS: u64 = 1_000_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    players: usize,
    initial: [u64; MAX_PLAYERS],
    small_blind: u64,
    big_blind: u64,
}

impl Config {
    pub fn new(stacks: &[u64], small_blind: u64, big_blind: u64) -> Result<Self, String> {
        if !(2..=MAX_PLAYERS).contains(&stacks.len()) {
            return Err("expected 2–6 players".into());
        }
        if stacks.iter().any(|&s| s == 0 || s > MAX_CHIPS)
            || small_blind == 0
            || small_blind > big_blind
            || big_blind > MAX_CHIPS
        {
            return Err("expected positive stacks <=10^12 chips and 0 < SB <= BB <=10^12".into());
        }
        let mut initial = [0; MAX_PLAYERS];
        initial[..stacks.len()].copy_from_slice(stacks);
        Ok(Self {
            players: stacks.len(),
            initial,
            small_blind,
            big_blind,
        })
    }

    pub fn players(&self) -> usize {
        self.players
    }
    pub fn small_blind(&self) -> u64 {
        self.small_blind
    }
    pub fn big_blind(&self) -> u64 {
        self.big_blind
    }
    pub fn stacks(&self) -> &[u64] {
        &self.initial[..self.players]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub kind: Kind,
    pub raise_to: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Legal {
    pub check: bool,
    pub call_amount: u64,
    /// None means raising is closed; otherwise inclusive absolute street targets.
    pub raise_to: Option<(u64, u64)>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Player {
    pub stack: u64,
    pub street_bet: u64,
    pub contributed: u64,
    pub folded: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pot {
    pub amount: u64,
    pub eligible: u8,
    pub winners: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settlement {
    pub contributions: [u64; MAX_PLAYERS],
    pub refunds: [u64; MAX_PLAYERS],
    pub rewards: [i64; MAX_PLAYERS],
    pots: [Pot; MAX_PLAYERS],
    count: usize,
}

impl Settlement {
    pub fn pots(&self) -> &[Pot] {
        &self.pots[..self.count]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Action {
        street: Street,
        actor: usize,
        action: Action,
        paid: u64,
    },
    Board(Street),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Core {
    config: Config,
    button: usize,
    holes: [[u8; 2]; MAX_PLAYERS],
    board: [u8; 5],
    dealt: usize,
    street: Street,
    players: [Player; MAX_PLAYERS],
    actor: Option<usize>,
    wager: u64,
    increment: u64,
    pending: u8,
    // An owed response and the right to raise are distinct after a short all-in.
    acted_at: [Option<u64>; MAX_PLAYERS],
    settlement: Option<Settlement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hand {
    core: Core,
    history: Vec<Event>,
}

pub struct Saved(Core, usize);

/// Borrowed, public-only projection. No simulator reference or unrevealed cards.
#[derive(Debug, PartialEq, Eq)]
pub struct Observation<'a> {
    pub config: Config,
    pub seat: usize,
    pub button: usize,
    pub hole: [u8; 2],
    pub board: &'a [u8],
    pub street: Street,
    pub players: &'a [Player],
    pub actor: Option<usize>,
    pub legal: Option<Legal>,
    pub history: &'a [Event],
}

impl Hand {
    pub fn from_deck(config: Config, button: usize, deck: &[u8]) -> Result<Self, String> {
        if button >= config.players {
            return Err("button outside table".into());
        }
        let mut seen = 0u64;
        for &card in deck {
            if card >= 52 || seen & (1u64 << card) != 0 {
                return Err("invalid or duplicate card".into());
            }
            seen |= 1u64 << card;
        }
        if deck.len() != 52 {
            return Err("expected a complete 52-card deck".into());
        }
        let n = config.players;
        let mut holes = [[0; 2]; MAX_PLAYERS];
        for round in 0..2 {
            for offset in 0..n {
                holes[(button + 1 + offset) % n][round] = deck[round * n + offset];
            }
        }
        let mut board = [0; 5];
        board.copy_from_slice(&deck[2 * n..2 * n + 5]);
        let mut players = [Player::default(); MAX_PLAYERS];
        for (seat, &stack) in config.stacks().iter().enumerate() {
            players[seat].stack = stack;
        }
        let sb = if n == 2 { button } else { (button + 1) % n };
        let bb = (sb + 1) % n;
        for (seat, blind) in [(sb, config.small_blind), (bb, config.big_blind)] {
            let paid = blind.min(players[seat].stack);
            players[seat].stack -= paid;
            players[seat].street_bet = paid;
            players[seat].contributed = paid;
        }
        let mut hand = Self {
            core: Core {
                config,
                button,
                holes,
                board,
                dealt: 0,
                street: Street::Preflop,
                players,
                actor: None,
                wager: config.big_blind,
                increment: config.big_blind,
                pending: (1 << n) - 1,
                acted_at: [None; MAX_PLAYERS],
                settlement: None,
            },
            history: Vec::with_capacity(32),
        };
        hand.advance(bb);
        Ok(hand)
    }

    pub fn config(&self) -> Config {
        self.core.config
    }
    pub fn button(&self) -> usize {
        self.core.button
    }
    pub fn actor(&self) -> Option<usize> {
        self.core.actor
    }
    pub fn street(&self) -> Street {
        self.core.street
    }
    pub fn finished(&self) -> bool {
        self.core.settlement.is_some()
    }
    pub fn players(&self) -> &[Player] {
        &self.core.players[..self.core.config.players]
    }
    pub fn board(&self) -> &[u8] {
        &self.core.board[..self.core.dealt]
    }
    pub fn pot(&self) -> u64 {
        self.players().iter().map(|p| p.contributed).sum()
    }
    pub fn wager(&self) -> u64 {
        self.core.wager
    }
    pub fn min_raise(&self) -> u64 {
        self.core.increment
    }
    pub fn settlement(&self) -> Option<&Settlement> {
        self.core.settlement.as_ref()
    }
    pub fn history(&self) -> &[Event] {
        &self.history
    }
    pub fn save(&self) -> Saved {
        Saved(self.core, self.history.len())
    }
    pub fn restore(&mut self, saved: Saved) {
        self.core = saved.0;
        self.history.truncate(saved.1);
    }

    pub fn legal(&self) -> Option<Legal> {
        let a = self.actor()?;
        let c = &self.core;
        let p = c.players[a];
        let owe = c.wager.saturating_sub(p.street_bet);
        let reopened =
            c.acted_at[a].is_none_or(|previous| c.wager.saturating_sub(previous) >= c.increment);
        let contested =
            self.players().iter().enumerate().any(|(i, other)| {
                i != a && !other.folded && other.stack + other.street_bet > c.wager
            });
        let raise_to = if reopened && contested && p.stack > owe {
            let high = p.street_bet + p.stack;
            Some(((c.wager + c.increment).min(high), high))
        } else {
            None
        };
        Some(Legal {
            check: owe == 0,
            call_amount: owe.min(p.stack),
            raise_to,
        })
    }

    pub fn observe(&self, seat: usize) -> Result<Observation<'_>, String> {
        if seat >= self.core.config.players {
            return Err("observer outside table".into());
        }
        Ok(Observation {
            config: self.config(),
            seat,
            button: self.button(),
            hole: self.core.holes[seat],
            board: self.board(),
            street: self.street(),
            players: self.players(),
            actor: self.actor(),
            legal: if self.actor() == Some(seat) {
                self.legal()
            } else {
                None
            },
            history: &self.history,
        })
    }

    /// Validate before mutation: a rejected action leaves the complete parent intact.
    pub fn apply_mut(&mut self, action: Action) -> Result<(), String> {
        let a = self.actor().ok_or("cannot act in a settled hand")?;
        let legal = self.legal().unwrap();
        let paid = match action.kind {
            Kind::Fold => 0,
            Kind::Check if legal.check => 0,
            Kind::Call if !legal.check => legal.call_amount,
            Kind::Raise
                if legal
                    .raise_to
                    .is_some_and(|(lo, hi)| (lo..=hi).contains(&action.raise_to)) =>
            {
                action.raise_to - self.core.players[a].street_bet
            }
            _ => return Err("illegal action or raise target".into()),
        };
        if action.kind != Kind::Raise && action.raise_to != 0 {
            return Err("non-raise target must be zero".into());
        }
        let c = &mut self.core;
        if action.kind == Kind::Fold {
            c.players[a].folded = true;
        }
        if action.kind == Kind::Raise {
            let increase = action.raise_to - c.wager;
            if increase >= c.increment {
                c.increment = increase;
            }
            c.wager = action.raise_to;
            for i in 0..c.config.players {
                let p = c.players[i];
                if !p.folded && p.stack > 0 && p.street_bet < c.wager {
                    c.pending |= 1 << i;
                }
            }
        }
        c.players[a].stack -= paid;
        c.players[a].street_bet += paid;
        c.players[a].contributed += paid;
        c.pending &= !(1 << a);
        // A check before an opening short all-in preserves the right to raise.
        if action.kind != Kind::Check {
            c.acted_at[a] = Some(c.wager);
        }
        self.history.push(Event::Action {
            street: c.street,
            actor: a,
            action,
            paid,
        });
        self.advance(a);
        Ok(())
    }

    fn advance(&mut self, mut after: usize) {
        let n = self.core.config.players;
        loop {
            let live = self.players().iter().filter(|p| !p.folded).count();
            if live == 1 {
                self.settle();
                return;
            }
            let able = self
                .players()
                .iter()
                .filter(|p| !p.folded && p.stack > 0)
                .count();
            if able == 1 {
                // Nominal BB bring-in applies only while two players can still bet.
                self.core.wager = self
                    .players()
                    .iter()
                    .filter(|p| !p.folded)
                    .map(|p| p.street_bet)
                    .max()
                    .unwrap();
                let a = self
                    .players()
                    .iter()
                    .position(|p| !p.folded && p.stack > 0)
                    .unwrap();
                if self.core.players[a].street_bet < self.core.wager {
                    self.core.pending |= 1 << a;
                } else {
                    self.core.pending &= !(1 << a);
                }
            }
            if let Some(a) = (1..=n).map(|offset| (after + offset) % n).find(|&i| {
                let p = self.core.players[i];
                !p.folded && p.stack > 0 && self.core.pending & (1 << i) != 0
            }) {
                self.core.actor = Some(a);
                return;
            }
            if self.core.street == Street::River {
                self.settle();
                return;
            }
            self.core.street = match self.core.street {
                Street::Preflop => Street::Flop,
                Street::Flop => Street::Turn,
                _ => Street::River,
            };
            self.core.dealt = self.core.street.board_cards();
            self.history.push(Event::Board(self.core.street));
            for p in &mut self.core.players[..n] {
                p.street_bet = 0;
            }
            self.core.wager = 0;
            self.core.increment = self.core.config.big_blind;
            self.core.acted_at = [None; MAX_PLAYERS];
            self.core.pending = if able > 1 { (1 << n) - 1 } else { 0 };
            after = self.core.button;
        }
    }

    fn settle(&mut self) {
        let n = self.core.config.players;
        let contributions = self.core.players.map(|p| p.contributed);
        let mut result = Settlement {
            contributions,
            refunds: [0; MAX_PLAYERS],
            rewards: [0; MAX_PLAYERS],
            pots: [Pot::default(); MAX_PLAYERS],
            count: 0,
        };
        let mut levels = contributions;
        levels[..n].sort_unstable();
        let mut previous = 0;
        for &level in &levels[..n] {
            if level == previous {
                continue;
            }
            let mut contributors = 0u8;
            let mut eligible = 0u8;
            for (i, &amount) in contributions[..n].iter().enumerate() {
                if amount >= level {
                    contributors |= 1 << i;
                    if !self.core.players[i].folded {
                        eligible |= 1 << i;
                    }
                }
            }
            let amount = (level - previous) * contributors.count_ones() as u64;
            previous = level;
            if contributors.count_ones() == 1 {
                result.refunds[contributors.trailing_zeros() as usize] += amount;
            } else if result.count > 0 && result.pots[result.count - 1].eligible == eligible {
                // Dead-money levels with unchanged eligibility are one rounding event.
                result.pots[result.count - 1].amount += amount;
            } else {
                result.pots[result.count] = Pot {
                    amount,
                    eligible,
                    winners: 0,
                };
                result.count += 1;
            }
        }
        let live: Vec<_> = (0..n).filter(|&i| !self.core.players[i].folded).collect();
        let mut ranks = [0; MAX_PLAYERS];
        if live.len() > 1 {
            assert_eq!(self.core.dealt, 5);
            for &seat in &live {
                let h = self.core.holes[seat];
                let b = self.core.board;
                ranks[seat] = evaluate(&[h[0], h[1], b[0], b[1], b[2], b[3], b[4]]);
            }
        }
        for pot in &mut result.pots[..result.count] {
            if live.len() == 1 {
                pot.winners = 1 << live[0];
            } else {
                let best = (0..n)
                    .filter(|i| pot.eligible & (1 << i) != 0)
                    .map(|i| ranks[i])
                    .max()
                    .expect("pot has no eligible seat");
                for (i, &rank) in ranks[..n].iter().enumerate() {
                    if pot.eligible & (1 << i) != 0 && rank == best {
                        pot.winners |= 1 << i;
                    }
                }
            }
            let winners = pot.winners.count_ones() as u64;
            let share = pot.amount / winners;
            let mut remainder = pot.amount % winners;
            for offset in 1..=n {
                let seat = (self.core.button + offset) % n;
                if pot.winners & (1 << seat) != 0 {
                    self.core.players[seat].stack += share + u64::from(remainder > 0);
                    remainder = remainder.saturating_sub(1);
                }
            }
        }
        for i in 0..n {
            self.core.players[i].stack += result.refunds[i];
            result.rewards[i] =
                self.core.players[i].stack as i64 - self.core.config.initial[i] as i64;
            self.core.players[i].street_bet = 0;
            self.core.players[i].contributed = 0;
        }
        assert_eq!(
            result.rewards.iter().sum::<i64>(),
            0,
            "settlement must conserve chips"
        );
        self.core.actor = None;
        self.core.pending = 0;
        self.core.settlement = Some(result);
    }
}
