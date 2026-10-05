//! The HU20 abstract menu and v1 information key, byte-identical to
//! `choices(..., raise_cap=None, free_fold=False)` and
//! `information_key(..., schema=HU20_UNCAPPED_SCHEMA)` in `src/blueprint/abstraction.py`.

use crate::game::{Action, Event, Hand, Kind, Street, BIG_BLIND, STACK};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use hu20_buckets::{evaluate, RANKS, SUITS};
use std::fmt::Write;

pub const SCHEMA: &str = "hu20-native-reopening-ordered-history-card-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub name: &'static str,
    pub action: Action,
}

/// Fold only when a check is unavailable; raises to min, pot and a conditional jam.
pub fn menu(hand: &Hand) -> Vec<Choice> {
    let legal = hand.legal();
    let a = hand.actor.unwrap() as usize;
    let mut out = Vec::with_capacity(5);
    if legal.fold && !legal.check {
        out.push(Choice { name: "fold", action: Action { kind: Kind::Fold, raise_to: 0 } });
    }
    if legal.check {
        out.push(Choice { name: "check", action: Action { kind: Kind::Check, raise_to: 0 } });
    }
    if legal.call {
        out.push(Choice { name: "call", action: Action { kind: Kind::Call, raise_to: 0 } });
    }
    if legal.raise {
        let matched = hand.street_bet[a] + legal.call_amount;
        let after_call = hand.pot() + legal.call_amount;
        let (low, high) = (legal.min_raise_to, legal.max_raise_to);
        let pot = high.min(low.max(matched + after_call));
        let mut used: Vec<u32> = Vec::with_capacity(3);
        for (name, target) in [("min", low), ("pot", pot), ("jam", high)] {
            // The all-in is useful once its size is at most twice the pot.
            if name == "jam" && high - matched > 2 * after_call && high != low {
                continue;
            }
            if used.contains(&target) {
                continue;
            }
            used.push(target);
            out.push(Choice { name, action: Action { kind: Kind::Raise, raise_to: target } });
        }
    }
    out
}

fn rank_char(card: u8) -> char {
    RANKS[(card / 4) as usize] as char
}

fn preflop_descriptor(hole: [u8; 2]) -> String {
    let (mut a, mut b) = (hole[0], hole[1]);
    // Python sorts by rank only, descending, and keeps the original order on ties.
    if b / 4 > a / 4 {
        std::mem::swap(&mut a, &mut b);
    }
    let tag = if a / 4 == b / 4 { 'p' } else if a % 4 == b % 4 { 's' } else { 'o' };
    format!("\"{}{}{}\"", rank_char(a), rank_char(b), tag)
}

/// `hand_value(cards)[1]` on the 2..14 rank scale: the primary rank of the best five.
fn primary_rank(value: u32) -> u32 {
    let low = value & 0xFFFFF;
    let primary = match value >> 20 {
        8 | 4 => low,
        7 | 6 => low >> 4,
        3 | 2 => low >> 8,
        1 => low >> 12,
        _ => low >> 16, // high card and flush: the top of five packed nibbles
    };
    primary + 2
}

fn postflop_descriptor(hole: [u8; 2], board: &[u8]) -> String {
    let mut cards: Vec<u8> = hole.to_vec();
    cards.extend_from_slice(board);
    let value = evaluate(&cards);
    let category = value >> 20;
    let top = primary_rank(value);
    let band = if top < 8 { 0 } else if top < 12 { 1 } else { 2 };
    let mut suits = [0u32; 4];
    let mut ranks = 0u16;
    for &c in &cards {
        suits[(c % 4) as usize] += 1;
        ranks |= 1 << (c / 4);
    }
    let early = board.len() < 5;
    let flush_draw = (early && *suits.iter().max().unwrap() >= 4) as u32;
    let wheel: u16 = (1 << 12) | 0b1111;
    let windows = std::iter::once(wheel).chain((0..9).map(|s| 0b11111u16 << s));
    let straight_draw = (early && windows.into_iter().any(|w| (w & !ranks).count_ones() == 1)) as u32;
    let mut board_ranks = 0u16;
    for &c in board {
        board_ranks |= 1 << (c / 4);
    }
    let board_paired = ((board_ranks.count_ones() as usize) < board.len()) as u32;
    format!("[{category},{band},{flush_draw},{straight_draw},{board_paired}]")
}

/// The ordered public history: board tokens, actors relative to the button, raise-size buckets.
fn history(hand: &Hand, out: &mut String) {
    let mut pot = 0u32;
    let mut remaining = [STACK; 2];
    out.push('[');
    let mut first = true;
    for event in &hand.events {
        let token = match *event {
            Event::Blind { seat, amount } => {
                pot += amount;
                remaining[seat as usize] -= amount;
                continue;
            }
            Event::Board { street } => format!("[\"{}\",\"board\"]", street.name()),
            Event::Act { street, seat, kind, paid } => {
                let label = if kind == Kind::Raise {
                    let base = pot.max(BIG_BLIND);
                    let size = if 2 * paid < base { 0 } else if 2 * paid < 3 * base { 1 } else if paid < 3 * base { 2 } else { 3 };
                    let all_in = if paid == remaining[seat as usize] { "-all-in" } else { "" };
                    format!("raise-{size}{all_in}")
                } else {
                    kind.name().to_string()
                };
                pot += paid;
                remaining[seat as usize] -= paid;
                let relative = (seat + 2 - hand.button) % 2;
                format!("[\"{}\",{},\"{}\"]", street.name(), relative, label)
            }
        };
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(&token);
    }
    out.push(']');
}

/// The JSON payload Python hashes, with `json.dumps(..., separators=(",", ":"))` formatting.
pub fn payload(hand: &Hand, menu: &[Choice]) -> String {
    let a = hand.actor.expect("a key needs a live decision") as usize;
    let mut s = String::with_capacity(256);
    let relative = (a as u8 + 2 - hand.button) % 2;
    let cards = if hand.street == Street::Preflop {
        preflop_descriptor(hand.holes[a])
    } else {
        postflop_descriptor(hand.holes[a], hand.visible_board())
    };
    write!(s, "[\"{SCHEMA}\",2,{relative},\"{}\",{cards},[", hand.street.name()).unwrap();
    for offset in 0..2 {
        let p = ((hand.button as usize) + offset) % 2;
        if offset > 0 {
            s.push(',');
        }
        write!(s, "[{},{}]", hand.folded[p], hand.all_in(p)).unwrap();
    }
    s.push_str("],");
    history(hand, &mut s);
    s.push_str(",[");
    for (i, choice) in menu.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write!(s, "\"{}\"", choice.name).unwrap();
    }
    s.push_str("]]");
    s
}

/// BLAKE2b with a 16-byte digest (a distinct parameter set, not a truncated 64-byte hash).
pub fn key_bytes(hand: &Hand, menu: &[Choice]) -> [u8; 16] {
    let mut hasher = Blake2bVar::new(16).unwrap();
    hasher.update(payload(hand, menu).as_bytes());
    let mut out = [0u8; 16];
    hasher.finalize_variable(&mut out).unwrap();
    out
}

pub fn key_hex(hand: &Hand, menu: &[Choice]) -> String {
    key_bytes(hand, menu).iter().map(|b| format!("{b:02x}")).collect()
}

pub fn suit_char(card: u8) -> char {
    SUITS[(card % 4) as usize] as char
}
