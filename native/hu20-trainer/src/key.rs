//! The HU20 abstract menu and v1 information key, byte-identical to
//! `choices(..., raise_cap=None, free_fold=False)` and
//! `information_key(..., schema=HU20_UNCAPPED_SCHEMA)` in `src/blueprint/abstraction.py`.

use crate::game::{Action, Core, Hand, Kind, Street};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use hu20_buckets::{evaluate, RANKS};
use std::cell::RefCell;

pub const SCHEMA: &str = "hu20-native-reopening-ordered-history-card-v1";
/// Menu names in their canonical order; a menu is a subsequence, so a bitmask names it.
pub const NAMES: [&str; 6] = ["fold", "check", "call", "min", "pot", "jam"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub name: &'static str,
    pub action: Action,
}

#[derive(Clone, Copy, Debug)]
pub struct Menu {
    items: [Choice; 5],
    pub len: usize,
    /// Bit i set when `NAMES[i]` is offered; identifies the name tuple exactly.
    pub code: u8,
}

impl Menu {
    pub fn as_slice(&self) -> &[Choice] {
        &self.items[..self.len]
    }
    fn push(&mut self, index: usize, action: Action) {
        self.items[self.len] = Choice { name: NAMES[index], action };
        self.len += 1;
        self.code |= 1 << index;
    }
}

pub fn names_of(code: u8) -> Vec<&'static str> {
    (0..6).filter(|i| code & (1 << i) != 0).map(|i| NAMES[i]).collect()
}

pub fn code_of(names: &[&str]) -> u8 {
    names.iter().fold(0, |code, name| code | 1 << NAMES.iter().position(|n| n == name).expect("menu name"))
}

/// Fold only when a check is unavailable; raises to min, pot and a conditional jam.
pub fn menu(hand: &Core) -> Menu {
    let legal = hand.legal();
    let a = hand.actor.unwrap() as usize;
    let none = Choice { name: "", action: Action { kind: Kind::Fold, raise_to: 0 } };
    let mut out = Menu { items: [none; 5], len: 0, code: 0 };
    if legal.fold && !legal.check {
        out.push(0, Action { kind: Kind::Fold, raise_to: 0 });
    }
    if legal.check {
        out.push(1, Action { kind: Kind::Check, raise_to: 0 });
    }
    if legal.call {
        out.push(2, Action { kind: Kind::Call, raise_to: 0 });
    }
    if legal.raise {
        let matched = hand.street_bet[a] + legal.call_amount;
        let after_call = hand.pot() + legal.call_amount;
        let (low, high) = (legal.min_raise_to, legal.max_raise_to);
        let pot = high.min(low.max(matched + after_call));
        let mut used = [u32::MAX; 3];
        for (slot, (index, target)) in [(3usize, low), (4, pot), (5, high)].into_iter().enumerate() {
            // The all-in is useful once its size is at most twice the pot.
            if index == 5 && high - matched > 2 * after_call && high != low {
                continue;
            }
            if used.contains(&target) {
                continue;
            }
            used[slot] = target;
            out.push(index, Action { kind: Kind::Raise, raise_to: target });
        }
    }
    out
}

fn rank_byte(card: u8) -> u8 {
    RANKS[(card / 4) as usize]
}

fn push_preflop(out: &mut Vec<u8>, hole: [u8; 2]) {
    let (mut a, mut b) = (hole[0], hole[1]);
    // Python sorts by rank only, descending, and keeps the original order on ties.
    if b / 4 > a / 4 {
        std::mem::swap(&mut a, &mut b);
    }
    let tag = if a / 4 == b / 4 { b'p' } else if a % 4 == b % 4 { b's' } else { b'o' };
    out.extend_from_slice(&[b'"', rank_byte(a), rank_byte(b), tag, b'"']);
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

fn push_postflop(out: &mut Vec<u8>, hole: [u8; 2], board: &[u8]) {
    let mut cards = [0u8; 7];
    cards[..2].copy_from_slice(&hole);
    cards[2..2 + board.len()].copy_from_slice(board);
    let cards = &cards[..2 + board.len()];
    let value = evaluate(cards);
    let category = value >> 20;
    let top = primary_rank(value);
    let band = if top < 8 { 0 } else if top < 12 { 1 } else { 2 };
    let mut suits = [0u32; 4];
    let mut ranks = 0u16;
    for &c in cards {
        suits[(c % 4) as usize] += 1;
        ranks |= 1 << (c / 4);
    }
    let early = board.len() < 5;
    let flush_draw = (early && *suits.iter().max().unwrap() >= 4) as u32;
    let wheel: u16 = (1 << 12) | 0b1111;
    let straight_draw = (early
        && (std::iter::once(wheel).chain((0..9).map(|s| 0b11111u16 << s)).any(|w| (w & !ranks).count_ones() == 1)))
        as u32;
    let mut board_ranks = 0u16;
    for &c in board {
        board_ranks |= 1 << (c / 4);
    }
    let board_paired = ((board_ranks.count_ones() as usize) < board.len()) as u32;
    let digit = |v: u32| b'0' + v as u8;
    out.extend_from_slice(&[b'[', digit(category), b',', digit(band), b',', digit(flush_draw), b',',
                            digit(straight_draw), b',', digit(board_paired), b']']);
}

fn push_bool(out: &mut Vec<u8>, value: bool) {
    out.extend_from_slice(if value { b"true" } else { b"false" });
}

/// The JSON payload Python hashes, with `json.dumps(..., separators=(",", ":"))` formatting.
pub fn write_payload(out: &mut Vec<u8>, hand: &Hand, menu: &[Choice]) {
    let a = hand.actor.expect("a key needs a live decision") as usize;
    out.clear();
    out.extend_from_slice(b"[\"");
    out.extend_from_slice(SCHEMA.as_bytes());
    out.extend_from_slice(b"\",2,");
    out.push(if (a as u8 + 2 - hand.button) % 2 == 0 { b'0' } else { b'1' });
    out.extend_from_slice(b",\"");
    out.extend_from_slice(hand.street.name().as_bytes());
    out.extend_from_slice(b"\",");
    if hand.street == Street::Preflop {
        push_preflop(out, hand.holes[a]);
    } else {
        push_postflop(out, hand.holes[a], hand.visible_board());
    }
    out.extend_from_slice(b",[");
    for offset in 0..2 {
        let p = (hand.button as usize + offset) % 2;
        if offset > 0 {
            out.push(b',');
        }
        out.push(b'[');
        push_bool(out, hand.folded[p]);
        out.push(b',');
        push_bool(out, hand.all_in(p));
        out.push(b']');
    }
    out.extend_from_slice(b"],[");
    out.extend_from_slice(&hand.history);
    out.extend_from_slice(b"],[");
    for (i, choice) in menu.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        out.push(b'"');
        out.extend_from_slice(choice.name.as_bytes());
        out.push(b'"');
    }
    out.extend_from_slice(b"]]");
}

pub fn payload(hand: &Hand, menu: &[Choice]) -> String {
    let mut out = Vec::new();
    write_payload(&mut out, hand, menu);
    String::from_utf8(out).unwrap()
}

thread_local! {
    static BUFFER: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(1024));
}

/// BLAKE2b with a 16-byte digest (a distinct parameter set, not a truncated 64-byte hash).
pub fn key_bytes(hand: &Hand, menu: &[Choice]) -> [u8; 16] {
    BUFFER.with(|buffer| {
        let mut buffer = buffer.borrow_mut();
        write_payload(&mut buffer, hand, menu);
        let mut hasher = Blake2bVar::new(16).unwrap();
        hasher.update(&buffer);
        let mut out = [0u8; 16];
        hasher.finalize_variable(&mut out).unwrap();
        out
    })
}

pub fn key_hex(hand: &Hand, menu: &[Choice]) -> String {
    key_bytes(hand, menu).iter().map(|b| format!("{b:02x}")).collect()
}
