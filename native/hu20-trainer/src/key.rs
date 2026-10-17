//! The HU20 abstract menu and v1 information key, byte-identical to
//! `choices(..., raise_cap=None, free_fold=False)` and
//! `information_key(..., schema=HU20_UNCAPPED_SCHEMA)` in `src/blueprint/abstraction.py`.

use crate::cards::Cards;
use crate::cfr::MAX_ACTIONS;
use crate::game::{Action, Core, Game, Hand, Kind, Street};
use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use hu20_buckets::{evaluate, RANKS};
use std::cell::RefCell;

pub const SCHEMA: &str = "hu20-native-reopening-ordered-history-card-v1";
/// Menu names, each with its own bit, so a bitmask names a menu. v1's six names keep their bits;
/// the overbet menu adds three. `RANK` gives the canonical order: by size, raises ascending.
pub const NAMES: [&str; 9] = ["fold", "check", "call", "min", "pot", "jam", "half", "pot-1.5", "pot-2"];
const RANK: [u8; 9] = [0, 1, 2, 3, 5, 8, 4, 6, 7];

/// Which raise sizes the abstract menu offers. V1 is every model so far; V2 adds half-pot and
/// 1.5× and 2× pot overbets, with history labels fine enough to tell every size apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuVersion {
    V1,
    V2,
}

impl MenuVersion {
    /// The checkpoint identity's `action_menu`.
    pub fn name(self, game: Game) -> &'static str {
        match (self, game) {
            (MenuVersion::V1, game) => game.menu(),
            (MenuVersion::V2, Game::Hu20) => "hu20-min-half-pot-overbet-conditional-jam-native-reopening-v2",
            (MenuVersion::V2, Game::Hu100) => "hu100-min-half-pot-overbet-conditional-jam-native-reopening-v2",
            (MenuVersion::V2, Game::Hu200) => "hu200-min-half-pot-overbet-conditional-jam-native-reopening-v2",
        }
    }

    pub fn parse(name: &str, game: Game) -> Option<MenuVersion> {
        [MenuVersion::V1, MenuVersion::V2].into_iter().find(|m| m.name(game) == name)
    }
}

/// The key schema for a game, card descriptor and menu.
pub fn schema(game: Game, equity: bool, menu: MenuVersion) -> &'static str {
    match (menu, equity, game) {
        (MenuVersion::V1, false, game) => game.schema(),
        (MenuVersion::V1, true, game) => crate::cards::equity_schema(game),
        (MenuVersion::V2, false, Game::Hu20) => "hu20-native-reopening-ordered-history-card-menu-v2",
        (MenuVersion::V2, false, Game::Hu100) => "hu100-native-reopening-ordered-history-card-menu-v2",
        (MenuVersion::V2, false, Game::Hu200) => "hu200-native-reopening-ordered-history-card-menu-v2",
        (MenuVersion::V2, true, Game::Hu20) => "hu20-native-reopening-ordered-history-equity-k50-menu-v2",
        (MenuVersion::V2, true, Game::Hu100) => "hu100-native-reopening-ordered-history-equity-k50-menu-v2",
        (MenuVersion::V2, true, Game::Hu200) => "hu200-native-reopening-ordered-history-equity-k50-menu-v2",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choice {
    pub name: &'static str,
    pub action: Action,
}

#[derive(Clone, Copy, Debug)]
pub struct Menu {
    items: [Choice; MAX_ACTIONS],
    pub len: usize,
    /// Bit i set when `NAMES[i]` is offered; identifies the name tuple exactly.
    pub code: u16,
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

/// A code's names in canonical order.
pub fn names_of(code: u16) -> Vec<&'static str> {
    let mut indices: Vec<usize> = (0..NAMES.len()).filter(|i| code & (1 << i) != 0).collect();
    indices.sort_by_key(|&i| RANK[i]);
    indices.into_iter().map(|i| NAMES[i]).collect()
}

pub fn code_of(names: &[&str]) -> u16 {
    names.iter().fold(0, |code, name| code | 1 << NAMES.iter().position(|n| n == name).expect("menu name"))
}

/// The bit of a menu name and its canonical rank.
pub fn bit_and_rank(name: &str) -> Option<(usize, u8)> {
    NAMES.iter().position(|n| *n == name).map(|i| (i, RANK[i]))
}

/// Fold only when a check is unavailable; raises to min, pot and a conditional jam, and under
/// the overbet menu also to half, 1.5× and 2× pot. Each size is the raise above a call as a
/// fraction of the pot after calling, clamped to the legal bounds; a target already offered by
/// an earlier name is skipped.
pub fn menu(hand: &Core) -> Menu {
    let legal = hand.legal();
    let a = hand.actor.unwrap() as usize;
    let none = Choice { name: "", action: Action { kind: Kind::Fold, raise_to: 0 } };
    let mut out = Menu { items: [none; MAX_ACTIONS], len: 0, code: 0 };
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
        let sized = |increment: u32| high.min(low.max(matched + increment));
        let v1 = [(3usize, low), (4, sized(after_call)), (5, high)];
        let v2 = [(3usize, low), (6, sized(after_call / 2)), (4, sized(after_call)),
                  (7, sized(3 * after_call / 2)), (8, sized(2 * after_call)), (5, high)];
        let targets: &[(usize, u32)] = match hand.menu { MenuVersion::V1 => &v1, MenuVersion::V2 => &v2 };
        let mut used = [u32::MAX; 6];
        for (slot, &(index, target)) in targets.iter().enumerate() {
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
    out.extend_from_slice(hand.schema().as_bytes());
    out.extend_from_slice(b"\",2,");
    out.push(if (a as u8 + 2 - hand.button) % 2 == 0 { b'0' } else { b'1' });
    out.extend_from_slice(b",\"");
    out.extend_from_slice(hand.street.name().as_bytes());
    out.extend_from_slice(b"\",");
    if hand.street == Street::Preflop {
        push_preflop(out, hand.holes[a]);
    } else {
        match hand.cards {
            Cards::V1 => push_postflop(out, hand.holes[a], hand.visible_board()),
            Cards::Equity(tables) => {
                out.extend_from_slice(tables.bucket(hand.holes[a], hand.visible_board()).to_string().as_bytes())
            }
        }
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
        digest16(&buffer)
    })
}

/// The 16-byte BLAKE2b digest every information key uses.
pub fn digest16(bytes: &[u8]) -> [u8; 16] {
    let mut hasher = Blake2bVar::new(16).unwrap();
    hasher.update(bytes);
    let mut out = [0u8; 16];
    hasher.finalize_variable(&mut out).unwrap();
    out
}

pub fn key_hex(hand: &Hand, menu: &[Choice]) -> String {
    key_bytes(hand, menu).iter().map(|b| format!("{b:02x}")).collect()
}
