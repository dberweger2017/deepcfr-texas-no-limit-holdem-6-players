//! Equity-distribution card abstraction tables for heads-up Hold'em.
//!
//! Cards are `rank * 4 + suit`, ranks 0..13 for 2..A and suits 0..4 for c, d, h, s.
//! A situation is a two-card holding plus a board. Suit-isomorphic situations
//! share one class, keyed by the sorted per-suit (holding mask, board mask)
//! signatures, so every class is computed once and weighted by its orbit size.

use rayon::prelude::*;

pub const RANKS: &[u8] = b"23456789TJQKA";
pub const SUITS: &[u8] = b"cdhs";

pub fn parse_card(text: &str) -> u8 {
    let bytes = text.as_bytes();
    assert!(bytes.len() == 2, "card {text}");
    let rank = RANKS.iter().position(|&r| r == bytes[0]).expect("rank") as u8;
    let suit = SUITS.iter().position(|&s| s == bytes[1]).expect("suit") as u8;
    rank * 4 + suit
}

pub fn card_name(card: u8) -> String {
    format!("{}{}", RANKS[(card / 4) as usize] as char, SUITS[(card % 4) as usize] as char)
}

/// Cards whose rank is among the top `ranks` ranks; 13 is the full deck.
pub fn deck(ranks: u8) -> Vec<u8> {
    (0..52u8).filter(|c| c / 4 >= 13 - ranks).collect()
}

fn straight_high(mask: u16) -> Option<u32> {
    for high in (4..13u32).rev() {
        let window = 0b11111u16 << (high - 4);
        if mask & window == window {
            return Some(high);
        }
    }
    if mask & 0x100F == 0x100F { Some(3) } else { None }
}

fn top(mask: u16, n: usize) -> u32 {
    let mut value = 0u32;
    let mut taken = 0;
    for rank in (0..13u32).rev() {
        if taken == n {
            break;
        }
        if mask & (1 << rank) != 0 {
            value = (value << 4) | rank;
            taken += 1;
        }
    }
    value
}

/// Hand strength of five to seven cards; larger is better, equal is a tie.
pub fn evaluate(cards: &[u8]) -> u32 {
    let mut suits = [0u16; 4];
    let mut counts = [0u8; 13];
    let mut all = 0u16;
    for &card in cards {
        let (rank, suit) = ((card / 4) as usize, (card % 4) as usize);
        suits[suit] |= 1 << rank;
        counts[rank] += 1;
        all |= 1 << rank;
    }
    // A flush excludes quads and full houses within seven cards.
    for mask in suits {
        if mask.count_ones() >= 5 {
            return match straight_high(mask) {
                Some(high) => (8 << 20) | high,
                None => (5 << 20) | top(mask, 5),
            };
        }
    }
    let (mut quads, mut trips, mut pairs) = (None, 0u16, 0u16);
    for rank in 0..13 {
        match counts[rank] {
            4 => quads = Some(rank as u32),
            3 => trips |= 1 << rank,
            2 => pairs |= 1 << rank,
            _ => {}
        }
    }
    if let Some(q) = quads {
        return (7 << 20) | (q << 4) | top(all & !(1 << q), 1);
    }
    if trips != 0 {
        let t = 15 - trips.leading_zeros();
        let rest = (trips & !(1 << t)) | pairs;
        if rest != 0 {
            return (6 << 20) | (t << 4) | (15 - rest.leading_zeros());
        }
    }
    if let Some(high) = straight_high(all) {
        return (4 << 20) | high;
    }
    if trips != 0 {
        let t = 15 - trips.leading_zeros();
        return (3 << 20) | (t << 8) | top(all & !(1 << t), 2);
    }
    if pairs.count_ones() >= 2 {
        let high = 15 - pairs.leading_zeros();
        let low = 15 - (pairs & !(1 << high)).leading_zeros();
        return (2 << 20) | (high << 8) | (low << 4) | top(all & !(1 << high) & !(1 << low), 1);
    }
    if pairs != 0 {
        let p = 15 - pairs.leading_zeros();
        return (1 << 20) | (p << 12) | top(all & !(1 << p), 3);
    }
    top(all, 5)
}

pub fn category(value: u32) -> u32 {
    value >> 20
}

fn signatures(hole: &[u8], board: &[u8]) -> [u32; 4] {
    let mut sig = [0u32; 4];
    for &card in hole {
        sig[(card % 4) as usize] |= 1 << (13 + card / 4);
    }
    for &card in board {
        sig[(card % 4) as usize] |= 1 << (card / 4);
    }
    sig
}

/// Suit-isomorphism class key: per-suit 26-bit signatures, sorted, packed.
pub fn class_key(hole: &[u8], board: &[u8]) -> u128 {
    let mut sig = signatures(hole, board);
    sig.sort_unstable_by(|a, b| b.cmp(a));
    sig.iter().fold(0u128, |key, &s| (key << 26) | s as u128)
}

/// Raw situations in the class: 24 suit permutations over the stabilizer size.
pub fn orbit_size(hole: &[u8], board: &[u8]) -> u64 {
    let mut sig = signatures(hole, board);
    sig.sort_unstable();
    let mut stabilizer = 1u64;
    let mut run = 1u64;
    for i in 1..4 {
        if sig[i] == sig[i - 1] {
            run += 1;
            stabilizer *= run;
        } else {
            run = 1;
        }
    }
    24 / stabilizer
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// The 64-bit table key; the build refuses any collision among its classes.
pub fn hash_key(key: u128) -> u64 {
    splitmix(((key >> 64) as u64) ^ splitmix(key as u64))
}

pub fn combinations(cards: &[u8], k: usize) -> Vec<Vec<u8>> {
    fn go(cards: &[u8], k: usize, start: usize, current: &mut Vec<u8>, out: &mut Vec<Vec<u8>>) {
        if current.len() == k {
            out.push(current.clone());
            return;
        }
        for i in start..cards.len() {
            if cards.len() - i < k - current.len() {
                break;
            }
            current.push(cards[i]);
            go(cards, k, i + 1, current, out);
            current.pop();
        }
    }
    let mut out = Vec::new();
    go(cards, k, 0, &mut Vec::new(), &mut out);
    out
}

/// One representative board per suit-isomorphism class.
pub fn canonical_boards(deck: &[u8], size: usize) -> Vec<Vec<u8>> {
    let mut keyed: Vec<(u128, Vec<u8>)> =
        combinations(deck, size).into_iter().map(|b| (class_key(&[], &b), b)).collect();
    keyed.par_sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    keyed.dedup_by(|a, b| a.0 == b.0);
    keyed.into_iter().map(|(_, b)| b).collect()
}

/// Equity of every holding on a complete board against a uniform random
/// opponent holding, with exact card removal; ties count one half.
pub fn river_equities(board: &[u8], deck: &[u8]) -> Vec<([u8; 2], f32)> {
    let rest: Vec<u8> = deck.iter().copied().filter(|c| !board.contains(c)).collect();
    let mut hands: Vec<([u8; 2], u32)> = Vec::with_capacity(rest.len() * rest.len() / 2);
    let mut cards = [0u8; 7];
    cards[2..].copy_from_slice(board);
    for i in 0..rest.len() {
        for j in i + 1..rest.len() {
            cards[0] = rest[i];
            cards[1] = rest[j];
            hands.push(([rest[i], rest[j]], evaluate(&cards)));
        }
    }
    hands.sort_unstable_by_key(|h| h.1);
    let opponents = ((rest.len() - 2) * (rest.len() - 3) / 2) as f32;
    let mut lower = [0u32; 52];
    let mut lower_total = 0u32;
    let mut out = Vec::with_capacity(hands.len());
    let mut start = 0;
    while start < hands.len() {
        let mut end = start;
        while end < hands.len() && hands[end].1 == hands[start].1 {
            end += 1;
        }
        let mut group = [0u32; 52];
        for h in &hands[start..end] {
            group[h.0[0] as usize] += 1;
            group[h.0[1] as usize] += 1;
        }
        let size = (end - start) as u32;
        for h in &hands[start..end] {
            let (a, b) = (h.0[0] as usize, h.0[1] as usize);
            let wins = lower_total - lower[a] - lower[b];
            let ties = size + 1 - group[a] - group[b];
            out.push((h.0, (wins as f32 + 0.5 * ties as f32) / opponents));
        }
        for h in &hands[start..end] {
            lower[h.0[0] as usize] += 1;
            lower[h.0[1] as usize] += 1;
        }
        lower_total += size;
        start = end;
    }
    out
}

/// River classes, sorted by key, with equity and orbit weight.
pub struct RiverTable {
    pub keys: Vec<u128>,
    pub equity: Vec<f32>,
    pub weight: Vec<u64>,
}

impl RiverTable {
    pub fn build(deck: &[u8]) -> RiverTable {
        let boards = canonical_boards(deck, 5);
        let mut rows: Vec<(u128, f32, u64)> = boards
            .par_iter()
            .flat_map_iter(|board| {
                river_equities(board, deck).into_iter().map(move |(hole, eq)| {
                    (class_key(&hole, board), eq, orbit_size(&hole, board))
                })
            })
            .collect();
        rows.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        rows.dedup_by(|a, b| {
            if a.0 == b.0 {
                assert!(a.1 == b.1, "isomorphic river situations disagree on equity");
                true
            } else {
                false
            }
        });
        RiverTable {
            keys: rows.iter().map(|r| r.0).collect(),
            equity: rows.iter().map(|r| r.1).collect(),
            weight: rows.iter().map(|r| r.2).collect(),
        }
    }

    pub fn equity_of(&self, hole: &[u8], board: &[u8]) -> f32 {
        let key = class_key(hole, board);
        let index = self.keys.binary_search(&key).expect("river class missing");
        self.equity[index]
    }
}

fn bin(equity: f32, bins: usize) -> usize {
    ((equity * bins as f32) as usize).min(bins - 1)
}

/// Classes of an earlier street with an equity histogram over all runouts.
pub struct StreetFeatures {
    pub keys: Vec<u128>,
    pub weight: Vec<u64>,
    pub mean_equity: Vec<f32>,
    /// Row-major cumulative histograms (CDFs), `bins` values per class.
    pub cdf: Vec<f32>,
    pub bins: usize,
}

impl StreetFeatures {
    pub fn build(deck: &[u8], board_size: usize, bins: usize, river: &RiverTable) -> StreetFeatures {
        let boards = canonical_boards(deck, board_size);
        let mut reps: Vec<(u128, [u8; 2], Vec<u8>)> = boards
            .par_iter()
            .flat_map_iter(|board| {
                let rest: Vec<u8> = deck.iter().copied().filter(|c| !board.contains(c)).collect();
                let mut found = Vec::new();
                for i in 0..rest.len() {
                    for j in i + 1..rest.len() {
                        let hole = [rest[i], rest[j]];
                        found.push((class_key(&hole, board), hole, board.clone()));
                    }
                }
                found
            })
            .collect();
        reps.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        reps.dedup_by(|a, b| a.0 == b.0);
        let rows: Vec<(u64, f32, Vec<f32>)> = reps
            .par_iter()
            .map(|(_, hole, board)| {
                let rest: Vec<u8> =
                    deck.iter().copied().filter(|c| !board.contains(c) && !hole.contains(c)).collect();
                let mut counts = vec![0u32; bins];
                let (mut total, mut sum) = (0u32, 0f64);
                for runout in combinations(&rest, 5 - board_size) {
                    let mut full = board.clone();
                    full.extend_from_slice(&runout);
                    let eq = river.equity_of(hole, &full);
                    counts[bin(eq, bins)] += 1;
                    total += 1;
                    sum += eq as f64;
                }
                let mut running = 0u32;
                let cdf = counts
                    .iter()
                    .map(|&c| {
                        running += c;
                        running as f32 / total as f32
                    })
                    .collect();
                (orbit_size(hole, board), (sum / total as f64) as f32, cdf)
            })
            .collect();
        StreetFeatures {
            keys: reps.iter().map(|r| r.0).collect(),
            weight: rows.iter().map(|r| r.0).collect(),
            mean_equity: rows.iter().map(|r| r.1).collect(),
            cdf: rows.into_iter().flat_map(|r| r.2).collect(),
            bins,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = splitmix(self.0);
        self.0
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn l1(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

pub struct Clustering {
    pub assignment: Vec<u16>,
    pub centroids: Vec<f32>,
    pub objective: Vec<f64>,
}

/// Weighted k-means under L1 between rows (earth mover's distance for CDFs),
/// with k-means++ seeding on a weighted sample and mean centroid updates.
pub fn kmeans(points: &[f32], dim: usize, weight: &[u64], k: usize, seed: u64, max_iterations: usize) -> Clustering {
    let n = weight.len();
    assert!(points.len() == n * dim && k <= n && k <= u16::MAX as usize);
    let row = |i: usize| &points[i * dim..(i + 1) * dim];
    let mut rng = Rng(seed);
    let total: f64 = weight.iter().map(|&w| w as f64).sum();
    // Weighted sample of candidates for seeding.
    let cumulative: Vec<f64> = weight
        .iter()
        .scan(0f64, |acc, &w| {
            *acc += w as f64;
            Some(*acc)
        })
        .collect();
    let draw = |rng: &mut Rng| -> usize {
        let target = rng.unit() * total;
        cumulative.partition_point(|&c| c <= target).min(n - 1)
    };
    let sample: Vec<usize> = (0..n.min(200_000)).map(|_| draw(&mut rng)).collect();
    let mut centroids: Vec<f32> = row(sample[0]).to_vec();
    let mut nearest: Vec<f32> = sample.iter().map(|&i| l1(row(i), &centroids[0..dim])).collect();
    while centroids.len() < k * dim {
        let sum: f64 = nearest.iter().map(|&d| d as f64).sum();
        let pick = if sum > 0.0 {
            let target = rng.unit() * sum;
            let mut acc = 0f64;
            let mut chosen = sample.len() - 1;
            for (s, &d) in nearest.iter().enumerate() {
                acc += d as f64;
                if acc > target {
                    chosen = s;
                    break;
                }
            }
            chosen
        } else {
            (rng.next() % sample.len() as u64) as usize
        };
        let start = centroids.len();
        centroids.extend_from_slice(row(sample[pick]));
        let fresh = centroids[start..start + dim].to_vec();
        nearest.par_iter_mut().zip(&sample).for_each(|(d, &i)| *d = d.min(l1(row(i), &fresh)));
    }
    let mut assignment = vec![u16::MAX; n];
    let mut objective = Vec::new();
    for _ in 0..max_iterations {
        let assigned: Vec<(u16, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let p = row(i);
                let mut best = (0u16, f32::INFINITY);
                for c in 0..k {
                    let d = l1(p, &centroids[c * dim..(c + 1) * dim]);
                    if d < best.1 {
                        best = (c as u16, d);
                    }
                }
                best
            })
            .collect();
        let moved: f64 = assigned
            .iter()
            .zip(&assignment)
            .zip(weight)
            .filter(|((a, old), _)| a.0 != **old)
            .map(|(_, &w)| w as f64)
            .sum();
        objective.push(assigned.iter().zip(weight).map(|(a, &w)| a.1 as f64 * w as f64).sum::<f64>() / total);
        for (slot, a) in assignment.iter_mut().zip(&assigned) {
            *slot = a.0;
        }
        let mut sums = vec![0f64; k * dim];
        let mut mass = vec![0f64; k];
        for i in 0..n {
            let c = assignment[i] as usize;
            let w = weight[i] as f64;
            mass[c] += w;
            for (s, &x) in sums[c * dim..(c + 1) * dim].iter_mut().zip(row(i)) {
                *s += w * x as f64;
            }
        }
        for c in 0..k {
            if mass[c] > 0.0 {
                for d in 0..dim {
                    centroids[c * dim + d] = (sums[c * dim + d] / mass[c]) as f32;
                }
            }
        }
        if moved / total < 1e-4 {
            break;
        }
    }
    Clustering { assignment, centroids, objective }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cards(text: &str) -> Vec<u8> {
        text.split_whitespace().map(parse_card).collect()
    }

    #[test]
    fn evaluator_orders_categories_and_kickers() {
        let order = [
            "2c 3d 5h 7s 9c Jd Kh", // high card
            "2c 2d 5h 7s 9c Jd Kh", // pair
            "2c 2d 5h 5s 9c Jd Kh", // two pair
            "2c 2d 2h 7s 9c Jd Kh", // trips
            "Ac 2d 3h 4s 5c Jd Kh", // wheel
            "6c 2d 3h 4s 5c Jd Kh", // six-high straight
            "2c 4c 6c 8c Tc Jd Kh", // flush
            "2c 2d 2h 7s 7c Jd Kh", // full house
            "2c 2d 2h 2s 7c Jd Kh", // quads
            "Ac 2c 3c 4c 5c Jd Kh", // steel wheel
            "Tc Jc Qc Kc Ac 2d 3h", // royal
        ];
        let values: Vec<u32> = order.iter().map(|h| evaluate(&cards(h))).collect();
        for pair in values.windows(2) {
            assert!(pair[0] < pair[1], "{values:?}");
        }
        // Kickers and board plays.
        assert!(evaluate(&cards("Ac Ad Kh 7s 5c 3d 2h")) > evaluate(&cards("Ac Ad Qh 7s 5c 3d 2h")));
        assert_eq!(evaluate(&cards("2c 3d Ah Ks Qc Jd Th")), evaluate(&cards("4c 5d Ah Ks Qc Jd Th")));
        // Two trips make a full house with the higher trips.
        assert!(evaluate(&cards("Kc Kd Kh 9s 9c 9d 2h")) > evaluate(&cards("Qc Qd Qh As Ac 3d 2h")));
    }

    /// Exact category counts over all 133,784,560 seven-card hands. Slow; release only.
    #[test]
    #[ignore]
    fn evaluator_matches_all_seven_card_category_counts() {
        let expected = [23_294_460u64, 58_627_800, 31_433_400, 6_461_620, 6_180_020, 4_047_644, 3_473_184, 224_848, 41_584];
        let counts = (0..52u8)
            .into_par_iter()
            .map(|a| {
                let mut local = [0u64; 9];
                for b in a + 1..52 {
                    for c in b + 1..52 {
                        for d in c + 1..52 {
                            for e in d + 1..52 {
                                for f in e + 1..52 {
                                    for g in f + 1..52 {
                                        local[category(evaluate(&[a, b, c, d, e, f, g])) as usize] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                local
            })
            .reduce(|| [0u64; 9], |mut x, y| {
                for i in 0..9 {
                    x[i] += y[i];
                }
                x
            });
        assert_eq!(counts, expected);
    }

    #[test]
    fn class_keys_respect_suit_symmetry_only() {
        let a = class_key(&cards("Ah Kh"), &cards("2h 7c 9d"));
        let b = class_key(&cards("As Ks"), &cards("2s 7d 9c"));
        let c = class_key(&cards("Ah Kd"), &cards("2h 7c 9d"));
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(orbit_size(&cards("Ah Kh"), &cards("2h 7c 9d")), 24);
        assert_eq!(orbit_size(&cards("Ah Ad"), &[]), 6);
    }

    #[test]
    fn river_equity_matches_brute_force() {
        let deck = deck(13);
        let board = cards("Ah Kd 7c 7s 2h");
        let table = river_equities(&board, &deck);
        for (hole, eq) in table.iter().step_by(97) {
            let mine = evaluate(&[hole[0], hole[1], board[0], board[1], board[2], board[3], board[4]]);
            let (mut score, mut count) = (0f64, 0f64);
            let rest: Vec<u8> = deck.iter().copied().filter(|c| !board.contains(c) && !hole.contains(c)).collect();
            for o in combinations(&rest, 2) {
                let theirs = evaluate(&[o[0], o[1], board[0], board[1], board[2], board[3], board[4]]);
                score += if mine > theirs { 1.0 } else if mine == theirs { 0.5 } else { 0.0 };
                count += 1.0;
            }
            assert!((score / count - *eq as f64).abs() < 1e-6, "{hole:?}");
        }
    }

    #[test]
    fn small_deck_orbits_cover_every_raw_situation() {
        let deck = deck(5);
        let river = RiverTable::build(&deck);
        let n = deck.len() as u64;
        let choose = |n: u64, k: u64| (0..k).fold(1u64, |acc, i| acc * (n - i) / (i + 1));
        assert_eq!(river.weight.iter().sum::<u64>(), choose(n, 5) * choose(n - 5, 2));
        let turn = StreetFeatures::build(&deck, 4, 10, &river);
        assert_eq!(turn.weight.iter().sum::<u64>(), choose(n, 4) * choose(n - 4, 2));
        let flop = StreetFeatures::build(&deck, 3, 10, &river);
        assert_eq!(flop.weight.iter().sum::<u64>(), choose(n, 3) * choose(n - 3, 2));
        let clusters = kmeans(&turn.cdf, turn.bins, &turn.weight, 8, 1, 50);
        assert_eq!(clusters.assignment.len(), turn.keys.len());
        assert!(clusters.assignment.iter().all(|&c| c < 8));
        assert!(clusters.objective.iter().all(|o| o.is_finite()));
        // Distinct classes must not collide in the 64-bit table key.
        let mut hashes: Vec<u64> = river.keys.iter().map(|&k| hash_key(k)).collect();
        hashes.sort_unstable();
        assert!(hashes.windows(2).all(|w| w[0] != w[1]));
    }
}
