//! The Python trainer's random streams, reproduced exactly.
//!
//! Each root `(iteration, seat, sample)` gets two seeds from `_seed`: a SHA-256 of the root's name.
//! The deck is the pinned engine's `State.from_seed` shuffle; opponent actions come from
//! CPython's `random.Random(seed).choices`. With them a native run equals the Python run with
//! the same seed, table entry for table entry.

pub struct Mt {
    state: [u32; 624],
    index: usize,
}

impl Mt {
    fn init_genrand(s: u32) -> Mt {
        let mut state = [0u32; 624];
        state[0] = s;
        for i in 1..624 {
            state[i] = 1812433253u32.wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt { state, index: 624 }
    }

    /// `Random(seed)` for a non-negative integer seed: `init_by_array` over its 32-bit words.
    pub fn new(seed: u64) -> Mt {
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        let mut mt = Mt::init_genrand(19650218);
        let s = &mut mt.state;
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                s[0] = s[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            s[i] = (s[i] ^ (s[i - 1] ^ (s[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                s[0] = s[623];
                i = 1;
            }
        }
        s[0] = 0x8000_0000;
        mt
    }

    /// The next output. Each state word is regenerated just before it's read, in the batch
    /// twist's order, so its inputs are the same old or new words the batch loop would see;
    /// a root that draws a few dozen numbers skips regenerating the other ~600.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.index = 0;
        }
        let k = self.index;
        let y = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % 624] & 0x7fff_ffff);
        let mut v = self.state[(k + 397) % 624] ^ (y >> 1);
        if y & 1 != 0 {
            v ^= 0x9908_b0df;
        }
        self.state[k] = v;
        self.index += 1;
        let mut y = v;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random()`: 53 bits from two draws.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `_randbelow_with_getrandbits(n)` for n < 2**32.
    pub fn below(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.next_u32() >> (32 - k);
            if r < n {
                return r;
            }
        }
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            items.swap(i, j);
        }
    }
}

impl crate::cfr::Sampler for Mt {
    /// `random.choices(range(n), weights=policy)`.
    fn sample(&mut self, policy: &[f64]) -> usize {
        let mut cumulative = [0f64; crate::cfr::MAX_ACTIONS];
        let mut running = 0.0;
        for (c, &p) in cumulative.iter_mut().zip(policy) {
            running += p;
            *c = running;
        }
        let target = self.random() * (running + 0.0);
        cumulative[..policy.len() - 1].partition_point(|&c| c <= target)
    }
}

/// The Python trainer's `_seed`: the first eight bytes, big-endian, of a SHA-256 over the root's name.
pub fn python_seed(seed: u64, iteration: u64, seat: usize, sample: usize, stream: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(format!("holdem-blueprint-v1/{seed}/{iteration}/{seat}/{sample}/{stream}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::Mt;

    #[test]
    fn matches_cpython() {
        // random.Random(s).shuffle(list(range(52))), then .random(), from CPython 3.11.
        for (seed, prefix, next) in [
            (0u64, [28, 12, 45, 41, 38, 7, 5, 36, 1, 49, 33, 0], 0.7302785763532248),
            (12345, [8, 25, 50, 40, 15, 3, 28, 1, 39, 34, 9, 24], 0.23262110322840768),
            ((1 << 40) + 7, [20, 22, 10, 12, 15, 21, 3, 13, 34, 6, 17, 7], 0.9429078635370134),
        ] {
            let mut mt = Mt::new(seed);
            let mut deck: Vec<u32> = (0..52).collect();
            mt.shuffle(&mut deck);
            assert_eq!(&deck[..12], &prefix[..]);
            assert_eq!(mt.random(), next);
        }
        // Past two state generations: Random(99).getrandbits(32) for draws 0, 623, 624 and 1500.
        let mut mt = Mt::new(99);
        let draws: Vec<u32> = (0..1501).map(|_| mt.next_u32()).collect();
        assert_eq!([draws[0], draws[623], draws[624], draws[1500]], [1735072617, 1744958591, 4259065050, 2968257009]);
    }
}

/// `pokers.State.from_seed`'s deck: `Card::collect()` (suit-major, clubs to spades, two to ace)
/// shuffled by rand 0.8.8 `StdRng::seed_from_u64(seed)`, in this crate's card numbering.
pub fn engine_deck(seed: u64) -> Vec<u8> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut deck: Vec<u8> = (0..52u8).map(|i| (i % 13) * 4 + i / 13).collect();
    deck.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
    deck
}
