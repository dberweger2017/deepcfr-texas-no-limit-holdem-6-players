//! The training table, stored compactly.
//!
//! A `HashMap<Key, Node>` spends 112 bytes on every slot, whatever the menu's length, and
//! leaves up to half its slots empty after each doubling: about 415 bytes per entry at HU100's
//! 39M-node peak. Here each shard keeps its entries' fixed fields in one segmented array, their
//! regrets and averages at their actual length in another, and a hash index of 32-bit positions.
//! Keys stored only from opponent visits never receive regrets, which stay exactly +0.0, so they
//! keep no regret storage at all. Segments grow by fixed chunks, so memory grows in proportion
//! to entries instead of in doublings, and nothing is moved when a shard grows.
//!
//! Shards split keys by their first byte, so sorting each shard and writing them in order
//! gives the checkpoint's sorted rows without sorting the whole table at once.

use crate::cfr::{Key, Lookup, Node, MAX_ACTIONS};
use hashbrown::HashTable;

pub const SHARDS: usize = 64;

/// Regrets of keys without stored ones: `Node::empty`'s.
static ZEROS: [f64; MAX_ACTIONS] = [0.0; MAX_ACTIONS];
/// A position with no regret storage.
const NONE: u32 = u32::MAX;

/// Fixed-size chunks, so growth never copies or doubles existing items.
struct Segments<T, const SHIFT: u32> {
    chunks: Vec<Vec<T>>,
}

impl<T: Copy, const SHIFT: u32> Segments<T, SHIFT> {
    const CHUNK: usize = 1 << SHIFT;

    fn new() -> Self {
        Segments { chunks: Vec::new() }
    }

    /// The position of `items`, kept contiguous inside one chunk.
    fn push(&mut self, items: &[T]) -> u32 {
        debug_assert!(items.len() <= Self::CHUNK);
        if self.chunks.last().map_or(true, |c| c.len() + items.len() > Self::CHUNK) {
            self.chunks.push(Vec::with_capacity(Self::CHUNK));
        }
        let chunk = self.chunks.len() - 1;
        let last = &mut self.chunks[chunk];
        let position = (chunk << SHIFT) + last.len();
        last.extend_from_slice(items);
        u32::try_from(position).expect("a table shard outgrew 32-bit positions")
    }

    fn get(&self, position: u32, len: usize) -> &[T] {
        let (chunk, offset) = ((position >> SHIFT) as usize, position as usize & (Self::CHUNK - 1));
        &self.chunks[chunk][offset..offset + len]
    }

    fn get_mut(&mut self, position: u32, len: usize) -> &mut [T] {
        let (chunk, offset) = ((position >> SHIFT) as usize, position as usize & (Self::CHUNK - 1));
        &mut self.chunks[chunk][offset..offset + len]
    }
}

#[derive(Clone, Copy)]
struct Entry {
    key: Key,
    visits: u64,
    average: u32,
    regrets: u32,
    stamp: u32,
    code: u8,
    len: u8,
}

/// Keys are uniform BLAKE2b digests; the bytes after the shard byte are the hash.
fn hash(key: &Key) -> u64 {
    u64::from_le_bytes(key[8..16].try_into().unwrap())
}

pub struct Shard {
    index: HashTable<u32>,
    entries: Segments<Entry, 13>,
    values: Segments<f64, 15>,
}

impl Shard {
    fn new() -> Shard {
        Shard { index: HashTable::new(), entries: Segments::new(), values: Segments::new() }
    }

    fn entry(&self, i: u32) -> &Entry {
        &self.entries.get(i, 1)[0]
    }

    fn find(&self, key: &Key) -> Option<u32> {
        self.index.find(hash(key), |&i| self.entry(i).key == *key).copied()
    }

    fn regrets(&self, e: &Entry) -> &[f64] {
        if e.regrets == NONE { &ZEROS[..e.len as usize] } else { self.values.get(e.regrets, e.len as usize) }
    }

    fn node(&self, i: u32) -> Node {
        let e = *self.entry(i);
        let n = e.len as usize;
        let mut node = Node { stamp: e.stamp, visits: e.visits, ..Node::empty(e.code, n) };
        node.regrets[..n].copy_from_slice(self.regrets(&e));
        node.average[..n].copy_from_slice(self.values.get(e.average, n));
        node
    }

    /// Stores a new entry; regret storage only when some regret differs from +0.0.
    fn add(&mut self, key: Key, node: &Node) -> u32 {
        let n = node.len as usize;
        let average = self.values.push(&node.average[..n]);
        let regrets = if node.regrets[..n].iter().all(|r| r.to_bits() == 0) { NONE } else { self.values.push(&node.regrets[..n]) };
        let i = self.entries.push(&[Entry { key, visits: node.visits, average, regrets, stamp: node.stamp, code: node.code, len: node.len }]);
        let entries = &self.entries;
        self.index.insert_unique(hash(&key), i, |&j| hash(&entries.get(j, 1)[0].key));
        i
    }

    fn store(&mut self, i: u32, node: &Node) {
        let mut e = *self.entry(i);
        let n = e.len as usize;
        debug_assert!(node.code == e.code && node.len == e.len);
        self.values.get_mut(e.average, n).copy_from_slice(&node.average[..n]);
        if e.regrets != NONE {
            self.values.get_mut(e.regrets, n).copy_from_slice(&node.regrets[..n]);
        } else if node.regrets[..n].iter().any(|r| r.to_bits() != 0) {
            e.regrets = self.values.push(&node.regrets[..n]);
        }
        e.visits = node.visits;
        e.stamp = node.stamp;
        self.entries.get_mut(i, 1)[0] = e;
    }

    /// Applies `update` to `key`'s node, created as `fresh` when the key is new.
    pub fn update(&mut self, key: Key, fresh: impl FnOnce() -> Node, update: impl FnOnce(&mut Node)) {
        let (i, mut node) = match self.find(&key) {
            Some(i) => (i, self.node(i)),
            None => {
                let node = fresh();
                (self.add(key, &node), node)
            }
        };
        update(&mut node);
        self.store(i, &node);
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Entries in key order.
    fn sorted(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        let mut order: Vec<u32> = self.index.iter().copied().collect();
        order.sort_unstable_by(|&a, &b| self.entry(a).key.cmp(&self.entry(b).key));
        order.into_iter().map(move |i| (self.entry(i).key, self.node(i)))
    }
}

/// The table, split by key so deltas merge and apply in parallel.
pub struct Store(pub Vec<Shard>);

impl Store {
    pub fn new() -> Store {
        Store((0..SHARDS).map(|_| Shard::new()).collect())
    }

    pub fn shard(key: &Key) -> usize {
        key[0] as usize * SHARDS / 256
    }

    pub fn len(&self) -> usize {
        self.0.iter().map(Shard::len).sum()
    }

    pub fn get(&self, key: &Key) -> Option<Node> {
        let shard = &self.0[Self::shard(key)];
        shard.find(key).map(|i| shard.node(i))
    }

    /// Adds a new key's node; false if the key is already stored.
    pub fn insert(&mut self, key: Key, node: Node) -> bool {
        let shard = &mut self.0[Self::shard(&key)];
        if shard.find(&key).is_some() {
            return false;
        }
        shard.add(key, &node);
        true
    }

    /// Every entry, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        self.0.iter().flat_map(|s| s.index.iter().map(move |&i| (s.entry(i).key, s.node(i))))
    }

    /// Every entry in key order, sorting one shard at a time.
    pub fn sorted(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        self.0.iter().flat_map(Shard::sorted)
    }
}

impl Lookup for Store {
    fn regrets(&self, key: &Key) -> Option<(u8, &[f64])> {
        let shard = &self.0[Self::shard(key)];
        shard.find(key).map(|i| {
            let e = shard.entry(i);
            (e.code, shard.regrets(e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(code: u8, len: usize, regrets: &[f64], average: &[f64], visits: u64) -> Node {
        let mut node = Node { visits, ..Node::empty(code, len) };
        node.regrets[..len].copy_from_slice(regrets);
        node.average[..len].copy_from_slice(average);
        node
    }

    fn bits(node: &Node) -> (u8, u8, Vec<u64>, Vec<u64>, u64, u32) {
        let n = node.len as usize;
        (node.code, node.len, node.regrets[..n].iter().map(|v| v.to_bits()).collect(),
         node.average[..n].iter().map(|v| v.to_bits()).collect(), node.visits, node.stamp)
    }

    #[test]
    fn stores_nodes_exactly_and_allocates_regrets_only_when_needed() {
        let mut store = Store::new();
        let opponent = node(0b110, 2, &[0.0, 0.0], &[3.5, 1.0], 0);
        let negative_zero = node(0b11, 2, &[-0.0, 0.0], &[0.0, 0.0], 1);
        assert!(store.insert([1; 16], opponent));
        assert!(store.insert([2; 16], negative_zero));
        assert!(!store.insert([1; 16], opponent), "duplicate keys are rejected");
        assert_eq!(bits(&store.get(&[1; 16]).unwrap()), bits(&opponent));
        assert_eq!(bits(&store.get(&[2; 16]).unwrap()), bits(&negative_zero), "-0.0 keeps its sign");
        assert_eq!(store.0[0].entry(store.0[0].find(&[1; 16]).unwrap()).regrets, NONE);
        let (code, regrets) = store.regrets(&[1; 16]).unwrap();
        assert_eq!((code, regrets.len()), (0b110, 2));
        // A first traverser visit gives an opponent-only key its regret storage.
        let shard = &mut store.0[Store::shard(&[1; 16])];
        shard.update([1; 16], || unreachable!(), |n| { n.regrets[1] = -2.25; n.visits = 1; n.stamp = 7; });
        let expected = node(0b110, 2, &[0.0, -2.25], &[3.5, 1.0], 1);
        assert_eq!(bits(&store.get(&[1; 16]).unwrap()), bits(&Node { stamp: 7, ..expected }));
        assert!(store.get(&[3; 16]).is_none() && store.regrets(&[3; 16]).is_none());
    }

    #[test]
    fn many_keys_survive_index_growth_and_sort_by_key() {
        let mut store = Store::new();
        let key = |i: u64| {
            let mut k = [0u8; 16];
            // Spread keys over shards and hashes like digests, with some chunk-crossing menus.
            let x = i.wrapping_mul(0x9E3779B97F4A7C15);
            k[..8].copy_from_slice(&x.to_be_bytes());
            k[8..].copy_from_slice(&x.rotate_left(29).to_le_bytes());
            k
        };
        let count = 50_000u64;
        for i in 0..count {
            let len = 1 + (i % 5) as usize;
            let values: Vec<f64> = (0..len).map(|j| (i * 7 + j as u64) as f64 - 3.0).collect();
            assert!(store.insert(key(i), node((1u8 << len) - 1, len, &values, &values, i)));
        }
        assert_eq!(store.len(), count as usize);
        for i in (0..count).step_by(97) {
            let n = store.get(&key(i)).unwrap();
            assert_eq!((n.visits, n.len as u64), (i, 1 + i % 5));
            assert_eq!(n.average[n.len as usize - 1], (i * 7 + i % 5) as f64 - 3.0);
        }
        let sorted: Vec<Key> = store.sorted().map(|(k, _)| k).collect();
        assert_eq!(sorted.len(), count as usize);
        assert!(sorted.windows(2).all(|w| w[0] < w[1]));
    }
}
