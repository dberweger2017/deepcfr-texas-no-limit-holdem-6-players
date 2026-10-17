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
//! Both trainers use it. Each entry carries one opaque 32-bit `meta` (the HU discount stamp, the
//! n-player owner and street), and averages are optional for streets the n-player trainer does
//! not average.
//!
//! Shards split keys by their first byte, so sorting each shard and writing them in order
//! gives the checkpoint's sorted rows without sorting the whole table at once.

use crate::cfr::{Key, Lookup, Node, MAX_ACTIONS};
use hashbrown::HashTable;

pub const SHARDS: usize = 64;

/// Regrets of keys without stored ones: `Node::empty`'s.
static ZEROS: [f64; MAX_ACTIONS] = [0.0; MAX_ACTIONS];
/// A position with no storage.
const NONE: u32 = u32::MAX;

/// Fixed-size chunks, so growth never copies or doubles existing items.
struct Segments<T, const SHIFT: u32> {
    chunks: Vec<Vec<T>>,
}

/// Copies keep every chunk's full capacity, so a copy also grows without moving items.
impl<T: Copy, const SHIFT: u32> Clone for Segments<T, SHIFT> {
    fn clone(&self) -> Self {
        let mut chunks = Vec::with_capacity(self.chunks.capacity());
        for chunk in &self.chunks {
            let mut copy = Vec::with_capacity(Self::CHUNK);
            copy.extend_from_slice(chunk);
            chunks.push(copy);
        }
        Segments { chunks }
    }
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

    fn allocation_size(&self) -> usize {
        self.chunks.capacity() * std::mem::size_of::<Vec<T>>()
            + self.chunks.iter().map(|c| c.capacity() * std::mem::size_of::<T>()).sum::<usize>()
    }
}

#[derive(Clone, Copy)]
struct Entry {
    key: Key,
    visits: u64,
    average: u32,
    regrets: u32,
    meta: u32,
    code: u16,
    len: u8,
}

/// Bytes of one entry's fixed fields.
pub const ENTRY_BYTES: usize = std::mem::size_of::<Entry>();

/// Heap bytes the table owns, by part; `values` and `entries` include unused chunk capacity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Allocation {
    pub index: usize,
    pub entries: usize,
    pub values: usize,
}

impl Allocation {
    pub fn total(self) -> usize {
        self.index + self.entries + self.values
    }
}

/// A stored entry, or one to store; `average` is `None` where none is kept.
#[derive(Clone, Copy, Debug)]
pub struct View<'a> {
    pub code: u16,
    pub meta: u32,
    pub visits: u64,
    pub regrets: &'a [f64],
    pub average: Option<&'a [f64]>,
}

/// Keys are uniform BLAKE2b digests; the bytes after the shard byte are the hash.
fn hash(key: &Key) -> u64 {
    u64::from_le_bytes(key[8..16].try_into().unwrap())
}

#[derive(Clone)]
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

    fn view(&self, i: u32) -> View<'_> {
        let e = self.entry(i);
        let average = (e.average != NONE).then(|| self.values.get(e.average, e.len as usize));
        View { code: e.code, meta: e.meta, visits: e.visits, regrets: self.regrets(e), average }
    }

    fn node(&self, i: u32) -> Node {
        let v = self.view(i);
        let n = v.regrets.len();
        let mut node = Node { stamp: v.meta, visits: v.visits, ..Node::empty(v.code, n) };
        node.regrets[..n].copy_from_slice(v.regrets);
        node.average[..n].copy_from_slice(v.average.expect("HU entries keep averages"));
        node
    }

    /// Stores a new entry; regret storage only when some regret differs from +0.0.
    fn add(&mut self, key: Key, entry: View) -> u32 {
        let n = entry.regrets.len();
        debug_assert!((1..=MAX_ACTIONS).contains(&n) && entry.average.map_or(true, |a| a.len() == n));
        let average = entry.average.map_or(NONE, |a| self.values.push(a));
        let regrets = if entry.regrets.iter().all(|r| r.to_bits() == 0) { NONE } else { self.values.push(entry.regrets) };
        let i = self.entries.push(&[Entry { key, visits: entry.visits, average, regrets, meta: entry.meta, code: entry.code, len: n as u8 }]);
        let entries = &self.entries;
        self.index.insert_unique(hash(&key), i, |&j| hash(&entries.get(j, 1)[0].key));
        i
    }

    fn add_node(&mut self, key: Key, node: &Node) -> u32 {
        let n = node.len as usize;
        self.add(key, View { code: node.code, meta: node.stamp, visits: node.visits, regrets: &node.regrets[..n], average: Some(&node.average[..n]) })
    }

    /// Writes `e`'s regrets, giving it regret storage once one differs from +0.0.
    fn set_regrets(&mut self, e: &mut Entry, regrets: &[f64]) {
        if e.regrets != NONE {
            self.values.get_mut(e.regrets, regrets.len()).copy_from_slice(regrets);
        } else if regrets.iter().any(|r| r.to_bits() != 0) {
            e.regrets = self.values.push(regrets);
        }
    }

    fn store(&mut self, i: u32, node: &Node) {
        let mut e = *self.entry(i);
        let n = e.len as usize;
        debug_assert!(node.code == e.code && node.len == e.len);
        self.values.get_mut(e.average, n).copy_from_slice(&node.average[..n]);
        self.set_regrets(&mut e, &node.regrets[..n]);
        e.visits = node.visits;
        e.meta = node.stamp;
        self.entries.get_mut(i, 1)[0] = e;
    }

    /// Applies `update` to `key`'s node, created as `fresh` when the key is new.
    pub fn update(&mut self, key: Key, fresh: impl FnOnce() -> Node, update: impl FnOnce(&mut Node)) {
        let (i, mut node) = match self.find(&key) {
            Some(i) => (i, self.node(i)),
            None => {
                let node = fresh();
                (self.add_node(key, &node), node)
            }
        };
        update(&mut node);
        self.store(i, &node);
    }

    /// Adds `delta`'s regrets, averages and visits to `key`'s entry, or stores `delta` as it is
    /// when the key is new.
    pub fn merge(&mut self, key: Key, delta: View) {
        let Some(i) = self.find(&key) else {
            self.add(key, delta);
            return;
        };
        let mut e = *self.entry(i);
        let n = e.len as usize;
        assert_eq!((e.code, n, e.meta, e.average != NONE), (delta.code, delta.regrets.len(), delta.meta, delta.average.is_some()),
            "a merged entry changed its menu, meta or average scope");
        let mut regrets = [0.0; MAX_ACTIONS];
        for ((sum, stored), inc) in regrets.iter_mut().zip(self.regrets(&e)).zip(delta.regrets) {
            *sum = stored + inc;
            assert!(sum.is_finite());
        }
        self.set_regrets(&mut e, &regrets[..n]);
        if let Some(increments) = delta.average {
            for (sum, inc) in self.values.get_mut(e.average, n).iter_mut().zip(increments) {
                *sum += inc;
                assert!(sum.is_finite());
            }
        }
        e.visits += delta.visits;
        self.entries.get_mut(i, 1)[0] = e;
    }

    pub fn contains(&self, key: &Key) -> bool {
        self.find(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn allocation(&self) -> Allocation {
        Allocation { index: self.index.allocation_size(), entries: self.entries.allocation_size(), values: self.values.allocation_size() }
    }

    /// Bytes of stored regrets and stored averages.
    pub fn payload_bytes(&self) -> (usize, usize) {
        self.index.iter().fold((0, 0), |(regrets, average), &i| {
            let e = self.entry(i);
            let bytes = e.len as usize * 8;
            (regrets + if e.regrets == NONE { 0 } else { bytes }, average + if e.average == NONE { 0 } else { bytes })
        })
    }

    /// Entry positions in key order.
    pub fn order(&self) -> Vec<u32> {
        let mut order: Vec<u32> = self.index.iter().copied().collect();
        order.sort_unstable_by(|&a, &b| self.entry(a).key.cmp(&self.entry(b).key));
        order
    }

    /// The key and node at a position from `order`.
    pub fn row(&self, i: u32) -> (Key, Node) {
        (self.entry(i).key, self.node(i))
    }

    /// The key and entry at a position from `order`.
    pub fn row_view(&self, i: u32) -> (Key, View<'_>) {
        (self.entry(i).key, self.view(i))
    }

    /// Entries in key order.
    fn sorted(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        self.order().into_iter().map(move |i| self.row(i))
    }

    fn sorted_views(&self) -> impl Iterator<Item = (Key, View<'_>)> + '_ {
        self.order().into_iter().map(move |i| self.row_view(i))
    }
}

/// The table, split by key so deltas merge and apply in parallel.
#[derive(Clone)]
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

    pub fn view(&self, key: &Key) -> Option<View<'_>> {
        let shard = &self.0[Self::shard(key)];
        shard.find(key).map(|i| shard.view(i))
    }

    pub fn contains(&self, key: &Key) -> bool {
        self.0[Self::shard(key)].contains(key)
    }

    /// Adds a new key's node; false if the key is already stored.
    pub fn insert(&mut self, key: Key, node: Node) -> bool {
        let shard = &mut self.0[Self::shard(&key)];
        if shard.find(&key).is_some() {
            return false;
        }
        shard.add_node(key, &node);
        true
    }

    /// Adds a new key's entry; false if the key is already stored.
    pub fn insert_view(&mut self, key: Key, entry: View) -> bool {
        let shard = &mut self.0[Self::shard(&key)];
        if shard.find(&key).is_some() {
            return false;
        }
        shard.add(key, entry);
        true
    }

    pub fn merge(&mut self, key: Key, delta: View) {
        self.0[Self::shard(&key)].merge(key, delta);
    }

    /// Heap bytes the shards own.
    pub fn allocation(&self) -> Allocation {
        self.0.iter().map(Shard::allocation).fold(Allocation::default(), |a, b| Allocation {
            index: a.index + b.index,
            entries: a.entries + b.entries,
            values: a.values + b.values,
        })
    }

    pub fn payload_bytes(&self) -> (usize, usize) {
        self.0.iter().map(Shard::payload_bytes).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    }

    /// Entries the hash indexes hold before they grow.
    pub fn index_capacity(&self) -> usize {
        self.0.iter().map(|s| s.index.capacity()).sum()
    }

    /// Every entry, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        self.0.iter().flat_map(|s| s.index.iter().map(move |&i| (s.entry(i).key, s.node(i))))
    }

    /// Every entry in key order, sorting one shard at a time.
    pub fn sorted(&self) -> impl Iterator<Item = (Key, Node)> + '_ {
        self.0.iter().flat_map(Shard::sorted)
    }

    pub fn views(&self) -> impl Iterator<Item = (Key, View<'_>)> + '_ {
        self.0.iter().flat_map(|s| s.index.iter().map(move |&i| s.row_view(i)))
    }

    pub fn sorted_views(&self) -> impl Iterator<Item = (Key, View<'_>)> + '_ {
        self.0.iter().flat_map(Shard::sorted_views)
    }
}

impl Default for Store {
    fn default() -> Store {
        Store::new()
    }
}

impl Lookup for Store {
    fn regrets(&self, key: &Key) -> Option<(u16, &[f64])> {
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

    fn node(code: u16, len: usize, regrets: &[f64], average: &[f64], visits: u64) -> Node {
        let mut node = Node { visits, ..Node::empty(code, len) };
        node.regrets[..len].copy_from_slice(regrets);
        node.average[..len].copy_from_slice(average);
        node
    }

    fn bits(node: &Node) -> (u16, u8, Vec<u64>, Vec<u64>, u64, u32) {
        let n = node.len as usize;
        (node.code, node.len, node.regrets[..n].iter().map(|v| v.to_bits()).collect(),
         node.average[..n].iter().map(|v| v.to_bits()).collect(), node.visits, node.stamp)
    }

    fn view_bits(v: View) -> (u16, u32, u64, Vec<u64>, Option<Vec<u64>>) {
        (v.code, v.meta, v.visits, v.regrets.iter().map(|r| r.to_bits()).collect(),
         v.average.map(|a| a.iter().map(|x| x.to_bits()).collect()))
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
    fn merge_adds_exactly_and_keeps_absent_averages_absent() {
        let mut store = Store::new();
        let delta = |regrets: &'static [f64], average: Option<&'static [f64]>, visits| View { code: 0b111, meta: 0x0201, visits, regrets, average };
        // New keys are stored as given, including -0.0.
        store.merge([5; 16], delta(&[-0.0, 0.0, 0.0], None, 0));
        assert_eq!(view_bits(store.view(&[5; 16]).unwrap()), view_bits(delta(&[-0.0, 0.0, 0.0], None, 0)));
        // +0.0 regrets keep no storage until a sum differs from +0.0.
        store.merge([6; 16], delta(&[0.0, 0.0, 0.0], Some(&[1.0, 0.5, 0.0]), 0));
        let shard = &store.0[Store::shard(&[6; 16])];
        assert_eq!(shard.entry(shard.find(&[6; 16]).unwrap()).regrets, NONE);
        store.merge([6; 16], delta(&[0.0, -0.0, 0.0], Some(&[0.25, 0.0, 2.0]), 0));
        let shard = &store.0[Store::shard(&[6; 16])];
        assert_eq!(shard.entry(shard.find(&[6; 16]).unwrap()).regrets, NONE, "+0.0 + -0.0 is +0.0");
        store.merge([6; 16], delta(&[1.5, -2.0, 0.0], Some(&[0.0, 0.0, 0.0]), 1));
        store.merge([6; 16], delta(&[0.25, 0.0, 3.0], Some(&[1.0, 1.0, 1.0]), 2));
        assert_eq!(view_bits(store.view(&[6; 16]).unwrap()),
            view_bits(delta(&[1.75, -2.0, 3.0], Some(&[2.25, 1.5, 3.0]), 3)));
    }

    #[test]
    #[should_panic(expected = "average scope")]
    fn merge_rejects_a_changed_entry() {
        let mut store = Store::new();
        store.merge([7; 16], View { code: 0b11, meta: 1, visits: 0, regrets: &[0.0, 0.0], average: None });
        store.merge([7; 16], View { code: 0b11, meta: 2, visits: 0, regrets: &[0.0, 0.0], average: None });
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
            assert!(store.insert(key(i), node((1u16 << len) - 1, len, &values, &values, i)));
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
        let views: Vec<Key> = store.sorted_views().map(|(k, _)| k).collect();
        assert_eq!(views, sorted);
    }
}
