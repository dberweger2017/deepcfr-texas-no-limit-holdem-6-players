//! The card part of the information key: v1's postflop descriptor, or #163's equity buckets.
//!
//! The equity schema changes only the postflop card part, to the holding's K=50 bucket on
//! the current street; preflop's 169 classes, flags, history and menus stay v1's (#189 rule 3).
//! Each schema name stands for one exact table set, so production loads refuse any table
//! whose SHA256 differs from #163's build.

use crate::game::Game;
use hu20_buckets::{class_key, hash_key, BucketTable};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::path::Path;

pub const V1_DESCRIPTOR: &str = "legacy-postflop-descriptor-v1";
pub const EQUITY_DESCRIPTOR: &str = "equity-histogram-k50-v1";
const STREETS: [&str; 3] = ["flop", "turn", "river"];
/// #163's K=50 tables (`docs/reports/hu20-equity-buckets.md`), validated by #190.
pub const EQUITY_K50_SHA256: [&str; 3] = [
    "084e243ded9fe93ad03cd59601f22f16c3210592f51e1e3646a60983b572c111",
    "c0bc6a8aa0535553118109d18a32d3b4dc6880e937c263cdc87472b1ae9f168f",
    "70c1cb5ecf4cd292e4c89dc2200f7c8381100be3b0288ceeb0a12a3e83977629",
];

/// Flop, turn and river tables, with each file's SHA256.
pub struct EquityTables {
    tables: [BucketTable; 3],
    pub sha256: [String; 3],
}

thread_local! {
    /// A traversal asks for the same few holdings and boards at every node of a street.
    /// Entries name their table set, so two sets in one thread never share one.
    static RECENT: RefCell<[Option<(usize, u64, u16)>; 8]> = const { RefCell::new([None; 8]) };
}

impl EquityTables {
    /// `{flop,turn,river}-k50.bin` from `directory`. Unless `unpinned` (test tables only),
    /// each file must be #163's.
    pub fn load(directory: &Path, unpinned: bool) -> Result<&'static EquityTables, String> {
        let mut tables = Vec::new();
        let mut sha256 = Vec::new();
        for (i, street) in STREETS.iter().enumerate() {
            let path = directory.join(format!("{street}-k50.bin"));
            let digest = crate::export::sha256_file(&path);
            if !unpinned && digest != EQUITY_K50_SHA256[i] {
                return Err(format!("{} is not #163's {street} K=50 table", path.display()));
            }
            let table = BucketTable::read(&path).map_err(|e| e.to_string())?;
            if table.street != i as u32 + 1 || table.k != 50 {
                return Err(format!("{} has street {} K {}, not {street} K=50", path.display(), table.street, table.k));
            }
            tables.push(table);
            sha256.push(digest);
        }
        let tables = tables.try_into().ok().unwrap();
        let sha256 = sha256.try_into().unwrap();
        Ok(Box::leak(Box::new(EquityTables { tables, sha256 })))
    }

    /// The bucket of `hole` on a flop, turn or river `board`. Every situation is in a
    /// complete table; a missing one means the wrong tables, so it panics.
    pub fn bucket(&self, hole: [u8; 2], board: &[u8]) -> u16 {
        let hash = hash_key(class_key(&hole, board));
        let (id, slot) = (self as *const _ as usize, (hash as usize) & 7);
        if let Some(bucket) = RECENT.with(|r| r.borrow()[slot].filter(|e| e.0 == id && e.1 == hash).map(|e| e.2)) {
            return bucket;
        }
        let bucket = self.tables[board.len() - 3].bucket(&hole, board).expect("situation missing from the equity bucket table");
        RECENT.with(|r| r.borrow_mut()[slot] = Some((id, hash, bucket)));
        bucket
    }
}

pub fn equity_schema(game: Game) -> &'static str {
    match game {
        Game::Hu20 => "hu20-native-reopening-ordered-history-equity-k50-v1",
        Game::Hu100 => "hu100-native-reopening-ordered-history-equity-k50-v1",
    }
}

/// Which card descriptor keys use.
#[derive(Clone, Copy)]
pub enum Cards {
    V1,
    Equity(&'static EquityTables),
}

impl std::fmt::Debug for Cards {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.descriptor())
    }
}

impl Cards {
    pub fn schema(self, game: Game) -> &'static str {
        match self {
            Cards::V1 => game.schema(),
            Cards::Equity(_) => equity_schema(game),
        }
    }

    pub fn descriptor(self) -> &'static str {
        match self {
            Cards::V1 => V1_DESCRIPTOR,
            Cards::Equity(_) => EQUITY_DESCRIPTOR,
        }
    }

    /// The bench's evaluator names its key metric; v1 keeps its original name.
    pub fn metric(self) -> &'static str {
        match self {
            Cards::V1 => "v1",
            Cards::Equity(_) => "equity-k50",
        }
    }

    /// The table hashes that checkpoints record, by street; none for v1.
    pub fn tables(self) -> Option<Value> {
        match self {
            Cards::V1 => None,
            Cards::Equity(t) => Some(json!({"flop": t.sha256[0], "turn": t.sha256[1], "river": t.sha256[2]})),
        }
    }
}
