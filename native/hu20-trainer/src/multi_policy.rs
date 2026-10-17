//! Minimal N-player research export/loader. Policy inputs cannot reach a Hand.
use crate::{
    cards::{EquityTables, EQUITY_K50_SHA256},
    cfr::{Key, KeyHasher},
    multi_cfr::Trainer,
    multi_checkpoint,
    multi_holdem::{self, MENU, SCHEMA},
    nplayer::{Action, Config, Observation},
};
use flate2::{read::MultiGzDecoder as GzDecoder, write::GzEncoder, Compression};
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    hash::BuildHasherDefault,
    io::{BufRead, BufReader, Write},
    path::Path,
    sync::Arc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Current,
    Average,
    ScopedAverage,
}
impl Mode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "current" => Ok(Self::Current),
            "average" => Ok(Self::Average),
            "scoped-average" => Ok(Self::ScopedAverage),
            _ => Err("mode must be current, average or scoped-average".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Average => "average",
            Self::ScopedAverage => "scoped-average",
        }
    }
    fn continuation(self) -> &'static str {
        if self == Self::ScopedAverage {
            "current-after-average-scope"
        } else {
            "none"
        }
    }
}
#[derive(Debug)]
pub struct Row {
    pub owner: u8,
    pub street: u8,
    pub code: u16,
    pub probabilities: Box<[f64]>,
}
pub struct PolicyTable {
    pub identity: Value,
    pub players: usize,
    pub scope: usize,
    pub mode: Mode,
    pub source_checkpoint_sha256: String,
    config: Config,
    pub rows: HashMap<Key, Row, BuildHasherDefault<KeyHasher>>,
}
fn cash_config(identity: &Value) -> Result<Config, String> {
    let n = identity["players"].as_u64().ok_or("missing N")?;
    if !(2..=6).contains(&n)
        || identity["schema"] != SCHEMA
        || identity["game"] != "nlhe-cash"
        || identity["menu"] != MENU
        || identity["utility_unit"] != "big-blind"
        || identity["card_tables"] != json!(EQUITY_K50_SHA256)
    {
        return Err("unsupported cash identity".into());
    }
    let stacks = identity["stacks"]
        .as_array()
        .ok_or("missing stacks")?
        .iter()
        .map(|v| v.as_u64().ok_or("invalid stack".to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if stacks.len() != n as usize {
        return Err("stack count differs from N".into());
    }
    Config::new(
        &stacks,
        identity["small_blind"].as_u64().ok_or("invalid SB")?,
        identity["big_blind"].as_u64().ok_or("invalid BB")?,
    )
}
fn verified(path: &Path, expected: &str) -> Result<(), String> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected complete SHA256".into());
    }
    std::fs::File::open(path).map_err(|e| e.to_string())?;
    if crate::export::sha256_file(path) != expected {
        return Err("SHA256 mismatch".into());
    }
    Ok(())
}
pub fn export(checkpoint: &Path, expected: &str, mode: Mode, out: &Path) -> Result<(), String> {
    verified(checkpoint, expected)?;
    let trainer = multi_checkpoint::load(checkpoint)?;
    write(&trainer, expected, mode, out)
}
fn write(trainer: &Trainer, source: &str, mode: Mode, out: &Path) -> Result<(), String> {
    cash_config(&trainer.identity)?;
    if mode == Mode::Average && trainer.last_average_street != 3 {
        return Err("full average unavailable; explicitly choose scoped-average or current".into());
    }
    if out.exists() {
        return Err("output already exists".into());
    }
    let temp = out.with_extension(format!("tmp-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|e| e.to_string())?;
    let result = (|| -> Result<(), String> {
        let mut w = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());
        let header = json!({"format":"nplayer-policy-v1","identity":trainer.identity,"players":trainer.players,"entries":trainer.table.len(),"strategy":mode.name(),"last_average_street":trainer.last_average_street,"continuation":mode.continuation(),"missing":"uniform","source_checkpoint_sha256":source});
        writeln!(w, "{header}").map_err(|e| e.to_string())?;
        for (key, e) in trainer.table.sorted() {
            let average = mode != Mode::Current && e.street as usize <= trainer.last_average_street;
            let policy = e.policy(average);
            writeln!(
                w,
                "{}",
                json!([
                    multi_checkpoint::key_hex(&key),
                    e.actor,
                    e.street,
                    e.code,
                    &policy[..e.regrets.len()]
                ])
            )
            .map_err(|e| e.to_string())?;
        }
        let mut writer = w.finish().map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temp, out).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(temp);
    }
    result
}
impl PolicyTable {
    pub fn load(path: &Path, expected: &str) -> Result<Self, String> {
        verified(path, expected)?;
        Self::read(path)
    }
    fn read(path: &Path) -> Result<Self, String> {
        let mut lines = BufReader::new(GzDecoder::new(
            std::fs::File::open(path).map_err(|e| e.to_string())?,
        ))
        .lines();
        let h: Value = serde_json::from_str(
            &lines
                .next()
                .ok_or("missing header")?
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        if h["format"] != "nplayer-policy-v1" || h["missing"] != "uniform" {
            return Err("unsupported policy".into());
        }
        let config = cash_config(&h["identity"])?;
        let players = h["players"].as_u64().ok_or("invalid N")?;
        let scope = h["last_average_street"].as_u64().ok_or("invalid scope")?;
        let mode = Mode::parse(h["strategy"].as_str().ok_or("missing strategy")?)?;
        let source = h["source_checkpoint_sha256"]
            .as_str()
            .ok_or("missing checkpoint pin")?;
        if players != config.players() as u64
            || scope > 3
            || (mode == Mode::Average && scope != 3)
            || h["continuation"] != mode.continuation()
            || source.len() != 64
            || !source.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err("invalid policy metadata".into());
        }
        let mut rows = HashMap::default();
        let mut previous = None;
        for line in lines {
            let value: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            let a = value.as_array().ok_or("invalid row")?;
            if a.len() != 5 {
                return Err("invalid row length".into());
            }
            let hex = a[0].as_str().ok_or("invalid key")?;
            if hex.len() != 32 || !hex.is_ascii() {
                return Err("invalid key".into());
            }
            let mut key = [0u8; 16];
            for (i, k) in key.iter_mut().enumerate() {
                *k = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?;
            }
            if previous.is_some_and(|p| p >= key) {
                return Err("keys must be strictly sorted".into());
            }
            previous = Some(key);
            let owner = a[1].as_u64().ok_or("invalid owner")?;
            let street = a[2].as_u64().ok_or("invalid street")?;
            let code = a[3].as_u64().ok_or("invalid actions")?;
            if owner >= players || street > 3 || code == 0 || code & !0b11_1111 != 0 {
                return Err("invalid row identity".into());
            }
            let p = a[4]
                .as_array()
                .ok_or("invalid probabilities")?
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|p| p.is_finite() && *p >= 0. && *p <= 1.)
                        .ok_or("invalid probability".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            if p.len() != code.count_ones() as usize || (p.iter().sum::<f64>() - 1.).abs() > 1e-12 {
                return Err("invalid distribution".into());
            }
            rows.insert(
                key,
                Row {
                    owner: owner as u8,
                    street: street as u8,
                    code: code as u16,
                    probabilities: p.into_boxed_slice(),
                },
            );
        }
        if h["entries"].as_u64() != Some(rows.len() as u64) {
            return Err("entry count mismatch".into());
        }
        Ok(Self {
            identity: h["identity"].clone(),
            players: players as usize,
            scope: scope as usize,
            mode,
            source_checkpoint_sha256: source.to_owned(),
            config,
            rows,
        })
    }
    pub fn config(&self) -> Config {
        self.config
    }
    pub fn distribution(
        &self,
        observation: &Observation<'_>,
        tables: &EquityTables,
    ) -> Result<(Vec<f64>, bool), String> {
        let c = observation.config;
        let expected = self.config();
        if c.players() != self.players
            || c.stacks() != expected.stacks()
            || c.small_blind() != expected.small_blind()
            || c.big_blind() != expected.big_blind()
            || tables.sha256 != EQUITY_K50_SHA256
        {
            return Err("policy/table configuration mismatch".into());
        }
        if observation.actor != Some(observation.seat) || observation.legal.is_none() {
            return Err("policy requires its acting-seat observation".into());
        }
        let d = multi_holdem::observation_decision(observation, tables);
        if let Some(row) = self.rows.get(&d.key) {
            if (row.owner as usize, row.street as usize, row.code) != (d.owner, d.street, d.code) {
                return Err("policy key/action identity mismatch".into());
            }
            Ok((row.probabilities.to_vec(), true))
        } else {
            Ok((vec![1. / d.len() as f64; d.len()], false))
        }
    }
}
/// Instances share immutable weights if desired, but each seat owns its action RNG.
pub struct Policy {
    pub table: Arc<PolicyTable>,
    rng: rand::rngs::StdRng,
    pub decisions: u64,
    pub misses: u64,
}
impl Policy {
    pub fn new(table: Arc<PolicyTable>, seed: u64) -> Self {
        Self {
            table,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            decisions: 0,
            misses: 0,
        }
    }
    pub fn choose(
        &mut self,
        observation: &Observation<'_>,
        tables: &EquityTables,
    ) -> Result<Action, String> {
        let (p, found) = self.table.distribution(observation, tables)?;
        let choices = multi_holdem::menu(observation);
        let draw: f64 = self.rng.gen();
        let mut sum = 0.;
        let mut chosen = p.iter().rposition(|&v| v > 0.).ok_or("empty support")?;
        for (i, v) in p.iter().enumerate() {
            sum += v;
            if draw < sum {
                chosen = i;
                break;
            }
        }
        self.decisions += 1;
        self.misses += u64::from(!found);
        Ok(choices[chosen].1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multi_cfr::{Decision, Entry};
    use crate::small_games::{Kind, SmallGame};
    fn fixture(n: usize, scope: usize) -> Trainer {
        let mut t = Trainer::new(&SmallGame::new(Kind::Kuhn, n), 11, scope, 100, false);
        t.identity = json!({"schema":SCHEMA,"game":"nlhe-cash","players":n,"stacks":vec![2000;n],"small_blind":50,"big_blind":100,"menu":MENU,"card_tables":EQUITY_K50_SHA256,"utility_unit":"big-blind"});
        for street in 0..2 {
            let d = Decision {
                key: [street as u8; 16],
                actor: n - 1,
                owner: n - 1,
                street,
                code: 0b1010,
            };
            let mut e = Entry::new(d, scope);
            e.regrets.copy_from_slice(&[1., 3.]);
            if let Some(a) = &mut e.average {
                a.copy_from_slice(&[4., 0.]);
            }
            t.table.insert(d.key, e);
        }
        t
    }
    #[test]
    fn strategy_scope_is_explicit_and_export_is_deterministic_for_all_n() {
        let dir = std::env::temp_dir().join(format!("nplayer-export-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        for n in 2..=6 {
            for mode in [Mode::Current, Mode::Average, Mode::ScopedAverage] {
                let t = fixture(n, if mode == Mode::Average { 3 } else { 0 });
                let a = dir.join(format!("{n}-{mode:?}-a.gz"));
                let b = dir.join(format!("{n}-{mode:?}-b.gz"));
                write(&t, &"a".repeat(64), mode, &a).unwrap();
                write(&t, &"a".repeat(64), mode, &b).unwrap();
                assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
                let table = PolicyTable::load(&a, &crate::export::sha256_file(&a)).unwrap();
                assert_eq!(table.players, n);
                assert_eq!(table.mode, mode);
                for street in 0..2 {
                    let expected = if mode != Mode::Current && street <= t.last_average_street {
                        vec![1., 0.]
                    } else {
                        vec![0.25, 0.75]
                    };
                    assert_eq!(&*table.rows[&[street as u8; 16]].probabilities, &expected);
                }
                assert!(write(&t, &"a".repeat(64), mode, &a).is_err());
                assert!(PolicyTable::load(&a, &"0".repeat(64)).is_err());
            }
        }
        assert!(write(
            &fixture(3, 0),
            &"a".repeat(64),
            Mode::Average,
            &dir.join("missing-average.gz")
        )
        .is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn loader_rejects_metadata_rows_and_probability_corruption() {
        let dir =
            std::env::temp_dir().join(format!("nplayer-invalid-policy-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let valid = dir.join("valid.gz");
        write(&fixture(3, 0), &"a".repeat(64), Mode::ScopedAverage, &valid).unwrap();
        let values: Vec<Value> =
            BufReader::new(GzDecoder::new(std::fs::File::open(&valid).unwrap()))
                .lines()
                .map(|s| serde_json::from_str(&s.unwrap()).unwrap())
                .collect();
        for mutation in 0..13 {
            let mut bad = values.clone();
            match mutation {
                0 => bad[0]["players"] = json!(2),
                1 => bad[0]["last_average_street"] = json!(4),
                2 => bad[0]["continuation"] = json!("none"),
                3 => bad[0]["identity"]["card_tables"][0] = json!("wrong"),
                4 => bad[1][1] = json!(3),
                5 => bad[1][2] = json!(4),
                6 => bad[1][3] = json!(64),
                7 => bad[1][4] = json!([-0.1, 1.1]),
                8 => bad[1][4] = json!([0.2, 0.2]),
                9 => bad[1][4] = json!([1.]),
                10 => bad[2] = bad[1].clone(),
                11 => {
                    bad.pop();
                }
                _ => bad[0]["source_checkpoint_sha256"] = json!("invalid"),
            }
            let path = dir.join(format!("bad-{mutation}.gz"));
            let mut w = GzEncoder::new(
                std::fs::File::create(&path).unwrap(),
                Compression::default(),
            );
            for row in bad {
                writeln!(w, "{row}").unwrap();
            }
            w.finish().unwrap();
            assert!(
                PolicyTable::load(&path, &crate::export::sha256_file(&path)).is_err(),
                "mutation {mutation}"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
