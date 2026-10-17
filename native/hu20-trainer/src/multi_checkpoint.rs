//! Versioned N-player checkpoint. Legacy HU readers/writers remain untouched.
use crate::multi_cfr::{Counts, Entry, Table, Trainer};
use flate2::{read::MultiGzDecoder as GzDecoder, write::GzEncoder, Compression};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub fn key_hex(key: &[u8; 16]) -> String {
    key.iter().map(|b| format!("{b:02x}")).collect()
}
pub fn counts_json(c: &Counts) -> Value {
    json!({"nodes":c.nodes,"terminals":c.terminals,"decisions":c.decisions,"missing":c.missing,
        "traverser":c.traverser,"discarded_keys":c.discarded_keys,"discarded_visits":c.discarded_visits})
}
pub fn save(trainer: &Trainer, path: &Path) -> Result<(), String> {
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)
        .map_err(|e| e.to_string())?;
    let result = (|| -> Result<(), String> {
        let mut out = GzEncoder::new(std::io::BufWriter::new(file), Compression::default());
        let header = json!({"format":"nplayer-cfr-v1","game":trainer.identity,"players":trainer.players,
            "seed":trainer.seed,"iteration":trainer.iteration,"last_average_street":trainer.last_average_street,
            "average_rule":"opponent-sampled","max_entries":trainer.max_entries,
            "freeze_at_cap":trainer.freeze_at_cap,"frozen_at":trainer.frozen_at,"entries":trainer.table.len(),
            "counts":counts_json(&trainer.counts)});
        writeln!(out, "{header}").map_err(|e| e.to_string())?;
        for (key, e) in trainer.table.sorted() {
            let row = json!([
                key_hex(&key),
                e.actor,
                e.street,
                e.code,
                e.visits,
                e.regrets,
                e.average
            ]);
            writeln!(out, "{row}").map_err(|e| e.to_string())?;
        }
        let mut writer = out.finish().map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        writer.get_ref().sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}
fn integer(v: &Value) -> Result<u64, String> {
    v.as_u64().ok_or_else(|| "expected unsigned integer".into())
}
fn fixed<const N: usize>(v: &Value) -> Result<[u64; N], String> {
    let a = v.as_array().ok_or("expected array")?;
    if a.len() != N {
        return Err("wrong array length".into());
    }
    let mut out = [0; N];
    for (i, x) in a.iter().enumerate() {
        out[i] = integer(x)?;
    }
    Ok(out)
}
fn floats(v: &Value, n: usize, nonnegative: bool) -> Result<Box<[f64]>, String> {
    let a = v.as_array().ok_or("expected float array")?;
    if a.len() != n {
        return Err("wrong action count".into());
    }
    a.iter()
        .map(|x| {
            x.as_f64()
                .filter(|f| f.is_finite() && (!nonnegative || *f >= 0.))
                .ok_or_else(|| "invalid accumulator".into())
        })
        .collect()
}
pub fn load(path: &Path) -> Result<Trainer, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut lines = BufReader::new(GzDecoder::new(file)).lines();
    let head = lines
        .next()
        .ok_or("missing header")?
        .map_err(|e| e.to_string())?;
    let h: Value = serde_json::from_str(&head).map_err(|e| e.to_string())?;
    if h["format"] != "nplayer-cfr-v1" || h["average_rule"] != "opponent-sampled" {
        return Err("unsupported checkpoint identity".into());
    }
    let players = integer(&h["players"])? as usize;
    let last_average_street = integer(&h["last_average_street"])? as usize;
    let max_entries = usize::try_from(integer(&h["max_entries"])?).map_err(|e| e.to_string())?;
    let iteration = integer(&h["iteration"])?;
    let frozen_at = if h["frozen_at"].is_null() {
        None
    } else {
        Some(integer(&h["frozen_at"])?)
    };
    let freeze_at_cap = h["freeze_at_cap"].as_bool().ok_or("missing freeze flag")?;
    if !(2..=6).contains(&players)
        || last_average_street > 3
        || max_entries == 0
        || iteration > 1 << 53
        || integer(&h["game"]["players"])? != players as u64
        || frozen_at.is_some_and(|t| t == 0 || t > iteration || !freeze_at_cap)
    {
        return Err("invalid run metadata".into());
    }
    let c = &h["counts"];
    let counts = Counts {
        nodes: integer(&c["nodes"])?,
        terminals: integer(&c["terminals"])?,
        decisions: fixed(&c["decisions"])?,
        missing: fixed(&c["missing"])?,
        traverser: fixed(&c["traverser"])?,
        discarded_keys: integer(&c["discarded_keys"])?,
        discarded_visits: integer(&c["discarded_visits"])?,
    };
    if counts.terminals.checked_add(
        counts
            .decisions
            .iter()
            .try_fold(0u64, |a, b| a.checked_add(*b))
            .ok_or("count overflow")?,
    ) != Some(counts.nodes)
        || counts
            .missing
            .iter()
            .zip(counts.decisions)
            .any(|(m, d)| *m > d)
        || counts.traverser[players..].iter().any(|&v| v != 0)
    {
        return Err("inconsistent telemetry".into());
    }
    let mut table = Table::default();
    let mut previous = None;
    for line in lines {
        let row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let a = row.as_array().ok_or("expected entry")?;
        if a.len() != 7 {
            return Err("invalid entry shape".into());
        }
        let hex = a[0].as_str().ok_or("missing key")?;
        if hex.len() != 32 || !hex.is_ascii() {
            return Err("invalid key".into());
        }
        let mut key = [0; 16];
        for (i, k) in key.iter_mut().enumerate() {
            *k = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?;
        }
        if previous.is_some_and(|p| p >= key) {
            return Err("keys not strictly sorted".into());
        }
        previous = Some(key);
        let owner = integer(&a[1])?;
        let street = integer(&a[2])?;
        let code = integer(&a[3])?;
        if owner >= players as u64
            || street > 3
            || code > u16::MAX as u64
            || !(1..=8).contains(&code.count_ones())
        {
            return Err("invalid entry identity".into());
        }
        let len = code.count_ones() as usize;
        let average = if street <= last_average_street as u64 {
            Some(floats(&a[6], len, true)?)
        } else {
            if !a[6].is_null() {
                return Err("average outside declared scope".into());
            }
            None
        };
        table.insert(
            key,
            Entry {
                actor: owner as u8,
                street: street as u8,
                code: code as u16,
                visits: integer(&a[4])?,
                regrets: floats(&a[5], len, false)?,
                average,
            },
        );
    }
    if table.len() as u64 != integer(&h["entries"])?
        || (frozen_at.is_some() && table.len() < max_entries)
        || (freeze_at_cap && table.len() >= max_entries && frozen_at.is_none())
    {
        return Err("entry count/freeze mismatch".into());
    }
    Ok(Trainer {
        identity: h["game"].clone(),
        players,
        seed: integer(&h["seed"])?,
        iteration,
        last_average_street,
        max_entries,
        freeze_at_cap,
        frozen_at,
        table,
        counts,
    })
}
