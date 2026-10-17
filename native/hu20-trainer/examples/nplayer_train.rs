//! Explicit experimental entry point; legacy HU commands/formats are unchanged.
use hu20_trainer::{
    cards::EquityTables,
    multi_cfr::{Game, Trainer},
    multi_checkpoint,
    multi_holdem::CashGame,
    nplayer::Config,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
};

fn train() -> Result<(), String> {
    let mut options = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    let mut freeze = false;
    let mut gzip = None;
    while let Some(key) = args.next() {
        if key == "--freeze-at-cap" {
            if freeze {
                return Err("duplicate --freeze-at-cap".into());
            }
            freeze = true;
            continue;
        }
        if key == "--fast-gzip" {
            if gzip.replace(hu20_trainer::gz::Gzip::Parallel).is_some() {
                return Err("duplicate --fast-gzip".into());
            }
            continue;
        }
        if ![
            "--players",
            "--stack-bb",
            "--stacks",
            "--small-blind",
            "--big-blind",
            "--menu",
            "--card-buckets",
            "--seed",
            "--nodes",
            "--last-average-street",
            "--max-entries",
            "--max-seconds",
            "--out",
            "--resume",
            "--resume-sha256",
            "--telemetry",
            "--stop-file",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("unknown option {key}"));
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if options.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate {key}"));
        }
    }
    let required = |key: &str| {
        options
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| format!("missing {key}"))
    };
    let unsigned = |key: &str, default: &str| {
        options
            .get(key)
            .map(String::as_str)
            .unwrap_or(default)
            .parse::<u64>()
            .map_err(|_| format!("invalid {key}"))
    };
    let players = required("--players")?
        .parse::<usize>()
        .map_err(|_| "invalid --players")?;
    if !(2..=6).contains(&players) {
        return Err("--players must be 2..6".into());
    }
    let sb = unsigned("--small-blind", "50")?;
    let bb = unsigned("--big-blind", "100")?;
    if options.contains_key("--stacks") && options.contains_key("--stack-bb") {
        return Err("choose --stacks or --stack-bb".into());
    }
    let stacks = if let Some(stacks) = options.get("--stacks") {
        stacks
            .split(',')
            .map(|s| s.parse::<u64>().map_err(|_| "invalid --stacks".to_string()))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![
            bb.checked_mul(unsigned("--stack-bb", "20")?)
                .ok_or("stack overflow")?;
            players
        ]
    };
    if stacks.len() != players {
        return Err("--stacks count differs from --players".into());
    }
    let config = Config::new(&stacks, sb, bb)?;
    if options.get("--menu").is_some_and(|m| m != "v1") {
        return Err("this phase supports --menu v1 only".into());
    }
    let nodes = required("--nodes")?
        .parse::<u64>()
        .map_err(|_| "invalid --nodes")?;
    let max_entries = required("--max-entries")?
        .parse::<usize>()
        .map_err(|_| "invalid --max-entries")?;
    let seconds = required("--max-seconds")?
        .parse::<f64>()
        .map_err(|_| "invalid --max-seconds")?;
    let scope = unsigned("--last-average-street", "3")? as usize;
    if nodes == 0 || max_entries == 0 || !seconds.is_finite() || seconds <= 0. || scope > 3 {
        return Err("invalid node, entry, time or averaging bound".into());
    }
    let seed = unsigned("--seed", "1")?;
    let out = PathBuf::from(required("--out")?);
    if out.exists() {
        return Err("output already exists; choose a fresh path".into());
    }
    if options.contains_key("--resume") != options.contains_key("--resume-sha256") {
        return Err("resume needs both --resume and --resume-sha256".into());
    }
    let tables = EquityTables::load(Path::new(required("--card-buckets")?), false)?;
    let game = CashGame { config, tables };
    let mut trainer = if let Some(path) = options.get("--resume") {
        let path = Path::new(path);
        let expected = required("--resume-sha256")?;
        if expected.len() != 64 || hu20_trainer::export::sha256_file(path) != expected {
            return Err("resume SHA256 mismatch".into());
        }
        let t = multi_checkpoint::load(path)?;
        if t.identity != game.identity()
            || t.players != players
            || t.seed != seed
            || t.last_average_street != scope
            || t.max_entries != max_entries
            || t.freeze_at_cap != freeze
        {
            return Err("resume configuration mismatch".into());
        }
        t
    } else {
        Trainer::new(&game, seed, scope, max_entries, freeze)
    };
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let mut telemetry = if let Some(path) = options.get("--telemetry") {
        Some(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| e.to_string())?,
        )
    } else {
        None
    };
    let start = std::time::Instant::now();
    let reason = loop {
        if trainer.counts.nodes >= nodes {
            break "nodes";
        }
        if !freeze && trainer.table.len() >= max_entries {
            break "capacity";
        }
        if start.elapsed().as_secs_f64() >= seconds {
            break "time";
        }
        if options
            .get("--stop-file")
            .is_some_and(|p| Path::new(p).exists())
        {
            break "stop-file";
        }
        trainer.step(&game);
        if trainer.iteration % 100 == 0 {
            if let Some(file) = telemetry.as_mut() {
                writeln!(file,"{}",json!({"iteration":trainer.iteration,"entries":trainer.table.len(),
                "frozen_at":trainer.frozen_at,"elapsed_seconds":start.elapsed().as_secs_f64(),"counts":multi_checkpoint::counts_json(&trainer.counts)})).map_err(|e|e.to_string())?;
            }
        }
    };
    let save_start = std::time::Instant::now();
    multi_checkpoint::save(&trainer, &out, gzip.unwrap_or_default())?;
    let report = json!({"identity":game.identity(),"stop_reason":reason,"iteration":trainer.iteration,
        "save_seconds":save_start.elapsed().as_secs_f64(),
        "entries":trainer.table.len(),"frozen_at":trainer.frozen_at,"counts":multi_checkpoint::counts_json(&trainer.counts),
        "elapsed_seconds":start.elapsed().as_secs_f64(),"checkpoint":out,
        "sha256":hu20_trainer::export::sha256_file(&out)});
    if let Some(file) = telemetry.as_mut() {
        writeln!(file, "{report}").map_err(|e| e.to_string())?;
    }
    println!("{report}");
    if reason == "capacity" {
        std::process::exit(3);
    }
    Ok(())
}
fn main() {
    if let Err(error) = train() {
        eprintln!("nplayer_train: {error}");
        std::process::exit(2);
    }
}
