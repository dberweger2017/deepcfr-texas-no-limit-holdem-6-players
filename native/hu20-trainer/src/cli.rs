//! Command-specific admission before opening inputs or creating outputs.
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

const CARDS: &[&str] = &["--card-buckets", "--card-tables-unpinned"];
const TRAIN: &[&str] = &["--nodes", "--iterations", "--seed", "--roots-per-seat", "--out", "--milestones",
    "--stack-bb", "--max-entries", "--max-seconds", "--stop-file", "--telemetry", "--resume",
    "--resume-sha256", "--completed-nodes", "--average-rule", "--regret-floor", "--recovery", "--menu", "--freeze-at-cap", "--fast-gzip"];
const BENCH: &[&str] = &["--roots", "--seed", "--iterations", "--lineage", "--variant", "--out",
    "--checkpoints", "--regret-floor", "--dcfr"];
const EXPORT: &[&str] = &["--current", "--average", "--zero-mass"];

pub struct Command {
    pub config: Value,
}

/// Canonical save thresholds: positive, below the final target, unique and ordered.
pub fn schedule(requested: &[u64], final_target: u64) -> Vec<u64> {
    let mut effective: Vec<u64> = requested.iter().copied().filter(|&n| n > 0 && n < final_target).collect();
    effective.push(final_target);
    effective.sort_unstable();
    effective.dedup();
    effective
}

fn integer(flag: &str, value: &str) -> Result<u64, String> {
    value.parse::<u64>().map_err(|_| format!("{flag} requires an unsigned integer, got {value:?}"))
}
fn float(flag: &str, value: &str, infinite: bool) -> Result<f64, String> {
    let number = value.parse::<f64>().map_err(|_| format!("{flag} requires a number, got {value:?}"))?;
    if number.is_nan() || (!infinite && !number.is_finite()) || number == f64::NEG_INFINITY {
        return Err(format!("{flag} has an invalid nonfinite value"));
    }
    Ok(number)
}
fn value(flag: &str, raw: &str) -> Result<Value, String> {
    Ok(match flag {
        "--nodes" | "--iterations" | "--seed" | "--roots-per-seat" | "--max-entries" | "--completed-nodes" => {
            let n = integer(flag, raw)?;
            if matches!(flag, "--nodes" | "--roots-per-seat" | "--max-entries") && n == 0 {
                return Err(format!("{flag} must be positive"));
            }
            if flag == "--roots-per-seat" && usize::try_from(n).is_err() {
                return Err(format!("{flag} exceeds this platform's usize"));
            }
            json!(n)
        }
        "--milestones" | "--checkpoints" => {
            // Empty lists retain the historical default; empty members in a supplied list are errors.
            let numbers: Vec<u64> = if raw.is_empty() { vec![] } else {
                raw.split(',').map(|s| integer(flag, s)).collect::<Result<_, _>>()?
            };
            json!(numbers)
        }
        "--regret-floor" => json!(float(flag, raw, false)?),
        "--max-seconds" => {
            let n = float(flag, raw, true)?;
            if n <= 0.0 { return Err(format!("{flag} must be positive")); }
            if n.is_infinite() { json!("inf") } else { json!(n) }
        }
        "--dcfr" => {
            let v: Vec<f64> = raw.split(',').map(|s| float(flag, s, false).map_err(|problem| {
                if problem.contains("nonfinite") { "--dcfr needs finite exponents (nonfinite value)".into() } else { problem }
            })).collect::<Result<_, _>>()?;
            if v.len() != 3 { return Err("--dcfr requires alpha,beta,gamma".into()); }
            json!(v)
        }
        "--stack-bb" => {
            let n = integer(flag, raw)?;
            if ![20, 100, 200].contains(&n) { return Err("--stack-bb must be 20, 100 or 200".into()); }
            json!(n)
        }
        "--menu" => {
            if !["v1", "v2"].contains(&raw) { return Err("--menu must be v1 or v2".into()); }
            json!(raw)
        }
        "--average-rule" => {
            if !["traverser-reach", "opponent-sampled"].contains(&raw) {
                return Err("--average-rule must be traverser-reach or opponent-sampled".into());
            }
            json!(raw)
        }
        "--zero-mass" => {
            if !["uniform", "current"].contains(&raw) { return Err("--zero-mass must be uniform or current".into()); }
            json!(raw)
        }
        "--resume-sha256" => {
            if raw.len() != 64 || !raw.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("--resume-sha256 requires a 64-digit hexadecimal SHA256".into());
            }
            json!(raw)
        }
        _ => {
            if raw.is_empty() { return Err(format!("{flag} needs a nonempty value")); }
            json!(raw)
        }
    })
}

pub fn parse(args: &[String]) -> Result<Command, String> {
    let command = args.get(1).map(String::as_str).ok_or("a command is required")?;
    let flags = match command {
        "train" => TRAIN,
        "bench-train" => BENCH,
        "parity" => &["--menu"][..],
        "export" => EXPORT,
        "traversal-parity" | "run-parity" | "engine-deal" => &[][..],
        _ => return Err(format!("unknown command {command:?}")),
    };
    let has_cards = matches!(command, "train" | "bench-train" | "parity");
    let mut supplied = BTreeMap::new();
    let mut positionals = vec![];
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag.starts_with('-') {
            if !flags.contains(&flag) && !(has_cards && CARDS.contains(&flag)) {
                return Err(format!("{command}: unknown or unsupported flag {flag}"));
            }
            if supplied.contains_key(flag) { return Err(format!("{command}: duplicate flag {flag}")); }
            let boolean = matches!(flag, "--recovery" | "--card-tables-unpinned" | "--freeze-at-cap" | "--fast-gzip");
            let parsed = if boolean { json!(true) } else {
                i += 1;
                let raw = args.get(i).filter(|s| !s.starts_with("--"))
                    .ok_or_else(|| format!("{command}: missing value for {flag}"))?;
                value(flag, raw)?
            };
            supplied.insert(flag, parsed);
        } else {
            // Fixed input paths must precede flags, so a misplaced value cannot be ignored.
            if !supplied.is_empty() { return Err(format!("{command}: misplaced positional argument {flag:?}")); }
            positionals.push(flag.to_string());
        }
        i += 1;
    }
    let count = match command {
        "parity" | "export" | "traversal-parity" | "run-parity" => 1,
        "engine-deal" => positionals.len(),
        _ => 0,
    };
    if positionals.len() != count || (command == "engine-deal" && count == 0) {
        return Err(format!("{command}: expected {} positional argument(s)", if command == "engine-deal" { "one or more seeds".into() } else { count.to_string() }));
    }
    if command == "engine-deal" {
        for seed in &positionals { integer("engine-deal seed", seed)?; }
    }
    if supplied.contains_key("--card-tables-unpinned") && !supplied.contains_key("--card-buckets") {
        return Err("--card-tables-unpinned requires --card-buckets".into());
    }
    if command == "train" {
        let resume = supplied.contains_key("--resume");
        if resume != supplied.contains_key("--resume-sha256") {
            return Err("--resume and --resume-sha256 must be supplied together".into());
        }
        if !resume && supplied.contains_key("--completed-nodes") {
            return Err("--completed-nodes requires --resume".into());
        }
        if resume && supplied.contains_key("--regret-floor") {
            return Err("resume requires linear CFR; --regret-floor is unsupported".into());
        }
    }
    if command == "bench-train" {
        for flag in ["--roots", "--seed", "--iterations", "--lineage", "--out"] {
            if !supplied.contains_key(flag) { return Err(format!("bench-train: {flag} is required")); }
        }
        if supplied["--iterations"] == json!(0) { return Err("bench-train: --iterations must be positive".into()); }
        let training = hu20_trainer::cfr::Options {
            regret_floor: supplied.get("--regret-floor").map(|v| v.as_f64().unwrap()),
            dcfr: supplied.get("--dcfr").map(|v| {
                let v = v.as_array().unwrap();
                [v[0].as_f64().unwrap(), v[1].as_f64().unwrap(), v[2].as_f64().unwrap()]
            }),
        };
        training.check_domain(supplied["--iterations"].as_u64().unwrap())?;
    }
    let mut options = Map::new();
    let defaults: &[(&str, &str)] = match command {
        "train" => &[("--nodes", "1000000"), ("--iterations", "0"), ("--seed", "1"), ("--roots-per-seat", "1"),
            ("--out", "native-checkpoint.json.gz"), ("--milestones", ""), ("--stack-bb", "20"),
            ("--max-entries", "1000000000"), ("--max-seconds", "inf"), ("--average-rule", "traverser-reach"), ("--menu", "v1")],
        "bench-train" => &[("--variant", "base"), ("--checkpoints", "")],
        "parity" => &[("--menu", "v1")],
        "export" => &[("--zero-mass", "uniform")],
        _ => &[],
    };
    for (flag, raw) in defaults { options.insert(flag[2..].replace('-', "_"), value(flag, raw)?); }
    for (flag, parsed) in supplied { options.insert(flag[2..].replace('-', "_"), parsed); }
    if has_cards {
        options.entry("card_buckets").or_insert(Value::Null);
        options.entry("card_tables_unpinned").or_insert(json!(false));
    }
    if command == "train" {
        options.entry("recovery").or_insert(json!(false));
        options.entry("freeze_at_cap").or_insert(json!(false));
        options.entry("fast_gzip").or_insert(json!(false));
    }
    if command == "bench-train" {
        options.insert("menu".into(), json!("v1"));
        options.insert("stack_bb".into(), json!(20));
        options.insert("average_rules".into(), json!(["traverser-reach", "opponent-sampled"]));
    }
    for flag in flags.iter().chain(if has_cards { CARDS.iter() } else { [].iter() }) {
        let key = flag[2..].replace('-', "_");
        options.entry(key).or_insert(Value::Null);
    }
    let mut config = json!({"command": command, "positionals": positionals, "options": options});
    if matches!(command, "train" | "bench-train") {
        let (field, target) = if command == "train" { ("milestones", "nodes") } else { ("checkpoints", "iterations") };
        let requested = config["options"][field].clone();
        let thresholds: Vec<u64> = requested.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        config["requested"] = json!({field: requested});
        config["options"][field] = json!(schedule(&thresholds, config["options"][target].as_u64().unwrap()));
        if command == "bench-train" {
            let raw = config["options"]["lineage"].as_str().unwrap().to_string();
            config["requested"]["lineage"] = json!(raw);
            config["options"]["lineage"] = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
        }
    }
    Ok(Command { config })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(s: &str) -> Vec<String> { std::iter::once("hu20-trainer").chain(s.split_whitespace()).map(str::to_string).collect() }
    #[test]
    fn rejects_ignored_or_malformed_arguments() {
        for s in ["train --dcfr 1.5,0,2", "bench-train --menu v2", "train --seed", "train --seed 1 --seed 2",
            "train --nodes nope", "train --max-seconds NaN", "train --menu v3", "train stray", "export cp --menu v1",
            "run-parity file --card-buckets dir", "parity --menu v2 file", "engine-deal nope", "train --resume cp",
            "train --completed-nodes 10", "train --card-tables-unpinned", "train --milestones 1,,2"] {
            assert!(parse(&args(s)).is_err(), "accepted {s}");
        }
    }
    #[test]
    fn accepts_supported_commands_and_defaults() {
        for s in ["train", "parity fixture --menu v2", "export checkpoint --average output", "run-parity fixture",
            "traversal-parity fixture", "engine-deal 1 2", "bench-train --roots roots --seed 1 --iterations 2 --lineage 3 --out out"] {
            assert!(parse(&args(s)).is_ok(), "rejected {s}");
        }
        let c = parse(&args("train --menu v2 --stack-bb 100")).unwrap();
        assert_eq!(c.config["options"]["menu"], "v2");
        assert_eq!(c.config["options"]["average_rule"], "traverser-reach");
    }
    #[test]
    fn resolves_schedule_and_lineage_with_requested_values_retained() {
        let c = parse(&args("bench-train --roots roots --seed 1 --iterations 3 --lineage 3 --checkpoints 2,2,0,99 --out out")).unwrap();
        assert_eq!(c.config["requested"]["checkpoints"], json!([2, 2, 0, 99]));
        assert_eq!(c.config["options"]["checkpoints"], json!([2, 3]));
        assert_eq!(c.config["requested"]["lineage"], "3");
        assert_eq!(c.config["options"]["lineage"], 3);
        let c = parse(&args("train --nodes 3 --milestones 2,2,0,99")).unwrap();
        assert_eq!(c.config["options"]["milestones"], json!([2, 3]));
    }
    #[test]
    fn refuses_dcfr_domains_during_admission() {
        for flags in ["--dcfr 1,0,2 --iterations 1", "--dcfr 1.5,0,9 --iterations 1",
                      "--dcfr 1.5,0,2 --iterations 4294967296"] {
            assert!(parse(&args(&format!("bench-train --roots absent --seed 1 --lineage 3 --out out {flags}"))).is_err());
        }
    }

}

#[cfg(test)]
mod freeze_tests {
    use super::*;
    fn args(s: &str) -> Vec<String> { std::iter::once("hu20-trainer").chain(s.split_whitespace()).map(str::to_string).collect() }
    #[test]
    fn freeze_is_a_train_only_boolean_with_an_explicit_default() {
        assert_eq!(parse(&args("train")).unwrap().config["options"]["freeze_at_cap"], false);
        assert_eq!(parse(&args("train --freeze-at-cap")).unwrap().config["options"]["freeze_at_cap"], true);
        assert_eq!(parse(&args("train")).unwrap().config["options"]["fast_gzip"], false);
        assert_eq!(parse(&args("train --fast-gzip")).unwrap().config["options"]["fast_gzip"], true);
        assert!(parse(&args("export cp --fast-gzip")).is_err());
        let resume = format!("train --resume cp --resume-sha256 {} --freeze-at-cap", "a".repeat(64));
        assert!(parse(&args(&resume)).is_ok());
        for s in ["train --freeze-at-cap true", "train --freeze-at-cap --freeze-at-cap",
            "export cp --freeze-at-cap", "parity cp --freeze-at-cap", "bench-train --freeze-at-cap"] {
            assert!(parse(&args(s)).is_err(), "accepted {s}");
        }
    }
}
