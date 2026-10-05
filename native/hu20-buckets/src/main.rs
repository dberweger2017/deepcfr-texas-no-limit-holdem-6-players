//! hu20-buckets build --out DIR [--ranks 13] [--ks 50,200] [--bins 50] [--seed 1] [--iterations 100]
//! hu20-buckets key HOLE BOARD     e.g. key "Ah Kh" "2h 7c 9d"
use hu20_buckets::*;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn arg(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or_else(|| default.to_string())
}

fn cards(text: &str) -> Vec<u8> {
    text.split_whitespace().map(parse_card).collect()
}

struct Log {
    started: Instant,
    status: PathBuf,
}
impl Log {
    fn stage(&self, stage: &str, detail: &str) {
        let elapsed = self.started.elapsed().as_secs_f64();
        eprintln!("[{elapsed:9.1}s] {stage} {detail}");
        let text = format!("{{\"stage\": \"{stage}\", \"detail\": \"{detail}\", \"elapsed_seconds\": {elapsed:.1}}}\n");
        fs::write(&self.status, text).expect("status");
    }
}

/// Table file: magic, street, k, count, sorted u64 class hashes, then u16 buckets.
fn write_table(path: &Path, street: u32, k: u32, keys: &[u128], assignment: &[u16]) {
    let mut rows: Vec<(u64, u16)> = keys.iter().map(|&key| hash_key(key)).zip(assignment.iter().copied()).collect();
    rows.sort_unstable_by_key(|r| r.0);
    assert!(rows.windows(2).all(|w| w[0].0 != w[1].0), "64-bit class hash collision");
    let mut out = Vec::with_capacity(24 + rows.len() * 10);
    out.extend_from_slice(b"HU20BKT1");
    out.extend_from_slice(&street.to_le_bytes());
    out.extend_from_slice(&k.to_le_bytes());
    out.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    for r in &rows {
        out.extend_from_slice(&r.0.to_le_bytes());
    }
    for r in &rows {
        out.extend_from_slice(&r.1.to_le_bytes());
    }
    fs::write(path, out).expect("table");
}

fn bucket_summary(k: usize, assignment: &[u16], weight: &[u64], equity: &[f32]) -> String {
    let mut mass = vec![0f64; k];
    let mut eq = vec![0f64; k];
    for ((&c, &w), &e) in assignment.iter().zip(weight).zip(equity) {
        mass[c as usize] += w as f64;
        eq[c as usize] += w as f64 * e as f64;
    }
    let total: f64 = mass.iter().sum();
    let parts: Vec<String> = (0..k)
        .map(|c| format!("[{:.8}, {:.6}]", mass[c] / total, if mass[c] > 0.0 { eq[c] / mass[c] } else { 0.0 }))
        .collect();
    format!("[{}]", parts.join(", "))
}

fn build(args: &[String]) {
    let out = PathBuf::from(arg(args, "--out", "buckets"));
    let ranks: u8 = arg(args, "--ranks", "13").parse().unwrap();
    let bins: usize = arg(args, "--bins", "50").parse().unwrap();
    let seed: u64 = arg(args, "--seed", "202610050003").parse().unwrap();
    let iterations: usize = arg(args, "--iterations", "100").parse().unwrap();
    let ks: Vec<usize> = arg(args, "--ks", "50,200").split(',').map(|k| k.parse().unwrap()).collect();
    fs::create_dir_all(&out).unwrap();
    let log = Log { started: Instant::now(), status: out.join("status.json") };
    let deck = deck(ranks);
    let mut summary = vec![format!(
        "\"format\": \"hu20-equity-buckets-v1\", \"ranks\": {ranks}, \"deck_cards\": {}, \"bins\": {bins}, \"seed\": {seed}, \"ks\": {ks:?}, \"threads\": {}",
        deck.len(), rayon::current_num_threads())];

    log.stage("river-equity", "computing");
    let river = RiverTable::build(&deck);
    log.stage("river-equity", &format!("{} classes", river.keys.len()));
    let mut streets = Vec::new();
    for (street, size) in [("turn", 4usize), ("flop", 3usize)] {
        log.stage(street, "features");
        let features = StreetFeatures::build(&deck, size, bins, &river);
        log.stage(street, &format!("{} classes", features.keys.len()));
        streets.push((street, features));
    }
    let mut parts = Vec::new();
    // River buckets cluster scalar equity.
    for &k in &ks {
        log.stage("river-kmeans", &format!("k={k}"));
        let c = kmeans(&river.equity, 1, &river.weight, k, seed, iterations);
        write_table(&out.join(format!("river-k{k}.bin")), 3, k as u32, &river.keys, &c.assignment);
        parts.push(format!(
            "{{\"street\": \"river\", \"k\": {k}, \"classes\": {}, \"weight\": {}, \"objective\": {:?}, \"buckets\": {}}}",
            river.keys.len(), river.weight.iter().sum::<u64>(), c.objective,
            bucket_summary(k, &c.assignment, &river.weight, &river.equity)));
    }
    for (street, f) in &streets {
        let code = if *street == "turn" { 2 } else { 1 };
        for &k in &ks {
            log.stage(&format!("{street}-kmeans"), &format!("k={k}"));
            let c = kmeans(&f.cdf, f.bins, &f.weight, k, seed, iterations);
            write_table(&out.join(format!("{street}-k{k}.bin")), code, k as u32, &f.keys, &c.assignment);
            parts.push(format!(
                "{{\"street\": \"{street}\", \"k\": {k}, \"classes\": {}, \"weight\": {}, \"objective\": {:?}, \"buckets\": {}}}",
                f.keys.len(), f.weight.iter().sum::<u64>(), c.objective,
                bucket_summary(k, &c.assignment, &f.weight, &f.mean_equity)));
        }
    }
    summary.push(format!("\"tables\": [{}]", parts.join(", ")));
    summary.push(format!("\"elapsed_seconds\": {:.1}", log.started.elapsed().as_secs_f64()));
    let mut file = fs::File::create(out.join("summary.json")).unwrap();
    writeln!(file, "{{{}}}", summary.join(", ")).unwrap();
    log.stage("complete", "all tables written");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") => build(&args),
        Some("key") => {
            let (hole, board) = (cards(&args[2]), cards(&args[3]));
            let key = class_key(&hole, &board);
            println!("{key:032x} {:016x} {}", hash_key(key), orbit_size(&hole, &board));
        }
        Some("eval") => {
            // One seven-card hand per stdin line; prints the comparable value.
            let stdin = std::io::stdin();
            let mut out = std::io::stdout().lock();
            for line in std::io::BufRead::lines(stdin.lock()) {
                writeln!(out, "{}", evaluate(&cards(&line.unwrap()))).unwrap();
            }
        }
        _ => {
            eprintln!("usage: hu20-buckets build --out DIR [--ranks 13] [--ks 50,200] [--bins 50] | key HOLE BOARD");
            std::process::exit(2);
        }
    }
}
