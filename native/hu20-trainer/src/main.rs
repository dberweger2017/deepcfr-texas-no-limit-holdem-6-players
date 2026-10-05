//! hu20-trainer parity FIXTURES.jsonl
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("parity") => {
            let file = std::fs::File::open(&args[2]).expect("fixtures");
            let (mut hands, mut decisions, mut failures) = (0u64, 0u64, 0u64);
            for line in std::io::BufReader::new(file).lines() {
                let record: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
                hands += 1;
                decisions += record["decisions"].as_array().unwrap().len() as u64;
                if let Some(problem) = hu20_trainer::parity::check_hand(&record) {
                    failures += 1;
                    if failures <= 5 {
                        eprintln!("{problem}");
                    }
                }
            }
            println!("hands {hands} decisions {decisions} mismatched_hands {failures}");
            if failures > 0 {
                std::process::exit(1);
            }
        }
        Some("traversal-parity") => {
            let document: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
            let (cases, problems) = hu20_trainer::parity::check_traversals(&document);
            for p in problems.iter().take(5) {
                eprintln!("{p}");
            }
            println!("traversals {cases} mismatched {}", problems.len());
            if !problems.is_empty() {
                std::process::exit(1);
            }
        }
        Some("train") => {
            let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or(default.into());
            let nodes: u64 = arg("--nodes", "1000000").parse().unwrap();
            let seed: u64 = arg("--seed", "1").parse().unwrap();
            let roots: usize = arg("--roots-per-seat", "1").parse().unwrap();
            let out = std::path::PathBuf::from(arg("--out", "native-checkpoint.json.gz"));
            let mut trainer = hu20_trainer::trainer::Trainer::new(seed, roots);
            let started = std::time::Instant::now();
            while trainer.nodes < nodes {
                trainer.step();
            }
            let seconds = started.elapsed().as_secs_f64();
            trainer.save(&out, 1_000_000_000, 1_000_000_000).unwrap();
            println!("iterations {} nodes {} entries {} seconds {:.2} nodes_per_second {:.0}",
                     trainer.iteration, trainer.nodes, trainer.table.len(), seconds, trainer.nodes as f64 / seconds);
        }
        Some("export") => {
            let arg = |name: &str| args.iter().position(|a| a == name).map(|i| std::path::PathBuf::from(&args[i + 1]));
            let checkpoint = std::path::PathBuf::from(&args[2]);
            let started = std::time::Instant::now();
            let count = hu20_trainer::export::export(&checkpoint, arg("--current").as_deref(), arg("--average").as_deref()).unwrap();
            println!("exported {count} entries in {:.2} s", started.elapsed().as_secs_f64());
        }
        _ => {
            eprintln!("usage: hu20-trainer parity FIXTURES.jsonl | traversal-parity FIXTURE.json | train --nodes N --seed S --out PATH");
            std::process::exit(2);
        }
    }
}
