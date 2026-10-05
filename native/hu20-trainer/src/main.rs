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
        Some("run-parity") => {
            let document: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
            match hu20_trainer::parity::check_run(&document) {
                Ok((iterations, entries)) => println!("iterations {iterations} entries {entries} identical"),
                Err(problem) => {
                    eprintln!("{problem}");
                    std::process::exit(1);
                }
            }
        }
        Some("engine-deal") => {
            // Hole cards by seat and the board for button 0, as the engine deals them from a seed.
            for seed in &args[2..] {
                let hand = hu20_trainer::game::Hand::from_deck(0, &hu20_trainer::streams::engine_deck(seed.parse().unwrap()));
                let name = |c: &u8| hu20_buckets::card_name(*c);
                println!("{} {}", hand.holes.iter().flatten().map(name).collect::<Vec<_>>().join(" "),
                         hand.board.iter().map(name).collect::<Vec<_>>().join(" "));
            }
        }
        Some("train") => {
            let arg = |name: &str, default: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone()).unwrap_or(default.into());
            let nodes: u64 = arg("--nodes", "1000000").parse().unwrap();
            // Alternatively stop after a fixed number of iterations.
            let iterations: u64 = arg("--iterations", "0").parse().unwrap();
            let seed: u64 = arg("--seed", "1").parse().unwrap();
            let roots: usize = arg("--roots-per-seat", "1").parse().unwrap();
            let out = arg("--out", "native-checkpoint.json.gz");
            // Optional earlier saves, like Python's milestones: the first complete iteration at or past each count.
            // A `{nodes}` placeholder in `--out` names each save by its milestone.
            let mut milestones: Vec<u64> = arg("--milestones", "").split(',').filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap()).filter(|&m| m < nodes).collect();
            milestones.push(nodes);
            let mut trainer = hu20_trainer::trainer::Trainer::new(seed, roots);
            trainer.average = hu20_trainer::cfr::AverageRule::parse(&arg("--average-rule", "traverser-reach"));
            let started = std::time::Instant::now();
            for milestone in milestones {
                while trainer.nodes < milestone && (iterations == 0 || trainer.iteration < iterations) {
                    trainer.step();
                }
                let seconds = started.elapsed().as_secs_f64();
                let path = std::path::PathBuf::from(out.replace("{nodes}", &milestone.to_string()));
                trainer.save(&path, 1_000_000_000, 1_000_000_000).unwrap();
                println!("milestone {} iterations {} nodes {} entries {} seconds {:.2} nodes_per_second {:.0} path {}",
                         milestone, trainer.iteration, trainer.nodes, trainer.table.len(), seconds,
                         trainer.nodes as f64 / seconds, path.display());
            }
        }
        Some("export") => {
            let arg = |name: &str| args.iter().position(|a| a == name).map(|i| std::path::PathBuf::from(&args[i + 1]));
            let checkpoint = std::path::PathBuf::from(&args[2]);
            let started = std::time::Instant::now();
            let count = hu20_trainer::export::export(&checkpoint, arg("--current").as_deref(), arg("--average").as_deref()).unwrap();
            println!("exported {count} entries in {:.2} s", started.elapsed().as_secs_f64());
        }
        _ => {
            eprintln!("usage: hu20-trainer parity FIXTURES.jsonl | traversal-parity FIXTURE.json | run-parity FIXTURE.json | train --nodes N [--iterations I] [--milestones N1,N2] --seed S [--roots-per-seat R] [--average-rule traverser-reach|opponent-sampled] --out PATH | export CHECKPOINT [--current PATH] [--average PATH]");
            std::process::exit(2);
        }
    }
}
