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
        _ => {
            eprintln!("usage: hu20-trainer parity FIXTURES.jsonl");
            std::process::exit(2);
        }
    }
}
