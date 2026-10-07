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
            milestones.sort_unstable();
            milestones.dedup();
            let game = hu20_trainer::game::Game::from_bb(arg("--stack-bb", "20").parse().unwrap());
            let max_entries: u64 = arg("--max-entries", "1000000000").parse().unwrap();
            let max_seconds: f64 = arg("--max-seconds", "inf").parse().unwrap();
            let stop_file = args.iter().position(|a| a == "--stop-file").map(|i| std::path::PathBuf::from(&args[i + 1]));
            let telemetry = args.iter().position(|a| a == "--telemetry").map(|i| std::path::PathBuf::from(&args[i + 1]));
            if let Some(path) = &stop_file { assert!(!path.exists(), "use a fresh stop-request path"); }
            if let Some(path) = &telemetry {
                std::fs::OpenOptions::new().write(true).create_new(true).open(path).expect("fresh telemetry path");
            }
            assert!(nodes > 0 && roots > 0 && max_entries > 0 && max_seconds > 0.0);
            let resume = args.iter().position(|a| a == "--resume");
            let mut trainer = if let Some(i) = resume {
                let path = std::path::Path::new(&args[i + 1]);
                let expected = arg("--resume-sha256", "");
                assert_eq!(expected.len(), 64, "resume requires --resume-sha256");
                assert_eq!(hu20_trainer::export::sha256_file(path), expected, "resume hash differs");
                let completed = args.iter().position(|a| a == "--completed-nodes").map(|i| args[i + 1].parse().unwrap());
                let loaded = hu20_trainer::checkpoint::load(path, game, completed, max_entries).expect("resume checkpoint");
                for (flag, value) in [("--seed", loaded.seed.to_string()), ("--roots-per-seat", loaded.roots_per_seat.to_string()),
                                      ("--average-rule", loaded.average.name().to_string())] {
                    if args.iter().any(|a| a == flag) { assert_eq!(arg(flag, ""), value, "resume config differs: {flag}"); }
                }
                assert!(!args.iter().any(|a| a == "--regret-floor"), "resume requires linear CFR");
                loaded
            } else {
                let mut fresh = hu20_trainer::trainer::Trainer::new(seed, roots);
                fresh.game = game;
                fresh.average = hu20_trainer::cfr::AverageRule::parse(&arg("--average-rule", "traverser-reach"));
                fresh
            };
            let recovery = args.iter().any(|a| a == "--recovery") || resume.is_some() || game == hu20_trainer::game::Game::Hu100;
            // CFR+'s floor keeps the production average weights, so exports and their bounds are unchanged.
            // DCFR reweights the average and stays bench-only.
            trainer.options.regret_floor = args.iter().position(|a| a == "--regret-floor").map(|i| args[i + 1].parse().unwrap());
            if let Err(problem) = trainer.options.check(0) {
                panic!("{problem}");
            }
            let started = std::time::Instant::now();
            let initial_nodes = trainer.nodes;
            let (mut previous_nodes, mut previous_entries, mut previous_seconds) = (trainer.nodes, trainer.table.len(), 0.0);
            for milestone in milestones {
                if milestone <= initial_nodes { continue; }
                let mut stopped = false;
                while trainer.nodes < milestone && (iterations == 0 || trainer.iteration < iterations) {
                    if started.elapsed().as_secs_f64() >= max_seconds
                        || stop_file.as_ref().map_or(false, |path| path.exists()) { stopped = true; break; }
                    trainer.step();
                    if trainer.table.len() as u64 >= max_entries { stopped = true; break; }
                }
                let seconds = started.elapsed().as_secs_f64();
                let path = std::path::PathBuf::from(out.replace("{nodes}", &milestone.to_string()));
                let write_started = hu20_trainer::telemetry::unix_seconds();
                let saving = std::time::Instant::now();
                if recovery { trainer.save_recoverable(&path, 1_000_000_000, max_entries.max(trainer.table.len() as u64)).unwrap(); }
                else { trainer.save(&path, 1_000_000_000, max_entries.max(trainer.table.len() as u64)).unwrap(); }
                let write_seconds = saving.elapsed().as_secs_f64();
                let write_finished = hu20_trainer::telemetry::unix_seconds();
                if let Some(receipts) = &telemetry {
                    let elapsed = started.elapsed().as_secs_f64();
                    let record = serde_json::json!({"version": 1,
                        "status": if trainer.nodes >= milestone { "saved" } else { "incomplete-target" },
                        "requested_nodes": milestone, "completed_nodes": trainer.nodes,
                        "overshoot_nodes": trainer.nodes.saturating_sub(milestone),
                        "iteration": trainer.iteration, "path": path,
                        "checkpoint_bytes": std::fs::metadata(&path).unwrap().len(),
                        "checkpoint_sha256": hu20_trainer::export::sha256_file(&path),
                        "write_started": write_started, "write_finished": write_finished, "write_seconds": write_seconds,
                        "elapsed_seconds_including_writes": elapsed,
                        "nodes_per_second_including_writes": (trainer.nodes - initial_nodes) as f64 / elapsed,
                        "recent_nodes_per_second_including_writes": (trainer.nodes - previous_nodes) as f64 / (elapsed - previous_seconds),
                        "new_entries": trainer.table.len() - previous_entries,
                        "nodes_since_previous_save": trainer.nodes - previous_nodes,
                        "diagnostics": hu20_trainer::telemetry::diagnostics(&trainer),
                        "stop_requested": stop_file.as_ref().map_or(false, |path| path.exists()),
                        "audit_status": "unaudited"});
                    hu20_trainer::telemetry::append(receipts, &record).expect("checkpoint telemetry");
                    previous_nodes = trainer.nodes; previous_entries = trainer.table.len(); previous_seconds = elapsed;
                }
                println!("milestone {} iterations {} nodes {} entries {} seconds {:.2} nodes_per_second {:.0} path {}",
                         milestone, trainer.iteration, trainer.nodes, trainer.table.len(), seconds,
                         (trainer.nodes - initial_nodes) as f64 / seconds, path.display());
                if stopped { eprintln!("resource limit reached at a complete iteration; inspect recovery checkpoint"); std::process::exit(3); }
                if iterations != 0 && trainer.iteration >= iterations { break; }
            }
        }
        Some("bench-train") => {
            // #162's bench: both averages in one run, like `SubgameTrainer`, as two lockstep trainers
            // whose regrets must agree. Writes `iteration-N/VARIANT.STRATEGY.json` for the bench's evaluator.
            use hu20_trainer::bench;
            use hu20_trainer::cfr::AverageRule;
            let arg = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
            let required = |name: &str| arg(name).unwrap_or_else(|| panic!("{name} is required"));
            let document: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(required("--roots")).unwrap()).unwrap();
            let roots: Vec<bench::FrozenRoot> = document.as_array().unwrap().iter().map(bench::FrozenRoot::from_json).collect();
            let seed: u64 = required("--seed").parse().unwrap();
            let iterations: u64 = required("--iterations").parse().unwrap();
            // A JSON value, as #149's manifests give it (an integer seed), else a plain string.
            let raw = required("--lineage");
            let lineage = serde_json::from_str(&raw).unwrap_or(serde_json::Value::String(raw));
            let variant = arg("--variant").unwrap_or("base".into());
            let out = std::path::PathBuf::from(required("--out"));
            // Like Python's sorted(set(checkpoints) | {iterations}); iteration 0 is never exported.
            let mut checkpoints: Vec<u64> = arg("--checkpoints").unwrap_or_default().split(',').filter(|s| !s.is_empty())
                .map(|s| s.parse().unwrap()).filter(|&c| 0 < c && c < iterations).collect();
            checkpoints.push(iterations);
            checkpoints.sort_unstable();
            checkpoints.dedup();
            let options = hu20_trainer::cfr::Options {
                regret_floor: arg("--regret-floor").map(|f| f.parse().unwrap()),
                dcfr: arg("--dcfr").map(|v| {
                    let v: Vec<f64> = v.split(',').map(|x| x.parse().unwrap()).collect();
                    v.try_into().expect("--dcfr alpha,beta,gamma")
                }),
            };
            if let Err(problem) = options.check(iterations) {
                panic!("{problem}");
            }
            let mut trainers = [AverageRule::TraverserReach, AverageRule::OpponentSampled].map(|rule| {
                let mut trainer = hu20_trainer::trainer::Trainer::new(seed, 1);
                trainer.average = rule;
                trainer.options = options;
                trainer
            });
            let write = |path: std::path::PathBuf, value: &serde_json::Value| {
                let temporary = path.with_extension("tmp");
                std::fs::write(&temporary, serde_json::to_string(value).unwrap()).unwrap();
                std::fs::rename(temporary, path).unwrap();
            };
            let started = std::time::Instant::now();
            for checkpoint in checkpoints {
                while trainers[0].iteration < checkpoint {
                    let [a, b] = &mut trainers;
                    let (x, y) = rayon::join(|| bench::step(a, &roots), || bench::step(b, &roots));
                    assert_eq!(x, y, "the averaging rule changed the traversal");
                }
                bench::check_lockstep(&trainers[0], &trainers[1]);
                let current = bench::export(&trainers[1], &trainers[1], &lineage, false);
                let folder = out.join(format!("iteration-{checkpoint}"));
                std::fs::create_dir_all(&folder).unwrap();
                write(folder.join(format!("{variant}.current.json")), &current);
                for trainer in &trainers {
                    let name = bench::strategy(trainer.average);
                    write(folder.join(format!("{variant}.{name}.json")), &bench::export(&trainers[1], trainer, &lineage, true));
                }
                let seconds = started.elapsed().as_secs_f64();
                println!("iteration {} nodes {} keys {} seconds {:.2} iterations_per_second {:.0}", checkpoint,
                         trainers[0].nodes, trainers[1].table.len(), seconds, checkpoint as f64 / seconds);
            }
        }
        Some("export") => {
            let arg = |name: &str| args.iter().position(|a| a == name).map(|i| std::path::PathBuf::from(&args[i + 1]));
            let checkpoint = std::path::PathBuf::from(&args[2]);
            let started = std::time::Instant::now();
            let zero_mass = args.iter().position(|a| a == "--zero-mass")
                .map_or(hu20_trainer::export::ZeroMass::Uniform, |i| hu20_trainer::export::ZeroMass::parse(&args[i + 1]));
            let count = hu20_trainer::export::export(&checkpoint, arg("--current").as_deref(), arg("--average").as_deref(), zero_mass).unwrap();
            println!("exported {count} entries in {:.2} s", started.elapsed().as_secs_f64());
        }
        _ => {
            eprintln!("usage: hu20-trainer parity FIXTURES.jsonl | traversal-parity FIXTURE.json | run-parity FIXTURE.json | train --nodes N [--stack-bb 20|100] [--resume PATH --resume-sha256 HASH [--completed-nodes N]] [--recovery] [--max-entries N] [--max-seconds S] [--iterations I] [--milestones N1,N2] --seed S [--roots-per-seat R] [--average-rule traverser-reach|opponent-sampled] [--regret-floor F] --out PATH | export CHECKPOINT [--current PATH] [--average PATH] [--zero-mass uniform|current] | bench-train --roots ROOTS.json --seed S --iterations N [--checkpoints a,b] --lineage NAME [--variant NAME] [--regret-floor F] [--dcfr A,B,G] --out FOLDER");
            std::process::exit(2);
        }
    }
}
