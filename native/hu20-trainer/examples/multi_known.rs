//! Bounded known-game experiment. No NLHE training or model selection.
use hu20_trainer::multi_cfr::{Game, Trainer};
use hu20_trainer::small_exact::Tree;
use hu20_trainer::small_games::{published_kuhn_policy, Kind, SmallGame};
use serde_json::json;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (5..=7).contains(&args.len()),
        "multi_known kuhn|leduc PLAYERS ITERATIONS SEED [NEW_OUTPUT_DIRECTORY [RESUME_CHECKPOINT]]"
    );
    let kind = match args[1].as_str() {
        "kuhn" => Kind::Kuhn,
        "leduc" => Kind::Leduc,
        _ => panic!("unknown game"),
    };
    let game = SmallGame::new(kind, args[2].parse().unwrap());
    let iterations: u64 = args[3].parse().unwrap();
    let seed: u64 = args[4].parse().unwrap();
    let tree = Tree::new(game, 10_000_000);
    let out = args.get(5).map(std::path::PathBuf::from);
    if let Some(p) = &out {
        std::fs::create_dir(p).unwrap();
    }
    let export = |name: &str, profile: &[Vec<f64>]| {
        if let Some(out) = &out {
            let policies: serde_json::Map<String, serde_json::Value> = tree
                .infos
                .iter()
                .zip(profile)
                .map(|(i, p)| {
                    (
                        hu20_trainer::multi_checkpoint::key_hex(&i.decision.key),
                        json!(p),
                    )
                })
                .collect();
            let e = tree.evaluate(profile);
            let result = json!({"identity":game.identity(),"policies":policies,"values":e.values,"best_response":e.best_response,"gains":e.gains});
            serde_json::to_writer(std::fs::File::create(out.join(name)).unwrap(), &result).unwrap();
        }
    };
    println!(
        "{}",
        json!({"identity":game.identity(),"tree_nodes":tree.nodes(),"information_sets":tree.infos.len()})
    );
    if kind == Kind::Kuhn && game.players == 3 {
        for beta in [0., 0.125, 0.25] {
            let profile = tree
                .infos
                .iter()
                .map(|i| published_kuhn_policy(&i.example, beta))
                .collect::<Vec<_>>();
            let e = tree.evaluate(&profile);
            println!(
                "{}",
                json!({"analytic_beta":beta,"values":e.values,"gains":e.gains})
            );
            assert!(e.gains.iter().all(|g| g.abs() < 1e-12));
            let expected = [-(0.5 + beta) / 24., -0.5 / 24., (1. + beta) / 24.];
            assert!(e
                .values
                .iter()
                .zip(expected)
                .all(|(v, e)| (v - e).abs() < 1e-12));
            export(&format!("analytic-{beta}.json"), &profile);
        }
    }
    let mut trainer = if let Some(path) = args.get(6) {
        let t = hu20_trainer::multi_checkpoint::load(std::path::Path::new(path)).unwrap();
        assert_eq!(t.identity, game.identity());
        assert_eq!(t.seed, seed);
        assert_eq!(t.last_average_street, 3);
        assert!(!t.freeze_at_cap && t.frozen_at.is_none());
        assert_eq!(t.max_entries, usize::MAX);
        assert!(t.iteration <= iterations);
        t
    } else {
        Trainer::new(&game, seed, 3, usize::MAX, false)
    };
    let start = std::time::Instant::now();
    let mut milestone = 1;
    while milestone <= trainer.iteration {
        milestone *= 10;
    }
    for t in trainer.iteration + 1..=iterations {
        trainer.step(&game);
        if t == milestone || t == iterations {
            let e = tree.evaluate(&tree.profile(&trainer.table, true));
            println!(
                "{}",
                json!({"iteration":t,"seed":seed,"nodes":trainer.counts.nodes,"entries":trainer.table.len(),
                "elapsed_seconds":start.elapsed().as_secs_f64(),"values":e.values,"best_response":e.best_response,"gains":e.gains})
            );
            milestone *= 10;
        }
    }
    export("average.json", &tree.profile(&trainer.table, true));
    if let Some(out) = &out {
        hu20_trainer::multi_checkpoint::save(&trainer, &out.join("checkpoint.gz"), hu20_trainer::gz::Gzip::Compatible).unwrap();
    }
}
