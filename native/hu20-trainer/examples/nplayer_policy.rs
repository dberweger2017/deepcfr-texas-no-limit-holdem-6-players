//! Explicit export and privileged replay harness; policy methods see observations only.
use hu20_trainer::{
    cards::EquityTables,
    multi_policy::{Mode, Policy, PolicyTable},
    nplayer::{Event, Hand},
};
use rand::{seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::Write, path::Path, sync::Arc};
fn snapshot(hand: &Hand) -> Value {
    let legal = hand
        .legal()
        .map(|l| json!({"check":l.check,"call":l.call_amount,"raise_to":l.raise_to}));
    json!({"actor":hand.actor(),"finished":hand.finished(),"board":hand.board(),
        "street":if hand.finished(){None}else{Some(hand.street().name())},"pot":hand.pot(),
        "wager":if hand.finished(){None}else{Some(hand.wager())},"min_raise":if hand.finished(){None}else{Some(hand.min_raise())},
        "players":hand.players().iter().map(|p|json!([p.stack,p.street_bet,p.contributed,p.folded])).collect::<Vec<_>>(),
        "holes":(0..hand.config().players()).map(|p|hand.observe(p).unwrap().hole).collect::<Vec<_>>(),"legal":legal,
        "rewards":hand.settlement().map(|s|s.rewards[..hand.config().players()].to_vec())})
}
fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("export") => {
            // Positional arguments never start with "--".
            let fast = a.len() == 7 && a[6] == "--fast-gzip";
            if !(a.len() == 6 || fast) || a[2..6].iter().any(|s| s.starts_with("--")) {
                return Err(
                    "export CHECKPOINT SHA256 current|average|scoped-average NEW_OUTPUT [--fast-gzip]".into(),
                );
            }
            let gzip = if fast { hu20_trainer::gz::Gzip::Parallel } else { hu20_trainer::gz::Gzip::Compatible };
            let mode = Mode::parse(&a[4])?;
            hu20_trainer::multi_policy::export(Path::new(&a[2]), &a[3], mode, Path::new(&a[5]), gzip)?;
            println!(
                "{}",
                json!({"mode":mode.name(),"bytes":std::fs::metadata(&a[5]).map_err(|e|e.to_string())?.len(),"sha256":hu20_trainer::export::sha256_file(Path::new(&a[5]))})
            );
        }
        Some("play") => {
            if a.len() < 10 || (a.len() - 6) % 2 != 0 {
                return Err("play K50 HANDS SEED NEW_TRACE POLICY SHA256 [POLICY SHA256 ...] (one pair per seat)".into());
            }
            let hands: usize = a[3].parse().map_err(|_| "invalid hand count")?;
            if !(1..=10000).contains(&hands) {
                return Err("hands must be 1..10000".into());
            }
            let seed: u64 = a[4].parse().map_err(|_| "invalid seed")?;
            let mut cache: HashMap<(String, String), Arc<PolicyTable>> = HashMap::new();
            let mut policies = Vec::new();
            let mut pins = Vec::new();
            for (seat, pair) in a[6..].chunks_exact(2).enumerate() {
                let id = (pair[0].clone(), pair[1].clone());
                let table = if let Some(t) = cache.get(&id) {
                    t.clone()
                } else {
                    let t = Arc::new(PolicyTable::load(Path::new(&pair[0]), &pair[1])?);
                    cache.insert(id, t.clone());
                    t
                };
                let mut hash = Sha256::new();
                hash.update(b"nplayer-policy-actions-v1");
                hash.update(seed.to_le_bytes());
                hash.update((seat as u64).to_le_bytes());
                let action_seed = u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap());
                pins.push(json!({"sha256":pair[1],"strategy":table.mode.name(),"average_scope":table.scope,"source_checkpoint_sha256":table.source_checkpoint_sha256}));
                policies.push(Policy::new(table, action_seed));
            }
            let n = policies.len();
            let identity = policies[0].table.identity.clone();
            if n != policies[0].table.players
                || policies.iter().any(|p| p.table.identity != identity)
            {
                return Err(
                    "need matching game identity and one distinct policy instance per seat".into(),
                );
            }
            let config = policies[0].table.config();
            let tables = EquityTables::load(Path::new(&a[2]), false)?;
            let mut out = std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&a[5])
                    .map_err(|e| e.to_string())?,
            );
            writeln!(out,"{}",json!({"format":"nplayer-privileged-policy-replay-v1","identity":identity,"hands":hands,"seed":seed,"policies":pins,"distinct_policy_instances":n,"loaded_weight_tables":cache.len()})).map_err(|e|e.to_string())?;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let mut decisions = [0u64; 4];
            let mut misses = [0u64; 4];
            for hand_id in 0..hands {
                let mut deck: Vec<u8> = (0..52).collect();
                deck.shuffle(&mut rng);
                let button = hand_id % n;
                let mut hand = Hand::from_deck(config, button, &deck)?;
                let mut replay = hand.clone();
                let mut states = vec![snapshot(&hand)];
                let mut actions = Vec::new();
                let mut actors = Vec::new();
                while let Some(actor) = hand.actor() {
                    if actions.len() >= 10000 {
                        return Err("hand exceeded action bound".into());
                    }
                    let observation = hand.observe(actor)?;
                    let street = observation.street as usize;
                    let old_misses = policies[actor].misses;
                    let action = policies[actor].choose(&observation, tables)?;
                    decisions[street] += 1;
                    misses[street] += policies[actor].misses - old_misses;
                    hand.apply_mut(action)?;
                    replay.apply_mut(action)?;
                    if hand != replay {
                        return Err("native action replay differs".into());
                    }
                    actors.push(actor);
                    actions.push(json!([action.kind.name(), action.raise_to]));
                    states.push(snapshot(&hand));
                }
                let s = hand.settlement().ok_or("hand did not settle")?;
                let history: Vec<_> = hand
                    .history()
                    .iter()
                    .map(|e| match e {
                        Event::Board(s) => json!([s.name(), "board"]),
                        Event::Action {
                            street,
                            actor,
                            action,
                            paid,
                        } => json!([
                            street.name(),
                            actor,
                            action.kind.name(),
                            action.raise_to,
                            paid
                        ]),
                    })
                    .collect();
                writeln!(out,"{}",json!({"hand":hand_id,"stacks":config.stacks(),"button":button,"sb":config.small_blind(),"bb":config.big_blind(),"deck":deck,"actors":actors,"actions":actions,"states":states,"history":history,"settlement":{"contributions":s.contributions[..n],"refunds":s.refunds[..n],"pots":s.pots().iter().map(|p|json!([p.amount,p.eligible])).collect::<Vec<_>>()}})).map_err(|e|e.to_string())?;
            }
            out.flush().map_err(|e| e.to_string())?;
            println!(
                "{}",
                json!({"hands":hands,"players":n,"distinct_policy_instances":n,"loaded_weight_tables":cache.len(),"decisions_by_street":decisions,"misses_by_street":misses,"seat_decisions":policies.iter().map(|p|p.decisions).collect::<Vec<_>>(),"trace_sha256":hu20_trainer::export::sha256_file(Path::new(&a[5]))})
            );
        }
        _ => return Err("expected export or play".into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("nplayer_policy: {e}");
        std::process::exit(2);
    }
}
