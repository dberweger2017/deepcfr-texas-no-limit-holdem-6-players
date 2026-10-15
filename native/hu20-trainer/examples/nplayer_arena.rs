//! Privileged duplicate host. Bots and models receive only acting observations.
use hu20_trainer::{
    cards::EquityTables,
    multi_arena::{stream_seed, Bot, PERMUTATIONS},
    multi_policy::{Policy, PolicyTable},
    nplayer::{Config, Event, Hand},
};
use rand::{seq::SliceRandom, SeedableRng};
use serde_json::{json, Value};
use std::{collections::HashMap, io::Write, path::Path, sync::Arc, time::Instant};

enum Player {
    Model(Policy),
    Script(Bot),
}
fn snapshot(hand: &Hand) -> Value {
    let legal = hand.legal().map(|l| json!({"check":l.check,"call":l.call_amount,"raise_to":l.raise_to}));
    json!({"actor":hand.actor(),"finished":hand.finished(),"board":hand.board(),
        "street":if hand.finished(){None}else{Some(hand.street().name())},"pot":hand.pot(),
        "wager":if hand.finished(){None}else{Some(hand.wager())},"min_raise":if hand.finished(){None}else{Some(hand.min_raise())},
        "players":hand.players().iter().map(|p|json!([p.stack,p.street_bet,p.contributed,p.folded])).collect::<Vec<_>>(),
        "holes":(0..hand.config().players()).map(|p|hand.observe(p).unwrap().hole).collect::<Vec<_>>(),"legal":legal,
        "rewards":hand.settlement().map(|s|s.rewards[..hand.config().players()].to_vec())})
}
fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 { return Err("K50 BLOCKS ROOT NEW_TRACE SINGLETON PAIR (policy=PATH@SHA256 or bot=NAME)".into()); }
    let blocks: u64 = a[2].parse().map_err(|_| "invalid blocks")?;
    if !(2..=1_000_000).contains(&blocks) { return Err("blocks must be 2..1000000".into()); }
    let root: u64 = a[3].parse().map_err(|_| "invalid root")?;
    let started = Instant::now();
    let mut cache = HashMap::new();
    let mut pins = Vec::new();
    let config = Config::new(&[2000; 3], 50, 100)?;
    let mut identity = None;
    for spec in [&a[5], &a[6], &a[6]] {
        if let Some(name) = spec.strip_prefix("bot=") {
            Bot::new(name, 0)?;
            pins.push(json!({"script":name,"version":"nplayer-scripts-v1"}));
        } else if let Some(pin) = spec.strip_prefix("policy=") {
            let (path, sha) = pin.rsplit_once('@').ok_or("policy requires PATH@SHA256")?;
            if !cache.contains_key(spec) {
                let table = Arc::new(PolicyTable::load(Path::new(path), sha)?);
                if table.config() != config { return Err("arena requires three-player 20 BB 50/100 identity".into()); }
                if identity.as_ref().is_some_and(|i| i != &table.identity) { return Err("mixed game identities".into()); }
                identity = Some(table.identity.clone());
                cache.insert(spec.clone(), table);
            }
            let t = &cache[spec];
            pins.push(json!({"sha256":sha,"strategy":t.mode.name(),"average_scope":t.scope,"source_checkpoint_sha256":t.source_checkpoint_sha256}));
        } else { return Err("expected policy=PATH@SHA256 or bot=NAME".into()); }
    }
    let tables = if cache.is_empty() { None } else { Some(EquityTables::load(Path::new(&a[1]), false)?) };
    let identity = identity.unwrap_or_else(|| json!({"players":3,"stacks":[2000,2000,2000],"small_blind":50,"big_blind":100}));
    let load_seconds = started.elapsed().as_secs_f64();
    let mut out = std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(&a[4]).map_err(|e|e.to_string())?);
    writeln!(out,"{}",json!({"format":"nplayer-privileged-policy-replay-v1","harness":"nplayer-duplicate-v1","identity":identity,"hands":blocks*6,"blocks":blocks,"seed":root,"policies":pins,"distinct_policy_instances":3,"loaded_weight_tables":cache.len(),"seat_semantics":"fixed physical seats; button=block%3; permutation maps seat to identity; identity0 singleton"})).map_err(|e|e.to_string())?;
    let playing = Instant::now();
    let mut decisions = [[0u64; 4]; 3];
    let mut misses = [[0u64; 4]; 3];
    for block in 0..blocks {
        if std::env::var_os("NPLAYER_STOP_FILE").is_some_and(|p| Path::new(&p).exists()) {
            out.flush().map_err(|e| e.to_string())?;
            return Err(format!("clean stop before block {block}; partial trace retained"));
        }
        let mut deck: Vec<u8> = (0..52).collect();
        deck.shuffle(&mut rand::rngs::StdRng::seed_from_u64(stream_seed(root,b"deck",block,0)));
        let button = block as usize % 3;
        for (permutation, ids) in PERMUTATIONS.iter().enumerate() {
            // Reset by labeled identity each duplicate. Action draws never advance the deck stream.
            let mut players = Vec::new();
            for id in 0..3 {
                let seed = stream_seed(root, b"action", block, id);
                let spec = if id == 0 { &a[5] } else { &a[6] };
                players.push(if let Some(name) = spec.strip_prefix("bot=") { Player::Script(Bot::new(name, seed)?) }
                    else { Player::Model(Policy::new(cache[spec].clone(), seed)) });
            }
            let mut hand = Hand::from_deck(config, button, &deck)?;
            let mut states = vec![snapshot(&hand)];
            let mut actions = Vec::new();
            let mut actors = Vec::new();
            while let Some(actor) = hand.actor() {
                if actions.len() >= 10000 { return Err("hand action bound".into()); }
                let observation = hand.observe(actor)?;
                let street = observation.street as usize;
                let id = ids[actor];
                let action = match &mut players[id] {
                    Player::Script(bot) => bot.choose(&observation)?,
                    Player::Model(policy) => {
                        let old = policy.misses;
                        let action = policy.choose(&observation, tables.unwrap())?;
                        misses[id][street] += policy.misses - old;
                        action
                    }
                };
                decisions[id][street] += 1;
                hand.apply_mut(action)?;
                actors.push(actor);
                actions.push(json!([action.kind.name(),action.raise_to]));
                states.push(snapshot(&hand));
            }
            let s = hand.settlement().ok_or("hand did not settle")?;
            if s.rewards.iter().sum::<i64>() != 0 { return Err("non-zero-sum settlement".into()); }
            let history: Vec<_> = hand.history().iter().map(|e|match e {
                Event::Board(s) => json!([s.name(),"board"]),
                Event::Action{street,actor,action,paid} => json!([street.name(),actor,action.kind.name(),action.raise_to,paid]),
            }).collect();
            writeln!(out,"{}",json!({"hand":block*6+permutation as u64,"block":block,"permutation":permutation,"identities":ids,"rewards":s.rewards[..3],"stacks":config.stacks(),"button":button,"sb":50,"bb":100,"deck":deck,"actors":actors,"actions":actions,"states":states,"history":history,"settlement":{"contributions":s.contributions[..3],"refunds":s.refunds[..3],"pots":s.pots().iter().map(|p|json!([p.amount,p.eligible])).collect::<Vec<_>>()}})).map_err(|e|e.to_string())?;
        }
    }
    out.flush().map_err(|e|e.to_string())?;
    println!("{}",json!({"status":"passed","blocks":blocks,"hands":blocks*6,"load_seconds":load_seconds,"play_seconds":playing.elapsed().as_secs_f64(),"decisions_by_identity_street":decisions,"misses_by_identity_street":misses,"trace_sha256":hu20_trainer::export::sha256_file(Path::new(&a[4]))}));
    Ok(())
}
fn main() {
    if let Err(error) = run() { eprintln!("nplayer_arena: {error}"); std::process::exit(2); }
}
