//! Streaming checkpoint validation shared by recovery and inference export.
use crate::cfr::{AverageRule, Node};
use crate::game::Game;
use crate::trainer::{Sharded, Trainer};
use flate2::read::GzDecoder;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn header_game(h: &Value) -> Result<Game, String> {
    let game = match h["config"]["game"].as_str() {
        Some(crate::trainer::GAME) => Game::Hu20,
        Some("hu100-native-reopening-100bb-52card-no-ante-rake-v1") => Game::Hu100,
        _ => return Err("unsupported native game".into()),
    };
    let identity = json!({"game": game.id(), "players": 2,
        "stacks": [game.stack(), game.stack()], "small_blind": 50, "big_blind": 100,
        "action_menu": game.menu(), "card_descriptor": "legacy-postflop-descriptor-v1",
        "raise_cap_semantics": "none; native minimum-raise/reopening/stack bounds"});
    let ids = h["table"]["player_ids"].as_array().ok_or("invalid table players")?;
    let config = &h["config"];
    if h["format"] != game.format() || h["abstraction"] != game.schema()
        || config["abstraction"] != game.schema() || h["kind"] != "training"
        || h["checkpoint_format"] != "jsonl-v2" || !config["raise_cap"].is_null()
        || !config.as_object().ok_or("invalid config")?.contains_key("raise_cap")
        || h["identity"] != identity || h["table"]["stacks"] != json!([game.stack(), game.stack()])
        || h["table"]["small_blind"] != 50 || h["table"]["big_blind"] != 100
        || h["table"]["chip_unit"] != "0.01" || !matches!(h["table"]["button"].as_u64(), Some(0 | 1))
        || ids.len() != 2 || ids[0] == ids[1] || ids.iter().any(|id| id.as_str().map_or(true, str::is_empty))
        || h["iteration"].as_u64().is_none() || config["seed"].as_u64().is_none()
        || config["roots_per_seat"].as_u64().map_or(true, |n| n == 0)
        || config["max_entries"].as_u64().map_or(true, |n| n == 0)
        || config["max_nodes"].as_u64().map_or(true, |n| n == 0)
    { return Err("native checkpoint game/schema/table identity differs".into()); }
    match h.get("average_rule").and_then(Value::as_str).unwrap_or("traverser-reach") {
        "traverser-reach" | "opponent-sampled" => (),
        _ => return Err("unknown average rule".into()),
    }
    Ok(game)
}

pub fn row_node(row: &Value) -> Result<([u8; 16], Node), String> {
    let row = row.as_array().ok_or("invalid checkpoint row")?;
    if row.len() != 5 { return Err("invalid checkpoint row length".into()); }
    let text = row[0].as_str().ok_or("invalid key")?;
    if text.len() != 32 || !text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("invalid key".into());
    }
    let names = row[1].as_array().ok_or("invalid menu")?;
    let regrets = row[2].as_array().ok_or("invalid regrets")?;
    let average = row[3].as_array().ok_or("invalid average")?;
    let visits = row[4].as_u64().ok_or("invalid visits")?;
    if names.is_empty() || names.len() > 5 || regrets.len() != names.len() || average.len() != names.len() {
        return Err("invalid node dimensions".into());
    }
    let mut code = 0u8;
    let mut last = None;
    for name in names {
        let i = crate::key::NAMES.iter().position(|&n| Some(n) == name.as_str()).ok_or("invalid menu name")?;
        if last.map_or(false, |p| p >= i) { return Err("noncanonical or duplicate menu".into()); }
        code |= 1 << i; last = Some(i);
    }
    let mut node = Node::empty(code, names.len());
    for i in 0..names.len() {
        node.regrets[i] = regrets[i].as_f64().filter(|v| v.is_finite()).ok_or("invalid regret")?;
        node.average[i] = average[i].as_f64().filter(|v| v.is_finite() && *v >= 0.0).ok_or("invalid accumulator")?;
    }
    node.visits = visits;
    Ok((crate::parity::hex_key(text), node))
}

/// Resume only the unchanged linear recipe. No partial iteration or RNG state is needed:
/// both independent root streams are derived from seed/iteration/seat/sample.
pub fn load(path: &Path, expected: Game, completed_nodes: Option<u64>, max_entries: u64) -> Result<Trainer, String> {
    let source = BufReader::new(GzDecoder::new(std::fs::File::open(path).map_err(|e| e.to_string())?));
    let mut lines = source.lines();
    let h: Value = serde_json::from_str(&lines.next().ok_or("empty checkpoint")?.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if header_game(&h)? != expected { return Err("resume game differs from --stack-bb".into()); }
    if h.get("training_options").is_some() { return Err("resume supports linear CFR without training options only".into()); }
    if h["table"]["button"] != 0 || h["table"]["player_ids"] != json!(["player-0", "player-1"]) {
        return Err("resume requires native root table".into());
    }
    let nodes = if let Some(state) = h.get("native_state") {
        if state["version"] != 1 { return Err("unknown native recovery version".into()); }
        let n = state["completed_nodes"].as_u64().ok_or("invalid completed nodes")?;
        if completed_nodes.map_or(false, |provided| provided != n) { return Err("completed nodes differ from checkpoint".into()); }
        n
    } else { completed_nodes.ok_or("legacy checkpoint requires manifest-verified --completed-nodes")? };
    let mut trainer = Trainer::new(h["config"]["seed"].as_u64().unwrap(), h["config"]["roots_per_seat"].as_u64().unwrap() as usize);
    trainer.game = expected;
    trainer.iteration = h["iteration"].as_u64().unwrap();
    trainer.nodes = nodes;
    if let Some(state) = h.get("native_state") {
        let baseline = state["coverage_start"].as_array().ok_or("missing coverage baseline")?;
        if baseline.len() != 3 { return Err("invalid coverage baseline".into()); }
        for i in 0..3 { trainer.coverage_start[i] = baseline[i].as_u64().ok_or("invalid coverage baseline")?; }
        let counts = |name: &str| -> Result<[u64; 4], String> {
            let values = state[name].as_array().ok_or("missing coverage counters")?;
            if values.len() != 4 { return Err("invalid coverage counters".into()); }
            values.iter().map(|v| v.as_u64().ok_or("invalid coverage count".to_string())).collect::<Result<Vec<_>, _>>()?
                .try_into().map_err(|_| "invalid coverage counters".into())
        };
        trainer.decisions_by_street = counts("decisions_by_street")?;
        trainer.traverser_visits_by_street = counts("traverser_visits_by_street")?;
    }
    if (trainer.iteration == 0) != (nodes == 0) { return Err("inconsistent iteration/node counters".into()); }
    trainer.average = AverageRule::parse(h.get("average_rule").and_then(Value::as_str).unwrap_or("traverser-reach"));
    for line in lines {
        let row: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let (key, node) = row_node(&row)?;
        if trainer.table.0[Sharded::shard(&key)].insert(key, node).is_some() { return Err("duplicate checkpoint key".into()); }
        if trainer.table.len() as u64 > max_entries.min(h["config"]["max_entries"].as_u64().unwrap()) {
            return Err("checkpoint exceeds entry cap".into());
        }
    }
    let total_visits = trainer.table.iter().try_fold(0u64, |sum, (_, node)| sum.checked_add(node.visits)).ok_or("visit counter overflow")?;
    if h.get("native_state").is_none() {
        trainer.coverage_start = [trainer.iteration, trainer.nodes, total_visits];
    }
    let decisions = trainer.decisions_by_street.iter().try_fold(0u64, |sum, &n| sum.checked_add(n)).ok_or("coverage overflow")?;
    let visits = trainer.traverser_visits_by_street.iter().try_fold(0u64, |sum, &n| sum.checked_add(n)).ok_or("coverage overflow")?;
    if trainer.coverage_start[0] > trainer.iteration || trainer.coverage_start[1] > trainer.nodes
        || decisions > trainer.nodes - trainer.coverage_start[1]
        || trainer.coverage_start[2].checked_add(visits) != Some(total_visits)
        || trainer.traverser_visits_by_street.iter().zip(trainer.decisions_by_street).any(|(&v, d)| v > d)
    { return Err("inconsistent native coverage counters".into()); }
    Ok(trainer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::{Forced, Lookup};
    use crate::game::{Action, Hand, Kind};

    fn river(game: Game) -> Hand {
        let mut hand = Hand::new_for(game, 0, [[48, 49], [44, 45]], [0, 5, 10, 15, 20]);
        hand.apply_mut(Action {kind: Kind::Raise, raise_to: game.stack() - 100});
        for kind in [Kind::Call, Kind::Check, Kind::Check, Kind::Check, Kind::Check] {
            hand.apply_mut(Action {kind, raise_to: 0});
        }
        // A legal river fixture with only one BB behind each player.
        hand
    }

    #[test]
    fn both_games_resume_exactly_and_reject_cross_game() {
        for game in [Game::Hu20, Game::Hu100] {
            let mut trainer = Trainer::new(123, 1);
            trainer.game = game;
            trainer.average = AverageRule::OpponentSampled;
            let step = |t: &mut Trainer| { t.step_with(|_, _, _| (river(game), Forced([0usize; 32].iter()))); };
            step(&mut trainer);
            let path = std::env::temp_dir().join(format!("hu-resume-{}-{}.gz", std::process::id(), game.stack()));
            trainer.save_recoverable(&path, 1000, 1000).unwrap();
            let other = if game == Game::Hu20 { Game::Hu100 } else { Game::Hu20 };
            assert!(load(&path, other, None, 1000).is_err());
            assert!(load(&path, game, Some(trainer.nodes + 1), 1000).is_err());
            let mut resumed = load(&path, game, None, 1000).unwrap();
            assert_eq!(resumed.coverage_start, [0; 3]);
            // Legacy checkpoints can recover only with independently retained completed nodes.
            // Their subsequent telemetry starts here, rather than claiming past street coverage.
            trainer.save(&path, 1000, 1000).unwrap();
            if game == Game::Hu20 {
                assert!(load(&path, game, None, 1000).is_err());
                let mut legacy = load(&path, game, Some(trainer.nodes), 1000).unwrap();
                assert_eq!(legacy.coverage_start[0], trainer.iteration);
                assert_eq!(legacy.coverage_start[1], trainer.nodes);
                step(&mut legacy);
                legacy.save_recoverable(&path, 1000, 1000).unwrap();
                assert!(load(&path, game, None, 1000).is_ok());
            }
            std::fs::remove_file(path).unwrap();
            step(&mut trainer); step(&mut resumed);
            assert_eq!(trainer.nodes, resumed.nodes);
            assert_eq!(trainer.iteration, resumed.iteration);
            assert_eq!(trainer.table.len(), resumed.table.len());
            for (key, node) in trainer.table.iter() {
                let other = resumed.table.lookup(key).unwrap();
                assert_eq!(node.regrets.map(f64::to_bits), other.regrets.map(f64::to_bits));
                assert_eq!(node.average.map(f64::to_bits), other.average.map(f64::to_bits));
                assert_eq!(node.visits, other.visits);
            }
        }
    }

    #[test]
    fn malformed_rows_are_rejected() {
        for row in [json!(["z".repeat(32), ["check"], [0], [0], 0]),
                    json!(["0".repeat(32), ["check", "check"], [0, 0], [0, 0], 0]),
                    json!(["0".repeat(32), ["call"], [0], [-1], 0])] {
            assert!(row_node(&row).is_err());
        }
    }
}
