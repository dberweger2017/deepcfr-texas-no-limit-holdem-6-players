//! Replay Python-engine fixtures (`scripts/native_parity_fixtures.py`) and
//! compare every decision and the settlement field by field.

use crate::game::{Action, Game, Hand, Kind};
use crate::key::{key_hex, menu};
use hu20_buckets::parse_card;
use serde_json::Value;

fn kind(name: &str) -> Kind {
    match name {
        "fold" => Kind::Fold,
        "check" => Kind::Check,
        "call" => Kind::Call,
        "raise" => Kind::Raise,
        other => panic!("unknown action kind {other}"),
    }
}

/// Returns a description of the first mismatch in one recorded hand, if any.
pub fn check_hand(record: &Value, cards: crate::cards::Cards) -> Option<String> {
    let deck: Vec<u8> = record["deck"].as_array().unwrap().iter().map(|c| parse_card(c.as_str().unwrap())).collect();
    let mut hand = Hand::from_deck_for(Game::from_bb(record["stack_bb"].as_u64().unwrap_or(20) as u32), record["button"].as_u64().unwrap() as u8, &deck);
    hand.core.cards = cards;
    let decisions = record["decisions"].as_array().unwrap();
    let actions = record["actions"].as_array().unwrap();
    for (index, (decision, action)) in decisions.iter().zip(actions).enumerate() {
        let at = |field: &str, mine: String, theirs: String| {
            Some(format!("seed {} decision {index} {field}: native {mine} python {theirs}", record["seed"]))
        };
        let actor = hand.actor.map(|a| a as u64);
        if actor != decision["actor"].as_u64() {
            return at("actor", format!("{actor:?}"), decision["actor"].to_string());
        }
        if hand.street.name() != decision["street"].as_str().unwrap() {
            return at("street", hand.street.name().into(), decision["street"].to_string());
        }
        if hand.pot() as u64 != decision["pot"].as_u64().unwrap() {
            return at("pot", hand.pot().to_string(), decision["pot"].to_string());
        }
        let legal = hand.legal();
        let mut kinds = vec!["fold"];
        kinds.push(if legal.check { "check" } else { "call" });
        if legal.raise {
            kinds.push("raise");
        }
        let theirs: Vec<&str> = decision["kinds"].as_array().unwrap().iter().map(|k| k.as_str().unwrap()).collect();
        if kinds != theirs {
            return at("kinds", format!("{kinds:?}"), format!("{theirs:?}"));
        }
        if legal.call_amount as u64 != decision["call"].as_u64().unwrap() {
            return at("call", legal.call_amount.to_string(), decision["call"].to_string());
        }
        if legal.raise
            && (Some(legal.min_raise_to as u64) != decision["min_raise_to"].as_u64()
                || Some(legal.max_raise_to as u64) != decision["max_raise_to"].as_u64())
        {
            return at("raise bounds", format!("{}..{}", legal.min_raise_to, legal.max_raise_to),
                      format!("{}..{}", decision["min_raise_to"], decision["max_raise_to"]));
        }
        let options = menu(&hand);
        let choices = options.as_slice();
        let mine: Vec<String> = choices.iter().map(|c| format!("{}:{}", c.name, c.action.raise_to)).collect();
        let theirs: Vec<String> = decision["menu"].as_array().unwrap().iter()
            .map(|c| format!("{}:{}", c[0].as_str().unwrap(), c[1].as_u64().unwrap_or(0))).collect();
        if mine != theirs {
            return at("menu", format!("{mine:?}"), format!("{theirs:?}"));
        }
        let key = key_hex(&hand, choices);
        if key != decision["key"].as_str().unwrap() {
            return at("key", format!("{key} {}", crate::key::payload(&hand, choices)), decision["key"].to_string());
        }
        let chosen = Action { kind: kind(action[0].as_str().unwrap()), raise_to: action[1].as_u64().unwrap_or(0) as u32 };
        hand.apply_mut(chosen);
    }
    if !hand.finished() {
        return Some(format!("seed {}: native hand still live after the recorded actions", record["seed"]));
    }
    let stacks = hand.final_stacks();
    let theirs: Vec<u64> = record["final_stacks"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    if stacks.iter().map(|&s| s as u64).collect::<Vec<_>>() != theirs {
        return Some(format!("seed {}: final stacks native {stacks:?} python {theirs:?}", record["seed"]));
    }
    None
}

/// Static menu names, so fixture nodes compare with traversal nodes.
pub fn static_name(name: &str) -> &'static str {
    match name {
        "fold" => "fold",
        "check" => "check",
        "call" => "call",
        "min" => "min",
        "pot" => "pot",
        "jam" => "jam",
        other => panic!("unknown menu name {other}"),
    }
}

pub fn hex_key(text: &str) -> crate::cfr::Key {
    let mut key = [0u8; 16];
    for i in 0..16 {
        key[i] = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap();
    }
    key
}

pub fn node_from(row: &Value) -> crate::cfr::Node {
    let names: Vec<&str> = row[0].as_array().unwrap().iter().map(|n| static_name(n.as_str().unwrap())).collect();
    let mut node = crate::cfr::Node::empty(crate::key::code_of(&names), names.len());
    for (i, x) in row[1].as_array().unwrap().iter().enumerate() {
        node.regrets[i] = x.as_f64().unwrap();
    }
    for (i, x) in row[2].as_array().unwrap().iter().enumerate() {
        node.average[i] = x.as_f64().unwrap();
    }
    node.visits = row[3].as_u64().unwrap();
    node
}

/// Compare native traversal deltas with recorded Python `_collect_root` deltas, bit for bit.
pub fn check_traversals(document: &Value) -> (usize, Vec<String>) {
    use crate::cfr::{Forced, Traversal};
    let table: crate::cfr::Table = document["nodes"].as_object().unwrap().iter()
        .map(|(k, row)| (hex_key(k), node_from(row))).collect();
    let button = document["table_button"].as_u64().unwrap() as u8;
    let mut problems = Vec::new();
    let cases = document["cases"].as_array().unwrap();
    for (index, case) in cases.iter().enumerate() {
        let deck: Vec<u8> = case["deck"].as_array().unwrap().iter().map(|c| parse_card(c.as_str().unwrap())).collect();
        let draws: Vec<usize> = case["draws"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
        let mut traversal = Traversal::new(&table, case["iteration"].as_u64().unwrap(), case["seat"].as_u64().unwrap() as usize,
                                           Forced(draws.iter()));
        traversal.run(&mut Hand::from_deck(button, &deck));
        let expected: std::collections::HashMap<_, _> = case["deltas"].as_object().unwrap().iter()
            .map(|(k, row)| (hex_key(k), node_from(row))).collect();
        let mut problem = None;
        if traversal.nodes != case["nodes"].as_u64().unwrap() || traversal.terminals != case["terminals"].as_u64().unwrap() {
            problem = Some(format!("nodes {}/{} terminals {}/{}", traversal.nodes, case["nodes"], traversal.terminals, case["terminals"]));
        } else if expected.len() != traversal.deltas.len() {
            problem = Some(format!("delta keys {} vs {}", traversal.deltas.len(), expected.len()));
        } else {
            for (key, theirs) in &expected {
                match traversal.deltas.get(key) {
                    None => { problem = Some("missing delta key".into()); break; }
                    Some(mine) => {
                        let same = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
                        if mine.code != theirs.code || mine.len != theirs.len || mine.visits != theirs.visits
                            || !same(&mine.regrets, &theirs.regrets) || !same(&mine.average, &theirs.average) {
                            problem = Some(format!("delta differs: native {mine:?} python {theirs:?}"));
                            break;
                        }
                    }
                }
            }
        }
        if let Some(p) = problem {
            problems.push(format!("case {index}: {p}"));
        }
    }
    (cases.len(), problems)
}

/// Replay a recorded Python training run (decks and opponent draws per root) and compare every
/// iteration's node count, then the final table, bit for bit.
pub fn check_run(document: &Value) -> Result<(usize, usize), String> {
    use crate::cfr::Forced;
    use crate::trainer::Trainer;
    assert_eq!(document["table_button"].as_u64(), Some(0), "native training uses button 0");
    let iterations = document["iterations"].as_array().unwrap();
    let roots: Vec<Vec<(Vec<u8>, Vec<usize>)>> = iterations.iter().map(|it| {
        it["roots"].as_array().unwrap().iter().map(|root| {
            let deck = root["deck"].as_array().unwrap().iter().map(|c| parse_card(c.as_str().unwrap())).collect();
            let draws = root["draws"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap() as usize).collect();
            (deck, draws)
        }).collect()
    }).collect();
    let mut trainer = Trainer::new(0, 1);
    for (index, recorded) in iterations.iter().enumerate() {
        let nodes = trainer.step_with(|iteration, seat, _sample| {
            let (deck, draws) = &roots[iteration as usize - 1][seat];
            (Hand::from_deck(0, deck), Forced(draws.iter()))
        });
        let expected = recorded["nodes"].as_u64().unwrap();
        if nodes != expected {
            return Err(format!("iteration {}: native {nodes} nodes, python {expected}", index + 1));
        }
    }
    let expected: std::collections::HashMap<_, _> = document["nodes"].as_object().unwrap().iter()
        .map(|(k, row)| (hex_key(k), node_from(row))).collect();
    if expected.len() != trainer.table.len() {
        return Err(format!("table has {} keys, python {}", trainer.table.len(), expected.len()));
    }
    let same = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let mut differing = 0;
    for (key, theirs) in &expected {
        match trainer.table.get(key) {
            None => return Err("a python key is missing natively".into()),
            Some(mine) => {
                if mine.code != theirs.code || mine.visits != theirs.visits
                    || !same(&mine.regrets, &theirs.regrets) || !same(&mine.average, &theirs.average) {
                    differing += 1;
                }
            }
        }
    }
    if differing > 0 {
        return Err(format!("{differing} of {} entries differ", expected.len()));
    }
    Ok((iterations.len(), expected.len()))
}
