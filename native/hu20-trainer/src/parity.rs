//! Replay Python-engine fixtures (`scripts/native_parity_fixtures.py`) and
//! compare every decision and the settlement field by field.

use crate::game::{Action, Hand, Kind};
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
pub fn check_hand(record: &Value) -> Option<String> {
    let deck: Vec<u8> = record["deck"].as_array().unwrap().iter().map(|c| parse_card(c.as_str().unwrap())).collect();
    let mut hand = Hand::from_deck(record["button"].as_u64().unwrap() as u8, &deck);
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
        let choices = menu(&hand);
        let mine: Vec<String> = choices.iter().map(|c| format!("{}:{}", c.name, c.action.raise_to)).collect();
        let theirs: Vec<String> = decision["menu"].as_array().unwrap().iter()
            .map(|c| format!("{}:{}", c[0].as_str().unwrap(), c[1].as_u64().unwrap_or(0))).collect();
        if mine != theirs {
            return at("menu", format!("{mine:?}"), format!("{theirs:?}"));
        }
        let key = key_hex(&hand, &choices);
        if key != decision["key"].as_str().unwrap() {
            return at("key", format!("{key} {}", crate::key::payload(&hand, &choices)), decision["key"].to_string());
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
