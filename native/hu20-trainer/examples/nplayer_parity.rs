//! Streaming test oracle: one engine-recorded hand per line, no training or policy access.
use hu20_trainer::game::Kind;
use hu20_trainer::nplayer::{Action, Config, Event, Hand};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

fn number(v: &Value) -> Result<u64, String> {
    v.as_u64()
        .ok_or_else(|| format!("expected unsigned integer, got {v}"))
}
fn array(v: &Value) -> Result<&Vec<Value>, String> {
    v.as_array().ok_or_else(|| "expected array".into())
}

fn snapshot(hand: &Hand) -> Value {
    let legal = hand
        .legal()
        .map(|l| json!({"check": l.check, "call": l.call_amount, "raise_to": l.raise_to}));
    json!({"actor": hand.actor(), "finished": hand.finished(), "board": hand.board(),
        "street": if hand.finished() { None } else { Some(hand.street().name()) },
        "pot": hand.pot(), "wager": if hand.finished() { None } else { Some(hand.wager()) },
        "min_raise": if hand.finished() { None } else { Some(hand.min_raise()) },
        "players": hand.players().iter().map(|p| json!([p.stack, p.street_bet, p.contributed, p.folded])).collect::<Vec<_>>(),
        "holes": (0..hand.config().players()).map(|s| hand.observe(s).unwrap().hole).collect::<Vec<_>>(),
        "legal": legal,
        "rewards": hand.settlement().map(|s| s.rewards[..hand.config().players()].to_vec())})
}

fn check(record: &Value) -> Result<usize, String> {
    let stacks = array(&record["stacks"])?
        .iter()
        .map(number)
        .collect::<Result<Vec<_>, _>>()?;
    let config = Config::new(&stacks, number(&record["sb"])?, number(&record["bb"])?)?;
    let deck = array(&record["deck"])?
        .iter()
        .map(|v| u8::try_from(number(v)?).map_err(|_| "card overflow".into()))
        .collect::<Result<Vec<_>, String>>()?;
    let button = usize::try_from(number(&record["button"])?).map_err(|_| "button overflow")?;
    let mut hand = Hand::from_deck(config, button, &deck)?;
    let states = array(&record["states"])?;
    let actions = array(&record["actions"])?;
    if states.len() != actions.len() + 1 {
        return Err("need initial and every post-action state".into());
    }
    for (i, expected) in states.iter().enumerate() {
        let actual = snapshot(&hand);
        if actual != *expected {
            return Err(format!("state {i}: native {actual}, engine {expected}"));
        }
        if i < actions.len() {
            let a = &actions[i];
            let kind = match a[0].as_str() {
                Some("fold") => Kind::Fold,
                Some("check") => Kind::Check,
                Some("call") => Kind::Call,
                Some("raise") => Kind::Raise,
                _ => return Err("invalid action kind".into()),
            };
            let action = Action {
                kind,
                raise_to: number(&a[1])?,
            };
            let before = hand.clone();
            let saved = hand.save();
            hand.apply_mut(action)?;
            let after = hand.clone();
            hand.restore(saved);
            if hand != before {
                return Err(format!("state {i}: save/restore changed parent"));
            }
            hand.apply_mut(action)?;
            if hand != after {
                return Err(format!("state {i}: replay changed child"));
            }
        }
    }
    if !hand.finished() {
        return Err("record did not settle".into());
    }
    let history = Value::Array(
        hand.history()
            .iter()
            .map(|event| match event {
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
                Event::Board(street) => json!([street.name(), "board"]),
            })
            .collect(),
    );
    if history != record["history"] {
        return Err(format!(
            "public history mismatch: native {history}, engine {}",
            record["history"]
        ));
    }
    let s = hand.settlement().unwrap();
    let actual = json!({"contributions": s.contributions[..stacks.len()], "refunds": s.refunds[..stacks.len()],
        "pots": s.pots().iter().map(|p| json!([p.amount, p.eligible])).collect::<Vec<_>>()});
    if actual != record["settlement"] {
        return Err(format!(
            "pot/refund mismatch: native {actual}, reference {}",
            record["settlement"]
        ));
    }
    Ok(actions.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let outcome = serde_json::from_str::<Value>(&line?)
            .map_err(|e| e.to_string())
            .and_then(|v| check(&v));
        match outcome {
            Ok(decisions) => writeln!(stdout, "{}", json!({"ok": true, "decisions": decisions}))?,
            Err(error) => {
                writeln!(stdout, "{}", json!({"ok": false, "error": error}))?;
                stdout.flush()?;
                std::process::exit(1);
            }
        }
        stdout.flush()?;
    }
    Ok(())
}
