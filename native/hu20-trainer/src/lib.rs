//! Native HU20 trainer: game rules, v1 keys and external-sampling CFR with
//! exact parity to the Python trainer.

pub mod bench;
pub mod cards;
pub mod cfr;
pub mod export;
pub mod game;
pub mod nplayer;
pub mod multi_cfr;
pub mod small_games;
pub mod small_exact;
pub mod multi_checkpoint;
pub mod multi_holdem;
#[cfg(test)]
mod multi_tests;
pub mod key;
pub mod parity;
pub mod store;
pub mod streams;
pub mod trainer;

pub mod checkpoint;
pub mod telemetry;
