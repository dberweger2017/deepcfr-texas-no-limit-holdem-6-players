//! Measure the unchanged production HU trainer, including its retained scratch.
#[path = "support/memory.rs"] mod memory;
#[global_allocator] static METER: memory::Meter=memory::Meter;
use hu20_trainer::{cards::{Cards,EquityTables},cfr::AverageRule,game::Game,trainer::Trainer};
use serde_json::json;
use std::{path::Path,time::Instant};
fn main() {
    let a:Vec<String>=std::env::args().collect();
    assert_eq!(a.len(),8,"hu_size_reference STACK_BB SEED NODES MAX_ENTRIES MAX_SECONDS K50_DIRECTORY NEW_CHECKPOINT");
    let game=match a[1].as_str(){"20"=>Game::Hu20,"100"=>Game::Hu100,_=>panic!("unsupported depth")};
    let seed=a[2].parse().unwrap();let nodes:u64=a[3].parse().unwrap();let cap:usize=a[4].parse().unwrap();let seconds:f64=a[5].parse().unwrap();
    assert!(nodes>0 && cap>0 && seconds.is_finite() && seconds>0.);
    let out=Path::new(&a[7]);assert!(!out.exists());
    let tables=EquityTables::load(Path::new(&a[6]),false).unwrap();
    let baseline=memory::live();
    let mut trainer=Trainer::new(seed,1);trainer.game=game;trainer.cards=Cards::Equity(tables);trainer.average=AverageRule::OpponentSampled;
    println!("{}",json!({"kind":"production-hu","stack_bb":a[1],"seed":seed,"schema":trainer.schema(),"target_nodes":nodes,"baseline_requested_bytes":baseline}));
    memory::reset_peak();let start=Instant::now();let milestones=[100_000,1_000_000,2_000_000,5_000_000,10_000_000,20_000_000];let mut next=0;
    let reason=loop {
        if trainer.nodes>=nodes{break "nodes";}
        if trainer.table.len()>=cap{break "capacity";}
        if start.elapsed().as_secs_f64()>=seconds{break "time";}
        trainer.step();
        if (next<milestones.len() && trainer.nodes>=milestones[next]) || trainer.nodes>=nodes || trainer.table.len()>=cap {
            println!("{}",json!({"nodes":trainer.nodes,"iteration":trainer.iteration,"entries":trainer.table.len(),"retained_training_requested_bytes":memory::live()-baseline,"requested_peak_bytes":memory::peak(),"rss_bytes":hu20_trainer::telemetry::process_rss_bytes().unwrap(),"elapsed_seconds":start.elapsed().as_secs_f64()}));
            while next<milestones.len() && milestones[next]<=trainer.nodes {next+=1;}
        }
    };
    let train_seconds=start.elapsed().as_secs_f64();
    trainer.save_recoverable(out,nodes,cap as u64).unwrap();
    println!("{}",json!({"stop_reason":reason,"nodes":trainer.nodes,"entries":trainer.table.len(),"training_seconds":train_seconds,"save_seconds":start.elapsed().as_secs_f64()-train_seconds,"elapsed_seconds":start.elapsed().as_secs_f64(),"checkpoint_bytes":out.metadata().unwrap().len(),"checkpoint_sha256":hu20_trainer::export::sha256_file(out),"requested_peak_bytes":memory::peak()}));
    if reason!="nodes"{std::process::exit(3);}
}
