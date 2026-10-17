//! Bounded, three-lookahead merge of hash-pinned raw HU200 averages. Inference
//! normalization remains exclusively in the existing checkpoint exporter.
// Multi-member: `train --fast-gzip` writes each piece as its own gzip member.
use flate2::{read::MultiGzDecoder, write::GzEncoder, Compression};
use hu20_trainer::{
    cards,
    cfr::{fsum, Node},
    checkpoint,
    export::sha256_file,
    game::Game,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

const RULE: &str =
    "fsum(raw Node.average_s[action] / (T_s*(T_s+1)/2)); absent keys contribute zero";
const SAMPLE_CAP: u64 = 100_000;

fn weight(t: u64) -> u64 {
    let w = t as u128 * (t as u128 + 1) / 2;
    assert!(
        t > 0 && w <= (1u128 << 53),
        "iteration weight must be exact in f64"
    );
    w as u64
}

struct Row {
    key: [u8; 16],
    text: String,
    node: Node,
}
struct Input {
    spec: Value,
    path: PathBuf,
    header: Value,
    lines: std::io::Lines<BufReader<MultiGzDecoder<File>>>,
    next: Option<Row>,
    previous: Option<[u8; 16]>,
    count: u64,
    action_slots: u64,
    keys: Sha256,
    roots: BTreeSet<String>,
    root_mass: BTreeMap<String, f64>,
    weight: u64,
}

impl Input {
    fn open(spec: &Value, roots: &BTreeSet<String>, input_dir: Option<&Path>) -> Self {
        // Keep the original spec in the output header: rebasing provenance paths
        // would change the frozen model bytes. Only the input file lookup moves.
        let path = input_dir.map_or_else(
            || PathBuf::from(spec["path"].as_str().expect("checkpoint path")),
            |directory| {
                directory.join(format!(
                    "HU200-frozen-20B-seed-{}.checkpoint.gz",
                    spec["seed"].as_u64().unwrap()
                ))
            },
        );
        assert_eq!(
            sha256_file(&path),
            spec["sha256"].as_str().unwrap(),
            "checkpoint SHA256"
        );
        let mut lines = BufReader::new(MultiGzDecoder::new(File::open(&path).unwrap())).lines();
        let header: Value = serde_json::from_str(&lines.next().expect("header").unwrap()).unwrap();
        assert_eq!(checkpoint::header_game(&header).unwrap(), Game::Hu200);
        assert_eq!(header["abstraction"], cards::equity_schema(Game::Hu200));
        assert_eq!(header["identity"]["action_menu"], Game::Hu200.menu());
        for (street, sha) in ["flop", "turn", "river"]
            .into_iter()
            .zip(cards::EQUITY_K50_SHA256)
        {
            assert_eq!(header["identity"]["card_tables"][street], sha);
        }
        assert_eq!(header["average_rule"], "opponent-sampled");
        assert_eq!(header["config"]["roots_per_seat"], 1);
        assert!(header.get("training_options").is_none(), "linear CFR only");
        assert_eq!(header["config"]["seed"], spec["seed"]);
        assert_eq!(header["iteration"], spec["iterations"]);
        let w = weight(spec["iterations"].as_u64().unwrap());
        assert_eq!(spec["weight"].as_u64().unwrap(), w, "predeclared weight");
        let mut input = Self {
            spec: spec.clone(),
            path,
            header,
            lines,
            next: None,
            previous: None,
            count: 0,
            action_slots: 0,
            keys: Sha256::new(),
            roots: roots.clone(),
            root_mass: BTreeMap::new(),
            weight: w,
        };
        input.advance();
        input
    }

    fn advance(&mut self) {
        self.next = self.lines.next().map(|line| {
            let row: Value = serde_json::from_str(&line.unwrap()).unwrap();
            let (key, node) = checkpoint::row_node(&row).expect("valid raw checkpoint row");
            assert!(
                self.previous.map_or(true, |p| p < key),
                "keys must strictly increase"
            );
            self.previous = Some(key);
            self.count += 1;
            self.action_slots += node.len as u64;
            assert!(
                self.count <= self.spec["entries"].as_u64().unwrap(),
                "input entry cap"
            );
            let text = row[0].as_str().unwrap().to_string();
            self.keys.update(text.as_bytes());
            if self.roots.contains(&text) {
                self.root_mass.insert(
                    text.clone(),
                    fsum(node.average[..node.len as usize].iter().copied()),
                );
            }
            Row { key, text, node }
        });
        if self.next.is_none() {
            assert_eq!(
                self.count,
                self.spec["entries"].as_u64().unwrap(),
                "input row count"
            );
        }
    }

    fn summary(&self) -> Value {
        assert_eq!(
            sha256_file(&self.path),
            self.spec["sha256"].as_str().unwrap(),
            "input changed while streaming"
        );
        let mass = fsum(self.root_mass.values().copied());
        assert!(mass.is_finite(), "nonfinite root mass");
        json!({"spec":self.spec,"entries":self.count,"action_slots":self.action_slots,
            "sorted_key_sha256":format!("{:x}",self.keys.clone().finalize()),
            "iteration_weight":self.weight,"root_mass":mass,
            "root_mass_minus_weight":mass-self.weight as f64,"root_masses":self.root_mass,
            "missing_root_keys":self.roots.iter().filter(|k| !self.root_mass.contains_key(*k)).collect::<Vec<_>>()})
    }
}

fn combined(rows: &[(&Node, u64)]) -> Node {
    assert!(!rows.is_empty());
    let first = rows[0].0;
    assert!(
        rows.iter()
            .all(|(r, _)| r.code == first.code && r.len == first.len),
        "same key changed its menu"
    );
    // Regrets and traverser visits deliberately carry no training interpretation.
    let mut output = Node::empty(first.code, first.len as usize);
    for action in 0..first.len as usize {
        output.average[action] = fsum(rows.iter().map(|(r, w)| {
            let raw = r.average[action];
            let scaled = raw / *w as f64;
            assert!(raw == 0.0 || scaled > 0.0, "positive accumulator underflow");
            scaled
        }));
        assert!(output.average[action].is_finite() && output.average[action] >= 0.0);
    }
    output
}

fn encoded(key: &str, node: &Node) -> Value {
    let n = node.len as usize;
    json!([key, node.names(), &node.regrets[..n], &node.average[..n], 0])
}

fn header(config: &Value, source: &Value, entries: u64) -> Value {
    let seed = config["blend_seed"]
        .as_u64()
        .expect("distinct synthetic identity");
    assert!(config["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .all(|s| s["seed"] != seed));
    let mut h = source.clone();
    for name in ["native_state", "frozen_table", "training_options"] {
        h.as_object_mut().unwrap().remove(name);
    }
    h["iteration"] = json!(1); // A format-required marker, not a training iteration count.
    h["config"]["seed"] = json!(seed);
    h["config"]["max_entries"] = json!(entries);
    h["config"]["max_nodes"] = json!(1);
    h["blend_provenance"] = json!({"version":1,"kind":"synthetic-sequence-form-average",
        "model_identity":format!("hu200-v055-sequence-blend-{seed}"),
        "rule":RULE,"inputs":config["inputs"],"regrets":0,"visits":0,
        "iteration_field":"synthetic marker 1; no training or resume claim"});
    checkpoint::header_game(&h).unwrap();
    h
}

fn fresh(path: &Path) -> BufWriter<File> {
    BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap(),
    )
}

fn write_json(path: &Path, value: &Value) {
    let mut f = fresh(path);
    serde_json::to_writer_pretty(&mut f, value).unwrap();
    writeln!(f).unwrap();
}

fn run(
    mode: &str,
    config_path: &Path,
    report_path: &Path,
    output_path: &Path,
    gate: Option<&Path>,
    input_dir: Option<&Path>,
) {
    let started = Instant::now();
    let config: Value = serde_json::from_reader(File::open(config_path).unwrap()).unwrap();
    let specs = config["inputs"].as_array().expect("three input specs");
    assert_eq!(specs.len(), 3);
    assert_eq!(
        specs
            .iter()
            .map(|s| s["seed"].as_u64().unwrap())
            .collect::<BTreeSet<_>>()
            .len(),
        3
    );
    let roots = config["root_keys"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect::<BTreeSet<_>>();
    assert_eq!(roots.len(), 169, "predeclare every initial root class");
    let (expected, cap) = if mode == "merge" {
        let gate_path = gate.expect("merge requires a capacity admission receipt");
        let g: Value = serde_json::from_reader(File::open(gate_path).unwrap()).unwrap();
        assert_eq!(g["status"], "admitted");
        assert_eq!(g["all_seed_averages_reproduced"], true);
        assert_eq!(g["configuration_sha256"], sha256_file(config_path));
        let scan_path = PathBuf::from(g["scan_report"].as_str().unwrap());
        assert_eq!(g["scan_sha256"], sha256_file(&scan_path));
        let scan: Value = serde_json::from_reader(File::open(scan_path).unwrap()).unwrap();
        assert_eq!(scan["configuration_sha256"], sha256_file(config_path));
        let n = scan["union_keys"].as_u64().unwrap();
        (Some(scan), n)
    } else {
        assert_eq!(mode, "scan");
        (None, SAMPLE_CAP)
    };
    assert!(
        !report_path.exists() && !output_path.exists(),
        "preserve previous outputs"
    );
    let mut inputs: Vec<_> = specs
        .iter()
        .map(|s| Input::open(s, &roots, input_dir))
        .collect();
    assert!(inputs
        .iter()
        .all(|s| s.header["identity"] == inputs[0].header["identity"]));
    let partial = output_path.with_extension("partial");
    let mut output = GzEncoder::new(fresh(&partial), Compression::default());
    writeln!(output, "{}", header(&config, &inputs[0].header, cap)).unwrap();
    let (mut union, mut slots, mut emitted) = (0u64, 0u64, 0u64);
    let mut membership = [0u64; 8];
    let mut menus = BTreeMap::<u16, u64>::new();
    while let Some(key) = inputs
        .iter()
        .filter_map(|s| s.next.as_ref().map(|r| r.key))
        .min()
    {
        let ids: Vec<_> = (0..3)
            .filter(|&i| inputs[i].next.as_ref().map_or(false, |r| r.key == key))
            .collect();
        let rows: Vec<_> = ids
            .iter()
            .map(|&i| (&inputs[i].next.as_ref().unwrap().node, inputs[i].weight))
            .collect();
        let code = rows[0].0.code;
        let n = rows[0].0.len;
        assert!(
            rows.iter().all(|(r, _)| r.code == code && r.len == n),
            "same key changed its menu"
        );
        let mask: usize = ids.iter().map(|i| 1 << i).sum();
        membership[mask] += 1;
        union += 1;
        slots += n as u64;
        *menus.entry(code).or_default() += 1;
        // A fixed key-based sample measures compression; it is never played.
        if mode == "merge" || (key[0] == 0 && key[1] < 16 && emitted < SAMPLE_CAP) {
            let node = combined(&rows);
            writeln!(
                output,
                "{}",
                encoded(&inputs[ids[0]].next.as_ref().unwrap().text, &node)
            )
            .unwrap();
            emitted += 1;
        }
        for i in ids {
            inputs[i].advance();
        }
    }
    let summaries: Vec<_> = inputs.iter().map(Input::summary).collect();
    assert_eq!(membership.iter().sum::<u64>(), union);
    assert_eq!(
        membership
            .iter()
            .enumerate()
            .map(|(mask, n)| mask.count_ones() as u64 * n)
            .sum::<u64>(),
        inputs.iter().map(|input| input.count).sum::<u64>()
    );
    let overlaps = (1..=3)
        .map(|n| {
            membership
                .iter()
                .enumerate()
                .filter(|(mask, _)| mask.count_ones() == n)
                .map(|(_, v)| v)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let stable = json!({"union_keys":union,"action_slots":slots,"keys_in_exactly_1_2_3_seeds":overlaps,
        "membership_masks":membership,"menus":menus.iter().map(|(code,n)|json!({"code":code,"names":hu20_trainer::key::names_of(*code),"keys":n})).collect::<Vec<_>>(),"inputs":summaries});
    if let Some(prior) = expected {
        for name in [
            "union_keys",
            "action_slots",
            "keys_in_exactly_1_2_3_seeds",
            "membership_masks",
            "menus",
            "inputs",
        ] {
            assert_eq!(
                stable[name], prior[name],
                "input/count changes after union gate"
            );
        }
        assert_eq!(emitted, cap);
    }
    output.finish().unwrap().flush().unwrap();
    std::fs::rename(&partial, output_path).unwrap();
    let mut result = stable;
    result["configuration_sha256"] = json!(sha256_file(config_path));
    result["rule"] = json!(RULE);
    result["model_identity"] = json!(format!(
        "hu200-v055-sequence-blend-{}",
        config["blend_seed"].as_u64().unwrap()
    ));
    result["mode"] = json!(mode);
    result["output"] = json!({"path":output_path,"bytes":output_path.metadata().unwrap().len(),"sha256":sha256_file(output_path),"rows":emitted});
    result["cache_array_bytes"] = json!(35 * union + 8 * slots + 6 * 128);
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    write_json(report_path, &result);
}

fn main() {
    let mut a: Vec<_> = std::env::args().collect();
    let input_dir = a
        .iter()
        .position(|argument| argument == "--input-dir")
        .map(|index| {
            assert_eq!(index + 2, a.len(), "--input-dir DIR must be last");
            let directory = PathBuf::from(a.pop().unwrap());
            a.pop();
            directory
        });
    assert!(
        a.len() == 5 || a.len() == 6,
        "hu200_blend scan|merge CONFIG NEW_REPORT NEW_CHECKPOINT [CAPACITY_GATE] [--input-dir DIR]"
    );
    run(
        &a[1],
        Path::new(&a[2]),
        Path::new(&a[3]),
        Path::new(&a[4]),
        a.get(5).map(Path::new),
        input_dir.as_deref(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn node(values: &[f64]) -> Node {
        let mut n = Node::empty(hu20_trainer::key::code_of(&["fold", "call"]), 2);
        n.average[..2].copy_from_slice(values);
        n
    }
    #[test]
    fn sequence_mass_is_preserved_instead_of_mixing_normalized_rows() {
        let a = node(&[90., 10.]);
        let b = node(&[0., 2.]);
        let out = combined(&[(&a, weight(4)), (&b, weight(1))]);
        assert_eq!(&out.average[..2], &[9., 3.]);
        assert_eq!(&out.regrets[..2], &[0., 0.]);
        assert_eq!(out.visits, 0);
        // Existing export must therefore produce 3/4, not the equal-row average 0.45.
        assert_eq!(
            out.average[0] / fsum(out.average[..2].iter().copied()),
            0.75
        );
    }
    #[test]
    fn absent_seeds_add_no_mass_and_zero_rows_stay_zero() {
        let a = node(&[0., 0.]);
        let b = node(&[2., 4.]);
        assert_eq!(&combined(&[(&a, 3)]).average[..2], &[0., 0.]);
        assert_eq!(&combined(&[(&b, 2)]).average[..2], &[1., 2.]);
        assert_eq!(&combined(&[(&b, 2), (&a, 3)]).average[..2], &[1., 2.]);
        assert_eq!(weight(16_670_844), 138_958_528_171_590);
        assert_eq!(weight(16_745_107), 140_199_312_593_278);
        assert_eq!(weight(16_827_728), 141_586_223_234_856);
    }
    #[test]
    #[should_panic(expected = "same key changed its menu")]
    fn key_collisions_with_different_menus_fail() {
        let a = node(&[1., 1.]);
        let b = Node::empty(hu20_trainer::key::code_of(&["check", "min"]), 2);
        combined(&[(&a, 1), (&b, 1)]);
    }

    #[test]
    #[should_panic(expected = "positive accumulator underflow")]
    fn numerical_underflow_is_reported_instead_of_becoming_zero_mass() {
        combined(&[(&node(&[f64::from_bits(1), 1.]), 3)]);
    }

    #[test]
    fn streamed_union_merge_and_existing_export_agree() {
        streamed_union_merge(false);
    }

    /// `train --fast-gzip` checkpoints hold the header and row pieces as separate gzip members.
    #[test]
    fn multi_member_checkpoints_blend_identically() {
        streamed_union_merge(true);
    }

    fn streamed_union_merge(multi_member: bool) {
        let directory = std::env::temp_dir().join(format!(
            "hu200-blend-test-{}-{multi_member}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let identity = checkpoint::identity(
            Game::Hu200,
            cards::EQUITY_DESCRIPTOR,
            Some(
                json!({"flop":cards::EQUITY_K50_SHA256[0],"turn":cards::EQUITY_K50_SHA256[1],"river":cards::EQUITY_K50_SHA256[2]}),
            ),
            hu20_trainer::key::MenuVersion::V1,
        );
        let groups = [
            vec![(1, [90., 10.]), (3, [0., 10.]), (5, [2., 0.])],
            vec![(2, [0., 0.]), (3, [2., 0.]), (5, [0., 3.])],
            vec![(3, [1., 1.]), (4, [0., 0.])],
        ];
        let mut specs = Vec::new();
        for (i, rows) in groups.iter().enumerate() {
            let seed = i as u64 + 1;
            let t = [4, 2, 1][i];
            let path = directory.join(format!("seed-{seed}.gz"));
            let h = json!({"format":Game::Hu200.format(),"abstraction":cards::equity_schema(Game::Hu200),
                "kind":"training","checkpoint_format":"jsonl-v2","average_rule":"opponent-sampled","iteration":t,
                "config":{"seed":seed,"roots_per_seat":1,"max_entries":rows.len(),"max_nodes":1,"max_seconds":900.,
                    "game":Game::Hu200.id(),"abstraction":cards::equity_schema(Game::Hu200),"raise_cap":null},
                "table":{"stacks":[20000,20000],"player_ids":["player-0","player-1"],"button":0,
                    "small_blind":50,"big_blind":100,"chip_unit":"0.01"},"identity":identity});
            let mut gz = GzEncoder::new(fresh(&path), Compression::default());
            writeln!(gz, "{h}").unwrap();
            if multi_member {
                gz = GzEncoder::new(gz.finish().unwrap(), Compression::default());
            }
            for (key, values) in rows {
                writeln!(gz, "{}", encoded(&format!("{key:032x}"), &node(values))).unwrap();
            }
            gz.finish().unwrap().flush().unwrap();
            specs.push(json!({"path":path,"sha256":sha256_file(&path),"seed":seed,"iterations":t,"weight":weight(t),"entries":rows.len()}));
        }
        let config = directory.join("config.json");
        write_json(
            &config,
            &json!({"blend_seed":99,"inputs":specs,"root_keys":(0..169).map(|i|format!("{i:032x}")).collect::<Vec<_>>()}),
        );
        let scan = directory.join("scan.json");
        run(
            "scan",
            &config,
            &scan,
            &directory.join("sample.gz"),
            None,
            None,
        );
        let statistics: Value = serde_json::from_reader(File::open(&scan).unwrap()).unwrap();
        assert_eq!(statistics["union_keys"], 5);
        assert_eq!(statistics["action_slots"], 10);
        assert_eq!(statistics["keys_in_exactly_1_2_3_seeds"], json!([3, 1, 1]));
        assert_eq!(statistics["cache_array_bytes"], 1023);
        let gate = directory.join("gate.json");
        write_json(
            &gate,
            &json!({"status":"admitted","configuration_sha256":sha256_file(&config),
            "all_seed_averages_reproduced":true,"scan_report":scan,"scan_sha256":sha256_file(&scan)}),
        );
        let merged = directory.join("blend.gz");
        run(
            "merge",
            &config,
            &directory.join("merged.json"),
            &merged,
            Some(&gate),
            None,
        );
        let relocated = directory.join("relocated");
        std::fs::create_dir(&relocated).unwrap();
        for seed in 1..=3 {
            std::fs::rename(
                directory.join(format!("seed-{seed}.gz")),
                relocated.join(format!("HU200-frozen-20B-seed-{seed}.checkpoint.gz")),
            )
            .unwrap();
        }
        let portable = directory.join("portable-blend.gz");
        run(
            "merge",
            &config,
            &directory.join("portable.json"),
            &portable,
            Some(&gate),
            Some(&relocated),
        );
        assert_eq!(
            sha256_file(&portable),
            sha256_file(&merged),
            "input relocation changes no compressed model bytes"
        );
        let average = directory.join("average.gz");
        assert_eq!(
            hu20_trainer::export::export(
                &merged,
                None,
                Some(&average),
                hu20_trainer::export::ZeroMass::Uniform
            )
            .unwrap(),
            5
        );
        let rows = BufReader::new(MultiGzDecoder::new(File::open(&average).unwrap()))
            .lines()
            .skip(1)
            .map(|s| serde_json::from_str::<Value>(&s.unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(rows[0][2], json!([0.9, 0.1]));
        assert_eq!(rows[1][2], json!([0.5, 0.5]));
        assert_eq!(rows[1][3], 0.0);
        let mass = fsum([2. / 3. + 1., 2.]);
        assert_eq!(rows[2][2], json!([(2. / 3. + 1.) / mass, 2. / mass]));
        assert_eq!(rows[3][2], json!([0.5, 0.5]));
        assert_eq!(rows[3][4], 0);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
