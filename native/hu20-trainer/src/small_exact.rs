//! Exact, information-set-constrained best responses, separate from training.
use crate::multi_cfr::{Decision, State, Table};
use crate::small_games::{Hand, SmallGame};
use std::collections::HashMap;

struct Node {
    info: Option<usize>,
    children: Vec<usize>,
    rewards: [f64; 6],
}
pub struct Info {
    pub decision: Decision,
    pub example: Hand,
    nodes: Vec<usize>,
    depth: usize,
}
pub struct Tree {
    pub players: usize,
    pub infos: Vec<Info>,
    nodes: Vec<Node>,
    roots: Vec<(usize, f64)>,
    levels: Vec<Vec<usize>>,
}
#[derive(Debug)]
pub struct Evaluation {
    pub values: Vec<f64>,
    pub best_response: Vec<f64>,
    pub gains: Vec<f64>,
}
impl Tree {
    pub fn new(game: SmallGame, max_nodes: usize) -> Self {
        fn add(
            tree: &mut Tree,
            hand: Hand,
            depth: usize,
            index: &mut HashMap<[u8; 16], usize>,
            max_nodes: usize,
        ) -> usize {
            assert!(
                tree.nodes.len() < max_nodes,
                "exact tree node budget exceeded"
            );
            let node = tree.nodes.len();
            tree.nodes.push(Node {
                info: None,
                children: Vec::new(),
                rewards: [0.; 6],
            });
            if tree.levels.len() <= depth {
                tree.levels.push(Vec::new());
            }
            tree.levels[depth].push(node);
            if let Some(d) = hand.decision() {
                let info = *index.entry(d.key).or_insert_with(|| {
                    let id = tree.infos.len();
                    tree.infos.push(Info {
                        decision: d,
                        example: hand,
                        nodes: Vec::new(),
                        depth,
                    });
                    id
                });
                let saved = &tree.infos[info];
                assert_eq!(
                    saved.depth, depth,
                    "same information set at different depths"
                );
                assert_eq!(saved.decision, d);
                tree.infos[info].nodes.push(node);
                tree.nodes[node].info = Some(info);
                for a in 0..d.len() {
                    let mut child = hand;
                    child.apply(a);
                    let id = add(tree, child, depth + 1, index, max_nodes);
                    tree.nodes[node].children.push(id);
                }
            } else {
                for p in 0..tree.players {
                    tree.nodes[node].rewards[p] = hand.payoff(p);
                }
                assert!(tree.nodes[node].rewards.iter().sum::<f64>().abs() < 1e-12);
            }
            node
        }
        let mut tree = Self {
            players: game.players,
            infos: Vec::new(),
            nodes: Vec::new(),
            roots: Vec::new(),
            levels: Vec::new(),
        };
        let mut index = HashMap::new();
        for (hand, probability) in game.deals() {
            let id = add(&mut tree, hand, 0, &mut index, max_nodes);
            tree.roots.push((id, probability));
        }
        assert!((tree.roots.iter().map(|r| r.1).sum::<f64>() - 1.).abs() < 1e-12);
        tree
    }
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn profile(&self, table: &Table, average: bool) -> Vec<Vec<f64>> {
        self.infos
            .iter()
            .map(|info| {
                let n = info.decision.len();
                if let Some(entry) = table.get(&info.decision.key) {
                    entry.check(info.decision);
                    entry.policy(average)[..n].to_vec()
                } else {
                    vec![1. / n as f64; n]
                }
            })
            .collect()
    }
    fn validate(&self, profile: &[Vec<f64>]) {
        assert_eq!(profile.len(), self.infos.len());
        for (info, p) in self.infos.iter().zip(profile) {
            assert_eq!(p.len(), info.decision.len());
            assert!(p.iter().all(|x| x.is_finite() && *x >= 0.));
            assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-12);
        }
    }
    pub fn evaluate(&self, profile: &[Vec<f64>]) -> Evaluation {
        self.validate(profile);
        let mut values = Vec::new();
        let mut best_response = Vec::new();
        for player in 0..self.players {
            let mut value = vec![0.; self.nodes.len()];
            for (id, node) in self.nodes.iter().enumerate().rev() {
                value[id] = if let Some(i) = node.info {
                    node.children
                        .iter()
                        .zip(&profile[i])
                        .map(|(&c, p)| p * value[c])
                        .sum()
                } else {
                    node.rewards[player]
                };
            }
            values.push(self.roots.iter().map(|&(r, p)| p * value[r]).sum::<f64>());

            // Opponent+chance reach excludes this player's own policy. Maximizing
            // per hidden node instead would be a clairvoyant, invalid response.
            let mut reach = vec![0.; self.nodes.len()];
            for &(root, p) in &self.roots {
                reach[root] = p;
            }
            for (id, node) in self.nodes.iter().enumerate() {
                if let Some(i) = node.info {
                    for (a, &child) in node.children.iter().enumerate() {
                        reach[child] = reach[id]
                            * if self.infos[i].decision.actor == player {
                                1.
                            } else {
                                profile[i][a]
                            };
                    }
                }
            }
            let mut chosen = vec![None; self.infos.len()];
            for level in self.levels.iter().rev() {
                for &id in level {
                    let node = &self.nodes[id];
                    let Some(i) = node.info else {
                        value[id] = node.rewards[player];
                        continue;
                    };
                    if self.infos[i].decision.actor != player {
                        value[id] = node
                            .children
                            .iter()
                            .zip(&profile[i])
                            .map(|(&c, p)| p * value[c])
                            .sum();
                    } else {
                        let action = *chosen[i].get_or_insert_with(|| {
                            let scores: Vec<f64> = (0..node.children.len())
                                .map(|a| {
                                    self.infos[i]
                                        .nodes
                                        .iter()
                                        .map(|&h| reach[h] * value[self.nodes[h].children[a]])
                                        .sum()
                                })
                                .collect();
                            (0..scores.len())
                                .max_by(|&a, &b| {
                                    scores[a].total_cmp(&scores[b]).then_with(|| b.cmp(&a))
                                })
                                .unwrap()
                        });
                        value[id] = value[node.children[action]];
                    }
                }
            }
            best_response.push(self.roots.iter().map(|&(r, p)| p * value[r]).sum::<f64>());
        }
        let gains = best_response
            .iter()
            .zip(&values)
            .map(|(b, v)| b - v)
            .collect();
        Evaluation {
            values,
            best_response,
            gains,
        }
    }
}
