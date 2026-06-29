# src/training/train.py
import pokers as pkrs
import random
import torch
import time
import os
import argparse
import sys
from tqdm import tqdm
from src.core.deep_cfr import DeepCFRAgent, traverse_agent_turn
from src.core.model import set_verbose
from src.utils import settings
from src.utils.agents import CheckpointAgent
from src.utils.checkpoints import find_checkpoints, load_checkpoint, standard_checkpoint_state
from src.utils.logging import apply_action_with_logging
from src.utils.evaluation import evaluate_agent_matchup
from src.utils.settings import set_strict_checking
from src.agents.random_agent import RandomAgent

def evaluate_against_random(agent, num_games=500, num_players=6):
    """Evaluate the trained agent against random opponents."""
    random_agents = [RandomAgent(i) for i in range(num_players)]
    metrics = evaluate_agent_matchup(
        agent,
        random_agents,
        num_games=num_games,
        seed_start=0,
        num_players=num_players,
        strict=settings.is_strict_checking(),
        label="evaluation vs random",
        print_warnings=True,
    )
    if metrics["completed_games"] == 0:
        print("WARNING: No games completed during evaluation!")
    return metrics["avg_profit"]


class _PerspectiveAgentWrapper:
    """Ensure each opponent acts using its own player_id."""

    def __init__(self, agent):
        self.agent = agent
        self.player_id = agent.player_id

    def choose_action(self, state):
        return self.agent.choose_action(state)


def _cfr_traverse_with_opponents(agent, state, iteration, opponent_agents, depth=0, verbose=False):
    """Mirror DeepCFRAgent.cfr_traverse while sourcing opponent actions from fixed agents."""
    max_depth = 1000
    if depth > max_depth:
        if verbose:
            print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
        return 0

    if state.final_state:
        return state.players_state[agent.player_id].reward

    current_player = state.current_player

    if current_player == agent.player_id:
        return traverse_agent_turn(
            agent,
            state,
            iteration,
            lambda new_state, next_depth: _cfr_traverse_with_opponents(
                agent,
                new_state,
                iteration,
                opponent_agents,
                next_depth,
                verbose,
            ),
            depth=depth,
            verbose=verbose,
        )

    try:
        opponent_agent = opponent_agents[current_player]
        if opponent_agent is None:
            if verbose:
                print(f"WARNING: No opponent agent for position {current_player}")
            return 0

        action = opponent_agent.choose_action(state)
        new_state, log_file, status = apply_action_with_logging(
            state,
            action,
            strict=settings.is_strict_checking(),
        )
        if new_state is None:
            if verbose:
                print(
                    f"WARNING: Opponent agent made invalid action at depth {depth}. "
                    f"Status: {status}"
                )
                print(f"Details logged to {log_file}")
            return 0

        return _cfr_traverse_with_opponents(
            agent, new_state, iteration, opponent_agents, depth + 1, verbose
        )
    except Exception as exc:
        if verbose:
            print(f"ERROR in opponent agent traversal: {exc}")
        if settings.is_strict_checking():
            raise
        return 0

def train_deep_cfr(num_iterations=1000, traversals_per_iteration=200, 
                   num_players=6, player_id=0, save_dir="models", 
                   log_dir="logs/deepcfr", verbose=False, debug_training=False,
                   progress_interval=100):
    """
    Train a Deep CFR agent in a 6-player No-Limit Texas Hold'em game
    against 5 random opponents.
    """
    # Import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Initialize the agent
    agent = DeepCFRAgent(player_id=player_id, num_players=num_players, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    agent.debug_training = debug_training
    
    # Create random agents for the opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # ADDED: Initial evaluation before training begins
    print("Initial evaluation...")
    initial_profit = evaluate_against_random(agent, num_games=500, num_players=num_players)
    profits.append(initial_profit)
    print(f"Initial average profit per game: {initial_profit:.2f}")
    writer.add_scalar('Performance/Profit', initial_profit, 0)
    
    # Checkpoint frequency
    checkpoint_frequency = 100  # Save a checkpoint every 100 iterations
    
    # Training loop
    progress = tqdm(
        range(1, num_iterations + 1),
        desc="Training",
        unit="iter",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    last_strat_loss = None
    last_profit = initial_profit

    for iteration in progress:
        agent.iteration_count = iteration
        start_time = time.time()

        # Run traversals to collect data
        for _ in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players-1),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Perform CFR traversal
            agent.cfr_traverse(state, iteration, random_agents)
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        
        # Log the loss to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        
        # Every few iterations, train the strategy network and evaluate
        if iteration % 10 == 0 or iteration == num_iterations:
            last_strat_loss = agent.train_strategy_network()
            writer.add_scalar('Loss/Strategy', last_strat_loss, iteration)
            
            # Evaluate the agent
            last_profit = evaluate_against_random(agent, num_games=500, num_players=num_players)
            profits.append(last_profit)
            writer.add_scalar('Performance/Profit', last_profit, iteration)
            
            # Save the model
            #model_path = f"{save_dir}/deep_cfr_iter_{iteration}.pt"
            #agent.save_model(model_path)
            #print(f"  Model saved to {model_path}")
        
        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0 or iteration == num_iterations:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save(
                standard_checkpoint_state(agent, losses=losses, profits=profits),
                checkpoint_path,
            )
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        progress.set_postfix(
            adv=f"{adv_loss:.3f}",
            strat="-" if last_strat_loss is None else f"{last_strat_loss:.3f}",
            profit=f"{last_profit:.2f}",
            adv_mem=len(agent.advantage_memory),
            strat_mem=len(agent.strategy_memory),
        )

        should_report = (
            progress_interval > 0
            and (iteration % progress_interval == 0 or iteration == num_iterations)
        )
        if should_report:
            tqdm.write(
                f"iter {iteration}/{num_iterations} | "
                f"adv={adv_loss:.4f} | "
                f"strat={'-' if last_strat_loss is None else f'{last_strat_loss:.4f}'} | "
                f"profit={last_profit:.2f} | "
                f"adv_mem={len(agent.advantage_memory)} | "
                f"strat_mem={len(agent.strategy_memory)} | "
                f"{elapsed:.2f}s"
            )
        
        # Commit the tensorboard logs
        writer.flush()
    
    # Final evaluation
    print("Final evaluation...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, losses, profits

def continue_training(checkpoint_path, additional_iterations=1000, 
                     traversals_per_iteration=200, save_dir="models", 
                     log_dir="logs/deepcfr_continued", verbose=False, debug_training=False):
    """
    Continue training a Deep CFR agent from a saved checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        additional_iterations: Number of additional iterations to train
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save new models
        log_dir: Directory for tensorboard logs
        verbose: Whether to print verbose output
    """
    # Import tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Load the checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Initialize the agent
    num_players = 6  # Assuming 6 players as in the original training
    player_id = 0    # Assuming player_id 0 as in the original training
    agent = DeepCFRAgent(player_id=player_id, num_players=num_players, 
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    agent.debug_training = debug_training
    
    # Load the model weights
    agent.advantage_net.load_state_dict(checkpoint['advantage_net'])
    agent.strategy_net.load_state_dict(checkpoint['strategy_net'])
    
    # Set the iteration count from the checkpoint
    start_iteration = checkpoint['iteration'] + 1
    agent.iteration_count = start_iteration - 1
    
    # Load the training history if available
    losses = checkpoint.get('losses', [])
    profits = checkpoint.get('profits', [])
    
    print(f"Loaded model from iteration {start_iteration-1}")
    print(f"Continuing training for {additional_iterations} more iterations")
    
    # Create random agents for the opponents
    random_agents = [RandomAgent(i) for i in range(num_players)]
    
    # ADDED: Initial evaluation before continuing training
    print("Initial evaluation of loaded model...")
    initial_profit = evaluate_against_random(agent, num_games=500, num_players=num_players)
    if not profits:  # Only append if profits list is empty
        profits.append(initial_profit)
    print(f"Initial average profit per game: {initial_profit:.2f}")
    writer.add_scalar('Performance/Profit', initial_profit, start_iteration-1)
    
    # Checkpoint frequency
    checkpoint_frequency = 100  # Save a checkpoint every 100 iterations
    
    # Training loop
    for iteration in range(start_iteration, start_iteration + additional_iterations):
        agent.iteration_count = iteration
        start_time = time.time()
        
        print(f"Iteration {iteration}/{start_iteration + additional_iterations - 1}")
        
        # Run traversals to collect data
        print("  Collecting data...")
        for _ in range(traversals_per_iteration):
            # Create a new poker game
            state = pkrs.State.from_seed(
                n_players=num_players,
                button=random.randint(0, num_players-1),
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )
            
            # Perform CFR traversal
            agent.cfr_traverse(state, iteration, random_agents)
        
        # Track traversal time
        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)
        
        # Train advantage network
        print("  Training advantage network...")
        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        
        # Log the loss to tensorboard
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)
        
        # Every few iterations, train the strategy network and evaluate
        if iteration % 10 == 0 or iteration == start_iteration + additional_iterations - 1:
            print("  Training strategy network...")
            strat_loss = agent.train_strategy_network()
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)
            
            # Evaluate the agent
            print("  Evaluating agent...")
            avg_profit = evaluate_against_random(agent, num_games=500, num_players=num_players)
            profits.append(avg_profit)
            print(f"  Average profit per game: {avg_profit:.2f}")
            writer.add_scalar('Performance/Profit', avg_profit, iteration)
            
            # Save the model
            model_path = f"{save_dir}/deep_cfr_iter_{iteration}.pt"
            agent.save_model(model_path)
            print(f"  Model saved to {model_path}")
        
        # Save checkpoint periodically
        if iteration % checkpoint_frequency == 0 or iteration == start_iteration + additional_iterations - 1:
            checkpoint_path = f"{save_dir}/checkpoint_iter_{iteration}.pt"
            torch.save(
                standard_checkpoint_state(agent, losses=losses, profits=profits),
                checkpoint_path,
            )
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print(f"  Advantage memory size: {len(agent.advantage_memory)}")
        print(f"  Strategy memory size: {len(agent.strategy_memory)}")
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)
        
        # Commit the tensorboard logs
        writer.flush()
        print()
    
    # Final evaluation
    print("Final evaluation...")
    avg_profit = evaluate_against_random(agent, num_games=1000)
    print(f"Final performance: Average profit per game: {avg_profit:.2f}")
    writer.add_scalar('Performance/FinalProfit', avg_profit, 0)
    
    # Close the tensorboard writer
    writer.close()
    
    return agent, losses, profits

def train_against_checkpoint(checkpoint_path, additional_iterations=1000, 
                           traversals_per_iteration=200, save_dir="models", 
                           log_dir="logs/deepcfr_selfplay", verbose=False, debug_training=False,
                           progress_interval=100):
    """
    Continue a Deep CFR agent from a checkpoint while training against a fixed
    opponent pool loaded from that same checkpoint.
    
    Args:
        checkpoint_path: Path to the saved checkpoint to use as an opponent
        additional_iterations: Number of iterations to train
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save new models
        log_dir: Directory for tensorboard logs
        verbose: Whether to print verbose output
    """
    from torch.utils.tensorboard import SummaryWriter
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    print(f"Loading opponent from checkpoint: {checkpoint_path}")
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create opponent agents for all positions
    opponent_agents = []
    for pos in range(6):
        # Create a new agent for each position
        pos_agent = DeepCFRAgent(player_id=pos, num_players=6, device=device)
        pos_agent.load_model(checkpoint_path)
        opponent_agents.append(pos_agent)
    
    # Continue the learning agent from the same checkpoint used for the fixed opponents.
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
    learning_agent.load_model(checkpoint_path)
    learning_agent.debug_training = debug_training
    starting_iteration = learning_agent.iteration_count + 1
    
    # For tracking learning progress
    losses = []
    profits = []
    
    # ADDED: Initial evaluation before training begins
    print("Initial evaluation...")
    initial_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
        learning_agent, opponent_agents, num_games=100)
    print(f"Initial average profit vs checkpoint: {initial_profit_vs_checkpoint:.2f}")
    writer.add_scalar(
        'Performance/ProfitVsCheckpoint', initial_profit_vs_checkpoint, starting_iteration - 1
    )
    
    initial_profit_random = evaluate_against_random(
        learning_agent, num_games=500, num_players=6)
    profits.append(initial_profit_random)
    print(f"Initial average profit vs random: {initial_profit_random:.2f}")
    writer.add_scalar('Performance/ProfitVsRandom', initial_profit_random, starting_iteration - 1)
    
    # Checkpoint frequency
    checkpoint_frequency = 100  # Save more frequently for self-play
    opponent_wrappers = [None] * 6
    for pos in range(6):
        if pos != learning_agent.player_id:
            opponent_wrappers[pos] = _PerspectiveAgentWrapper(opponent_agents[pos])

    # Training loop
    final_iteration = starting_iteration + additional_iterations - 1
    progress = tqdm(
        range(starting_iteration, final_iteration + 1),
        desc="Self-play",
        unit="iter",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    last_strat_loss = None
    last_profit_vs_checkpoint = initial_profit_vs_checkpoint
    last_profit_random = initial_profit_random

    for iteration in progress:
        learning_agent.iteration_count = iteration
        start_time = time.time()

        # Run traversals to collect data
        for t in range(traversals_per_iteration):
            button_pos = t % 6

            state = pkrs.State.from_seed(
                n_players=6,
                button=button_pos,
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )

            _cfr_traverse_with_opponents(
                learning_agent, state, iteration, opponent_wrappers, verbose=verbose
            )

        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)

        adv_loss = learning_agent.train_advantage_network()
        losses.append(adv_loss)

        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(learning_agent.advantage_memory), iteration)

        if iteration % 10 == 0 or iteration == final_iteration:
            last_strat_loss = learning_agent.train_strategy_network()
            writer.add_scalar('Loss/Strategy', last_strat_loss, iteration)

            last_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
                learning_agent, opponent_agents, num_games=100
            )
            writer.add_scalar(
                'Performance/ProfitVsCheckpoint', last_profit_vs_checkpoint, iteration
            )

            last_profit_random = evaluate_against_random(
                learning_agent, num_games=500, num_players=6
            )
            profits.append(last_profit_random)
            writer.add_scalar('Performance/ProfitVsRandom', last_profit_random, iteration)

        if iteration % checkpoint_frequency == 0 or iteration == final_iteration:
            checkpoint_path = f"{save_dir}/selfplay_checkpoint_iter_{iteration}.pt"
            torch.save(
                standard_checkpoint_state(learning_agent, losses=losses, profits=profits),
                checkpoint_path,
            )
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        writer.add_scalar('Memory/Strategy', len(learning_agent.strategy_memory), iteration)
        progress.set_postfix(
            adv=f"{adv_loss:.3f}",
            strat="-" if last_strat_loss is None else f"{last_strat_loss:.3f}",
            vs_ckpt=f"{last_profit_vs_checkpoint:.2f}",
            vs_rand=f"{last_profit_random:.2f}",
            adv_mem=len(learning_agent.advantage_memory),
            strat_mem=len(learning_agent.strategy_memory),
        )

        should_report = (
            progress_interval > 0
            and (iteration % progress_interval == 0 or iteration == final_iteration)
        )
        if should_report:
            tqdm.write(
                f"iter {iteration}/{final_iteration} | "
                f"adv={adv_loss:.4f} | "
                f"strat={'-' if last_strat_loss is None else f'{last_strat_loss:.4f}'} | "
                f"vs_ckpt={last_profit_vs_checkpoint:.2f} | "
                f"vs_random={last_profit_random:.2f} | "
                f"adv_mem={len(learning_agent.advantage_memory)} | "
                f"strat_mem={len(learning_agent.strategy_memory)} | "
                f"{elapsed:.2f}s"
            )

        writer.flush()

    print("Final evaluation...")
    avg_profit_vs_checkpoint = evaluate_against_checkpoint_agents(
        learning_agent, opponent_agents, num_games=500
    )
    print(f"Final performance vs checkpoint: Average profit per game: {avg_profit_vs_checkpoint:.2f}")
    writer.add_scalar('Performance/FinalProfitVsCheckpoint', avg_profit_vs_checkpoint, 0)

    avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)

    writer.close()

    return learning_agent, losses, profits

def evaluate_against_checkpoint_agents(agent, opponent_agents, num_games=100):
    """
    Evaluate the trained agent against opponent agents.
    Each agent will receive and process observations from its own perspective.
    """
    opponent_wrappers = [None] * 6
    for pos in range(6):
        if pos != agent.player_id:
            opponent_wrappers[pos] = _PerspectiveAgentWrapper(opponent_agents[pos])

    metrics = evaluate_agent_matchup(
        agent,
        opponent_wrappers,
        num_games=num_games,
        seed_start=10000,
        strict=settings.is_strict_checking(),
        label="evaluation vs checkpoint",
        print_warnings=True,
    )
    if metrics["completed_games"] == 0:
        print("WARNING: No games completed during evaluation!")
    return metrics["avg_profit"]

def evaluate_against_agent(agent, opponent_agent, num_games=100):
    """Evaluate the trained agent against an opponent agent."""
    # This is kept for backward compatibility
    # But we recommend using evaluate_against_checkpoint_agents instead
    
    # Create a list of opponent agents for all positions
    opponent_agents = []
    for pos in range(6):
        if pos == agent.player_id:
            opponent_agents.append(None)  # Will be filled with the main agent
        else:
            # Create a copy of the opponent agent for this position
            pos_agent = DeepCFRAgent(player_id=pos, num_players=6,
                                    device=opponent_agent.device)
            pos_agent.advantage_net.load_state_dict(opponent_agent.advantage_net.state_dict())
            pos_agent.strategy_net.load_state_dict(opponent_agent.strategy_net.state_dict())
            opponent_agents.append(pos_agent)
    
    return evaluate_against_checkpoint_agents(agent, opponent_agents, num_games)

def train_with_mixed_checkpoints(checkpoint_dir, training_model_prefix="*checkpoint_iter_",
                           additional_iterations=1000, traversals_per_iteration=200,
                           save_dir="models", log_dir="logs/deepcfr_mixed",
                           refresh_interval=1000, num_opponents=5, verbose=False,
                           checkpoint_path=None, debug_training=False,
                           progress_interval=100):
    """
    Train a Deep CFR agent against opponents randomly selected from a pool of checkpoints.
    
    Args:
        checkpoint_dir: Directory containing checkpoint models
        training_model_prefix: Prefix or glob fragment for models in the selection pool
        additional_iterations: Number of iterations to train
        traversals_per_iteration: Number of traversals per iteration
        save_dir: Directory to save new models
        log_dir: Directory for tensorboard logs
        refresh_interval: How often to refresh the opponent pool (in iterations)
        num_opponents: Number of opponents to select from the pool
        verbose: Whether to print verbose output
    """
    from torch.utils.tensorboard import SummaryWriter
    import os
    import random
    
    # Set verbosity
    set_verbose(verbose)
    
    # Create the directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize the learning agent
    learning_agent = DeepCFRAgent(player_id=0, num_players=6, device=device)
    learning_agent.debug_training = debug_training
    starting_iteration = 1
    
    # For tracking learning progress
    losses = []
    profits = []
    profits_vs_checkpoints = []
    
    # Checkpoint frequency for saving our learning agent
    checkpoint_frequency = 100

    if checkpoint_path:
        print(f"Loading learning agent from checkpoint: {checkpoint_path}")
        learning_agent.load_model(checkpoint_path)
        try:
            checkpoint_state = load_checkpoint(checkpoint_path, map_location=device)
            starting_iteration = checkpoint_state.get("iteration", learning_agent.iteration_count) + 1
            losses = checkpoint_state.get("losses", losses)
            profits = checkpoint_state.get("profits", profits)
            profits_vs_checkpoints = checkpoint_state.get(
                "profits_vs_checkpoints",
                profits_vs_checkpoints,
            )
        except Exception:
            starting_iteration = learning_agent.iteration_count + 1

    def select_random_checkpoints():
        checkpoint_files = [str(path) for path in find_checkpoints(checkpoint_dir, training_model_prefix)]

        if not checkpoint_files:
            print(
                f"WARNING: No checkpoint files found matching '{training_model_prefix}' "
                f"in {checkpoint_dir}"
            )
            return [RandomAgent(i) if i != 0 else None for i in range(6)]

        selected_files = random.sample(checkpoint_files, min(num_opponents, len(checkpoint_files)))
        print(f"Selected checkpoints: {[os.path.basename(f) for f in selected_files]}")

        selected_agents = []
        current_pos = 1
        for checkpoint_file in selected_files:
            checkpoint_agent = CheckpointAgent(
                player_id=current_pos,
                model_path=checkpoint_file,
                device=device,
                sanitize_actions=True,
            )
            selected_agents.append((current_pos, checkpoint_agent))
            current_pos += 1
            if current_pos >= 6:
                current_pos = 1

        opponent_agents = [None] * 6
        opponent_agents[1] = RandomAgent(1)

        for pos, checkpoint_agent in selected_agents:
            if pos != 1:
                opponent_agents[pos] = checkpoint_agent

        for pos in range(1, 6):
            if opponent_agents[pos] is None:
                opponent_agents[pos] = RandomAgent(pos)

        return opponent_agents

    def wrap_opponents(opponent_agents):
        opponent_wrappers = [None] * 6
        for pos in range(6):
            if pos != learning_agent.player_id and opponent_agents[pos] is not None:
                opponent_wrappers[pos] = _PerspectiveAgentWrapper(opponent_agents[pos])
        return opponent_wrappers

    opponent_agents = select_random_checkpoints()

    print("Initial evaluation...")
    initial_profit_vs_mixed = evaluate_against_checkpoint_agents(
        learning_agent, opponent_agents, num_games=100
    )
    if not profits_vs_checkpoints:
        profits_vs_checkpoints.append(initial_profit_vs_mixed)
    print(f"Initial average profit vs mixed opponents: {initial_profit_vs_mixed:.2f}")
    writer.add_scalar('Performance/ProfitVsMixed', initial_profit_vs_mixed, starting_iteration - 1)

    initial_profit_random = evaluate_against_random(
        learning_agent, num_games=500, num_players=6
    )
    if not profits:
        profits.append(initial_profit_random)
    print(f"Initial average profit vs random: {initial_profit_random:.2f}")
    writer.add_scalar('Performance/ProfitVsRandom', initial_profit_random, starting_iteration - 1)

    opponent_wrappers = wrap_opponents(opponent_agents)

    final_iteration = starting_iteration + additional_iterations - 1
    progress = tqdm(
        range(starting_iteration, final_iteration + 1),
        desc="Mixed training",
        unit="iter",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    last_strat_loss = None
    last_profit_vs_mixed = initial_profit_vs_mixed
    last_profit_random = initial_profit_random

    for iteration in progress:
        learning_agent.iteration_count = iteration
        start_time = time.time()

        if iteration % refresh_interval == 1:
            tqdm.write(f"Refreshing opponent pool at iteration {iteration}")
            opponent_agents = select_random_checkpoints()
            opponent_wrappers = wrap_opponents(opponent_agents)

        for t in range(traversals_per_iteration):
            button_pos = t % 6
            state = pkrs.State.from_seed(
                n_players=6,
                button=button_pos,
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000)
            )

            _cfr_traverse_with_opponents(
                learning_agent, state, iteration, opponent_wrappers, verbose=verbose
            )

        traversal_time = time.time() - start_time
        writer.add_scalar('Time/Traversal', traversal_time, iteration)

        adv_loss = learning_agent.train_advantage_network()
        losses.append(adv_loss)

        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(learning_agent.advantage_memory), iteration)

        if iteration % 10 == 0 or iteration == final_iteration:
            last_strat_loss = learning_agent.train_strategy_network()
            writer.add_scalar('Loss/Strategy', last_strat_loss, iteration)

            last_profit_vs_mixed = evaluate_against_checkpoint_agents(
                learning_agent, opponent_agents, num_games=100
            )
            profits_vs_checkpoints.append(last_profit_vs_mixed)
            writer.add_scalar('Performance/ProfitVsMixed', last_profit_vs_mixed, iteration)

            last_profit_random = evaluate_against_random(
                learning_agent, num_games=500, num_players=6
            )
            profits.append(last_profit_random)
            writer.add_scalar('Performance/ProfitVsRandom', last_profit_random, iteration)

            if iteration % 100 == 0:
                t_model_path = f"{checkpoint_dir}/{training_model_prefix}mixed_iter_{iteration}.pt"
                torch.save(standard_checkpoint_state(learning_agent), t_model_path)
                tqdm.write(f"Training model saved: {t_model_path}")

        if iteration % checkpoint_frequency == 0 or iteration == final_iteration:
            checkpoint_path = f"{save_dir}/mixed_checkpoint_iter_{iteration}.pt"
            torch.save(
                standard_checkpoint_state(
                    learning_agent,
                    losses=losses,
                    profits=profits,
                    profits_vs_checkpoints=profits_vs_checkpoints,
                ),
                checkpoint_path,
            )
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        writer.add_scalar('Memory/Strategy', len(learning_agent.strategy_memory), iteration)
        progress.set_postfix(
            adv=f"{adv_loss:.3f}",
            strat="-" if last_strat_loss is None else f"{last_strat_loss:.3f}",
            vs_mixed=f"{last_profit_vs_mixed:.2f}",
            vs_rand=f"{last_profit_random:.2f}",
            adv_mem=len(learning_agent.advantage_memory),
            strat_mem=len(learning_agent.strategy_memory),
        )

        should_report = (
            progress_interval > 0
            and (iteration % progress_interval == 0 or iteration == final_iteration)
        )
        if should_report:
            tqdm.write(
                f"iter {iteration}/{final_iteration} | "
                f"adv={adv_loss:.4f} | "
                f"strat={'-' if last_strat_loss is None else f'{last_strat_loss:.4f}'} | "
                f"vs_mixed={last_profit_vs_mixed:.2f} | "
                f"vs_random={last_profit_random:.2f} | "
                f"adv_mem={len(learning_agent.advantage_memory)} | "
                f"strat_mem={len(learning_agent.strategy_memory)} | "
                f"{elapsed:.2f}s"
            )

        writer.flush()

    print("Final evaluation...")

    avg_profit_random = evaluate_against_random(learning_agent, num_games=500)
    print(f"Final performance vs random: Average profit per game: {avg_profit_random:.2f}")
    writer.add_scalar('Performance/FinalProfitVsRandom', avg_profit_random, 0)

    avg_profit_vs_mixed = evaluate_against_checkpoint_agents(
        learning_agent, opponent_agents, num_games=500
    )
    print(f"Final performance vs mixed opponents: Average profit per game: {avg_profit_vs_mixed:.2f}")
    writer.add_scalar('Performance/FinalProfitVsMixed', avg_profit_vs_mixed, 0)

    writer.close()

    return learning_agent, losses, profits, profits_vs_checkpoints


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train a Deep CFR agent for poker')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr', help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to continue training from')
    parser.add_argument('--self-play', action='store_true', help='Train against checkpoint instead of random agents')
    parser.add_argument('--mixed', action='store_true', help='Train against mixed checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='models', help='Directory containing checkpoint models')
    parser.add_argument(
        '--model-prefix',
        type=str,
        default='*checkpoint_iter_',
        help='Checkpoint prefix or glob fragment to include in the mixed-opponent pool',
    )
    parser.add_argument('--refresh-interval', type=int, default=1000, help='Interval to refresh opponent pool')
    parser.add_argument('--num-opponents', type=int, default=5, help='Number of checkpoint opponents to select')
    parser.add_argument('--strict', action='store_true', help='Enable strict error checking that raises exceptions for invalid game states')
    parser.add_argument('--debug-training', action='store_true', help='Print detailed training diagnostics from network updates')
    parser.add_argument('--progress-interval', type=int, default=100, help='Print compact training summaries every N iterations; set 0 to disable')
    args = parser.parse_args(argv)

    # Strict training for debug
    set_strict_checking(args.strict)
    
    if args.mixed:
        print(f"Starting mixed checkpoint training with models from: {args.checkpoint_dir}")
        agent, losses, profits, profits_vs_checkpoints = train_with_mixed_checkpoints(
            checkpoint_dir=args.checkpoint_dir,
            training_model_prefix=args.model_prefix,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_mixed",
            refresh_interval=args.refresh_interval,
            num_opponents=args.num_opponents,
            verbose=args.verbose,
            checkpoint_path=args.checkpoint,
            debug_training=args.debug_training,
            progress_interval=args.progress_interval,
        )
    elif args.checkpoint and args.self_play:
        print(f"Starting self-play training against checkpoint: {args.checkpoint}")
        agent, losses, profits = train_against_checkpoint(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_selfplay",
            verbose=args.verbose,
            debug_training=args.debug_training,
            progress_interval=args.progress_interval,
        )
    elif args.checkpoint:
        print(f"Continuing training from checkpoint: {args.checkpoint}")
        agent, losses, profits = continue_training(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=args.log_dir + "_continued",
            verbose=args.verbose,
            debug_training=args.debug_training,
        )
    else:
        print(f"Starting Deep CFR training for {args.iterations} iterations")
        print(f"Using {args.traversals} traversals per iteration")
        print(f"Logs will be saved to: {args.log_dir}")
        print(f"Models will be saved to: {args.save_dir}")
        
        # Train the Deep CFR agent
        agent, losses, profits = train_deep_cfr(
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            num_players=6,
            player_id=0,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            verbose=args.verbose,
            debug_training=args.debug_training,
            progress_interval=args.progress_interval,
        )
    
    print("\nTraining Summary:")
    print(f"Final loss: {losses[-1]:.6f}")
    if profits:
        print(f"Final average profit vs random: {profits[-1]:.2f}")
    if 'profits_vs_checkpoints' in locals() and profits_vs_checkpoints:
        print(f"Final average profit vs mixed checkpoints: {profits_vs_checkpoints[-1]:.2f}")
    
    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={args.log_dir}")
    print("Then open http://localhost:6006 in your browser")


if __name__ == "__main__":
    main()
