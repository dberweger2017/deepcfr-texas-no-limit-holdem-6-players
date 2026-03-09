import argparse
import os
import random
import time

import pokers as pkrs
import torch

from src.agents.random_agent import RandomAgent
from src.core.model import set_verbose
from src.opponent_modeling.deep_cfr_with_opponent_modeling import (
    DeepCFRAgentWithOpponentModeling,
)
from src.training.train_mixed_with_opponent_modeling import (
    ModelAgent,
    checkpoint_uses_opponent_modeling,
    evaluate_against_opponents,
    train_mixed_with_opponent_modeling,
)
from src.training.train_with_opponent_modeling import (
    evaluate_against_random,
    train_deep_cfr_with_opponent_modeling,
)
from src.utils.settings import set_strict_checking


def _build_random_opponents(player_id):
    opponents = [None] * 6
    for position in range(6):
        if position != player_id:
            opponents[position] = RandomAgent(position)
    return opponents


def _build_fixed_checkpoint_opponents(checkpoint_path, player_id, device):
    is_om_checkpoint = checkpoint_uses_opponent_modeling(checkpoint_path)
    opponents = [None] * 6
    for position in range(6):
        if position == player_id:
            continue
        opponents[position] = ModelAgent(
            player_id=position,
            model_path=checkpoint_path,
            device=device,
            with_opponent_modeling=is_om_checkpoint,
        )
    return opponents


def train_against_checkpoint_with_opponent_modeling(
    checkpoint_path,
    additional_iterations=1000,
    traversals_per_iteration=200,
    save_dir="models_om",
    log_dir="logs/deepcfr_om_selfplay",
    player_id=0,
    verbose=False,
):
    """Continue an opponent-modeling agent against a fixed checkpoint opponent pool."""
    from torch.utils.tensorboard import SummaryWriter

    try:
        from scripts.telegram_notifier import TelegramNotifier
    except Exception:
        TelegramNotifier = None

    set_verbose(verbose)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not checkpoint_uses_opponent_modeling(checkpoint_path):
        raise ValueError(
            "Opponent-model self-play requires an opponent-model checkpoint. "
            "Use a phase-1 checkpoint created by src.training.train_opponent_modeling."
        )

    notifier = None
    if TelegramNotifier is not None:
        try:
            notifier = TelegramNotifier()
        except Exception as exc:
            print(f"Warning: Could not initialize Telegram notifier: {exc}")

    print(f"Loading fixed self-play opponents from checkpoint: {checkpoint_path}")
    opponents = _build_fixed_checkpoint_opponents(checkpoint_path, player_id, device)

    learning_agent = DeepCFRAgentWithOpponentModeling(
        player_id=player_id,
        num_players=6,
        device=device,
    )
    learning_agent.load_model(checkpoint_path)
    starting_iteration = learning_agent.iteration_count + 1

    advantage_losses = []
    strategy_losses = []
    opponent_model_losses = []
    profits = []
    profits_vs_checkpoints = []
    checkpoint_frequency = 100

    print("Initial evaluation...")
    initial_profit_vs_checkpoint = evaluate_against_opponents(
        learning_agent,
        opponents,
        num_games=100,
        iteration=0,
        notifier=notifier,
    )
    profits_vs_checkpoints.append(initial_profit_vs_checkpoint)
    print(f"Initial average profit vs checkpoint: {initial_profit_vs_checkpoint:.2f}")
    writer.add_scalar(
        "Performance/ProfitVsCheckpoint",
        initial_profit_vs_checkpoint,
        starting_iteration - 1,
    )

    initial_profit_random = evaluate_against_random(
        learning_agent,
        num_games=200,
        iteration=0,
        notifier=notifier,
    )
    profits.append(initial_profit_random)
    print(f"Initial average profit vs random: {initial_profit_random:.2f}")
    writer.add_scalar("Performance/ProfitVsRandom", initial_profit_random, starting_iteration - 1)

    final_iteration = starting_iteration + additional_iterations - 1
    for iteration in range(starting_iteration, final_iteration + 1):
        learning_agent.iteration_count = iteration
        start_time = time.time()

        print(f"Self-play iteration {iteration}/{final_iteration}")
        print("  Collecting data...")
        for traversal in range(traversals_per_iteration):
            state = pkrs.State.from_seed(
                n_players=6,
                button=traversal % 6,
                sb=1,
                bb=2,
                stake=200.0,
                seed=random.randint(0, 10000),
            )
            try:
                learning_agent.cfr_traverse(state, iteration, opponents)
            except Exception as exc:
                print(f"Error during traversal: {exc}")
                if notifier and traversal % 50 == 0:
                    notifier.send_message(
                        f"⚠️ <b>SELF-PLAY TRAVERSAL ERROR</b>\nIteration: {iteration}\nError: {exc}"
                    )

        traversal_time = time.time() - start_time
        writer.add_scalar("Time/Traversal", traversal_time, iteration)

        print("  Training advantage network...")
        adv_loss = learning_agent.train_advantage_network()
        advantage_losses.append(adv_loss)
        print(f"  Advantage network loss: {adv_loss:.6f}")
        writer.add_scalar("Loss/Advantage", adv_loss, iteration)

        if iteration % 5 == 0 or iteration == final_iteration:
            print("  Training strategy network...")
            strat_loss = learning_agent.train_strategy_network()
            strategy_losses.append(strat_loss)
            print(f"  Strategy network loss: {strat_loss:.6f}")
            writer.add_scalar("Loss/Strategy", strat_loss, iteration)

        if iteration % 10 == 0 or iteration == final_iteration:
            print("  Training opponent modeling...")
            opp_loss = learning_agent.train_opponent_modeling()
            opponent_model_losses.append(opp_loss)
            print(f"  Opponent modeling loss: {opp_loss:.6f}")
            writer.add_scalar("Loss/OpponentModeling", opp_loss, iteration)

        if iteration % 20 == 0 or iteration == final_iteration:
            print("  Evaluating against checkpoint opponents...")
            avg_profit_vs_checkpoint = evaluate_against_opponents(
                learning_agent,
                opponents,
                num_games=100,
                iteration=iteration,
                notifier=notifier,
            )
            profits_vs_checkpoints.append(avg_profit_vs_checkpoint)
            print(f"  Average profit vs checkpoint: {avg_profit_vs_checkpoint:.2f}")
            writer.add_scalar("Performance/ProfitVsCheckpoint", avg_profit_vs_checkpoint, iteration)

            print("  Evaluating against random opponents...")
            avg_profit_random = evaluate_against_random(
                learning_agent,
                num_games=200,
                iteration=iteration,
                notifier=notifier,
            )
            profits.append(avg_profit_random)
            print(f"  Average profit vs random: {avg_profit_random:.2f}")
            writer.add_scalar("Performance/ProfitVsRandom", avg_profit_random, iteration)

        if iteration % checkpoint_frequency == 0 or iteration == final_iteration:
            save_path = f"{save_dir}/selfplay_checkpoint_iter_{iteration}.pt"
            torch.save(
                {
                    "iteration": iteration,
                    "advantage_net": learning_agent.advantage_net.state_dict(),
                    "strategy_net": learning_agent.strategy_net.state_dict(),
                    "history_encoder": learning_agent.opponent_modeling.history_encoder.state_dict(),
                    "opponent_model": learning_agent.opponent_modeling.opponent_model.state_dict(),
                },
                save_path,
            )
            print(f"  Checkpoint saved to {save_path}")

        writer.add_scalar("Memory/Advantage", len(learning_agent.advantage_memory), iteration)
        writer.add_scalar("Memory/Strategy", len(learning_agent.strategy_memory), iteration)

        elapsed = time.time() - start_time
        writer.add_scalar("Time/Iteration", elapsed, iteration)
        print(f"  Iteration completed in {elapsed:.2f} seconds")
        print()

    print("Final evaluation...")
    final_profit_random = evaluate_against_random(
        learning_agent,
        num_games=500,
        iteration=final_iteration,
        notifier=notifier,
    )
    print(f"Final performance vs random: Average profit per game: {final_profit_random:.2f}")
    writer.add_scalar("Performance/FinalProfitVsRandom", final_profit_random, 0)

    final_profit_checkpoint = evaluate_against_opponents(
        learning_agent,
        opponents,
        num_games=500,
        iteration=final_iteration,
        notifier=notifier,
    )
    print(
        "Final performance vs checkpoint opponents: "
        f"Average profit per game: {final_profit_checkpoint:.2f}"
    )
    writer.add_scalar("Performance/FinalProfitVsCheckpoint", final_profit_checkpoint, 0)

    writer.close()

    return (
        learning_agent,
        advantage_losses,
        strategy_losses,
        opponent_model_losses,
        profits,
        profits_vs_checkpoints,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a Deep CFR agent with opponent modeling",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of CFR iterations")
    parser.add_argument("--traversals", type=int, default=200, help="Traversals per iteration")
    parser.add_argument("--save-dir", type=str, default="models_om", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs/deepcfr_om", help="Directory for logs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--self-play", action="store_true", help="Train against a fixed checkpoint opponent")
    parser.add_argument("--mixed", action="store_true", help="Train against a checkpoint pool")
    parser.add_argument("--checkpoint-dir", type=str, default="models_om", help="Directory containing checkpoint models")
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="*checkpoint_iter_",
        help="Checkpoint prefix or glob fragment for selection",
    )
    parser.add_argument("--refresh-interval", type=int, default=1000, help="Interval to refresh opponent pool")
    parser.add_argument("--num-opponents", type=int, default=5, help="Number of checkpoint opponents to select")
    parser.add_argument("--strict", action="store_true", help="Raise exceptions for invalid game states")
    args = parser.parse_args(argv)

    set_strict_checking(args.strict)

    if args.mixed:
        print(f"Starting mixed checkpoint training with opponent modeling from: {args.checkpoint_dir}")
        effective_log_dir = args.log_dir + "_mixed"
        result = train_mixed_with_opponent_modeling(
            checkpoint_dir=args.checkpoint_dir,
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            refresh_interval=args.refresh_interval,
            num_opponents=args.num_opponents,
            save_dir=args.save_dir,
            log_dir=effective_log_dir,
            player_id=0,
            model_prefix=args.model_prefix,
            verbose=args.verbose,
            checkpoint_path=args.checkpoint,
        )
        agent, adv_losses, strat_losses, om_losses, profits, profits_vs_checkpoints = result
    elif args.checkpoint and args.self_play:
        print(f"Starting opponent-model self-play against checkpoint: {args.checkpoint}")
        effective_log_dir = args.log_dir + "_selfplay"
        result = train_against_checkpoint_with_opponent_modeling(
            checkpoint_path=args.checkpoint,
            additional_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=effective_log_dir,
            player_id=0,
            verbose=args.verbose,
        )
        agent, adv_losses, strat_losses, om_losses, profits, profits_vs_checkpoints = result
    else:
        effective_log_dir = args.log_dir + "_continued" if args.checkpoint else args.log_dir
        if args.checkpoint:
            print(f"Continuing opponent-model training from checkpoint: {args.checkpoint}")
        else:
            print(f"Starting opponent-model training for {args.iterations} iterations")
        result = train_deep_cfr_with_opponent_modeling(
            num_iterations=args.iterations,
            traversals_per_iteration=args.traversals,
            save_dir=args.save_dir,
            log_dir=effective_log_dir,
            verbose=args.verbose,
            checkpoint_path=args.checkpoint,
        )
        agent, adv_losses, strat_losses, om_losses, profits = result
        profits_vs_checkpoints = []

    print("\nTraining Summary:")
    if adv_losses:
        print(f"Final advantage network loss: {adv_losses[-1]:.6f}")
    if strat_losses:
        print(f"Final strategy network loss: {strat_losses[-1]:.6f}")
    if om_losses:
        print(f"Final opponent-model loss: {om_losses[-1]:.6f}")
    if profits:
        print(f"Final average profit vs random: {profits[-1]:.2f}")
    if profits_vs_checkpoints:
        print(f"Final average profit vs checkpoints: {profits_vs_checkpoints[-1]:.2f}")

    print("\nTo view training progress:")
    print(f"Run: tensorboard --logdir={effective_log_dir}")
    print("Then open http://localhost:6006 in your browser")

    return agent


if __name__ == "__main__":
    main()
