import pokers as pkrs
import random

class RandomAgent:
    """Simple random agent for poker that ensures valid bet sizing."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"Player {player_id}"
        
    def choose_action(self, state):
        """Choose a random legal action with a raise amount clamped by the available balance."""
        if not state.legal_actions:
            raise ValueError(f"No legal actions available for player {self.player_id}")
        
        action_enum = random.choice(state.legal_actions)
        
        # For non-raise actions, return as is.
        if action_enum in (pkrs.ActionEnum.Fold, pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
            return pkrs.Action(action_enum)
        
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            
            # Compute legal total bet bounds.
            lower_bound = current_bet + state.min_bet
            upper_bound = current_bet + player_state.stake  # maximum total bet if all-in.
            
            # If the player's remaining stake is less than the required min raise, just call.
            if player_state.stake < state.min_bet:
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            # Compute candidate total bets using pot-based heuristics.
            candidate_half = current_bet + state.pot * 0.5
            candidate_full = current_bet + state.pot
            
            candidates = []
            if lower_bound <= candidate_half <= upper_bound:
                candidates.append(candidate_half)
            if lower_bound <= candidate_full <= upper_bound:
                candidates.append(candidate_full)
            
            if candidates:
                chosen_total = random.choice(candidates)
            else:
                chosen_total = random.uniform(lower_bound, upper_bound)
            
            # Desired raise is the extra amount over current bet.
            desired_raise = chosen_total - current_bet
            
            # Clamp the raise to the player's available chips.
            final_raise = min(desired_raise, player_state.stake)
            
            return pkrs.Action(action_enum, final_raise)
        
        raise ValueError(f"Unexpected action type: {action_enum}")


def card_to_string(card):
    """Convert a card to a readable string."""
    suits = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
    ranks = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 
             7: "9", 8: "10", 9: "J", 10: "Q", 11: "K", 12: "A"}
    return f"{ranks[int(card.rank)]}{suits[int(card.suit)]}"

def debug_poker_game(num_games=3):
    """Run detailed poker game simulations and print all state transitions."""
    
    for game_num in range(num_games):
        print(f"\n{'='*80}\nGAME {game_num+1}\n{'='*80}")
        
        # Create agents
        agents = [RandomAgent(i) for i in range(6)]
        
        # Initialize game
        initial_state = pkrs.State.from_seed(
            n_players=len(agents), 
            button=0, 
            sb=1.0, 
            bb=2.0, 
            stake=100.0, 
            seed=random.randint(0, 10000)
        )
        
        # Print initial state
        print("INITIAL STATE:")
        print(f"Status: {initial_state.status}")
        print(f"Stage: {initial_state.stage}")
        print(f"Pot: {initial_state.pot}")
        print(f"Min bet: {initial_state.min_bet}")
        print("Player hands:")
        for i, player in enumerate(initial_state.players_state):
            hand = " ".join([card_to_string(card) for card in player.hand])
            print(f"  Player {i}: {hand} (Stake: {player.stake}, Bet: {player.bet_chips})")
        print(f"Public cards: {'None' if not initial_state.public_cards else ' '.join([card_to_string(card) for card in initial_state.public_cards])}")
        print(f"Legal actions: {initial_state.legal_actions}")
        
        state = initial_state
        action_count = 0
        
        # Main game loop
        while not state.final_state:
            action_count += 1
            print(f"\nACTION {action_count}:")
            
            current_player = state.current_player
            print(f"Current player: {current_player}")
            print(f"Legal actions: {state.legal_actions}")
            
            # Get player stake and bet info
            player_state = state.players_state[current_player]
            print(f"Player stake: {player_state.stake}, Current bet: {player_state.bet_chips}")
            print(f"Pot: {state.pot}, Min bet: {state.min_bet}")
            
            # Choose action
            try:
                action = agents[current_player].choose_action(state)
                action_str = f"Action chosen: {action.action}" + (f" with amount {action.amount}" if action.action == pkrs.ActionEnum.Raise else "")
                print(action_str)
                
                # Apply action
                new_state = state.apply_action(action)
                
                # Check if the action was successful
                print(f"New state status: {new_state.status}")
                
                if new_state.status != pkrs.StateStatus.Ok:
                    print(f"ERROR: Invalid action resulted in status {new_state.status}")
                    print("Details:")
                    print(f"  Player stake: {player_state.stake}")
                    print(f"  Current bet: {player_state.bet_chips}")
                    print(f"  Min bet: {state.min_bet}")
                    if action.action == pkrs.ActionEnum.Raise:
                        print(f"  Attempted raise amount: {action.amount}")
                    
                    # Try to continue with a default action
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        action = pkrs.Action(pkrs.ActionEnum.Call)
                    elif pkrs.ActionEnum.Check in state.legal_actions:
                        action = pkrs.Action(pkrs.ActionEnum.Check)
                    else:
                        action = pkrs.Action(pkrs.ActionEnum.Fold)
                    
                    print(f"Falling back to action: {action.action}")
                    new_state = state.apply_action(action)
                    print(f"New fallback state status: {new_state.status}")
                
                # Update state
                state = new_state
                
                # Print current state info
                print(f"Stage: {state.stage}")
                print(f"Pot: {state.pot}")
                print(f"Public cards: {'None' if not state.public_cards else ' '.join([card_to_string(card) for card in state.public_cards])}")
                
                # Print player status
                print("Player status:")
                for i, p in enumerate(state.players_state):
                    status = "Active" if p.active else "Folded"
                    print(f"  Player {i}: Stake: {p.stake}, Bet: {p.bet_chips}, Status: {status}")
                
            except Exception as e:
                print(f"ERROR: Exception occurred: {e}")
                break
        
        # Game is over
        print(f"\nGAME OVER (after {action_count} actions)")
        print(f"Final status: {state.status}")
        print(f"Final stage: {state.stage}")
        print(f"Final pot: {state.pot}")
        
        # Print player hands
        print("Final player hands:")
        for i, player in enumerate(state.players_state):
            if player.active:
                hand = " ".join([card_to_string(card) for card in player.hand])
                print(f"  Player {i}: {hand} (Reward: {player.reward})")
            else:
                print(f"  Player {i}: Folded (Reward: {player.reward})")
        
        # Print community cards
        print(f"Community cards: {'None' if not state.public_cards else ' '.join([card_to_string(card) for card in state.public_cards])}")
        
        # Check if rewards sum to zero (as expected in a zero-sum game)
        total_reward = sum(p.reward for p in state.players_state)
        print(f"Total rewards: {total_reward} (should be approximately 0 for a zero-sum game)")

if __name__ == "__main__":
    debug_poker_game(num_games=4)