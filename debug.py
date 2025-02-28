import pokers as pkrs
import random

class RandomAgent:
    """Simple random agent for poker that ensures valid bet sizing."""
    def __init__(self, player_id):
        self.player_id = player_id
        self.name = f"Player {player_id}"
        
    def choose_action(self, state):
        """Choose a random legal action with proper bet sizing."""
        if not state.legal_actions:
            raise ValueError(f"No legal actions available for player {self.player_id}")
        
        # Select a random legal action
        action_enum = random.choice(state.legal_actions)
        
        # For fold, check, and call, no amount is needed
        if action_enum == pkrs.ActionEnum.Fold:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Check:
            return pkrs.Action(action_enum)
        elif action_enum == pkrs.ActionEnum.Call:
            return pkrs.Action(action_enum)
        # For raises, carefully calculate a valid amount
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            
            # Get the minimum required raise
            min_raise = state.min_bet
            
            # Calculate maximum raise based on player's available chips
            max_raise = player_state.stake
            
            # If player doesn't have enough chips to meet the minimum raise,
            # they can still go all-in with whatever they have left
            if max_raise < min_raise:
                # Go all-in with their remaining stake
                return pkrs.Action(action_enum, max_raise)
            
            # Calculate potential raise amounts (half pot and full pot)
            half_pot = state.pot * 0.5
            full_pot = state.pot
            
            # Create a list of valid raise amounts
            valid_amounts = []
            
            # Add half pot if it's a valid amount
            if half_pot >= min_raise:
                valid_amounts.append(min(half_pot, max_raise))
                
            # Add full pot if it's a valid amount
            if full_pot >= min_raise:
                valid_amounts.append(min(full_pot, max_raise))
                
            # If we have valid pot-based raise options, choose one randomly
            if valid_amounts:
                raise_amount = random.choice(valid_amounts)
            else:
                # Otherwise use a random amount between min and max
                raise_amount = random.uniform(min(min_raise, max_raise), max_raise)
                
            # Make sure the final amount is within valid bounds
            raise_amount = min(raise_amount, max_raise)
            
            # Create and return the raise action
            return pkrs.Action(action_enum, raise_amount)
        
        # If we get an unexpected action type, raise an exception
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
                print(f"Action chosen: {action.action}" + (f" with amount {action.amount}" if action.action == pkrs.ActionEnum.Raise else ""))
                
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
    debug_poker_game(num_games=3)