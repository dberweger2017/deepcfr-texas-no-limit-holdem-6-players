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
        if action_enum in (pkrs.ActionEnum.Fold, pkrs.ActionEnum.Check, pkrs.ActionEnum.Call):
            return pkrs.Action(action_enum)
        
        # For raises, calculate a valid additional raise amount.
        # The environment expects the raise amount to be the extra chips put in on top of the current bet.
        elif action_enum == pkrs.ActionEnum.Raise:
            player_state = state.players_state[state.current_player]
            current_bet = player_state.bet_chips
            # Minimum total bet required is current bet plus the min additional bet
            min_total_bet = current_bet + state.min_bet
            # Maximum total bet is what the player already bet plus their remaining chips (all-in)
            max_total_bet = current_bet + player_state.stake

            # If the player's remaining chips are less than the min required raise,
            # treat the situation as all-in (i.e. simply call)
            if player_state.stake < state.min_bet:
                return pkrs.Action(pkrs.ActionEnum.Call)
            
            # Propose potential total bet amounts based on pot size
            half_pot_total = current_bet + state.pot * 0.5
            full_pot_total = current_bet + state.pot
            valid_total_bets = []
            
            if half_pot_total >= min_total_bet:
                valid_total_bets.append(min(half_pot_total, max_total_bet))
            if full_pot_total >= min_total_bet:
                valid_total_bets.append(min(full_pot_total, max_total_bet))
            
            if valid_total_bets:
                chosen_total_bet = random.choice(valid_total_bets)
            else:
                chosen_total_bet = random.uniform(min_total_bet, max_total_bet)
            
            # The additional raise is the difference between the desired total bet and the current bet.
            additional_raise = chosen_total_bet - current_bet
            return pkrs.Action(action_enum, additional_raise)
        
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