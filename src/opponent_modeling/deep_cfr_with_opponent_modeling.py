# src/opponent_modeling/deep_cfr_with_opponent_modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from collections import deque
from src.core.model import encode_state, VERBOSE, set_verbose
from src.opponent_modeling.opponent_model import OpponentModelingSystem
from src.core.deep_cfr import PrioritizedMemory
from src.utils.settings import STRICT_CHECKING
from src.utils.logging import log_game_error

class EnhancedPokerNetwork(nn.Module):
    """
    Enhanced network that incorporates opponent modeling features
    and continuous bet sizing.
    """
    def __init__(self, input_size=500, opponent_feature_size=20, hidden_size=256, num_actions=3):
        super().__init__()
        # Standard game state processing
        self.base_state = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Process opponent features
        self.opponent_fc = nn.Linear(opponent_feature_size, hidden_size // 2)
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        # Action type prediction (fold, check/call, raise)
        self.action_head = nn.Linear(hidden_size, num_actions)
        
        # Continuous bet sizing prediction
        self.sizing_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
    def forward(self, state_input, opponent_features=None):
        # Process game state
        x = self.base_state(state_input)
        
        # If opponent features are provided, incorporate them
        if opponent_features is not None:
            opponent_encoding = F.relu(self.opponent_fc(opponent_features))
            x = torch.cat([x, opponent_encoding], dim=1)
        else:
            # If no opponent features, use zeros
            batch_size = state_input.size(0)
            x = torch.cat([x, torch.zeros(batch_size, self.opponent_fc.out_features, device=state_input.device)], dim=1)
        
        # Continue processing the combined features
        x = self.combined(x)
        
        # Output action logits and bet sizing
        action_logits = self.action_head(x)
        bet_size = 0.1 + 2.9 * self.sizing_head(x)  # Output between 0.1x and 3x pot
        
        return action_logits, bet_size

class DeepCFRAgentWithOpponentModeling:
    def __init__(self, player_id=0, num_players=6, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = 3
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Create advantage network with opponent modeling and bet sizing
        self.advantage_net = EnhancedPokerNetwork(
            input_size=input_size, 
            opponent_feature_size=20, 
            hidden_size=256, 
            num_actions=self.num_actions
        ).to(device)
        
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=1e-6, weight_decay=1e-5)
        
        # Create prioritized memory buffer
        self.advantage_memory = PrioritizedMemory(memory_size)
        
        # Strategy network (also with opponent modeling)
        self.strategy_net = EnhancedPokerNetwork(
            input_size=input_size, 
            opponent_feature_size=20, 
            hidden_size=256, 
            num_actions=self.num_actions
        ).to(device)
        
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # Initialize opponent modeling system with enhanced features
        self.opponent_modeling = OpponentModelingSystem(
            max_history_per_opponent=20,
            action_dim=4,  # Still tracking 4 discrete actions for history
            state_dim=25,  # Expanded to include bet sizing features
            device=device
        )
        
        # For tracking game history during play
        self.current_game_history = {}  # Maps opponent_id -> (actions, contexts)
        
        # For keeping statistics
        self.iteration_count = 0
        
        # Bet sizing bounds (as multipliers of pot)
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
    
    def action_type_to_pokers_action(self, action_type, state, bet_size_multiplier=None):
        """Convert action type and optional bet size to Pokers action."""
        # NOTE: This function is identical to the one in DeepCFRAgent.
        #       Consider refactoring into a shared utility if possible.
        try:
            if action_type == 0:  # Fold
                # Ensure Fold is legal before returning
                if pkrs.ActionEnum.Fold in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Fold)
                else:
                    # Fallback if Fold is somehow illegal (shouldn't happen often)
                    if pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    elif pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        # If nothing else is legal, something is very wrong
                        print("WARNING: No legal actions found, even Fold!")
                        # Attempt Fold anyway as a last resort
                        return pkrs.Action(pkrs.ActionEnum.Fold)

            elif action_type == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                elif pkrs.ActionEnum.Call in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                else:
                    # Fallback if neither Check nor Call is legal
                    if pkrs.ActionEnum.Fold in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Fold)
                    else:
                        print("WARNING: Check/Call chosen but neither is legal!")
                        # Attempt Check as a last resort if available
                        return pkrs.Action(pkrs.ActionEnum.Check)


            elif action_type == 2:  # Raise
                # First, check if Raise itself is a legal action type
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    # If Raise is not legal, fall back to Call or Check
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    elif pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    else:
                        # If neither Call nor Check is available, Fold
                        return pkrs.Action(pkrs.ActionEnum.Fold)

                # Get current player state
                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips
                available_stake = player_state.stake

                # Calculate what's needed to call (match the current min_bet)
                call_amount = max(0, state.min_bet - current_bet)

                # *** CORRECTED LOGIC HERE ***
                # If player doesn't have enough chips *beyond* the call amount to make a valid raise,
                # or if they are going all-in just to call, it should be a Call action.
                # A raise requires putting in *more* than the call amount.
                # The minimum additional raise amount is typically the big blind or 1 chip.
                min_raise_increment = 1.0 # A small default
                if hasattr(state, 'bb'):
                    min_raise_increment = max(1.0, state.bb) # Usually BB, but at least 1

                if available_stake <= call_amount + min_raise_increment:
                    # Player cannot make a valid raise (or is all-in just to call).
                    # This action should be treated as a Call.
                    if VERBOSE:
                        print(f"Action type 2 (Raise) chosen, but player cannot make a valid raise. "
                              f"Stake: {available_stake}, Call Amount: {call_amount}. Switching to Call.")
                    # Ensure Call is legal before returning it
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    else:
                        # If Call is not legal (edge case, e.g., already all-in matching bet), Fold.
                         if VERBOSE:
                             print(f"WARNING: Cannot Call (not legal), falling back to Fold.")
                         return pkrs.Action(pkrs.ActionEnum.Fold)
                # *** END OF CORRECTION ***

                # If we reach here, the player *can* make a valid raise.
                remaining_stake_after_call = available_stake - call_amount

                # Calculate target raise amount based on pot multiplier
                pot_size = max(1.0, state.pot) # Avoid division by zero
                if bet_size_multiplier is None:
                    # Default to 1x pot if no multiplier provided
                    bet_size_multiplier = 1.0

                # Ensure multiplier is within bounds
                bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))
                target_additional_raise = pot_size * bet_size_multiplier

                # Ensure minimum raise increment is met
                target_additional_raise = max(target_additional_raise, min_raise_increment)

                # Ensure we don't exceed available stake after calling
                additional_amount = min(target_additional_raise, remaining_stake_after_call)

                # Final check: Ensure the additional amount is at least the minimum required increment
                if additional_amount < min_raise_increment:
                     # This case should be rare due to the check above, but as a safeguard:
                     if VERBOSE:
                         print(f"Calculated raise amount {additional_amount} is less than min increment {min_raise_increment}. Falling back to Call.")
                     if pkrs.ActionEnum.Call in state.legal_actions:
                         return pkrs.Action(pkrs.ActionEnum.Call)
                     else:
                         return pkrs.Action(pkrs.ActionEnum.Fold)

                if VERBOSE:
                    print(f"\nRAISE CALCULATION DETAILS:")
                    print(f"  Player ID: {state.current_player}")
                    print(f"  Action type: {action_type}")
                    print(f"  Current bet: {current_bet}")
                    print(f"  Available stake: {available_stake}")
                    print(f"  Min bet: {state.min_bet}")
                    print(f"  Call amount: {call_amount}")
                    print(f"  Pot size: {state.pot}")
                    print(f"  Bet multiplier: {bet_size_multiplier}x pot")
                    print(f"  Calculated additional raise amount: {additional_amount}")
                    print(f"  Total player bet will be: {current_bet + call_amount + additional_amount}")

                # Return the Raise action with the calculated *additional* amount
                return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)

            else:
                raise ValueError(f"Unknown action type: {action_type}")

        except Exception as e:
            if VERBOSE:
                print(f"ERROR creating action {action_type}: {e}")
                print(f"State: current_player={state.current_player}, legal_actions={state.legal_actions}")
                print(f"Player stake: {state.players_state[state.current_player].stake}")
            # Fall back to call as safe option
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)

    def get_legal_action_types(self, state):
        """Get the legal action types for the current state."""
        legal_action_types = []
        
        # Check each action type
        if pkrs.ActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)
            
        if pkrs.ActionEnum.Check in state.legal_actions or pkrs.ActionEnum.Call in state.legal_actions:
            legal_action_types.append(1)
            
        if pkrs.ActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)
        
        return legal_action_types
    
    def extract_state_context(self, state):
        """
        Extract a simplified state context for opponent modeling.
        Returns a compact representation of the current state with bet sizing features.
        """
        # For simplicity, we'll use a fixed-size feature vector
        context = np.zeros(25)  # Expanded to 25 for bet size features
        
        # Game stage (one-hot encoded)
        stage_idx = int(state.stage)
        if 0 <= stage_idx < 5:
            context[stage_idx] = 1
        
        # Pot size (normalized)
        initial_stake = max(1.0, state.players_state[0].stake + state.players_state[0].bet_chips)
        context[5] = state.pot / initial_stake
        
        # Number of active players
        active_count = sum(1 for p in state.players_state if p.active)
        context[6] = active_count / self.num_players
        
        # Position relative to button
        btn_distance = (state.current_player - state.button) % self.num_players
        context[7] = btn_distance / self.num_players
        
        # Community card count
        context[8] = len(state.public_cards) / 5
        
        # Previous action type and size
        if state.from_action is not None:
            prev_action_type = int(state.from_action.action.action)
            if 0 <= prev_action_type < 4:
                context[9 + prev_action_type] = 1
            
            if prev_action_type == int(pkrs.ActionEnum.Raise):
                context[13] = state.from_action.action.amount / initial_stake
        
        # Min bet relative to pot
        context[14] = state.min_bet / max(1.0, state.pot)
        
        # Player stack sizes
        avg_stack = sum(p.stake for p in state.players_state) / self.num_players
        context[15] = state.players_state[state.current_player].stake / max(1.0, avg_stack)
        
        # Current bet relative to pot
        current_bet = state.players_state[state.current_player].bet_chips
        context[16] = current_bet / max(1.0, state.pot)
        
        # Add bet size features
        if state.from_action is not None and state.from_action.action.action == pkrs.ActionEnum.Raise:
            # Normalize bet size as a fraction of the pot
            normalized_bet_size = state.from_action.action.amount / max(1.0, state.pot)
            context[20] = normalized_bet_size
            
            # Add bucketed bet size indicators
            if normalized_bet_size < 0.5:
                context[21] = 1  # Small bet (less than half pot)
            elif normalized_bet_size < 1.0:
                context[22] = 1  # Medium bet (half to full pot)
            elif normalized_bet_size < 2.0:
                context[23] = 1  # Large bet (1-2x pot)
            else:
                context[24] = 1  # Very large bet (2x+ pot)
        
        return context
    
    def record_opponent_action(self, state, action_id, opponent_id):
        """
        Record an action taken by an opponent for later opponent modeling.
        """
        # Initialize history for this opponent if needed
        if opponent_id not in self.current_game_history:
            self.current_game_history[opponent_id] = {
                'actions': [],
                'contexts': []
            }
        
        # Convert action to one-hot encoding
        action_encoded = np.zeros(4)  # Use original 4 action encoding for history
        action_encoded[action_id] = 1
        
        # Get state context
        context = self.extract_state_context(state)
        
        # Record action and context
        self.current_game_history[opponent_id]['actions'].append(action_encoded)
        self.current_game_history[opponent_id]['contexts'].append(context)
    
    def end_game_recording(self, state):
        """
        Finalize recording of the current game and add to opponent histories.
        """
        for opponent_id, history in self.current_game_history.items():
            # Skip if no actions recorded
            if not history['actions']:
                continue
            
            # Get the outcome for this opponent
            outcome = state.players_state[opponent_id].reward
            
            # Record to opponent modeling system
            self.opponent_modeling.record_game(
                opponent_id=opponent_id,
                action_sequence=history['actions'],
                state_contexts=history['contexts'],
                outcome=outcome
            )
        
        # Clear the current game history
        self.current_game_history = {}

    def cfr_traverse(self, state, iteration, opponents, depth=0):
        """
        Traverse the game tree using external sampling MCCFR with continuous bet sizing.
        Modified to work with both RandomAgent and ModelAgent opponents.
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            # Record the end of the game for opponent modeling
            self.end_game_recording(state)
            # Return payoff for the trained agent
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        # If it's the trained agent's turn
        if current_player == self.player_id:
            legal_action_types = self.get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            # Encode the base state
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            # Get opponent features for the current opponent
            opponent_features = None
            if current_player != self.player_id:
                opponent_features = self.opponent_modeling.get_opponent_features(current_player)
                opponent_features = torch.FloatTensor(opponent_features).to(self.device)
            
            # Get advantages and bet sizing prediction from network
            with torch.no_grad():
                # Use opponent features if available
                if opponent_features is not None:
                    advantages, bet_size_pred = self.advantage_net(
                        state_tensor.unsqueeze(0), 
                        opponent_features.unsqueeze(0)
                    )
                    advantages = advantages[0].cpu().numpy()
                    bet_size_multiplier = bet_size_pred[0][0].item()
                else:
                    advantages, bet_size_pred = self.advantage_net(state_tensor.unsqueeze(0))
                    advantages = advantages[0].cpu().numpy()
                    bet_size_multiplier = bet_size_pred[0][0].item()
                
            # Use regret matching to compute strategy for action types
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_types:
                advantages_masked[a] = max(advantages[a], 0)
                
            # Choose an action based on the strategy
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_types:
                    strategy[a] = 1.0 / len(legal_action_types)
            
            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_type in legal_action_types:
                try:
                    # Use the predicted bet size for raise actions
                    if action_type == 2:  # Raise
                        pokers_action = self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
                    else:
                        pokers_action = self.action_type_to_pokers_action(action_type, state)
                    
                    new_state = state.apply_action(pokers_action)
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        if STRICT_CHECKING:
                            log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Player: {current_player}, Action: {pokers_action.action}, Amount: {pokers_action.amount if pokers_action.action == pkrs.ActionEnum.Raise else 'N/A'}")
                            print(f"Current bet: {state.players_state[current_player].bet_chips}, Stake: {state.players_state[current_player].stake}")
                        continue  # Skip this action and try others
                        
                    action_values[action_type] = self.cfr_traverse(new_state, iteration, opponents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_type}: {e}")
                    action_values[action_type] = 0
                    if STRICT_CHECKING:
                        raise  # Re-raise in strict mode
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
            
            # Calculate normalization factor
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_type in legal_action_types:
                # Calculate regret
                regret = action_values[action_type] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                # Apply scaling
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                weighted_regret = clipped_regret * scale_factor
                
                # Store in prioritized memory with regret magnitude as priority
                priority = abs(weighted_regret) + 0.01  # Add small constant to ensure non-zero priority
                
                # For raise actions, store the bet size multiplier
                if action_type == 2:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id), 
                         opponent_features.cpu().numpy() if opponent_features is not None else np.zeros(20),
                         action_type, 
                         bet_size_multiplier, 
                         weighted_regret),
                        priority
                    )
                else:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id),
                         opponent_features.cpu().numpy() if opponent_features is not None else np.zeros(20),
                         action_type, 
                         0.0, 
                         weighted_regret),
                        priority
                    )
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            # Store strategy memory with opponent features if available
            self.strategy_memory.append((
                encode_state(state, self.player_id),
                opponent_features.cpu().numpy() if opponent_features is not None else np.zeros(20),
                strategy_full,
                bet_size_multiplier if 2 in legal_action_types else 0.0,
                iteration
            ))
            
            return ev
            
        # If it's another player's turn (model opponent or random agent)
        else:
            try:
                # Get the opponent object
                opponent = opponents[current_player]
                
                # Handle the case if we have no opponent at this position (shouldn't happen)
                if opponent is None:
                    if VERBOSE:
                        print(f"WARNING: No opponent at position {current_player}, using random action")
                    # Create a temporary random agent for this position
                    from src.training.train_with_opponent_modeling import RandomAgent
                    opponent = RandomAgent(current_player)
                
                # Let the opponent choose an action
                action = opponent.choose_action(state)
                
                # Record this action for opponent modeling
                # First, determine which action ID it corresponds to
                if action.action == pkrs.ActionEnum.Fold:
                    action_id = 0
                elif action.action == pkrs.ActionEnum.Check or action.action == pkrs.ActionEnum.Call:
                    action_id = 1
                elif action.action == pkrs.ActionEnum.Raise:
                    # Determine which raise size it's closest to
                    if action.amount <= state.pot * 0.75:
                        action_id = 2  # 0.5x pot raise
                    else:
                        action_id = 3  # 1x pot raise
                else:
                    action_id = 1  # Default to call if unrecognized
                
                # Record the action
                self.record_opponent_action(state, action_id, current_player)
                
                # Apply the action
                new_state = state.apply_action(action)
                
                # Check if the action was valid
                if new_state.status != pkrs.StateStatus.Ok:
                    if VERBOSE:
                        print(f"WARNING: Opponent made invalid action at depth {depth}. Status: {new_state.status}")
                    return 0
                    
                return self.cfr_traverse(new_state, iteration, opponents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in opponent traversal: {e}")
                return 0
    
    def train_advantage_network(self, batch_size=128, epochs=3):
        """Train the advantage network using collected samples with opponent modeling."""
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from prioritized memory
            batch, indices, weights = self.advantage_memory.sample(batch_size)
            states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass with opponent features
            action_advantages, bet_size_preds = self.advantage_net(state_tensors, opponent_feature_tensors)
            
            # Compute action type loss (for all actions)
            predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * weight_tensors).mean()
            
            # Compute bet sizing loss (only for raise actions)
            raise_mask = (action_type_tensors == 2)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weight_tensors[raise_indices]
                
                bet_size_loss = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = (bet_size_loss.squeeze() * raise_weights).mean()
                
                # Combine losses
                loss = weighted_action_loss + weighted_bet_size_loss
            else:
                loss = weighted_action_loss
            
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
            
            self.optimizer.step()
            
            # Update priorities based on new TD errors
            with torch.no_grad():
                new_action_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                
                new_priorities = new_action_errors.detach().cpu().numpy()
                if raise_mask.sum() > 0:
                    # If we have raise actions, incorporate their loss in the priorities
                    new_bet_errors = torch.zeros_like(new_action_errors)
                    new_bet_errors[raise_mask] = F.smooth_l1_loss(raise_bet_preds, raise_bet_targets, reduction='none').squeeze()
                    new_priorities += new_bet_errors.detach().cpu().numpy()
                
                # Update memory priorities
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, new_priorities[i] + 0.01)  # Small constant for stability
        
        avg_loss = total_loss / epochs
        return avg_loss
    
    def train_strategy_network(self, batch_size=128, epochs=3):
        """Train the strategy network using collected samples with opponent modeling."""
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.strategy_memory, batch_size)
            states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass with opponent features
            action_logits, bet_size_preds = self.strategy_net(state_tensors, opponent_feature_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)
            
            # Action type loss (weighted cross-entropy)
            action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
            
            # Bet size loss (only for states with raise actions)
            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                bet_size_loss = F.mse_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze())
                
                # Combine losses
                loss = action_loss + 0.5 * weighted_bet_size_loss  # Less weight on bet sizing to balance learning
            else:
                loss = action_loss
            
            total_loss += loss.item()
            
            self.strategy_optimizer.zero_grad()
            loss.backward()
            self.strategy_optimizer.step()
            
        return total_loss / epochs
    
    def train_opponent_modeling(self, batch_size=64, epochs=2):
        """Train the opponent modeling system."""
        return self.opponent_modeling.train(batch_size=batch_size, epochs=epochs)
    
    def choose_action(self, state, opponent_id=None):
        """
        Choose an action for the given state during actual play.
        Fixed to properly handle bet sizing according to poker rules.
        """
        legal_action_types = self.get_legal_action_types(state)
        
        if not legal_action_types:
            # Default to call if no legal actions
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)
                
        # Encode the base state
        state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).unsqueeze(0).to(self.device)
        
        # Get opponent features if available
        opponent_features = None
        if opponent_id is not None:
            opponent_features = self.opponent_modeling.get_opponent_features(opponent_id)
            opponent_features = torch.FloatTensor(opponent_features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use opponent features if available
            if opponent_features is not None:
                logits, bet_size_pred = self.strategy_net(state_tensor, opponent_features)
            else:
                logits, bet_size_pred = self.strategy_net(state_tensor)
                
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
        
        # Filter to only legal actions
        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)
        
        # Choose action based on probabilities
        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]
        
        # Use the predicted bet size for raise actions
        if action_type == 2:  # Raise
            return self.action_type_to_pokers_action(action_type, state, bet_size_multiplier)
        else:
            return self.action_type_to_pokers_action(action_type, state)
    
    def save_model(self, path_prefix):
        """Save the model to disk, including opponent modeling."""
        torch.save({
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'history_encoder': self.opponent_modeling.history_encoder.state_dict(),
            'opponent_model': self.opponent_modeling.opponent_model.state_dict()
        }, f"{path_prefix}_iteration_{self.iteration_count}.pt")
        
    def load_model(self, path):
        """Load the model from disk, including opponent modeling if available."""
        checkpoint = torch.load(path)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        
        # Load opponent modeling if available
        if 'history_encoder' in checkpoint and 'opponent_model' in checkpoint:
            self.opponent_modeling.history_encoder.load_state_dict(checkpoint['history_encoder'])
            self.opponent_modeling.opponent_model.load_state_dict(checkpoint['opponent_model'])