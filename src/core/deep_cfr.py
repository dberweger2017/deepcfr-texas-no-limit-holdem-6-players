# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from collections import deque
from src.core.model import PokerNetwork, encode_state, VERBOSE, set_verbose
from src.utils.settings import STRICT_CHECKING

class PrioritizedMemory:
    """Memory buffer with prioritized experience replay."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, experience, priority):
        """Add a new experience to memory with its priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            # Replace the oldest entry
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, priority_scale=0.7):
        """Sample a batch of experiences based on their priorities."""
        # Scale priorities to adjust the importance of high-priority experiences
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * sample_probabilities[indices]) ** -1
        weights = weights / weights.max()  # Normalize
        
        return samples, indices, weights
        
    def update_priority(self, index, priority):
        """Update the priority of an experience."""
        self.priorities[index] = priority
        
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.buffer)

class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=6, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action types (Fold, Check/Call, Raise)
        self.num_actions = 3
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Create advantage network with bet sizing
        self.advantage_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        
        # Use a smaller learning rate for more stable training
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=0.00005, weight_decay=1e-5)
        
        # Create prioritized memory buffer
        self.advantage_memory = PrioritizedMemory(memory_size)
        
        # Strategy network
        self.strategy_net = PokerNetwork(input_size=input_size, hidden_size=256, num_actions=self.num_actions).to(device)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # For keeping statistics
        self.iteration_count = 0
        
        # Regret normalization tracker
        self.max_regret_seen = 1.0
        
        # Bet sizing bounds (as multipliers of pot)
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0

    def action_type_to_pokers_action(self, action_type, state, bet_size_multiplier=None):
        """Convert action type and optional bet size to Pokers action."""
        try:
            if action_type == 0:  # Fold
                return pkrs.Action(pkrs.ActionEnum.Fold)
            elif action_type == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                else:
                    return pkrs.Action(pkrs.ActionEnum.Call)
            elif action_type == 2:  # Raise
                if pkrs.ActionEnum.Raise not in state.legal_actions:
                    if pkrs.ActionEnum.Call in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Call)
                    elif pkrs.ActionEnum.Check in state.legal_actions:
                        return pkrs.Action(pkrs.ActionEnum.Check)
                    else:
                        return pkrs.Action(pkrs.ActionEnum.Fold)
                    
                # Get current player state
                player_state = state.players_state[state.current_player]
                current_bet = player_state.bet_chips
                available_stake = player_state.stake
                
                # Calculate what's needed to call
                call_amount = max(0, state.min_bet - current_bet)
                
                # If player can't even call, go all-in
                if available_stake <= call_amount:
                    if VERBOSE:
                        print(f"All-in raise with {available_stake} chips (below min_bet {state.min_bet})")
                    return pkrs.Action(pkrs.ActionEnum.Raise, available_stake)
                
                # Check if player can actually afford to raise
                remaining_stake = available_stake - call_amount
                if remaining_stake <= 0:
                    # Can't raise at all, just call
                    return pkrs.Action(pkrs.ActionEnum.Call)
                
                # Calculate target raise amount based on pot multiplier
                pot_size = max(1.0, state.pot)
                if bet_size_multiplier is None:
                    # Default to 1x pot if no multiplier provided
                    bet_size_multiplier = 1.0
                
                # Ensure multiplier is within bounds
                bet_size_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_size_multiplier))
                target_raise = pot_size * bet_size_multiplier
                
                # Ensure minimum raise
                min_raise = 1.0
                if hasattr(state, 'bb'):
                    min_raise = state.bb
                    
                target_raise = max(target_raise, min_raise)
                
                # Ensure we don't exceed available stake
                additional_amount = min(target_raise, remaining_stake)
                
                # If we can't meet minimum raise, fall back to call
                if additional_amount < min_raise:
                    return pkrs.Action(pkrs.ActionEnum.Call)
                    
                if VERBOSE:
                    print(f"Creating raise action: {bet_size_multiplier}x pot, amount={additional_amount}, pot={state.pot}")
                
                return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)
                    
            else:
                raise ValueError(f"Unknown action type: {action_type}")
        except Exception as e:
            if VERBOSE:
                print(f"ERROR creating action {action_type}: {e}")
                print(f"State: current_player={state.current_player}, legal_actions={state.legal_actions}")
                print(f"Player stake: {state.players_state[state.current_player].stake}")
            raise

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

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        """
        Traverse the game tree using external sampling MCCFR with continuous bet sizing.
        Returns the expected value for the traverser.
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if VERBOSE:
                print(f"WARNING: Max recursion depth reached ({max_depth}). Returning zero value.")
            return 0
        
        if state.final_state:
            # Return payoff for the trained agent
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        # Debug information for the current state
        if VERBOSE and depth % 100 == 0:
            print(f"Depth: {depth}, Player: {current_player}, Stage: {state.stage}")
        
        # If it's the trained agent's turn
        if current_player == self.player_id:
            legal_action_types = self.get_legal_action_types(state)
            
            if not legal_action_types:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            # Get advantages and bet sizing prediction from network
            with torch.no_grad():
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
                        log_file = log_game_error(state, pokers_action, f"State status not OK ({new_state.status})")
                        if STRICT_CHECKING:
                            raise ValueError(f"State status not OK ({new_state.status}) during CFR traversal. Details logged to {log_file}")
                        elif VERBOSE:
                            print(f"WARNING: Invalid action {action_type} at depth {depth}. Status: {new_state.status}")
                            print(f"Player: {current_player}, Action: {pokers_action.action}, Amount: {pokers_action.amount if pokers_action.action == pkrs.ActionEnum.Raise else 'N/A'}")
                            print(f"Details logged to {log_file}")
                        continue  # Skip this action and try others in non-strict mode
                        
                    action_values[action_type] = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
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
                        (encode_state(state, self.player_id), action_type, bet_size_multiplier, weighted_regret),
                        priority
                    )
                else:
                    self.advantage_memory.add(
                        (encode_state(state, self.player_id), action_type, 0.0, weighted_regret),
                        priority
                    )
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                encode_state(state, self.player_id),
                strategy_full,
                bet_size_multiplier if 2 in legal_action_types else 0.0,
                iteration  # Keep linear weighting for strategy
            ))
            
            return ev
            
        # If it's another player's turn (random agent)
        else:
            try:
                # Let the random agent choose an action
                action = random_agents[current_player].choose_action(state)
                new_state = state.apply_action(action)
                
                # Check if the action was valid
                if new_state.status != pkrs.StateStatus.Ok:
                    log_file = log_game_error(state, action, f"State status not OK ({new_state.status})")
                    if STRICT_CHECKING:
                        raise ValueError(f"State status not OK ({new_state.status}) from random agent. Details logged to {log_file}")
                    if VERBOSE:
                        print(f"WARNING: Random agent made invalid action at depth {depth}. Status: {new_state.status}")
                        print(f"Details logged to {log_file}")
                    return 0
                    
                return self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in random agent traversal: {e}")
                if STRICT_CHECKING:
                    raise  # Re-raise in strict mode
                return 0

    def train_advantage_network(self, batch_size=128, epochs=3):
        """Train the advantage network using collected samples."""
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from prioritized memory
            batch, indices, weights = self.advantage_memory.sample(batch_size)
            states, action_types, bet_sizes, regrets = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass
            action_advantages, bet_size_preds = self.advantage_net(state_tensors)
            
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
            
            # Add gradient clipping to prevent exploding gradients
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
        
        return total_loss / epochs

    def train_strategy_network(self, batch_size=128, epochs=3):
        """Train the strategy network using collected samples."""
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.strategy_memory, batch_size)
            states, strategies, bet_sizes, iterations = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass
            action_logits, bet_size_preds = self.strategy_net(state_tensors)
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

    def choose_action(self, state):
        """Choose an action for the given state during actual play."""
        legal_action_types = self.get_legal_action_types(state)
        
        if not legal_action_types:
            # Default to call if no legal actions (shouldn't happen)
            return pkrs.Action(pkrs.ActionEnum.Call)
            
        state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
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
        """Save the model to disk."""
        torch.save({
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict()
        }, f"{path_prefix}_iteration_{self.iteration_count}.pt")
        
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])