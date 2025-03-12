# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from src.core.model import PokerNetwork, encode_state, VERBOSE, set_verbose
from collections import deque

class DeepCFRAgent:
    def __init__(self, player_id=0, num_players=6, memory_size=200000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action space (Fold, Check/Call, Raise 0.5x pot, Raise 1x pot)
        self.num_actions = 4
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Create advantage network
        self.advantage_net = PokerNetwork(input_size=input_size, num_actions=self.num_actions).to(device)
        
        # Use a smaller learning rate for more stable training
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # Create memory buffer
        self.advantage_memory = deque(maxlen=memory_size)
        
        # Strategy network
        self.strategy_net = PokerNetwork(input_size=input_size, num_actions=self.num_actions).to(device)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.0001, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # For keeping statistics
        self.iteration_count = 0
        
        # Regret normalization tracker
        self.max_regret_seen = 1.0

    def action_id_to_pokers_action(self, action_id, state):
        """Convert our action ID to Pokers action."""
        try:
            if action_id == 0:  # Fold
                return pkrs.Action(pkrs.ActionEnum.Fold)
            elif action_id == 1:  # Check/Call
                if pkrs.ActionEnum.Check in state.legal_actions:
                    return pkrs.Action(pkrs.ActionEnum.Check)
                else:
                    return pkrs.Action(pkrs.ActionEnum.Call)
            elif action_id == 2 or action_id == 3:  # Raise actions
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
                
                # Calculate what's needed to call (match the current min_bet)
                call_amount = max(0, state.min_bet - current_bet)
                
                # Before applying minimum raise enforcement, check if player can call at all
                if available_stake <= call_amount:
                    # Player can't even call, so go all-in
                    if VERBOSE:
                        print(f"All-in raise with {available_stake} chips (below min_bet {state.min_bet})")
                    return pkrs.Action(pkrs.ActionEnum.Raise, available_stake)
                
                # Check if player can actually afford to raise
                remaining_stake = available_stake - call_amount
                if remaining_stake <= 0:
                    # Can't raise at all, just call
                    return pkrs.Action(pkrs.ActionEnum.Call)
                
                # Calculate target raise amounts
                pot_size = max(1.0, state.pot)  # Avoid division by zero
                if action_id == 2:  # 0.5x pot
                    target_raise = pot_size * 0.5
                else:  # 1x pot
                    target_raise = pot_size
                
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
                    print(f"Creating raise action ({'0.5x pot' if action_id == 2 else '1x pot'}): amount={additional_amount}, pot={state.pot}")
                
                return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)
                    
            else:
                raise ValueError(f"Unknown action ID: {action_id}")
        except Exception as e:
            if VERBOSE:
                print(f"ERROR creating action {action_id}: {e}")
                print(f"State: current_player={state.current_player}, legal_actions={state.legal_actions}")
                print(f"Player stake: {state.players_state[state.current_player].stake}")
            raise

    def get_legal_action_ids(self, state):
        """Get the legal action IDs for the current state."""
        legal_action_ids = []
        
        # Check each action type
        for action_enum in state.legal_actions:
            if action_enum == pkrs.ActionEnum.Fold:
                legal_action_ids.append(0)
            elif action_enum == pkrs.ActionEnum.Check or action_enum == pkrs.ActionEnum.Call:
                legal_action_ids.append(1)
            elif action_enum == pkrs.ActionEnum.Raise:
                # Add our different raise sizes
                legal_action_ids.append(2)  # 0.5x pot
                legal_action_ids.append(3)  # 1x pot
        
        return legal_action_ids

    def cfr_traverse(self, state, iteration, random_agents, depth=0):
        """
        Traverse the game tree using external sampling MCCFR.
        Returns the expected value for the traverser.
        
        Args:
            state: Current game state
            iteration: Current training iteration
            random_agents: List of random agents for other players
            depth: Current recursion depth
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
            legal_action_ids = self.get_legal_action_ids(state)
            
            if not legal_action_ids:
                if VERBOSE:
                    print(f"WARNING: No legal actions found for player {current_player} at depth {depth}")
                return 0
                
            state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).to(self.device)
            
            # Get advantages from network
            with torch.no_grad():
                advantages = self.advantage_net(state_tensor.unsqueeze(0))[0]
                
            # Use regret matching to compute strategy
            advantages_np = advantages.cpu().numpy()
            advantages_masked = np.zeros(self.num_actions)
            for a in legal_action_ids:
                advantages_masked[a] = max(advantages_np[a], 0)
                
            # Choose an action based on the strategy
            if sum(advantages_masked) > 0:
                strategy = advantages_masked / sum(advantages_masked)
            else:
                strategy = np.zeros(self.num_actions)
                for a in legal_action_ids:
                    strategy[a] = 1.0 / len(legal_action_ids)
            
            # Choose actions and traverse
            action_values = np.zeros(self.num_actions)
            for action_id in legal_action_ids:
                try:
                    pokers_action = self.action_id_to_pokers_action(action_id, state)
                    new_state = state.apply_action(pokers_action)
                    
                    # Check if the action was valid
                    if new_state.status != pkrs.StateStatus.Ok:
                        if VERBOSE:
                            print(f"WARNING: Invalid action {action_id} at depth {depth}. Status: {new_state.status}")
                            print(f"Player: {current_player}, Action: {pokers_action.action}, Amount: {pokers_action.amount if pokers_action.action == pkrs.ActionEnum.Raise else 'N/A'}")
                            print(f"Current bet: {state.players_state[current_player].bet_chips}, Stake: {state.players_state[current_player].stake}")
                        continue  # Skip this action and try others
                        
                    action_values[action_id] = self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_id}: {e}")
                    action_values[action_id] = 0
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_ids)
            
            # Calculate normalization factor (max absolute action value)
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_id in legal_action_ids:
                # Calculate regret
                regret = action_values[action_id] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                # Apply a more stable scaling (sqrt of iteration instead of linear)
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                
                self.advantage_memory.append((
                    encode_state(state, self.player_id),
                    action_id,
                    clipped_regret * scale_factor
                ))
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_ids:
                strategy_full[a] = strategy[a]
            
            self.strategy_memory.append((
                encode_state(state, self.player_id),
                strategy_full,
                iteration  # Keep linear weighting for strategy (this is fine)
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
                    if VERBOSE:
                        raise f"Error detected of type {new_state.status}"
                        print(f"WARNING: Random agent made invalid action at depth {depth}. Status: {new_state.status}")
                    return 0
                    
                return self.cfr_traverse(new_state, iteration, random_agents, depth + 1)
            except Exception as e:
                if VERBOSE:
                    print(f"ERROR in random agent traversal: {e}")
                return 0

    def train_advantage_network(self, batch_size=128, epochs=3):
        """Train the advantage network using collected samples."""
        if len(self.advantage_memory) < batch_size:
            return 0
        
        self.advantage_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.advantage_memory, batch_size)
            states, actions, regrets = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            action_tensors = torch.LongTensor(np.array(actions)).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            
            # Forward pass
            advantages = self.advantage_net(state_tensors)
            predicted_regrets = advantages.gather(1, action_tensors.unsqueeze(1)).squeeze(1)
            
            # Compute Huber loss instead of MSE for better stability with outliers
            loss = F.smooth_l1_loss(predicted_regrets, regret_tensors)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        avg_loss = total_loss / epochs
        
        # Calculate some statistics on regrets for debugging
        if len(self.advantage_memory) > 100:
            # Sample regrets
            sample = random.sample(self.advantage_memory, 100)
            regrets = [r for _, _, r in sample]
            avg_regret = sum(regrets) / len(regrets)
            max_regret = max(regrets)
            min_regret = min(regrets)
            
            if VERBOSE:
                print(f"  Regret stats: min={min_regret:.4f}, max={max_regret:.4f}, avg={avg_regret:.4f}")
        
        return avg_loss

    def train_strategy_network(self, batch_size=128, epochs=3):
        """Train the strategy network using collected samples."""
        if len(self.strategy_memory) < batch_size:
            return 0
        
        self.strategy_net.train()
        total_loss = 0
        
        for _ in range(epochs):
            # Sample batch from memory
            batch = random.sample(self.strategy_memory, batch_size)
            states, strategies, iterations = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass
            logits = self.strategy_net(state_tensors)
            predicted_strategies = F.softmax(logits, dim=1)
            
            # Compute weighted cross-entropy loss
            loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
            total_loss += loss.item()
            
            self.strategy_optimizer.zero_grad()
            loss.backward()
            self.strategy_optimizer.step()
            
        return total_loss / epochs

    def choose_action(self, state):
        """Choose an action for the given state during actual play."""
        legal_action_ids = self.get_legal_action_ids(state)
        
        if not legal_action_ids:
            # Default to call if no legal actions (shouldn't happen)
            return pkrs.Action(pkrs.ActionEnum.Call)
            
        state_tensor = torch.FloatTensor(encode_state(state, self.player_id)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Filter to only legal actions
        legal_probs = np.array([probs[a] for a in legal_action_ids])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_ids)) / len(legal_action_ids)
        
        # Choose action based on probabilities
        action_idx = np.random.choice(len(legal_action_ids), p=legal_probs)
        action_id = legal_action_ids[action_idx]
        
        return self.action_id_to_pokers_action(action_id, state)

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