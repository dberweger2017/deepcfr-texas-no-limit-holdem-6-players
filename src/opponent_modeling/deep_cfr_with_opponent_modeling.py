# deep_cfr_with_opponent_modeling.py
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

class EnhancedPokerNetwork(nn.Module):
    """
    Enhanced network that can incorporate opponent modeling features.
    """
    def __init__(self, input_size=500, opponent_feature_size=20, hidden_size=256, num_actions=4):
        super().__init__()
        # Standard game state processing
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Process opponent features
        self.opponent_fc = nn.Linear(opponent_feature_size, hidden_size // 2)
        
        # Combined processing after opponent features are incorporated
        self.fc3 = nn.Linear(hidden_size + hidden_size // 2, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, num_actions)
        
    def forward(self, state_input, opponent_features=None):
        # Process game state
        x = F.relu(self.fc1(state_input))
        x = F.relu(self.fc2(x))
        
        # If opponent features are provided, incorporate them
        if opponent_features is not None:
            opponent_encoding = F.relu(self.opponent_fc(opponent_features))
            x = torch.cat([x, opponent_encoding], dim=1)
        else:
            # If no opponent features, use zeros
            batch_size = state_input.size(0)
            x = torch.cat([x, torch.zeros(batch_size, self.opponent_fc.out_features, device=state_input.device)], dim=1)
        
        # Continue processing the combined features
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)

class DeepCFRAgentWithOpponentModeling:
    def __init__(self, player_id=0, num_players=6, memory_size=200000, device='cpu'):
        self.player_id = player_id
        self.num_players = num_players
        self.device = device
        
        # Define action space (Fold, Check/Call, Raise 0.5x pot, Raise 1x pot)
        self.num_actions = 4
        
        # Calculate input size based on state encoding
        input_size = 52 + 52 + 5 + 1 + num_players + num_players + num_players*4 + 1 + 4 + 5
        
        # Create advantage network with opponent modeling capabilities
        self.advantage_net = EnhancedPokerNetwork(
            input_size=input_size, 
            opponent_feature_size=20, 
            hidden_size=256, 
            num_actions=self.num_actions
        ).to(device)
        
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=0.0001, weight_decay=1e-5)
        
        # Create memory buffer
        self.advantage_memory = deque(maxlen=memory_size)
        
        # Strategy network (also with opponent modeling)
        self.strategy_net = EnhancedPokerNetwork(
            input_size=input_size, 
            opponent_feature_size=20, 
            hidden_size=256, 
            num_actions=self.num_actions
        ).to(device)
        
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.0001, weight_decay=1e-5)
        self.strategy_memory = deque(maxlen=memory_size)
        
        # Initialize opponent modeling system
        self.opponent_modeling = OpponentModelingSystem(
            max_history_per_opponent=20,
            action_dim=4,
            state_dim=20,
            device=device
        )
        
        # For tracking game history during play
        self.current_game_history = {}  # Maps opponent_id -> (actions, contexts)
        
        # For keeping statistics
        self.iteration_count = 0
        
        # Regret normalization tracker
        self.max_regret_seen = 1.0
    
    def action_id_to_pokers_action(self, action_id, state):
        """Convert our action ID to Pokers action with fixed bet calculation."""
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
                    return pkrs.Action(pkrs.ActionEnum.Call)
                
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
                    print(f"\nRAISE CALCULATION DETAILS:")
                    print(f"  Player ID: {state.current_player}")
                    print(f"  Action ID: {action_id}")
                    print(f"  Current bet: {current_bet}")
                    print(f"  Available stake: {available_stake}")
                    print(f"  Min bet: {state.min_bet}")
                    print(f"  Call amount: {call_amount}")
                    print(f"  Pot size: {state.pot}")
                    print(f"  Additional raise amount: {additional_amount}")
                    print(f"  Total player bet will be: {current_bet + call_amount + additional_amount}")
                
                # Return the action with the additional amount
                return pkrs.Action(pkrs.ActionEnum.Raise, additional_amount)
                        
            else:
                raise ValueError(f"Unknown action ID: {action_id}")
                        
        except Exception as e:
            if VERBOSE:
                print(f"ERROR creating action {action_id}: {e}")
                print(f"State: current_player={state.current_player}, legal_actions={state.legal_actions}")
                print(f"Player stake: {state.players_state[state.current_player].stake}")
            # Fall back to call as safe option
            if pkrs.ActionEnum.Call in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Call)
            elif pkrs.ActionEnum.Check in state.legal_actions:
                return pkrs.Action(pkrs.ActionEnum.Check)
            else:
                return pkrs.Action(pkrs.ActionEnum.Fold)

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
    
    def extract_state_context(self, state):
        """
        Extract a simplified state context for opponent modeling.
        Returns a compact representation of the current state.
        """
        # For simplicity, we'll use a fixed-size feature vector
        # In practice, you might want to engineer more sophisticated features
        context = np.zeros(20)
        
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
        
        # Last two slots for custom features
        
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
        action_encoded = np.zeros(self.num_actions)
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
        Traverse the game tree using external sampling MCCFR.
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
            legal_action_ids = self.get_legal_action_ids(state)
            
            if not legal_action_ids:
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
            
            # Get advantages from network
            with torch.no_grad():
                # Use opponent features if available
                if opponent_features is not None:
                    advantages = self.advantage_net(
                        state_tensor.unsqueeze(0), 
                        opponent_features.unsqueeze(0)
                    )[0]
                else:
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
                        
                    action_values[action_id] = self.cfr_traverse(new_state, iteration, opponents, depth + 1)
                except Exception as e:
                    if VERBOSE:
                        print(f"ERROR in traversal for action {action_id}: {e}")
                    action_values[action_id] = 0
            
            # Compute counterfactual regrets and add to memory
            ev = sum(strategy[a] * action_values[a] for a in legal_action_ids)
            
            # Calculate normalization factor
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            
            for action_id in legal_action_ids:
                # Calculate regret
                regret = action_values[action_id] - ev
                
                # Normalize and clip regret
                normalized_regret = regret / max_abs_val
                clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
                
                # Apply scaling
                scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0
                
                # Store advantage memory with opponent features if available
                if opponent_features is not None:
                    self.advantage_memory.append((
                        encode_state(state, self.player_id),
                        opponent_features.cpu().numpy(),
                        action_id,
                        clipped_regret * scale_factor
                    ))
                else:
                    self.advantage_memory.append((
                        encode_state(state, self.player_id),
                        np.zeros(20),  # Default opponent features
                        action_id,
                        clipped_regret * scale_factor
                    ))
            
            # Add to strategy memory
            strategy_full = np.zeros(self.num_actions)
            for a in legal_action_ids:
                strategy_full[a] = strategy[a]
            
            # Store strategy memory with opponent features if available
            if opponent_features is not None:
                self.strategy_memory.append((
                    encode_state(state, self.player_id),
                    opponent_features.cpu().numpy(),
                    strategy_full,
                    iteration
                ))
            else:
                self.strategy_memory.append((
                    encode_state(state, self.player_id),
                    np.zeros(20),  # Default opponent features
                    strategy_full,
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
            # Sample batch from memory
            batch = random.sample(self.advantage_memory, batch_size)
            states, opponent_features, actions, regrets = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_tensors = torch.LongTensor(np.array(actions)).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            
            # Forward pass with opponent features
            advantages = self.advantage_net(state_tensors, opponent_feature_tensors)
            predicted_regrets = advantages.gather(1, action_tensors.unsqueeze(1)).squeeze(1)
            
            # Compute Huber loss
            loss = F.smooth_l1_loss(predicted_regrets, regret_tensors)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
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
            states, opponent_features, strategies, iterations = zip(*batch)
            
            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            
            # Weight samples by iteration (Linear CFR)
            weights = iteration_tensors / torch.sum(iteration_tensors)
            
            # Forward pass with opponent features
            logits = self.strategy_net(state_tensors, opponent_feature_tensors)
            predicted_strategies = F.softmax(logits, dim=1)
            
            # Compute weighted cross-entropy loss
            loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
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
        legal_action_ids = self.get_legal_action_ids(state)
        
        if not legal_action_ids:
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
                logits = self.strategy_net(state_tensor, opponent_features)
            else:
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
        
        # Convert to poker action with proper bet calculations
        return self.action_id_to_pokers_action(action_id, state)
    
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