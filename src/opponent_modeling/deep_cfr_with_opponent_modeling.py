# src/opponent_modeling/deep_cfr_with_opponent_modeling.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import pokers as pkrs
from collections import deque
from src.core import model as model_settings
from src.core.model import encode_state, set_verbose
from src.opponent_modeling.opponent_model import OpponentModelingSystem
from src.core.deep_cfr import (
    DeepCFRAgent,
    PrioritizedMemory,
    _resolve_model_save_path,
    linear_cfr_weights,
    traverse_agent_turn,
    weighted_sample_loss,
    weighted_strategy_cross_entropy,
)
from src.utils.checkpoints import (
    AGENT_TYPE_OPPONENT_MODELING,
    load_checkpoint,
    opponent_modeling_checkpoint_state,
    validate_checkpoint_compatibility,
)
from src.utils import settings
from src.utils.logging import apply_action_with_logging
from src.utils.traversal_diagnostics import (
    TraversalDiagnostics,
    TraversalFailure,
    fail_traversal,
    failure_context,
)

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
        self.local_training_iteration = 0
        self.traversal_diagnostics = TraversalDiagnostics()
        
        # Bet sizing bounds (as multipliers of pot)
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
    
    action_type_to_pokers_action = DeepCFRAgent.action_type_to_pokers_action
    get_legal_action_types = DeepCFRAgent.get_legal_action_types
    
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

    def get_table_opponent_features(self):
        """Average available opponent features into the fixed 20-feature input."""
        opponent_ids = [
            opponent_id
            for opponent_id in range(self.num_players)
            if opponent_id != self.player_id
        ]
        feature_rows = [
            self.opponent_modeling.get_opponent_features(opponent_id)
            for opponent_id in opponent_ids
            if opponent_id in self.opponent_modeling.opponent_histories
        ]
        if not feature_rows:
            return np.zeros(20, dtype=np.float32)
        return np.mean(np.array(feature_rows, dtype=np.float32), axis=0)

    def cfr_traverse(self, state, iteration, opponents, depth=0):
        """
        Traverse the game tree using external sampling MCCFR with continuous bet sizing.
        Modified to work with both RandomAgent and ModelAgent opponents.
        """
        # Add recursion depth protection
        max_depth = 1000
        if depth > max_depth:
            if model_settings.is_verbose():
                print(f"WARNING: Max recursion depth reached ({max_depth}). Raising traversal failure.")
            fail_traversal(
                self,
                "max_depth",
                **failure_context(
                    state=state,
                    depth=depth,
                    iteration=iteration,
                    player_id=self.player_id,
                    message=f"Max recursion depth reached ({max_depth})",
                ),
            )
        
        if state.final_state:
            # Record the end of the game for opponent modeling
            self.end_game_recording(state)
            # Return payoff for the trained agent
            return state.players_state[self.player_id].reward
        
        current_player = state.current_player
        
        # If it's the trained agent's turn
        if current_player == self.player_id:
            opponent_feature_array = self.get_table_opponent_features()
            return traverse_agent_turn(
                self,
                state,
                iteration,
                lambda new_state, next_depth: self.cfr_traverse(
                    new_state,
                    iteration,
                    opponents,
                    next_depth,
                ),
                depth=depth,
                verbose=model_settings.is_verbose(),
                opponent_features=opponent_feature_array,
                network_forward_fn=lambda state_tensor, opponent_feature_tensor: self.advantage_net(
                    state_tensor.unsqueeze(0),
                    opponent_feature_tensor.unsqueeze(0),
                ),
            )
            
        # If it's another player's turn (model opponent or random agent)
        else:
            try:
                # Get the opponent object
                opponent = opponents[current_player]
                
                # Handle the case if we have no opponent at this position (shouldn't happen)
                if opponent is None:
                    if model_settings.is_verbose():
                        print(f"WARNING: No opponent at position {current_player}. Raising traversal failure.")
                    fail_traversal(
                        self,
                        "opponent_missing",
                        **failure_context(
                            state=state,
                            depth=depth,
                            iteration=iteration,
                            player_id=self.player_id,
                            message=f"No opponent at position {current_player}",
                        ),
                    )
                
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
                new_state, log_file, status = apply_action_with_logging(
                    state,
                    action,
                    strict=settings.is_strict_checking(),
                )

                # Check if the action was valid
                if new_state is None:
                    if model_settings.is_verbose():
                        print(f"WARNING: Opponent made invalid action at depth {depth}. Status: {status}")
                        print(f"Details logged to {log_file}")
                    fail_traversal(
                        self,
                        "opponent_invalid_action",
                        **failure_context(
                            state=state,
                            depth=depth,
                            iteration=iteration,
                            player_id=self.player_id,
                            action=action,
                            status=status,
                            log_file=log_file,
                        ),
                    )
                    
                return self.cfr_traverse(new_state, iteration, opponents, depth + 1)
            except TraversalFailure:
                raise
            except Exception as e:
                if model_settings.is_verbose():
                    print(f"ERROR in opponent traversal: {e}")
                fail_traversal(
                    self,
                    "opponent_exception",
                    exception=e,
                    **failure_context(
                        state=state,
                        depth=depth,
                        iteration=iteration,
                        player_id=self.player_id,
                        message=str(e),
                    ),
                )
    
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
            iteration_tensors = torch.FloatTensor(iterations).to(self.device)
            
            # Weight samples by iteration (Linear CFR)
            weights = linear_cfr_weights(iteration_tensors, device=self.device)
            
            # Forward pass with opponent features
            action_logits, bet_size_preds = self.strategy_net(state_tensors, opponent_feature_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)
            
            # Action type loss (weighted cross-entropy)
            action_loss = weighted_strategy_cross_entropy(
                strategy_tensors,
                predicted_strategies,
                weights,
            )
            
            # Bet size loss (only for states with raise actions)
            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                bet_size_loss = F.mse_loss(
                    raise_bet_preds,
                    raise_bet_targets,
                    reduction='none',
                ).squeeze(1)
                weighted_bet_size_loss = weighted_sample_loss(bet_size_loss, raise_weights)
                
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
            return self.action_type_to_pokers_action(
                action_type,
                state,
                bet_size_multiplier,
                strict=settings.is_strict_checking(),
            )
        else:
            return self.action_type_to_pokers_action(
                action_type,
                state,
                strict=settings.is_strict_checking(),
            )
    
    def save_model(self, path_prefix):
        """Save the model to disk, including opponent modeling."""
        model_path = _resolve_model_save_path(path_prefix, self.iteration_count)
        torch.save(opponent_modeling_checkpoint_state(self), model_path)
        
    def load_model(self, path):
        """Load the model from disk, including opponent modeling if available."""
        checkpoint = load_checkpoint(path, map_location=self.device)
        validate_checkpoint_compatibility(
            checkpoint,
            expected_agent_type=AGENT_TYPE_OPPONENT_MODELING,
            expected_num_players=self.num_players,
        )
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        
        # Load opponent modeling if available
        if 'history_encoder' in checkpoint and 'opponent_model' in checkpoint:
            self.opponent_modeling.history_encoder.load_state_dict(checkpoint['history_encoder'])
            self.opponent_modeling.opponent_model.load_state_dict(checkpoint['opponent_model'])
