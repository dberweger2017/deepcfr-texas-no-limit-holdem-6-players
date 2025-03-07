# opponent_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class ActionHistoryEncoder(nn.Module):
    """
    Encodes a sequence of opponent actions using an RNN.
    """
    def __init__(self, action_dim=4, state_dim=20, hidden_dim=128, output_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Action embedding
        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        
        # State context embedding
        self.state_embedding = nn.Linear(state_dim, hidden_dim)
        
        # Combined embedding
        self.combined_embedding = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # GRU for sequence processing (more efficient than LSTM for this task)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, action_sequences, state_contexts):
        """
        Process a batch of action sequences with their state contexts.
        
        Args:
            action_sequences: Tensor of shape [batch_size, seq_len, action_dim]
            state_contexts: Tensor of shape [batch_size, seq_len, state_dim]
        
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = action_sequences.shape
        
        # Embed actions and states
        action_embedded = F.relu(self.action_embedding(action_sequences))
        state_embedded = F.relu(self.state_embedding(state_contexts))
        
        # Combine embeddings
        combined = torch.cat([action_embedded, state_embedded], dim=-1)
        combined = F.relu(self.combined_embedding(combined))
        
        # Process sequence with GRU
        output, hidden = self.gru(combined)
        
        # Use the final hidden state as the encoding
        encoding = F.relu(self.output(hidden.squeeze(0)))
        
        return encoding

class OpponentModel(nn.Module):
    """
    Model that predicts opponent tendencies based on their action history.
    """
    def __init__(self, input_size=64, hidden_size=128, output_size=20):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OpponentModelingSystem:
    """
    System to track opponent history and generate opponent models.
    """
    def __init__(self, max_history_per_opponent=20, action_dim=4, state_dim=20, device='cpu'):
        self.device = device
        self.max_history = max_history_per_opponent
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        # Initialize history storage
        self.opponent_histories = {}  # Maps opponent_id -> list of (action_sequence, state_context, outcome)
        
        # Initialize models
        self.history_encoder = ActionHistoryEncoder(
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_dim=128,
            output_dim=64
        ).to(device)
        
        self.opponent_model = OpponentModel(
            input_size=64,
            hidden_size=128,
            output_size=20  # Various tendencies we want to track
        ).to(device)
        
        # Optimizer
        params = list(self.history_encoder.parameters()) + list(self.opponent_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=0.001)
    
    def record_game(self, opponent_id, action_sequence, state_contexts, outcome):
        """
        Record a game history for an opponent.
        
        Args:
            opponent_id: Unique identifier for the opponent
            action_sequence: List of encoded actions
            state_contexts: List of encoded state contexts for each action
            outcome: Game outcome (e.g., chips won/lost)
        """
        if opponent_id not in self.opponent_histories:
            self.opponent_histories[opponent_id] = deque(maxlen=self.max_history)
        
        self.opponent_histories[opponent_id].append((action_sequence, state_contexts, outcome))
    
    def get_opponent_encoding(self, opponent_id):
        """
        Get the encoding for an opponent based on their history.
        Returns a zero tensor for new opponents.
        """
        if opponent_id not in self.opponent_histories or not self.opponent_histories[opponent_id]:
            # Return zero encoding for new opponents
            return torch.zeros(64, dtype=torch.float32, device=self.device)
        
        # Prepare batch for the encoder
        histories = self.opponent_histories[opponent_id]
        action_seqs = [h[0] for h in histories]
        state_contexts = [h[1] for h in histories]
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in action_seqs)
        
        # Pad action sequences - with explicit dtype
        padded_actions = torch.zeros(len(histories), max_len, self.action_dim, 
                                dtype=torch.float32, device=self.device)
        for i, seq in enumerate(action_seqs):
            # Convert to numpy array with explicit dtype first
            seq_array = np.array(seq, dtype=np.float32)
            padded_actions[i, :len(seq)] = torch.tensor(seq_array, 
                                                    dtype=torch.float32, device=self.device)
        
        # Pad state contexts - with explicit dtype
        padded_contexts = torch.zeros(len(histories), max_len, self.state_dim, 
                                    dtype=torch.float32, device=self.device)
        for i, context in enumerate(state_contexts):
            # Convert to numpy array with explicit dtype first
            context_array = np.array(context, dtype=np.float32)
            padded_contexts[i, :len(context)] = torch.tensor(context_array, 
                                                        dtype=torch.float32, device=self.device)
        
        # Get encoding from the history encoder
        with torch.no_grad():
            encoding = self.history_encoder(padded_actions, padded_contexts).mean(dim=0)  # Average over all games
        
        return encoding
    
    def get_opponent_features(self, opponent_id):
        """
        Get predicted features/tendencies for an opponent.
        """
        encoding = self.get_opponent_encoding(opponent_id)
        
        with torch.no_grad():
            features = self.opponent_model(encoding.unsqueeze(0)).squeeze(0)
        
        return features.cpu().numpy()
    
    def train(self, batch_size=32, epochs=1):
        """
        Train the opponent modeling system on recorded histories.
        """
        if not self.opponent_histories:
            return 0.0  # No data to train on
        
        total_loss = 0.0
        
        for _ in range(epochs):
            # Sample batch of opponent histories
            batch_opponents = np.random.choice(list(self.opponent_histories.keys()), 
                                            min(batch_size, len(self.opponent_histories)))
            
            all_encodings = []
            all_outcomes = []
            
            for opponent_id in batch_opponents:
                histories = self.opponent_histories[opponent_id]
                if not histories:
                    continue
                
                # Sample a game from this opponent's history
                game_idx = np.random.randint(0, len(histories))
                action_seq, state_context, outcome = histories[game_idx]
                
                # Convert sequences to numpy arrays first
                action_array = np.array(action_seq, dtype=np.float32)  # Explicitly use float32
                context_array = np.array(state_context, dtype=np.float32)  # Explicitly use float32
                
                # Prepare tensors - make sure to specify the dtype as float32
                action_tensor = torch.tensor(action_array, dtype=torch.float32, device=self.device).unsqueeze(0)
                context_tensor = torch.tensor(context_array, dtype=torch.float32, device=self.device).unsqueeze(0)
                
                # Get encoding
                encoding = self.history_encoder(action_tensor, context_tensor)
                all_encodings.append(encoding)
                all_outcomes.append(outcome)
            
            if not all_encodings:
                continue
                
            # Stack encodings and outcomes
            encodings_tensor = torch.cat(all_encodings, dim=0)
            outcomes_tensor = torch.tensor(all_outcomes, dtype=torch.float32, device=self.device)
            
            # Forward pass through opponent model
            predicted_features = self.opponent_model(encodings_tensor)
            
            # For simplicity, we'll just use the outcome as the target for now
            # In practice, you'd want more sophisticated tendencies to predict
            target = outcomes_tensor.unsqueeze(1).expand(-1, predicted_features.size(1))
            
            # Calculate loss
            loss = F.mse_loss(predicted_features, target)
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / epochs if epochs > 0 else 0.0