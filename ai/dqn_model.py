"""Deep Q-Network model for Mahjong AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network for Mahjong.

    Separates value and advantage estimation for better learning stability.
    The final Q-values are computed as: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(
        self, state_size: int, action_size: int, hidden_sizes: list = [512, 256, 128]
    ):
        """
        Initialize the Dueling DQN.

        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Shared feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_size = state_size

        for hidden_size in hidden_sizes:
            self.feature_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 128), nn.ReLU(), nn.Linear(128, 1)
        )

        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 128), nn.ReLU(), nn.Linear(128, action_size)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Batch of state vectors [batch_size, state_size]

        Returns:
            Q-values for each action [batch_size, action_size]
        """
        # Shared feature extraction
        x = state
        for layer in self.feature_layers:
            x = F.relu(layer(x))

        # Value and advantage streams
        value = self.value_stream(x)  # [batch_size, 1]
        advantage = self.advantage_stream(x)  # [batch_size, action_size]

        # Combine value and advantage using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
    ) -> Tuple[int, torch.Tensor]:
        """
        Select action using epsilon-greedy policy with action masking.

        Args:
            state: Single state vector [state_size]
            action_mask: Boolean mask for valid actions [action_size]
            epsilon: Exploration probability

        Returns:
            Tuple of (selected_action, q_values)
        """
        # Ensure state is on the correct device
        device = next(self.parameters()).device
        state = state.to(device)
        if action_mask is not None:
            action_mask = action_mask.to(device)
        
        if np.random.random() < epsilon:
            # Exploration: random valid action
            if action_mask is not None:
                valid_actions = torch.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions.cpu().numpy())
                else:
                    action = 0  # Fallback
            else:
                action = np.random.randint(self.action_size)

            # Still compute Q-values for consistency
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))
        else:
            # Exploitation: best valid action
            with torch.no_grad():
                q_values = self.forward(state.unsqueeze(0))

                if action_mask is not None:
                    # Mask invalid actions with large negative values
                    masked_q_values = q_values.clone()
                    masked_q_values[0, ~action_mask] = -float("inf")
                    action = masked_q_values.argmax().item()
                else:
                    action = q_values.argmax().item()

        return action, q_values.squeeze(0)


class MahjongStateEncoder:
    """
    Encodes Mahjong game state into neural network input format.

    State representation includes:
    - Player's hand tiles (one-hot encoded)
    - Exposed sets information
    - Game context (current player, wall remaining, etc.)
    - Other players' visible information
    """

    def __init__(self):
        # Mahjong tile types mapping
        self.tile_to_index = self._create_tile_mapping()
        self.index_to_tile = {v: k for k, v in self.tile_to_index.items()}
        self.num_tile_types = len(self.tile_to_index)

    def _create_tile_mapping(self) -> dict:
        """Create mapping from tile names to indices."""
        mapping = {}
        idx = 0

        # Numbered tiles (萬, 筒, 條)
        for suit in ["萬", "筒", "條"]:
            for num in range(1, 10):
                mapping[f"{num}_{suit}"] = idx
                idx += 1

        # Honor tiles (風)
        for wind in ["東", "南", "西", "北"]:
            mapping[wind] = idx
            idx += 1

        # Dragon tiles (元)
        for dragon in ["中", "發", "白"]:
            mapping[dragon] = idx
            idx += 1

        return mapping

    def encode_hand(self, tiles: list) -> np.ndarray:
        """
        Encode hand tiles as count vector.

        Args:
            tiles: List of MahjongTile objects

        Returns:
            Count vector [num_tile_types * 4] for up to 4 of each tile
        """
        # Count each tile type
        tile_counts = {}
        for tile in tiles:
            tile_name = tile.name
            if tile_name in self.tile_to_index:
                tile_counts[tile_name] = tile_counts.get(tile_name, 0) + 1

        # Create one-hot style encoding for counts (0, 1, 2, 3, 4)
        encoding = np.zeros(self.num_tile_types * 4)

        for tile_name, count in tile_counts.items():
            if tile_name in self.tile_to_index:
                base_idx = self.tile_to_index[tile_name] * 4
                # One-hot encode the count (capped at 4)
                count = min(count, 4)
                if count > 0:
                    encoding[base_idx + count - 1] = 1.0

        return encoding

    def encode_sets(self, sets: list) -> np.ndarray:
        """
        Encode exposed sets.

        Args:
            sets: List of meld sets (each set is list of tiles)

        Returns:
            Sets encoding vector
        """
        # Simple encoding: count of each set type per tile
        encoding = np.zeros(self.num_tile_types * 3)  # pung, chow, kong

        for meld_set in sets:
            if not meld_set:
                continue

            set_size = len(meld_set)
            first_tile = meld_set[0]

            if first_tile.name in self.tile_to_index:
                tile_idx = self.tile_to_index[first_tile.name]

                if set_size == 4:  # Kong
                    encoding[tile_idx * 3 + 2] = 1.0
                elif set_size == 3:
                    # Check if pung or chow
                    if all(tile.name == first_tile.name for tile in meld_set):
                        # Pung
                        encoding[tile_idx * 3 + 0] = 1.0
                    else:
                        # Chow
                        encoding[tile_idx * 3 + 1] = 1.0

        return encoding

    def encode_game_context(self, game_state: dict) -> np.ndarray:
        """
        Encode game context information.

        Args:
            game_state: Dictionary with game information

        Returns:
            Context encoding vector
        """
        context = np.zeros(20)

        # Player index (one-hot)
        player_idx = game_state.get("current_player", 0)
        if player_idx < 4:
            context[player_idx] = 1.0

        # Wall remaining (normalized)
        wall_remaining = game_state.get("wall_remaining", 0)
        context[4] = wall_remaining / 144.0

        # Turn count (normalized)
        turn_count = game_state.get("turn_count", 0)
        context[5] = min(turn_count / 100.0, 1.0)

        # Hand size (normalized)
        hand_size = game_state.get("hand_size", 13)
        context[6] = hand_size / 14.0

        # Number of exposed sets
        num_sets = len(game_state.get("sets", []))
        context[7] = num_sets / 5.0

        return context

    def encode_other_players(self, other_players: list) -> np.ndarray:
        """
        Encode information about other players.

        Args:
            other_players: List of other player information

        Returns:
            Other players encoding vector
        """
        encoding = np.zeros(3 * 30)  # 3 other players, 30 features each

        for i, player_info in enumerate(other_players[:3]):
            base_idx = i * 30

            # Hand size
            hand_size = player_info.get("hand_size", 13)
            encoding[base_idx] = hand_size / 14.0

            # Number of exposed sets
            num_sets = len(player_info.get("sets", []))
            encoding[base_idx + 1] = num_sets / 5.0

            # Recent discards (simplified)
            discards = player_info.get("discards", [])
            recent_discards = discards[-5:]  # Last 5 discards
            for j, discard in enumerate(recent_discards):
                if discard in self.tile_to_index and j < 10:
                    discard_idx = self.tile_to_index[discard]
                    encoding[base_idx + 2 + j] = discard_idx / self.num_tile_types

        return encoding

    def encode_state(self, game_state: dict) -> np.ndarray:
        """
        Encode complete game state.

        Args:
            game_state: Complete game state dictionary

        Returns:
            State vector for neural network
        """
        # Extract components
        hand_tiles = game_state.get("hand_tiles", [])
        sets = game_state.get("sets", [])
        other_players = game_state.get("other_players", [])

        # Encode each component
        hand_encoding = self.encode_hand(hand_tiles)
        sets_encoding = self.encode_sets(sets)
        context_encoding = self.encode_game_context(game_state)
        others_encoding = self.encode_other_players(other_players)

        # Combine all encodings
        state_vector = np.concatenate(
            [hand_encoding, sets_encoding, context_encoding, others_encoding]
        )

        return state_vector.astype(np.float32)

    def create_action_mask(self, valid_actions: list, hand_size: int) -> torch.Tensor:
        """
        Create action mask for valid discard actions.

        Args:
            valid_actions: List of valid tile indices to discard
            hand_size: Current hand size

        Returns:
            Boolean mask tensor
        """
        mask = torch.zeros(14, dtype=torch.bool)

        for action in valid_actions:
            if 0 <= action < hand_size:
                mask[action] = True

        return mask
