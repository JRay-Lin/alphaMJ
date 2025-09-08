"""Training environment wrapper for Mahjong Deep Q-Learning."""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from copy import deepcopy

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MahjongGame, MahjongRule
from modules.tile import MahjongTile
from modules.player import MahjongPlayer
from ai.dqn_model import MahjongStateEncoder
from ai.reward_system import Stage1RewardSystem, Stage2RewardSystem


class MahjongTrainingEnv:
    """
    Training environment wrapper for Mahjong game.
    
    Provides RL interface for training DQN agents:
    - State encoding for neural networks
    - Action masking for valid moves
    - Reward calculation and episode management
    - Statistics tracking
    """
    
    def __init__(self, rule_name: str = "standard", num_players: int = 4,
                 ai_player_indices: List[int] = None, reward_system: str = "stage1"):
        """
        Initialize training environment.
        
        Args:
            rule_name: Mahjong rule variant ("standard" or "taiwan")
            num_players: Number of players
            ai_player_indices: Indices of AI players (None = all AI)
            reward_system: Reward system to use ("stage1" or "stage2")
        """
        self.rule_name = rule_name
        self.num_players = num_players
        self.ai_player_indices = ai_player_indices or list(range(num_players))
        
        # Initialize game components
        self.rule = MahjongRule(rule_name, num_players)
        self.game = None
        self.state_encoder = MahjongStateEncoder()
        
        # Initialize reward system
        if reward_system == "stage1":
            self.reward_system = Stage1RewardSystem()
        elif reward_system == "stage2":
            self.reward_system = Stage2RewardSystem()
        else:
            raise ValueError(f"Unknown reward system: {reward_system}")
        
        # Episode tracking
        self.current_episode = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = {i: 0 for i in range(num_players)}
        self.total_games = 0
        
        # Training state
        self.last_states = {}
        self.last_actions = {}
    
    def reset(self) -> Dict[int, np.ndarray]:
        """
        Reset environment for new episode.
        
        Returns:
            Dictionary mapping player indices to initial states
        """
        # Create new game
        self.game = MahjongGame(self.rule)
        self.game.setup_game(ai_players=self.ai_player_indices)
        
        # Clear episode state
        self.last_states.clear()
        self.last_actions.clear()
        
        # Get initial states for all AI players
        initial_states = {}
        for player_idx in self.ai_player_indices:
            game_state = self.game.get_game_state_for_ai(player_idx)
            state_vector = self._convert_game_state_to_vector(game_state, player_idx)
            initial_states[player_idx] = state_vector
            self.last_states[player_idx] = state_vector
        
        return initial_states
    
    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], 
                                                   Dict[int, bool], Dict[str, Any]]:
        """
        Execute actions and return results.
        
        Args:
            actions: Dictionary mapping player indices to actions
            
        Returns:
            Tuple of (next_states, rewards, dones, info)
        """
        if not self.game:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Store actions
        for player_idx, action in actions.items():
            self.last_actions[player_idx] = action
        
        # Execute game turns until we need AI decisions
        next_states = {}
        rewards = {}
        dones = {}
        info = {'game_result': None, 'winner': None, 'turn_count': self.game.turn_count}
        
        try:
            # Process one turn at a time
            if not self.game.game_over:
                current_player = self.game.players[self.game.current_player]
                
                # First, player draws a tile to start their turn (13 → 14 tiles)
                print(f"\n--- {current_player.name}'s Turn ---")
                print(f"Before draw: {len(current_player.hand.hands)} tiles")
                if not self.game.draw_tile(current_player):
                    # Wall empty, game ends in draw
                    self.game.game_over = True
                    return next_states, rewards, dones, info
                print(f"After draw: {len(current_player.hand.hands)} tiles")
                print(f"Hand: {[str(tile) for tile in current_player.hand.hands]}")
                if current_player.hand.discards:
                    print(f"Discards: {[str(tile) for tile in current_player.hand.discards[-3:]]}")  # Last 3
                
                if current_player.index in self.ai_player_indices:
                    # AI player's turn - use provided action
                    if current_player.index in actions:
                        action = actions[current_player.index]
                        self._execute_ai_action(current_player, action)
                    else:
                        # No action provided - use random
                        self._execute_random_action(current_player)
                else:
                    # Non-AI player - use simple heuristic
                    self._execute_random_action(current_player)
            
            # Calculate rewards and states
            for player_idx in self.ai_player_indices:
                if self.game.game_over:
                    # Terminal rewards
                    reward = self._calculate_terminal_reward(player_idx)
                    dones[player_idx] = True
                    
                    # Final state (can be same as current)
                    game_state = self.game.get_game_state_for_ai(player_idx)
                    next_states[player_idx] = self._convert_game_state_to_vector(game_state, player_idx)
                else:
                    # Step reward
                    reward = self.reward_system.calculate_step_reward('discard', True, {})
                    dones[player_idx] = False
                    
                    # Next state
                    game_state = self.game.get_game_state_for_ai(player_idx)
                    next_states[player_idx] = self._convert_game_state_to_vector(game_state, player_idx)
                
                rewards[player_idx] = reward
            
            # Update info
            if self.game.game_over:
                info['game_result'] = 'win' if self.game.winner else 'draw'
                info['winner'] = self.game.winner.index if self.game.winner else -1
                self._update_episode_stats(info['winner'])
        
        except Exception as e:
            # Handle game errors gracefully
            print(f"Game error: {e}")
            for player_idx in self.ai_player_indices:
                rewards[player_idx] = -1.0
                dones[player_idx] = True
                if player_idx in self.last_states:
                    next_states[player_idx] = self.last_states[player_idx]
                else:
                    next_states[player_idx] = np.zeros(self.state_encoder.total_features)
        
        # Update stored states
        for player_idx, state in next_states.items():
            self.last_states[player_idx] = state
        
        return next_states, rewards, dones, info
    
    def _player_just_acted(self, player: MahjongPlayer) -> bool:
        """Check if player just performed a pong/chi/kong and shouldn't draw."""
        # For now, assume players should always draw at start of turn
        # This could be enhanced to track recent actions
        return False
    
    def _execute_ai_action(self, player: MahjongPlayer, action: int):
        """Execute AI action (discard tile at given index)."""
        if 0 <= action < len(player.hand.hands):
            discarded_tile = self.game.discard_tile(player, action)
            print(f"🎯 {player.name} discards: {discarded_tile}")
            print(f"After discard: {len(player.hand.hands)} tiles")
            print(f"Remaining hand: {[str(tile) for tile in player.hand.hands]}")
            if discarded_tile and not self.game.game_over:
                # Handle potential actions from other players
                action_taken = self.game.handle_discard_actions(discarded_tile, player.index)
                
                # Only advance turn if no other player took action
                if not action_taken and not self.game.game_over:
                    self.game.current_player = (self.game.current_player + 1) % self.num_players
                    self.game.turn_count += 1
    
    def _execute_random_action(self, player: MahjongPlayer):
        """Execute random action for non-AI players."""
        import random
        
        if player.hand.hands:
            action = random.randint(0, len(player.hand.hands) - 1)
            self._execute_ai_action(player, action)
    
    def _calculate_terminal_reward(self, player_idx: int) -> float:
        """Calculate reward for game end."""
        if not self.game.game_over:
            return 0.0
        
        if self.game.winner:
            if self.game.winner.index == player_idx:
                game_result = 'win'
            else:
                game_result = 'loss'
            winner_idx = self.game.winner.index
        else:
            game_result = 'draw'
            winner_idx = -1
        
        return self.reward_system.calculate_reward(
            game_result, player_idx, winner_idx, self.game.turn_count
        )
    
    def _convert_game_state_to_vector(self, game_state: Dict, player_idx: int) -> np.ndarray:
        """Convert game state dictionary to state vector."""
        # Convert tile names back to MahjongTile objects for encoding
        hand_tiles = []
        for tile_name in game_state.get('hand', []):
            # Parse tile name back to MahjongTile
            if '_' in tile_name:
                number_str, class_name = tile_name.split('_', 1)
                number = int(number_str)
            else:
                number = 0
                class_name = tile_name
            
            hand_tiles.append(MahjongTile(number, class_name))
        
        # Convert sets
        sets = []
        for meld_set_names in game_state.get('sets', []):
            meld_set = []
            for tile_name in meld_set_names:
                if '_' in tile_name:
                    number_str, class_name = tile_name.split('_', 1)
                    number = int(number_str)
                else:
                    number = 0
                    class_name = tile_name
                meld_set.append(MahjongTile(number, class_name))
            sets.append(meld_set)
        
        # Create state dictionary for encoder
        encoder_state = {
            'hand_tiles': hand_tiles,
            'sets': sets,
            'current_player': game_state.get('current_player', 0),
            'wall_remaining': game_state.get('wall_remaining', 0),
            'turn_count': game_state.get('turn_count', 0),
            'hand_size': game_state.get('hand_size', 0),
            'other_players': game_state.get('other_players', [])
        }
        
        return self.state_encoder.encode_state(encoder_state)
    
    def get_valid_actions(self, player_idx: int) -> List[int]:
        """Get list of valid actions for player."""
        if not self.game or player_idx >= len(self.game.players):
            return []
        
        player = self.game.players[player_idx]
        hand_size = len(player.hand.hands)
        
        # Valid actions are tile indices in hand
        return list(range(hand_size))
    
    def get_action_mask(self, player_idx: int) -> torch.Tensor:
        """Get action mask tensor for player."""
        valid_actions = self.get_valid_actions(player_idx)
        return self.state_encoder.create_action_mask(valid_actions, 14)
    
    def _update_episode_stats(self, winner_idx: int):
        """Update episode statistics."""
        self.total_games += 1
        
        if winner_idx >= 0:
            # Update win rates
            for i in range(self.num_players):
                if i == winner_idx:
                    self.win_rates[i] = ((self.win_rates[i] * (self.total_games - 1)) + 1) / self.total_games
                else:
                    self.win_rates[i] = (self.win_rates[i] * (self.total_games - 1)) / self.total_games
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_games': self.total_games,
            'win_rates': self.win_rates.copy(),
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }
    
    def set_reward_system(self, reward_system):
        """Set new reward system."""
        self.reward_system = reward_system


class MultiAgentTrainingEnv:
    """
    Multi-agent training environment for self-play training.
    
    Manages multiple DQN agents playing against each other.
    """
    
    def __init__(self, rule_name: str = "standard", reward_system: str = "stage1"):
        """
        Initialize multi-agent environment.
        
        Args:
            rule_name: Mahjong rule variant
            reward_system: Reward system type
        """
        self.base_env = MahjongTrainingEnv(
            rule_name=rule_name,
            num_players=4,
            ai_player_indices=[0, 1, 2, 3],
            reward_system=reward_system
        )
        
        self.agents = {}  # Will store agent instances
        
    def register_agent(self, player_idx: int, agent):
        """Register an agent for a player position."""
        self.agents[player_idx] = agent
    
    def run_episode(self) -> Dict[str, Any]:
        """
        Run one complete episode with all agents.
        
        Returns:
            Episode results and statistics
        """
        # Reset environment
        states = self.base_env.reset()
        episode_data = []
        step_count = 0
        max_steps = 200  # Reduced to encourage more decisive games
        
        while step_count < max_steps:
            # Get current player from the game
            if not self.base_env.game or self.base_env.game.game_over:
                break
                
            current_player_idx = self.base_env.game.current_player
            
            # Only get action from current player
            if current_player_idx in self.agents and current_player_idx in states:
                agent = self.agents[current_player_idx]
                
                # Get agent's device and create tensors on that device
                device = next(agent.q_network.parameters()).device
                state = torch.FloatTensor(states[current_player_idx]).to(device)
                action_mask = self.base_env.get_action_mask(current_player_idx).to(device)
                
                # Get action from current player's agent
                with torch.no_grad():
                    action, q_values = agent.get_action(state, action_mask, agent.epsilon)
                
                # Execute only the current player's action
                actions = {current_player_idx: action}
                next_states, rewards, dones, info = self.base_env.step(actions)
                
                # Store transition data for current player
                if current_player_idx in next_states:
                    episode_data.append({
                        'player_idx': current_player_idx,
                        'state': states[current_player_idx],
                        'action': action,
                        'reward': rewards.get(current_player_idx, 0),
                        'next_state': next_states[current_player_idx],
                        'done': dones.get(current_player_idx, False)
                    })
                
                # Update states
                states.update(next_states)
                
                # Check if episode is done
                if self.base_env.game.game_over or any(dones.values()):
                    # Add final transitions for all other players
                    for player_idx in self.agents.keys():
                        if player_idx != current_player_idx and player_idx in states:
                            final_reward = rewards.get(player_idx, 0)
                            episode_data.append({
                                'player_idx': player_idx,
                                'state': states[player_idx],
                                'action': 0,  # Dummy action
                                'reward': final_reward,
                                'next_state': states[player_idx],  # Same state
                                'done': True
                            })
                    break
            else:
                # Fallback: advance game with random action
                try:
                    actions = {current_player_idx: 0}
                    next_states, rewards, dones, info = self.base_env.step(actions)
                    states.update(next_states)
                    if self.base_env.game.game_over:
                        break
                except Exception as e:
                    print(f"Game step error: {e}")
                    break
            
            step_count += 1
        
        return {
            'episode_data': episode_data,
            'info': info,
            'statistics': self.base_env.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return self.base_env.get_statistics()