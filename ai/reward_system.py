"""Reward systems for Mahjong AI training."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from modules.winChecker import win_checker
from modules.tile import MahjongTile


class Stage1RewardSystem:
    """
    Stage 1 reward system focusing on basic win/loss outcomes.
    
    This system teaches the AI fundamental game knowledge:
    - Learning to win (+1 reward)
    - Avoiding losses (-0.5 reward)
    - Minimizing game length (small negative per turn)
    """
    
    def __init__(self, win_reward: float = 1.0, loss_reward: float = -0.5,
                 turn_penalty: float = -0.005, draw_reward: float = 0.0):
        """
        Initialize Stage 1 reward system.
        
        Args:
            win_reward: Reward for winning
            loss_reward: Penalty for losing
            turn_penalty: Small penalty per turn to encourage faster wins
            draw_reward: Reward for draw games
        """
        self.win_reward = win_reward
        self.loss_reward = loss_reward
        self.turn_penalty = turn_penalty
        self.draw_reward = draw_reward
    
    def calculate_reward(self, game_result: str, player_index: int, winner_index: int,
                        turn_count: int = 0, game_state: Dict = None) -> float:
        """
        Calculate reward for a game outcome.
        
        Args:
            game_result: 'win', 'loss', or 'draw'
            player_index: Index of the player being rewarded
            winner_index: Index of the winning player (-1 for draw)
            turn_count: Number of turns taken
            game_state: Additional game state information
            
        Returns:
            Reward value
        """
        if game_result == 'win':
            return self.win_reward
        elif game_result == 'loss':
            return self.loss_reward
        elif game_result == 'draw':
            return self.draw_reward
        else:
            # Ongoing game - apply turn penalty
            return self.turn_penalty
    
    def calculate_step_reward(self, action_type: str, action_success: bool,
                            game_state: Dict) -> float:
        """
        Calculate reward for individual actions during the game.
        
        Args:
            action_type: Type of action taken ('discard', 'chi', 'pong', 'kong', 'win')
            action_success: Whether the action was successful
            game_state: Current game state
            
        Returns:
            Step reward
        """
        # In Stage 1, we primarily rely on terminal rewards
        # Small penalties for unsuccessful actions
        if not action_success:
            return -0.05
        
        # Small positive reward for successful special actions
        if action_type in ['chi', 'pong', 'kong']:
            return 0.02
        
        # Regular discard - small turn penalty to encourage faster games
        return self.turn_penalty


class Stage2RewardSystem:
    """
    Stage 2 reward system incorporating Mahjong scoring.
    
    This system teaches the AI to maximize hand values and consider
    risk-reward tradeoffs for higher scoring combinations.
    """
    
    def __init__(self, base_win_reward: float = 0.5, score_multiplier: float = 0.1,
                 opponent_score_penalty: float = -0.05, risk_factor: float = 0.02):
        """
        Initialize Stage 2 reward system.
        
        Args:
            base_win_reward: Base reward for any win
            score_multiplier: Multiplier for hand score
            opponent_score_penalty: Penalty based on opponent's potential score
            risk_factor: Factor for risk-reward calculations
        """
        self.base_win_reward = base_win_reward
        self.score_multiplier = score_multiplier
        self.opponent_score_penalty = opponent_score_penalty
        self.risk_factor = risk_factor
        
        # Initialize scoring system
        from ai.scoring_system import MahjongScoring
        self.scorer = MahjongScoring()
    
    def calculate_reward(self, game_result: str, player_index: int, winner_index: int,
                        final_hands: List[List[MahjongTile]] = None,
                        final_sets: List[List[List[MahjongTile]]] = None,
                        game_state: Dict = None) -> float:
        """
        Calculate score-aware reward for game outcome.
        
        Args:
            game_result: Game outcome
            player_index: Player being rewarded
            winner_index: Winner index
            final_hands: Final hands of all players
            final_sets: Final sets of all players
            game_state: Game state information
            
        Returns:
            Reward value incorporating scoring
        """
        if game_result == 'win' and winner_index == player_index:
            # Calculate hand score
            if final_hands and final_sets:
                hand_score = self.scorer.calculate_hand_score(
                    final_hands[player_index], 
                    final_sets[player_index],
                    game_state or {}
                )
                return self.base_win_reward + self.score_multiplier * hand_score
            else:
                return self.base_win_reward
                
        elif game_result == 'loss':
            # Penalty based on winner's score
            if final_hands and final_sets and 0 <= winner_index < len(final_hands):
                winner_score = self.scorer.calculate_hand_score(
                    final_hands[winner_index],
                    final_sets[winner_index], 
                    game_state or {}
                )
                return self.opponent_score_penalty * winner_score
            else:
                return -0.5
                
        elif game_result == 'draw':
            return 0.0
        
        return -0.01  # Turn penalty
    
    def calculate_step_reward(self, action_type: str, action_success: bool,
                            game_state: Dict, hand_before: List[MahjongTile],
                            hand_after: List[MahjongTile]) -> float:
        """
        Calculate score-aware step rewards.
        
        Args:
            action_type: Action type
            action_success: Whether action succeeded
            game_state: Current game state
            hand_before: Hand before action
            hand_after: Hand after action
            
        Returns:
            Step reward
        """
        if not action_success:
            return -0.05
        
        # Calculate potential score improvement
        if action_type in ['chi', 'pong', 'kong']:
            # Positive reward for forming sets (they increase scoring potential)
            return 0.05
        
        elif action_type == 'discard':
            # Analyze if discard improves hand potential
            potential_before = self._calculate_hand_potential(hand_before, game_state)
            potential_after = self._calculate_hand_potential(hand_after, game_state)
            
            improvement = potential_after - potential_before
            return self.risk_factor * improvement
        
        return 0.0
    
    def _calculate_hand_potential(self, hand: List[MahjongTile], 
                                 game_state: Dict) -> float:
        """
        Calculate the scoring potential of a hand.
        
        This is a heuristic estimate of how likely the hand is to form
        high-scoring combinations.
        """
        if not hand:
            return 0.0
        
        # Simple heuristic: count useful tiles for scoring combinations
        potential = 0.0
        
        # Count honor tiles (dragons, winds) - higher scoring potential
        honor_tiles = 0
        for tile in hand:
            if tile.type in ['風', '元']:
                honor_tiles += 1
        
        potential += honor_tiles * 0.1
        
        # Check for same suit concentration (needed for flush)
        suit_counts = {'萬': 0, '筒': 0, '條': 0}
        for tile in hand:
            if tile.type in suit_counts:
                suit_counts[tile.type] += 1
        
        max_suit_concentration = max(suit_counts.values()) / len(hand)
        potential += max_suit_concentration * 0.2
        
        # Check for sequential tiles (easier to complete)
        sequences = self._count_potential_sequences(hand)
        potential += sequences * 0.05
        
        return potential
    
    def _count_potential_sequences(self, hand: List[MahjongTile]) -> int:
        """Count potential sequences in hand."""
        sequences = 0
        
        for suit in ['萬', '筒', '條']:
            suit_tiles = [tile for tile in hand if tile.class_name == suit]
            suit_numbers = [tile.number for tile in suit_tiles]
            suit_numbers.sort()
            
            # Count consecutive runs
            i = 0
            while i < len(suit_numbers) - 1:
                if suit_numbers[i + 1] == suit_numbers[i] + 1:
                    sequences += 1
                    # Skip ahead to avoid double counting
                    while (i < len(suit_numbers) - 1 and 
                           suit_numbers[i + 1] == suit_numbers[i] + 1):
                        i += 1
                i += 1
        
        return sequences


class AdaptiveRewardSystem:
    """
    Adaptive reward system that transitions from Stage 1 to Stage 2.
    
    Gradually shifts focus from basic win/loss to score optimization
    based on training progress.
    """
    
    def __init__(self, transition_episodes: int = 5000):
        """
        Initialize adaptive reward system.
        
        Args:
            transition_episodes: Episodes over which to transition from Stage 1 to Stage 2
        """
        self.transition_episodes = transition_episodes
        self.stage1_system = Stage1RewardSystem()
        self.stage2_system = Stage2RewardSystem()
        self.current_episode = 0
    
    def set_episode(self, episode: int):
        """Set current episode for transition calculation."""
        self.current_episode = episode
    
    def get_transition_weight(self) -> float:
        """Get current transition weight (0=Stage1, 1=Stage2)."""
        if self.current_episode >= self.transition_episodes:
            return 1.0
        return self.current_episode / self.transition_episodes
    
    def calculate_reward(self, *args, **kwargs) -> float:
        """Calculate blended reward based on transition weight."""
        weight = self.get_transition_weight()
        
        stage1_reward = self.stage1_system.calculate_reward(*args, **kwargs)
        stage2_reward = self.stage2_system.calculate_reward(*args, **kwargs)
        
        return (1 - weight) * stage1_reward + weight * stage2_reward
    
    def calculate_step_reward(self, *args, **kwargs) -> float:
        """Calculate blended step reward."""
        weight = self.get_transition_weight()
        
        stage1_step = self.stage1_system.calculate_step_reward(*args, **kwargs)
        stage2_step = self.stage2_system.calculate_step_reward(*args, **kwargs)
        
        return (1 - weight) * stage1_step + weight * stage2_step


class RewardNormalizer:
    """
    Utility class for normalizing and tracking reward statistics.
    
    Helps maintain stable training by normalizing rewards and tracking
    their distribution over time.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize reward normalizer.
        
        Args:
            window_size: Size of rolling window for statistics
        """
        self.window_size = window_size
        self.reward_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
        
    def add_reward(self, reward: float):
        """Add reward to history and update statistics."""
        self.reward_history.append(reward)
        
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
        
        # Update running statistics
        if len(self.reward_history) > 10:
            self.running_mean = np.mean(self.reward_history)
            self.running_std = max(np.std(self.reward_history), 0.01)
    
    def normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        return (reward - self.running_mean) / self.running_std
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current reward statistics."""
        return {
            'mean': self.running_mean,
            'std': self.running_std,
            'min': min(self.reward_history) if self.reward_history else 0,
            'max': max(self.reward_history) if self.reward_history else 0,
            'count': len(self.reward_history)
        }