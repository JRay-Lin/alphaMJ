"""Model evaluation script for trained Mahjong DQN agents."""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.dqn_agent import DQNAgent
from ai.training_env import MultiAgentTrainingEnv
from ai.scoring_system import MahjongScoring
from main import MahjongGame, MahjongRule


class MahjongEvaluator:
    """
    Comprehensive evaluation system for trained Mahjong agents.
    
    Provides detailed analysis including:
    - Win rates and game statistics
    - Scoring analysis and hand quality
    - Playstyle analysis
    - Performance against different opponents
    """
    
    def __init__(self, model_path: str, rule: str = "standard", device: str = "auto"):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            rule: Mahjong rule variant
            device: Computing device
        """
        # Device selection: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        self.rule = rule
        self.model_path = model_path
        
        # Load model
        self.agent = DQNAgent(device=self.device)
        if os.path.exists(model_path):
            self.agent.load_model(model_path, load_optimizer=False)
            self.agent.set_training_mode(False)
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize scoring system
        self.scoring_system = MahjongScoring(rule=rule)
        
        # Evaluation statistics
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset evaluation statistics."""
        self.statistics = {
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0,
            'average_score': 0.0,
            'score_distribution': [],
            'game_lengths': [],
            'action_counts': defaultdict(int),
            'opponent_statistics': defaultdict(lambda: {'wins': 0, 'games': 0}),
            'hand_analysis': {
                'flush_attempts': 0,
                'honor_focus': 0,
                'quick_wins': 0,
                'high_score_wins': 0
            }
        }
    
    def evaluate_against_random(self, num_games: int = 1000) -> Dict[str, Any]:
        """
        Evaluate agent against random players.
        
        Args:
            num_games: Number of games to play
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating against random players ({num_games} games)...")
        self.reset_statistics()
        
        # Create evaluation environment
        env = MultiAgentTrainingEnv(rule_name=self.rule, reward_system="stage1")
        
        # Register agents (our agent vs 3 random agents)
        env.register_agent(0, self.agent)
        
        # Create simple random agents
        random_agents = []
        for i in range(3):
            random_agent = RandomAgent()
            random_agents.append(random_agent)
            env.register_agent(i + 1, random_agent)
        
        # Run evaluation games
        for game_num in range(num_games):
            result = env.run_episode()
            self._process_game_result(result, agent_index=0)
            
            if (game_num + 1) % 100 == 0:
                print(f"Completed {game_num + 1}/{num_games} games")
        
        # Calculate final statistics
        self._calculate_final_statistics()
        
        return self.statistics
    
    def evaluate_against_models(self, opponent_models: List[str], num_games: int = 400) -> Dict[str, Any]:
        """
        Evaluate agent against other trained models.
        
        Args:
            opponent_models: List of paths to opponent models
            num_games: Number of games to play
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating against trained models ({num_games} games)...")
        self.reset_statistics()
        
        # Load opponent agents
        opponents = []
        for model_path in opponent_models:
            if os.path.exists(model_path):
                opponent = DQNAgent(device=self.device)
                opponent.load_model(model_path, load_optimizer=False)
                opponent.set_training_mode(False)
                opponents.append(opponent)
            else:
                print(f"Warning: Opponent model not found: {model_path}")
        
        if len(opponents) < 3:
            print("Not enough opponent models, using random agents for remaining slots")
            while len(opponents) < 3:
                opponents.append(RandomAgent())
        
        # Create evaluation environment
        env = MultiAgentTrainingEnv(rule_name=self.rule, reward_system="stage2")
        
        # Register agents
        env.register_agent(0, self.agent)
        for i, opponent in enumerate(opponents[:3]):
            env.register_agent(i + 1, opponent)
        
        # Run evaluation games
        for game_num in range(num_games):
            result = env.run_episode()
            self._process_game_result(result, agent_index=0)
            
            if (game_num + 1) % 50 == 0:
                print(f"Completed {game_num + 1}/{num_games} games")
        
        self._calculate_final_statistics()
        
        return self.statistics
    
    def analyze_playstyle(self, num_games: int = 200) -> Dict[str, Any]:
        """
        Analyze agent's playstyle and decision patterns.
        
        Args:
            num_games: Number of games for analysis
            
        Returns:
            Playstyle analysis
        """
        print(f"Analyzing playstyle ({num_games} games)...")
        
        # Detailed game tracking
        detailed_stats = {
            'tile_preferences': Counter(),
            'discard_patterns': [],
            'meld_preferences': {'chi': 0, 'pong': 0, 'kong': 0},
            'winning_hands': [],
            'risk_assessment': {'safe_plays': 0, 'aggressive_plays': 0},
            'endgame_behavior': []
        }
        
        # Run analysis games with detailed tracking
        env = MultiAgentTrainingEnv(rule_name=self.rule, reward_system="stage2")
        env.register_agent(0, self.agent)
        
        # Add simple opponents
        for i in range(3):
            env.register_agent(i + 1, RandomAgent())
        
        for game_num in range(num_games):
            result = env.run_episode()
            self._analyze_game_decisions(result, detailed_stats)
        
        # Process analysis results
        analysis = {
            'preferred_tiles': dict(detailed_stats['tile_preferences'].most_common(10)),
            'meld_distribution': detailed_stats['meld_preferences'],
            'risk_profile': detailed_stats['risk_assessment'],
            'average_hand_quality': self._calculate_hand_quality(detailed_stats['winning_hands']),
            'playstyle_summary': self._summarize_playstyle(detailed_stats)
        }
        
        return analysis
    
    def _process_game_result(self, result: Dict[str, Any], agent_index: int = 0):
        """Process individual game result for statistics."""
        info = result['info']
        episode_data = result['episode_data']
        
        self.statistics['games_played'] += 1
        
        # Determine outcome for our agent
        winner = info.get('winner', -1)
        if winner == agent_index:
            self.statistics['wins'] += 1
            
            # Calculate game score if possible
            # This is simplified - would need actual game state
            game_score = self._estimate_game_score(episode_data, agent_index)
            self.statistics['score_distribution'].append(game_score)
            
        elif winner == -1:
            self.statistics['draws'] += 1
        else:
            self.statistics['losses'] += 1
            self.statistics['opponent_statistics'][winner]['wins'] += 1
        
        # Track game length
        game_length = len([t for t in episode_data if t['player_idx'] == agent_index])
        self.statistics['game_lengths'].append(game_length)
        
        # Update opponent statistics
        for i in range(4):
            if i != agent_index:
                self.statistics['opponent_statistics'][i]['games'] += 1
    
    def _estimate_game_score(self, episode_data: List[Dict], agent_index: int) -> float:
        """Estimate game score from episode data."""
        # This is a simplified estimation
        # In practice, would need access to final game state
        agent_transitions = [t for t in episode_data if t['player_idx'] == agent_index]
        
        if agent_transitions:
            total_reward = sum(t['reward'] for t in agent_transitions)
            # Convert reward to estimated score
            return max(total_reward * 25, 10)  # Rough scaling
        
        return 20  # Default minimum score
    
    def _analyze_game_decisions(self, result: Dict[str, Any], detailed_stats: Dict):
        """Analyze detailed game decisions for playstyle analysis."""
        episode_data = result['episode_data']
        agent_data = [t for t in episode_data if t['player_idx'] == 0]
        
        # Analyze decision patterns
        for transition in agent_data:
            action = transition['action']
            
            # Track action types
            if action < 14:  # Discard action
                detailed_stats['discard_patterns'].append(action)
            
            # Simple risk assessment based on reward
            reward = transition['reward']
            if reward > 0.05:
                detailed_stats['risk_assessment']['aggressive_plays'] += 1
            else:
                detailed_stats['risk_assessment']['safe_plays'] += 1
    
    def _calculate_hand_quality(self, winning_hands: List) -> float:
        """Calculate average quality of winning hands."""
        if not winning_hands:
            return 0.0
        
        # Simplified quality calculation
        return np.mean([len(hand) for hand in winning_hands])
    
    def _summarize_playstyle(self, detailed_stats: Dict) -> str:
        """Generate playstyle summary."""
        risk_ratio = detailed_stats['risk_assessment']['aggressive_plays'] / max(
            detailed_stats['risk_assessment']['safe_plays'] + 
            detailed_stats['risk_assessment']['aggressive_plays'], 1
        )
        
        if risk_ratio > 0.6:
            style = "Aggressive"
        elif risk_ratio < 0.4:
            style = "Conservative"
        else:
            style = "Balanced"
        
        return f"{style} player with {risk_ratio:.2f} aggression ratio"
    
    def _calculate_final_statistics(self):
        """Calculate final evaluation statistics."""
        games = self.statistics['games_played']
        if games > 0:
            self.statistics['win_rate'] = self.statistics['wins'] / games
            
            if self.statistics['score_distribution']:
                self.statistics['average_score'] = np.mean(self.statistics['score_distribution'])
    
    def generate_report(self, output_dir: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save report
            
        Returns:
            Report text
        """
        if not output_dir:
            output_dir = os.path.dirname(self.model_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate text report
        report = []
        report.append("=== Mahjong DQN Agent Evaluation Report ===")
        report.append(f"Model: {self.model_path}")
        report.append(f"Rule: {self.rule}")
        report.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Basic statistics
        report.append("=== Performance Statistics ===")
        report.append(f"Games Played: {self.statistics['games_played']}")
        report.append(f"Wins: {self.statistics['wins']}")
        report.append(f"Losses: {self.statistics['losses']}")
        report.append(f"Draws: {self.statistics['draws']}")
        report.append(f"Win Rate: {self.statistics['win_rate']:.3f}")
        report.append(f"Average Score: {self.statistics['average_score']:.1f}")
        
        if self.statistics['game_lengths']:
            report.append(f"Average Game Length: {np.mean(self.statistics['game_lengths']):.1f} turns")
        
        report.append("")
        
        # Score distribution
        if self.statistics['score_distribution']:
            report.append("=== Score Analysis ===")
            scores = self.statistics['score_distribution']
            report.append(f"Score Range: {min(scores):.1f} - {max(scores):.1f}")
            report.append(f"Score Std: {np.std(scores):.1f}")
            
            # Score categories
            high_scores = len([s for s in scores if s >= 80])
            medium_scores = len([s for s in scores if 40 <= s < 80])
            low_scores = len([s for s in scores if s < 40])
            
            report.append(f"High Scores (80+): {high_scores} ({high_scores/len(scores)*100:.1f}%)")
            report.append(f"Medium Scores (40-79): {medium_scores} ({medium_scores/len(scores)*100:.1f}%)")
            report.append(f"Low Scores (<40): {low_scores} ({low_scores/len(scores)*100:.1f}%)")
        
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Save statistics as JSON
        stats_path = os.path.join(output_dir, "evaluation_stats.json")
        with open(stats_path, 'w') as f:
            # Convert numpy types for JSON serialization
            stats_json = {}
            for key, value in self.statistics.items():
                if isinstance(value, np.ndarray):
                    stats_json[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    stats_json[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    stats_json[key] = int(value)
                else:
                    stats_json[key] = value
            
            json.dump(stats_json, f, indent=2, default=str)
        
        # Generate visualization
        self._generate_evaluation_plots(output_dir)
        
        print(f"Evaluation report saved to {output_dir}")
        return report_text


    def _generate_evaluation_plots(self, output_dir: str):
        """Generate evaluation visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Win rate pie chart
        wins = self.statistics['wins']
        losses = self.statistics['losses']
        draws = self.statistics['draws']
        
        if wins + losses + draws > 0:
            axes[0, 0].pie([wins, losses, draws], 
                          labels=['Wins', 'Losses', 'Draws'],
                          autopct='%1.1f%%',
                          startangle=90)
            axes[0, 0].set_title('Game Outcomes')
        
        # Score distribution
        if self.statistics['score_distribution']:
            axes[0, 1].hist(self.statistics['score_distribution'], bins=20, alpha=0.7)
            axes[0, 1].set_title('Score Distribution')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
        
        # Game length distribution
        if self.statistics['game_lengths']:
            axes[1, 0].hist(self.statistics['game_lengths'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Game Length Distribution')
            axes[1, 0].set_xlabel('Turns')
            axes[1, 0].set_ylabel('Frequency')
        
        # Performance summary
        metrics = ['Win Rate', 'Avg Score']
        values = [self.statistics['win_rate'], 
                 self.statistics['average_score'] / 100]  # Scale for visualization
        
        axes[1, 1].bar(metrics, values)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Value')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'evaluation_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()


class RandomAgent:
    """Simple random agent for evaluation."""
    
    def __init__(self):
        self.epsilon = 0.0
        
    def get_action(self, state, action_mask=None, epsilon=0.0):
        """Select random valid action."""
        if action_mask is not None:
            valid_actions = torch.where(action_mask)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions.cpu().numpy())
            else:
                action = 0
        else:
            action = np.random.randint(14)
        
        # Return dummy Q-values
        q_values = torch.zeros(14)
        return action, q_values
    
    def set_training_mode(self, training):
        """Compatibility method."""
        pass


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Mahjong DQN Agent')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--rule', choices=['standard', 'taiwan'], default='standard',
                       help='Mahjong rule variant')
    parser.add_argument('--games', type=int, default=1000,
                       help='Number of evaluation games')
    parser.add_argument('--opponent-models', nargs='*',
                       help='Paths to opponent models for evaluation')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device')
    parser.add_argument('--analysis', action='store_true',
                       help='Run detailed playstyle analysis')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MahjongEvaluator(args.model, args.rule, args.device)
    
    # Run evaluation
    if args.opponent_models:
        print("Evaluating against trained models...")
        results = evaluator.evaluate_against_models(args.opponent_models, args.games)
    else:
        print("Evaluating against random players...")
        results = evaluator.evaluate_against_random(args.games)
    
    # Run playstyle analysis if requested
    if args.analysis:
        print("Running playstyle analysis...")
        analysis = evaluator.analyze_playstyle(min(args.games // 5, 200))
        print("Playstyle Analysis:")
        print(f"  Summary: {analysis['playstyle_summary']}")
        print(f"  Average Hand Quality: {analysis['average_hand_quality']:.2f}")
    
    # Generate report
    output_dir = args.output_dir or os.path.dirname(args.model)
    report = evaluator.generate_report(output_dir)
    
    print("\nEvaluation Summary:")
    print(f"Win Rate: {results['win_rate']:.3f}")
    print(f"Average Score: {results['average_score']:.1f}")
    print(f"Games Played: {results['games_played']}")


if __name__ == "__main__":
    main()