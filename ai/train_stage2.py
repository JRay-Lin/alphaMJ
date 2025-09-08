"""Stage 2 training script for Mahjong DQN - Scoring system integration."""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import deque
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.dqn_agent import DQNAgent, MultiAgentDQN
from ai.training_env import MultiAgentTrainingEnv
from ai.utils.config import dqn_config
from ai.reward_system import Stage2RewardSystem
from ai.scoring_system import MahjongScoring


class Stage2Trainer:
    """
    Stage 2 trainer for Mahjong DQN.
    
    Focuses on scoring system integration:
    - Learning to maximize hand values
    - Risk-reward tradeoffs for higher scoring combinations
    - Understanding strategic depth beyond basic win/loss
    """
    
    def __init__(self, config: dict = None, pretrained_model_path: str = None):
        """
        Initialize Stage 2 trainer.
        
        Args:
            config: Training configuration dictionary
            pretrained_model_path: Path to Stage 1 pretrained model
        """
        self.config = config or self._get_default_config()
        self.pretrained_model_path = pretrained_model_path
        
        # Device selection: CUDA > MPS > CPU
        device_name = config.get('device', 'auto')
        if device_name == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)
        print(f"Training on device: {self.device}")
        
        # Create scoring system
        self.scoring_system = MahjongScoring(rule=self.config['rule_name'])
        
        # Create multi-agent system
        self.multi_agent = MultiAgentDQN(
            num_agents=4,
            shared_replay=self.config['shared_replay'],
            model_dir=self.config['model_dir'],
            device=self.device.type
        )
        
        # Load pretrained models if provided
        if pretrained_model_path:
            self._load_pretrained_models()
        
        # Create training environment with Stage 2 rewards
        self.env = MultiAgentTrainingEnv(
            rule_name=self.config['rule_name'],
            reward_system="stage2"
        )
        
        # Register agents with environment
        for i in range(4):
            self.env.register_agent(i, self.multi_agent.get_agent(i))
        
        # Training statistics
        self.episode_rewards = deque(maxlen=1000)
        self.episode_scores = deque(maxlen=1000)  # Actual game scores
        self.episode_lengths = deque(maxlen=1000)
        self.win_rates = deque(maxlen=100)
        self.loss_history = []
        self.score_distributions = []
        
        # Scoring statistics
        self.high_score_games = deque(maxlen=100)
        self.average_winning_scores = deque(maxlen=100)
        
        # Setup logging
        self.setup_logging()
    
    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            'num_episodes': 15000,  # More episodes for Stage 2
            'max_steps_per_episode': 200,
            'save_frequency': 100,
            'eval_frequency': 500,
            'log_frequency': 10,
            'model_dir': 'ai/models/stage2',
            'log_dir': 'ai/logs/stage2',
            'rule_name': 'standard',
            'shared_replay': False,
            'target_win_rate': 0.25,
            'target_avg_score': 50,  # Target average winning score
            'early_stopping_patience': 3000,
            'learning_rate_decay': 0.95,
            'decay_frequency': 1000,
            'curriculum_learning': True,
            'score_thresholds': [30, 50, 80, 120]  # Progressive scoring targets
        }
    
    def _load_pretrained_models(self):
        """Load Stage 1 pretrained models for transfer learning."""
        print(f"Loading pretrained models from {self.pretrained_model_path}")
        
        # Try to load the best model first
        best_model_path = os.path.join(self.pretrained_model_path, 'best_model.pth')
        
        if os.path.exists(best_model_path):
            # Load best model for all agents initially
            for i in range(4):
                agent = self.multi_agent.get_agent(i)
                agent.load_model(best_model_path, load_optimizer=False)
                
                # Reset some training parameters for Stage 2
                agent.epsilon = 0.3  # Start with some exploration
                agent.learning_rate = self.config.get('stage2_learning_rate', 0.0005)
                
                # Update optimizer with new learning rate
                agent.optimizer = torch.optim.Adam(
                    agent.q_network.parameters(), 
                    lr=agent.learning_rate
                )
                
            print("Loaded best model for all agents")
        else:
            print("No best model found, starting from scratch")
    
    def setup_logging(self):
        """Setup logging directories and files."""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Save config
        with open(os.path.join(self.config['log_dir'], 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def train(self):
        """Run Stage 2 training."""
        print("Starting Stage 2 Training: Scoring System Integration")
        print(f"Target episodes: {self.config['num_episodes']}")
        print(f"Rule: {self.config['rule_name']}")
        print(f"Target average score: {self.config['target_avg_score']}")
        print("-" * 60)
        
        start_time = time.time()
        best_avg_score = 0
        patience_counter = 0
        current_score_threshold = self.config['score_thresholds'][0]
        threshold_index = 0
        
        # Training loop
        for episode in tqdm(range(self.config['num_episodes']), desc="Training"):
            # Curriculum learning - adjust difficulty
            if self.config['curriculum_learning']:
                self._update_curriculum(episode)
            
            # Run episode
            episode_result = self.env.run_episode()
            
            # Process episode data with scoring analysis
            self._process_episode_with_scoring(episode_result, episode)
            
            # Learning rate decay
            if episode > 0 and episode % self.config['decay_frequency'] == 0:
                self._decay_learning_rate()
            
            # Logging
            if episode % self.config['log_frequency'] == 0:
                self._log_progress_with_scoring(episode)
            
            # Save models
            if episode % self.config['save_frequency'] == 0 and episode > 0:
                self._save_checkpoint(episode)
            
            # Evaluation
            if episode % self.config['eval_frequency'] == 0 and episode > 0:
                eval_results = self._evaluate_with_scoring()
                self._log_evaluation_with_scoring(episode, eval_results)
                
                # Early stopping and curriculum progression
                avg_score = np.mean(self.average_winning_scores) if self.average_winning_scores else 0
                
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    patience_counter = 0
                    self._save_best_model(episode)
                    
                    # Progress curriculum if target reached
                    if (avg_score >= current_score_threshold and 
                        threshold_index < len(self.config['score_thresholds']) - 1):
                        threshold_index += 1
                        current_score_threshold = self.config['score_thresholds'][threshold_index]
                        print(f"Curriculum progressed! New target score: {current_score_threshold}")
                        
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config['early_stopping_patience'] / self.config['eval_frequency']:
                    print(f"Early stopping at episode {episode}")
                    break
        
        # Final save and evaluation
        self._save_checkpoint(self.config['num_episodes'])
        final_eval = self._evaluate_with_scoring()
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final evaluation: {final_eval}")
        
        # Generate training report
        self._generate_training_report_with_scoring(training_time)
    
    def _update_curriculum(self, episode: int):
        """Update curriculum learning parameters."""
        # Gradually increase focus on high-scoring hands
        progress = episode / self.config['num_episodes']
        
        # Adjust reward system parameters dynamically
        # This is a placeholder for more sophisticated curriculum learning
        pass
    
    def _process_episode_with_scoring(self, episode_result: dict, episode_num: int):
        """Process episode results with scoring analysis."""
        episode_data = episode_result['episode_data']
        info = episode_result['info']
        
        # Track episode statistics
        episode_reward = 0
        episode_length = len(episode_data)
        episode_game_scores = []
        
        # Process transitions for each agent
        agent_transitions = {i: [] for i in range(4)}
        
        for transition in episode_data:
            player_idx = transition['player_idx']
            agent_transitions[player_idx].append(transition)
            episode_reward += transition['reward']
        
        # Calculate actual game scores if game ended with a winner
        if info.get('winner', -1) >= 0:
            winner_idx = info['winner']
            
            # Get final game state to calculate scores
            final_hands, final_sets = self._extract_final_hands_and_sets(episode_result)
            
            if final_hands and final_sets:
                for player_idx in range(4):
                    if player_idx < len(final_hands) and player_idx < len(final_sets):
                        game_context = {
                            'self_draw': player_idx == winner_idx,
                            'is_dealer': player_idx == 0  # Simplified
                        }
                        
                        score = self.scoring_system.calculate_hand_score(
                            final_hands[player_idx],
                            final_sets[player_idx],
                            game_context
                        )
                        episode_game_scores.append(score)
                
                # Track winning score
                if winner_idx < len(episode_game_scores):
                    winning_score = episode_game_scores[winner_idx]
                    self.average_winning_scores.append(winning_score)
                    
                    if winning_score >= 80:  # High score threshold
                        self.high_score_games.append({
                            'episode': episode_num,
                            'winner': winner_idx,
                            'score': winning_score,
                            'scores': episode_game_scores.copy()
                        })
        
        # Update agents with enhanced scoring-aware feedback
        for player_idx, transitions in agent_transitions.items():
            agent = self.multi_agent.get_agent(player_idx)
            
            # Add experiences with scoring context
            for transition in transitions:
                # Enhance reward with scoring information if available
                enhanced_reward = transition['reward']
                if episode_game_scores and player_idx < len(episode_game_scores):
                    score_bonus = episode_game_scores[player_idx] * 0.01  # Scale score to reasonable range
                    enhanced_reward += score_bonus
                
                agent.step(
                    transition['state'],
                    transition['action'],
                    enhanced_reward,
                    transition['next_state'],
                    transition['done']
                )
            
            # Update performance tracking
            if transitions:
                avg_reward = np.mean([t['reward'] for t in transitions])
                if episode_game_scores and player_idx < len(episode_game_scores):
                    # Weight performance by actual game score
                    score_weight = episode_game_scores[player_idx] / 50.0  # Normalize
                    avg_reward = avg_reward + score_weight
                
                self.multi_agent.update_agent_performance(player_idx, avg_reward)
        
        # Store statistics
        self.episode_rewards.append(episode_reward / 4)
        self.episode_lengths.append(episode_length)
        if episode_game_scores:
            self.episode_scores.append(np.mean(episode_game_scores))
            self.score_distributions.append(episode_game_scores.copy())
    
    def _extract_final_hands_and_sets(self, episode_result: dict) -> tuple:
        """Extract final hands and sets from episode result."""
        # This is a simplified extraction
        # In practice, we'd need to track game state throughout the episode
        return [], []  # Placeholder
    
    def _decay_learning_rate(self):
        """Decay learning rate for all agents."""
        for agent in self.multi_agent.agents:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= self.config['learning_rate_decay']
        
        current_lr = self.multi_agent.agents[0].optimizer.param_groups[0]['lr']
        print(f"Learning rate decayed to: {current_lr:.6f}")
    
    def _log_progress_with_scoring(self, episode: int):
        """Log training progress with scoring information."""
        if not self.episode_rewards:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        avg_score = np.mean(self.episode_scores) if self.episode_scores else 0
        avg_winning_score = np.mean(self.average_winning_scores) if self.average_winning_scores else 0
        
        # Get agent statistics
        agent_stats = self.multi_agent.get_statistics()
        avg_epsilon = np.mean([stat['epsilon'] for stat in agent_stats['agent_stats']])
        avg_loss = np.mean([stat['average_loss'] for stat in agent_stats['agent_stats']])
        
        print(f"Episode {episode:5d} | "
              f"Reward: {avg_reward:6.3f} | "
              f"Game Score: {avg_score:5.1f} | "
              f"Win Score: {avg_winning_score:5.1f} | "
              f"Length: {avg_length:5.1f} | "
              f"ε: {avg_epsilon:.3f} | "
              f"Loss: {avg_loss:.4f}")
    
    def _evaluate_with_scoring(self) -> dict:
        """Run evaluation tournament with scoring analysis."""
        print("Running scoring evaluation tournament...")
        
        # Set agents to evaluation mode
        for i in range(4):
            self.multi_agent.get_agent(i).set_training_mode(False)
        
        # Run tournament with scoring tracking
        win_rates = self.multi_agent.tournament_evaluation(num_games=100)
        
        # Additional scoring metrics would be calculated here
        # This is simplified for now
        
        eval_results = {
            'win_rates': win_rates,
            'balance_score': 1.0 - np.std(list(win_rates.values())),
            'average_game_score': np.mean(self.episode_scores) if self.episode_scores else 0,
            'average_winning_score': np.mean(self.average_winning_scores) if self.average_winning_scores else 0,
            'high_score_games': len(self.high_score_games),
            'agent_performances': self.multi_agent.agent_performances.copy()
        }
        
        # Restore training mode
        for i in range(4):
            self.multi_agent.get_agent(i).set_training_mode(True)
        
        return eval_results
    
    def _log_evaluation_with_scoring(self, episode: int, eval_results: dict):
        """Log evaluation results with scoring information."""
        win_rates = eval_results['win_rates']
        balance = eval_results['balance_score']
        avg_score = eval_results['average_game_score']
        avg_win_score = eval_results['average_winning_score']
        high_scores = eval_results['high_score_games']
        
        print(f"\nEvaluation at episode {episode}:")
        print(f"Win rates: {[f'{rate:.3f}' for rate in win_rates.values()]}")
        print(f"Balance score: {balance:.3f}")
        print(f"Average game score: {avg_score:.1f}")
        print(f"Average winning score: {avg_win_score:.1f}")
        print(f"High-score games: {high_scores}")
        print(f"Agent performances: {[f'{perf:.3f}' for perf in eval_results['agent_performances']]}")
        print("-" * 40)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint with scoring data."""
        # Save all agents
        self.multi_agent.save_all_agents(episode)
        
        # Save training statistics including scoring
        stats = {
            'episode': episode,
            'episode_rewards': list(self.episode_rewards),
            'episode_scores': list(self.episode_scores),
            'episode_lengths': list(self.episode_lengths),
            'win_rates': list(self.win_rates),
            'average_winning_scores': list(self.average_winning_scores),
            'high_score_games': list(self.high_score_games),
            'score_distributions': self.score_distributions[-100:],  # Last 100 games
            'multi_agent_stats': self.multi_agent.get_statistics()
        }
        
        stats_path = os.path.join(self.config['log_dir'], f'training_stats_{episode}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Checkpoint saved at episode {episode}")
    
    def _save_best_model(self, episode: int):
        """Save the best performing model."""
        best_agent = self.multi_agent.get_best_agent()
        best_path = os.path.join(self.config['model_dir'], 'best_model.pth')
        best_agent.save_model(best_path)
        
        # Save metadata with scoring information
        metadata = {
            'episode': episode,
            'performance': max(self.multi_agent.agent_performances),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_winning_score': np.mean(self.average_winning_scores) if self.average_winning_scores else 0,
            'high_score_games': len(self.high_score_games),
            'timestamp': time.time()
        }
        
        metadata_path = os.path.join(self.config['model_dir'], 'best_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_training_report_with_scoring(self, training_time: float):
        """Generate comprehensive training report with scoring analysis."""
        print("\nGenerating training report with scoring analysis...")
        
        # Create plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Reward progress
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        
        # Game scores
        if self.episode_scores:
            axes[0, 1].plot(self.episode_scores)
            axes[0, 1].set_title('Game Scores')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Score')
        
        # Winning scores
        if self.average_winning_scores:
            axes[1, 0].plot(self.average_winning_scores)
            axes[1, 0].set_title('Winning Scores')
            axes[1, 0].set_xlabel('Game')
            axes[1, 0].set_ylabel('Winner Score')
            axes[1, 0].axhline(y=self.config['target_avg_score'], color='r', linestyle='--', label='Target')
            axes[1, 0].legend()
        
        # Episode lengths
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title('Episode Lengths')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        
        # Score distribution (recent games)
        if self.score_distributions:
            recent_scores = [score for game_scores in self.score_distributions[-50:] for score in game_scores]
            axes[2, 0].hist(recent_scores, bins=20, alpha=0.7)
            axes[2, 0].set_title('Score Distribution (Recent 50 Games)')
            axes[2, 0].set_xlabel('Score')
            axes[2, 0].set_ylabel('Frequency')
        
        # Agent performance comparison
        performances = self.multi_agent.agent_performances
        axes[2, 1].bar(range(len(performances)), performances)
        axes[2, 1].set_title('Final Agent Performances')
        axes[2, 1].set_xlabel('Agent Index')
        axes[2, 1].set_ylabel('Performance Score')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 'training_report_stage2.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate detailed report
        report = {
            'training_time': training_time,
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': np.mean(self.episode_rewards),
            'final_avg_game_score': np.mean(self.episode_scores) if self.episode_scores else 0,
            'final_avg_winning_score': np.mean(self.average_winning_scores) if self.average_winning_scores else 0,
            'final_avg_length': np.mean(self.episode_lengths),
            'high_score_games': len(self.high_score_games),
            'best_agent_performance': max(self.multi_agent.agent_performances),
            'agent_balance': 1.0 - np.std(self.multi_agent.agent_performances),
            'score_improvement': {
                'initial_avg': np.mean(self.average_winning_scores[:10]) if len(self.average_winning_scores) >= 10 else 0,
                'final_avg': np.mean(self.average_winning_scores[-10:]) if len(self.average_winning_scores) >= 10 else 0
            },
            'config': self.config
        }
        
        report_path = os.path.join(self.config['log_dir'], 'final_report_stage2.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Stage 2 training report saved to {self.config['log_dir']}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Stage 2 Mahjong DQN Training')
    parser.add_argument('--episodes', type=int, default=15000,
                       help='Number of training episodes')
    parser.add_argument('--rule', choices=['standard', 'taiwan'], default='standard',
                       help='Mahjong rule variant')
    parser.add_argument('--pretrained', type=str,
                       help='Path to Stage 1 pretrained models directory')
    parser.add_argument('--model-dir', type=str, default='ai/models/stage2',
                       help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='ai/logs/stage2',
                       help='Log directory')
    parser.add_argument('--shared-replay', action='store_true',
                       help='Use shared experience replay')
    parser.add_argument('--target-score', type=int, default=50,
                       help='Target average winning score')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Configure device: CUDA > MPS > CPU
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Create training configuration
    config = {
        'num_episodes': args.episodes,
        'rule_name': args.rule,
        'model_dir': args.model_dir,
        'log_dir': args.log_dir,
        'shared_replay': args.shared_replay,
        'target_avg_score': args.target_score,
        'max_steps_per_episode': 200,
        'save_frequency': 100,
        'eval_frequency': 500,
        'log_frequency': 10,
        'target_win_rate': 0.25,
        'early_stopping_patience': 3000,
        'learning_rate_decay': 0.95,
        'decay_frequency': 1000,
        'curriculum_learning': True,
        'score_thresholds': [30, 50, 80, 120]
    }
    
    # Create and run trainer
    trainer = Stage2Trainer(config, args.pretrained)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer._save_checkpoint(len(trainer.episode_rewards))
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    print("\nStage 2 training completed!")


if __name__ == "__main__":
    main()