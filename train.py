"""Main training script for Mahjong DQN with both Stage 1 and Stage 2 training."""

import os
import sys
import argparse
import json
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.train_stage1 import Stage1Trainer
from ai.train_stage2 import Stage2Trainer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_default_config(stage: str, rule: str = "standard") -> Dict[str, Any]:
    """Create default configuration for training stage."""
    base_config = {
        'rule_name': rule,
        'shared_replay': False,
        'save_frequency': 100,
        'eval_frequency': 500,
        'log_frequency': 10,
        'target_win_rate': 0.25,
        'early_stopping_patience': 2000 if stage == "stage1" else 3000,
        'max_steps_per_episode': 200
    }
    
    if stage == "stage1":
        stage1_config = {
            'num_episodes': 10000,
            'model_dir': 'ai/models/stage1',
            'log_dir': 'ai/logs/stage1'
        }
        base_config.update(stage1_config)
        
    elif stage == "stage2":
        stage2_config = {
            'num_episodes': 15000,
            'model_dir': 'ai/models/stage2',
            'log_dir': 'ai/logs/stage2',
            'target_avg_score': 50,
            'learning_rate_decay': 0.95,
            'decay_frequency': 1000,
            'curriculum_learning': True,
            'score_thresholds': [30, 50, 80, 120]
        }
        base_config.update(stage2_config)
    
    return base_config


def run_stage1_training(config: Dict[str, Any], resume: bool = False) -> str | None:
    """
    Run Stage 1 training.
    
    Args:
        config: Training configuration
        resume: Whether to resume from checkpoint
        
    Returns:
        Path to best model from Stage 1
    """
    print("=" * 60)
    print("STAGE 1 TRAINING: Basic Win/Loss Learning")
    print("=" * 60)
    
    trainer = Stage1Trainer(config)
    
    if resume:
        # TODO: Implement checkpoint resuming
        print("Note: Resume functionality not yet implemented")
    
    try:
        trainer.train()
        
        # Return path to best model
        best_model_path = os.path.join(config['model_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"Stage 1 training completed. Best model saved to: {best_model_path}")
            return best_model_path
        else:
            print("Stage 1 training completed, but best model not found.")
            return None
            
    except Exception as e:
        print(f"Stage 1 training failed: {e}")
        raise


def run_stage2_training(config: Dict[str, Any], pretrained_path: str | None = None, resume: bool = False) -> str | None:
    """
    Run Stage 2 training.
    
    Args:
        config: Training configuration
        pretrained_path: Path to Stage 1 pretrained model
        resume: Whether to resume from checkpoint
        
    Returns:
        Path to best model from Stage 2
    """
    print("=" * 60)
    print("STAGE 2 TRAINING: Scoring System Integration")
    print("=" * 60)
    
    if pretrained_path:
        pretrained_dir = os.path.dirname(pretrained_path)
        print(f"Using pretrained models from: {pretrained_dir}")
    else:
        print("Starting Stage 2 training from scratch")
        pretrained_dir = None
    
    trainer = Stage2Trainer(config, pretrained_dir)
    
    if resume:
        # TODO: Implement checkpoint resuming
        print("Note: Resume functionality not yet implemented")
    
    try:
        trainer.train()
        
        # Return path to best model
        best_model_path = os.path.join(config['model_dir'], 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"Stage 2 training completed. Best model saved to: {best_model_path}")
            return best_model_path
        else:
            print("Stage 2 training completed, but best model not found.")
            return None
            
    except Exception as e:
        print(f"Stage 2 training failed: {e}")
        raise


def run_full_training_pipeline(stage1_config: Dict[str, Any], stage2_config: Dict[str, Any],
                             resume_stage1: bool = False, resume_stage2: bool = False) -> Dict[str, str | None]:
    """
    Run complete training pipeline (Stage 1 followed by Stage 2).
    
    Args:
        stage1_config: Stage 1 configuration
        stage2_config: Stage 2 configuration
        resume_stage1: Whether to resume Stage 1
        resume_stage2: Whether to resume Stage 2
        
    Returns:
        Dictionary with paths to final models
    """
    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    results = {}
    
    # Stage 1 Training
    print("\nStarting Stage 1 training...")
    stage1_best = run_stage1_training(stage1_config, resume_stage1)
    results['stage1_best'] = stage1_best
    
    if stage1_best:
        print(f"✓ Stage 1 completed successfully")
    else:
        print("⚠ Stage 1 completed but no best model found")
    
    # Brief pause between stages
    print("\nTransitioning to Stage 2...")
    time.sleep(2)
    
    # Stage 2 Training
    print("\nStarting Stage 2 training...")
    stage2_best = run_stage2_training(stage2_config, stage1_best, resume_stage2)
    results['stage2_best'] = stage2_best
    
    if stage2_best:
        print(f"✓ Stage 2 completed successfully")
    else:
        print("⚠ Stage 2 completed but no best model found")
    
    # Training summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Total training time: {total_time:.1f} seconds ({total_time/3600:.1f} hours)")
    print(f"Stage 1 best model: {results.get('stage1_best', 'Not found')}")
    print(f"Stage 2 best model: {results.get('stage2_best', 'Not found')}")
    
    # Save pipeline results
    pipeline_results = {
        'training_time': total_time,
        'stage1_best': results.get('stage1_best'),
        'stage2_best': results.get('stage2_best'),
        'timestamp': time.time(),
        'stage1_config': stage1_config,
        'stage2_config': stage2_config
    }
    
    results_path = 'ai/pipeline_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    print(f"Pipeline results saved to: {results_path}")
    
    return results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Mahjong DQN Agent')
    
    # Training stage selection
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'full'], default='full',
                       help='Training stage to run')
    
    # Basic parameters
    parser.add_argument('--rule', choices=['standard', 'taiwan'], default='standard',
                       help='Mahjong rule variant')
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    
    # Stage 1 parameters
    parser.add_argument('--stage1-episodes', type=int, default=10000,
                       help='Number of Stage 1 episodes')
    parser.add_argument('--stage1-model-dir', type=str, default='ai/models/stage1',
                       help='Stage 1 model directory')
    
    # Stage 2 parameters
    parser.add_argument('--stage2-episodes', type=int, default=15000,
                       help='Number of Stage 2 episodes')
    parser.add_argument('--stage2-model-dir', type=str, default='ai/models/stage2',
                       help='Stage 2 model directory')
    parser.add_argument('--pretrained', type=str,
                       help='Path to Stage 1 pretrained model for Stage 2')
    parser.add_argument('--target-score', type=int, default=50,
                       help='Target average winning score for Stage 2')
    
    # Training options
    parser.add_argument('--shared-replay', action='store_true',
                       help='Use shared experience replay')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Training device')
    
    args = parser.parse_args()
    
    print("=== Mahjong DQN Training ===")
    print(f"Training stage: {args.stage}")
    print(f"Rule: {args.rule}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Load or create configuration
        if args.config and os.path.exists(args.config):
            print(f"Loading configuration from: {args.config}")
            stage1_config = load_config(args.config)
            stage2_config = load_config(args.config)
        else:
            print("Using default configuration")
            stage1_config = create_default_config('stage1', args.rule)
            stage2_config = create_default_config('stage2', args.rule)
        
        # Override with command line arguments
        stage1_config.update({
            'num_episodes': args.stage1_episodes,
            'model_dir': args.stage1_model_dir,
            'shared_replay': args.shared_replay,
            'rule_name': args.rule
        })
        
        stage2_config.update({
            'num_episodes': args.stage2_episodes,
            'model_dir': args.stage2_model_dir,
            'shared_replay': args.shared_replay,
            'rule_name': args.rule,
            'target_avg_score': args.target_score
        })
        
        # Run training based on stage selection
        if args.stage == 'stage1':
            run_stage1_training(stage1_config, args.resume)
            
        elif args.stage == 'stage2':
            if not args.pretrained:
                print("Warning: No pretrained model specified for Stage 2")
                print("Stage 2 will start from random initialization")
            
            run_stage2_training(stage2_config, args.pretrained, args.resume)
            
        elif args.stage == 'full':
            run_full_training_pipeline(stage1_config, stage2_config, args.resume, args.resume)
        
        print("\n🎉 Training completed successfully! 🎉")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()