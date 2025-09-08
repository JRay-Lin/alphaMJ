"""Interactive script for playing against trained AI agents."""

import os
import sys
import argparse
import torch
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MahjongGame, MahjongRule
from ai.dqn_agent import DQNAgent
from ai.dqn_model import MahjongStateEncoder
from modules.player import MahjongPlayer


class AIPlayer(MahjongPlayer):
    """AI player wrapper for trained DQN agents."""
    
    def __init__(self, name: str, index: int, model_path: str, device: str = "auto"):
        """
        Initialize AI player.
        
        Args:
            name: Player name
            index: Player index
            model_path: Path to trained model
            device: Computing device
        """
        super().__init__(name, index, is_ai=True)
        
        # Set device: CUDA > MPS > CPU
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load AI agent
        self.agent = DQNAgent(device=self.device)
        if os.path.exists(model_path):
            self.agent.load_model(model_path, load_optimizer=False)
            self.agent.set_training_mode(False)
            print(f"AI agent loaded: {name} from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # State encoder for converting game state
        self.state_encoder = MahjongStateEncoder()
    
    def get_ai_action(self, game_state: dict) -> int:
        """
        Get AI action for current game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Action index (tile to discard)
        """
        # Convert game state to neural network input
        state_vector = self._convert_game_state(game_state)
        state_tensor = torch.FloatTensor(state_vector).to(self.device)
        
        # Create action mask for valid actions
        valid_actions = list(range(len(self.hand.hands)))
        action_mask = self.state_encoder.create_action_mask(valid_actions, 14)
        action_mask = action_mask.to(self.device)
        
        # Get action from agent (no exploration)
        with torch.no_grad():
            action, q_values = self.agent.get_action(state_tensor, action_mask, epsilon=0.0)
        
        # Ensure action is valid
        if action >= len(self.hand.hands):
            action = len(self.hand.hands) - 1
        
        return action
    
    def _convert_game_state(self, game_state: dict) -> list:
        """Convert game state to state vector format."""
        # Convert tile names back to MahjongTile objects
        hand_tiles = []
        for tile_name in game_state.get('hand', []):
            if '_' in tile_name:
                number_str, class_name = tile_name.split('_', 1)
                number = int(number_str)
            else:
                number = 0
                class_name = tile_name
            
            from modules.tile import MahjongTile
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
                
                from modules.tile import MahjongTile
                meld_set.append(MahjongTile(number, class_name))
            sets.append(meld_set)
        
        # Create encoder state
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


class HumanVsAIGame(MahjongGame):
    """Extended Mahjong game for human vs AI play."""
    
    def __init__(self, rule: MahjongRule, ai_models: List[str], ai_positions: List[int] = None):
        """
        Initialize human vs AI game.
        
        Args:
            rule: Mahjong rule
            ai_models: List of paths to AI models
            ai_positions: List of positions for AI players (default: [1,2,3])
        """
        super().__init__(rule)
        
        self.ai_models = ai_models
        self.ai_positions = ai_positions or [1, 2, 3]  # Default: human is player 0
        
        # Validate inputs
        if len(ai_models) < len(self.ai_positions):
            raise ValueError("Not enough AI models for specified positions")
    
    def setup_game(self, ai_players: list = None):
        """Setup game with human and AI players."""
        print("Setting up Human vs AI game...")
        
        # Create players
        player_names = ["Human Player", "AI Player 1", "AI Player 2", "AI Player 3"]
        
        for i in range(self.num_players):
            if i in self.ai_positions and i - (1 if 0 not in self.ai_positions else 0) < len(self.ai_models):
                # Create AI player
                model_index = self.ai_positions.index(i)
                model_path = self.ai_models[model_index]
                
                try:
                    ai_player = AIPlayer(player_names[i], i, model_path)
                    self.players.append(ai_player)
                except FileNotFoundError as e:
                    print(f"Error loading AI model: {e}")
                    # Fallback to regular AI player
                    self.players.append(MahjongPlayer(f"{player_names[i]} (Random)", i, True))
            else:
                # Create human player
                self.players.append(MahjongPlayer(player_names[i], i, False))
        
        # Print player setup
        for player in self.players:
            player_type = "AI (Trained)" if isinstance(player, AIPlayer) else "AI (Random)" if player.is_ai else "Human"
            print(f"Created {player.name} - {player_type} with wind: {player.self_wind}")
        
        # Shuffle wall and deal initial hands
        self.wall.shuffle()
        self.deal_initial_hands()
        
        print(f"Game setup complete. Starting player: {self.players[self.current_player].name}")
    
    def get_ai_discard_choice(self, player_obj) -> int:
        """Get AI discard choice for trained AI agents."""
        if isinstance(player_obj, AIPlayer):
            # Use trained AI agent
            game_state = self.get_game_state_for_ai(player_obj.index)
            action = player_obj.get_ai_action(game_state)
            
            print(f"🤖 {player_obj.name} (Trained AI) chooses to discard tile {action}: {player_obj.hand.hands[action]}")
            return action
        else:
            # Use default random AI
            return super().get_ai_discard_choice(player_obj)
    
    def get_ai_action_decision(self, player_obj, action_type: str, discarded_tile, action_data) -> bool:
        """Get AI action decision for trained agents."""
        if isinstance(player_obj, AIPlayer):
            # For now, use simple heuristics for special actions
            # A full implementation would extend the AI to handle these decisions
            import random
            
            action_probabilities = {
                "kong": 0.9,
                "pong": 0.7,
                "chi": 0.5,
            }
            
            probability = action_probabilities.get(action_type, 0.5)
            decision = random.random() < probability
            
            print(f"🤖 {player_obj.name} (Trained AI) {'accepts' if decision else 'declines'} {action_type}")
            return decision
        else:
            return super().get_ai_action_decision(player_obj, action_type, discarded_tile, action_data)
    
    def display_game_statistics(self):
        """Display game statistics including AI performance."""
        print(f"\n--- Game Statistics ---")
        print(f"Turn count: {self.turn_count}")
        print(f"Tiles remaining: {len(self.wall.tiles)}")
        
        for player in self.players:
            hand_size = len(player.hand.hands)
            discard_count = len(player.hand.discards)
            sets_count = len(player.hand.sets)
            
            player_type = "Trained AI" if isinstance(player, AIPlayer) else "Human" if not player.is_ai else "Random AI"
            print(f"{player.name} ({player_type}): {hand_size} tiles, {discard_count} discards, {sets_count} sets")
    
    def run_interactive_game(self):
        """Run interactive game with enhanced display."""
        self.setup_game()
        
        print(f"\n🀄 Starting Interactive Mahjong Game! 🀄")
        print(f"Rule: {self.rule.rule_name.title()}")
        print("=" * 60)
        
        # Show initial game state
        self.display_game_statistics()
        
        # Main game loop
        while not self.game_over:
            try:
                self.play_turn()
                
                # Show statistics every few turns for human player awareness
                if self.turn_count % 8 == 0:
                    self.display_game_statistics()
                
                # Safety check
                if self.turn_count > 200:
                    print("Game has gone on too long. Ending in draw.")
                    self.game_over = True
                    
            except KeyboardInterrupt:
                print("\n\nGame interrupted by user. Goodbye!")
                break
        
        # Game end
        if self.winner:
            print(f"\n🎉 Game Over! {self.winner.name} wins! 🎉")
            
            # Show winner's hand
            self.display_player_hand(self.winner)
            self.display_player_sets(self.winner)
            
            # Calculate and display score if possible
            try:
                from ai.scoring_system import MahjongScoring
                scorer = MahjongScoring(rule=self.rule.rule_name)
                
                game_context = {
                    'self_draw': True,  # Simplified
                    'is_dealer': self.winner.index == 0
                }
                
                score = scorer.calculate_hand_score(
                    self.winner.hand.hands,
                    self.winner.hand.sets,
                    game_context
                )
                print(f"Winning hand score: {score} points")
                
            except Exception as e:
                print(f"Could not calculate score: {e}")
                
        else:
            print("\n Game ended in a draw.")
        
        # Final statistics
        self.display_game_statistics()
        print("Thanks for playing!")


def main():
    """Main function for human vs AI play."""
    parser = argparse.ArgumentParser(description='Play Mahjong against trained AI')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Paths to AI model files')
    parser.add_argument('--rule', choices=['standard', 'taiwan'], default='standard',
                       help='Mahjong rule variant')
    parser.add_argument('--ai-positions', nargs='+', type=int, default=[1, 2, 3],
                       help='Positions for AI players (0-3, default: 1 2 3)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device for AI')
    
    args = parser.parse_args()
    
    print("=== Mahjong: Human vs AI ===")
    print(f"Rule: {args.rule}")
    print(f"AI Models: {args.models}")
    print(f"AI Positions: {args.ai_positions}")
    print()
    
    try:
        # Create game
        rule = MahjongRule(args.rule, 4)
        game = HumanVsAIGame(rule, args.models, args.ai_positions)
        
        # Show instructions
        print("Instructions:")
        print("- You are the human player")
        print("- When it's your turn, you'll see your tiles with indices")
        print("- Enter the index number of the tile you want to discard")
        print("- Press Ctrl+C to quit anytime")
        print()
        
        input("Press Enter to start the game...")
        
        # Run game
        game.run_interactive_game()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure all AI model files exist.")
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()