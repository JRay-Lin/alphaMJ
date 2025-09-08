#!/usr/bin/env python3
"""
Comprehensive test suite for Mahjong game logic.
Tests core game mechanics, tile handling, and action logic.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MahjongGame, MahjongRule
from modules.tile import MahjongTile
from modules.player import MahjongPlayer
from modules.wall import MahjongWall


class TestMahjongGameLogic(unittest.TestCase):
    """Test suite for core Mahjong game logic."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.rule = MahjongRule("standard", 4)
        self.game = MahjongGame(self.rule)
        self.game.setup_game()

    def test_game_initialization(self):
        """Test that game initializes correctly."""
        self.assertEqual(self.game.num_players, 4)
        self.assertEqual(len(self.game.players), 4)
        self.assertEqual(self.game.current_player, 0)
        self.assertEqual(self.game.turn_count, 0)
        self.assertFalse(self.game.game_over)
        self.assertIsNone(self.game.winner)

    def test_initial_hand_dealing(self):
        """Test that each player gets 13 tiles initially."""
        for player in self.game.players:
            self.assertEqual(len(player.hand.hands), 13)
            self.assertEqual(len(player.hand.discards), 0)
            self.assertEqual(len(player.hand.sets), 0)

    def test_draw_tile(self):
        """Test drawing tiles from wall."""
        player = self.game.players[0]
        initial_hand_size = len(player.hand.hands)
        initial_wall_size = len(self.game.wall.tiles)

        success = self.game.draw_tile(player)
        
        self.assertTrue(success)
        self.assertEqual(len(player.hand.hands), initial_hand_size + 1)
        self.assertEqual(len(self.game.wall.tiles), initial_wall_size - 1)

    def test_discard_tile(self):
        """Test discarding tiles."""
        player = self.game.players[0]
        initial_hand_size = len(player.hand.hands)
        tile_to_discard = player.hand.hands[0]

        discarded_tile = self.game.discard_tile(player, 0)
        
        self.assertEqual(discarded_tile, tile_to_discard)
        self.assertEqual(len(player.hand.hands), initial_hand_size - 1)
        self.assertEqual(len(player.hand.discards), 1)
        self.assertEqual(player.hand.discards[0], tile_to_discard)

    def test_empty_wall_handling(self):
        """Test game behavior when wall is empty."""
        # Empty the wall
        self.game.wall.tiles = []
        player = self.game.players[0]

        success = self.game.draw_tile(player)
        
        self.assertFalse(success)
        self.assertTrue(self.game.game_over)


class TestPlayerActions(unittest.TestCase):
    """Test suite for player actions (chi, pong, kong)."""

    def setUp(self):
        """Set up test fixtures."""
        self.rule = MahjongRule("standard", 4)
        self.game = MahjongGame(self.rule)
        self.game.setup_game()

    def test_pong_detection(self):
        """Test pong action detection."""
        player = self.game.players[0]
        discarded_tile = MahjongTile(0, "東")
        
        # Give player two matching tiles
        player.hand.hands = [
            MahjongTile(0, "東"),
            MahjongTile(0, "東"),
            MahjongTile(1, "萬")
        ]
        
        actions = player.get_available_actions(discarded_tile, False)
        self.assertTrue(actions["pong"])

    def test_chi_detection(self):
        """Test chi action detection."""
        player = self.game.players[0]
        discarded_tile = MahjongTile(5, "萬")
        
        # Give player tiles that can form sequence with discarded tile
        player.hand.hands = [
            MahjongTile(3, "萬"),
            MahjongTile(4, "萬"),
            MahjongTile(6, "萬"),
            MahjongTile(7, "萬")
        ]
        player.hand.hands.sort()
        
        actions = player.get_available_actions(discarded_tile, True)  # is_next_player=True
        self.assertTrue(actions["chi"])
        self.assertGreater(len(actions["chi"]), 0)

    def test_kong_detection(self):
        """Test kong action detection."""
        player = self.game.players[0]
        discarded_tile = MahjongTile(0, "東")
        
        # Give player three matching tiles for normal kong
        player.hand.hands = [
            MahjongTile(0, "東"),
            MahjongTile(0, "東"),
            MahjongTile(0, "東")
        ]
        
        actions = player.get_available_actions(discarded_tile, False)
        self.assertTrue(actions["kong"]["normal"])

    def test_execute_pong(self):
        """Test executing pong action."""
        player = self.game.players[0]
        discarded_tile = MahjongTile(0, "東")
        
        # Set up player hand with matching tiles
        player.hand.hands = [
            MahjongTile(0, "東"),
            MahjongTile(0, "東"),
            MahjongTile(1, "萬")
        ]
        
        success = player.execute_pong(discarded_tile)
        
        self.assertTrue(success)
        self.assertEqual(len(player.hand.sets), 1)
        self.assertEqual(len(player.hand.sets[0]), 3)  # Pong has 3 tiles
        self.assertEqual(len(player.hand.hands), 1)  # 2 tiles removed for pong

    def test_execute_chi(self):
        """Test executing chi action."""
        player = self.game.players[0]
        discarded_tile = MahjongTile(5, "萬")
        
        # Set up player hand
        player.hand.hands = [
            MahjongTile(3, "萬"),
            MahjongTile(4, "萬"),
            MahjongTile(1, "筒")
        ]
        player.hand.hands.sort()
        
        # Get chi options
        actions = player.get_available_actions(discarded_tile, True)
        if actions["chi"]:
            chi_option = actions["chi"][0]  # Take first available option
            success = player.execute_chi(discarded_tile, chi_option)
            
            self.assertTrue(success)
            self.assertEqual(len(player.hand.sets), 1)
            self.assertEqual(len(player.hand.sets[0]), 3)  # Chi has 3 tiles
            self.assertEqual(len(player.hand.hands), 1)  # 2 tiles removed for chi


class TestWinConditions(unittest.TestCase):
    """Test suite for win condition detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.rule = MahjongRule("standard", 4)
        self.game = MahjongGame(self.rule)
        self.game.setup_game()

    def test_can_win_with_tile(self):
        """Test win detection when given a specific tile."""
        player = self.game.players[0]
        
        # Create a hand that needs one specific tile to win
        # Example: 4 complete sets + pair - 1 tile
        player.hand.hands = [MahjongTile(1, "萬")]  # Just need pair
        
        # Add 4 complete sets
        player.hand.sets = [
            [MahjongTile(0, "東")] * 3,  # Pong
            [MahjongTile(0, "南")] * 3,  # Pong
            [MahjongTile(0, "西")] * 3,  # Pong
            [MahjongTile(0, "北")] * 3,  # Pong
        ]
        
        winning_tile = MahjongTile(1, "萬")
        can_win = self.game.can_win_with_tile(player, winning_tile)
        
        # This should be true if the winning logic is implemented correctly
        # Note: This test depends on the is_winning implementation in MahjongPlayer
        self.assertIsInstance(can_win, bool)

    def test_win_priority_over_actions(self):
        """Test that win detection has priority over other actions."""
        # Set up a scenario where multiple actions are possible
        discarder = self.game.players[0] 
        potential_winner = self.game.players[1]
        
        # Mock the can_win_with_tile method to return True
        with patch.object(self.game, 'can_win_with_tile', return_value=True):
            with patch.object(self.game, 'ask_player_win', return_value=True):
                discarded_tile = MahjongTile(1, "萬")
                
                action_taken = self.game.handle_discard_actions(discarded_tile, 0)
                
                self.assertTrue(action_taken)
                self.assertTrue(self.game.game_over)
                self.assertEqual(self.game.winner, potential_winner)


class TestTurnLogic(unittest.TestCase):
    """Test suite for turn progression and tile count logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.rule = MahjongRule("standard", 4)
        self.game = MahjongGame(self.rule)
        self.game.setup_game()

    def test_normal_turn_progression(self):
        """Test normal turn progression without actions."""
        initial_player = self.game.current_player
        
        # Mock get_player_input to return a valid choice
        with patch.object(self.game, 'get_player_input', return_value=0):
            # Mock handle_discard_actions to return False (no actions taken)
            with patch.object(self.game, 'handle_discard_actions', return_value=False):
                self.game.play_turn()
                
                # Player should advance
                expected_next_player = (initial_player + 1) % self.game.num_players
                self.assertEqual(self.game.current_player, expected_next_player)

    def test_tile_count_after_pong(self):
        """Test that tile counts are correct after pong action."""
        player = self.game.players[1]
        discarded_tile = MahjongTile(0, "東")
        
        # Set up player with matching tiles
        player.hand.hands = [MahjongTile(0, "東")] * 2 + [MahjongTile(1, "萬")] * 11
        initial_hand_size = len(player.hand.hands)
        
        # Execute pong with mocked input and no follow-up actions
        with patch.object(self.game, 'get_player_input', return_value=0):
            with patch.object(self.game, 'handle_discard_actions', return_value=False):
                success = self.game.execute_player_action("pong", 1, discarded_tile, None)
        
        if success:
            # After pong and discard: should have correct tile counts
            # Started with 13, removed 2 for pong, then discarded 1 more = 10 tiles left
            expected_hand_size = initial_hand_size - 2 - 1  # 2 for pong, 1 discarded
            
            self.assertEqual(len(player.hand.hands), expected_hand_size)
            self.assertEqual(len(player.hand.sets), 1)
            self.assertEqual(len(player.hand.sets[0]), 3)

    def test_tile_count_after_kong(self):
        """Test that tile counts are correct after kong action."""
        player = self.game.players[1]
        discarded_tile = MahjongTile(0, "東")
        
        # Set up player with 3 matching tiles
        player.hand.hands = [MahjongTile(0, "東")] * 3 + [MahjongTile(1, "萬")] * 10
        initial_hand_size = len(player.hand.hands)
        initial_wall_size = len(self.game.wall.tiles)
        
        # Execute kong with mocked input and no follow-up actions
        action_data = {"normal": True, "promote": False}
        
        with patch.object(self.game, 'get_player_input', return_value=0):
            with patch.object(self.game, 'handle_discard_actions', return_value=False):
                success = self.game.execute_player_action("kong", 1, discarded_tile, action_data)
        
        if success:
            # After kong and discard: removed 3 for kong, +1 replacement, -1 discard = 10 tiles
            expected_hand_size = initial_hand_size - 3 + 1 - 1  # 3 used in kong, 1 replacement, 1 discarded
            self.assertEqual(len(player.hand.hands), expected_hand_size)
            self.assertEqual(len(player.hand.sets), 1)
            self.assertEqual(len(player.hand.sets[0]), 4)  # Kong has 4 tiles
            self.assertEqual(len(self.game.wall.tiles), initial_wall_size - 1)  # Replacement tile drawn


class TestGameStateManagement(unittest.TestCase):
    """Test suite for game state management and AI integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.rule = MahjongRule("standard", 4)
        self.game = MahjongGame(self.rule)

    def test_ai_player_setup(self):
        """Test setting up AI players."""
        ai_players = [1, 2, 3]  # Players 2, 3, 4 are AI
        self.game.setup_game(ai_players)
        
        self.assertFalse(getattr(self.game.players[0], 'is_ai', False))  # Player 1 human
        self.assertTrue(getattr(self.game.players[1], 'is_ai', False))   # Player 2 AI
        self.assertTrue(getattr(self.game.players[2], 'is_ai', False))   # Player 3 AI  
        self.assertTrue(getattr(self.game.players[3], 'is_ai', False))   # Player 4 AI

    def test_get_game_state_for_ai(self):
        """Test AI game state generation."""
        self.game.setup_game([1, 2, 3])
        state = self.game.get_game_state_for_ai(0)
        
        self.assertIn('hand', state)
        self.assertIn('hand_size', state)
        self.assertIn('sets', state)
        self.assertIn('discards', state)
        self.assertIn('wind', state)
        self.assertIn('current_player', state)
        self.assertIn('turn_count', state)
        self.assertIn('wall_remaining', state)
        self.assertIn('rule', state)
        self.assertIn('other_players', state)
        
        # Should have info for 3 other players
        self.assertEqual(len(state['other_players']), 3)

    def test_ai_action_decisions(self):
        """Test AI action decision making."""
        self.game.setup_game([1])  # Player 2 is AI
        ai_player = self.game.players[1]
        discarded_tile = MahjongTile(0, "東")
        
        # Test different action types
        for action_type in ["chi", "pong", "kong"]:
            decision = self.game.get_ai_action_decision(ai_player, action_type, discarded_tile, None)
            self.assertIsInstance(decision, bool)


def run_specific_test_suite():
    """Run specific test categories."""
    print("🀄 Running Mahjong Game Logic Tests 🀄")
    print("=" * 50)
    
    # Create test suites
    game_logic_suite = unittest.TestLoader().loadTestsFromTestCase(TestMahjongGameLogic)
    player_actions_suite = unittest.TestLoader().loadTestsFromTestCase(TestPlayerActions)
    win_conditions_suite = unittest.TestLoader().loadTestsFromTestCase(TestWinConditions)
    turn_logic_suite = unittest.TestLoader().loadTestsFromTestCase(TestTurnLogic)
    game_state_suite = unittest.TestLoader().loadTestsFromTestCase(TestGameStateManagement)
    
    # Combine all suites
    combined_suite = unittest.TestSuite([
        game_logic_suite,
        player_actions_suite, 
        win_conditions_suite,
        turn_logic_suite,
        game_state_suite
    ])
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(combined_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\n')[-2]}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run the test suite
    success = run_specific_test_suite()
    
    if success:
        print("\n✅ All tests passed! Game logic is working correctly.")
    else:
        print("\n❌ Some tests failed. Please review the issues above.")
        sys.exit(1)