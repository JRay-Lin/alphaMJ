#!/usr/bin/env python3
"""
Test runner for Mahjong game logic tests.
Provides easy interface to run different test categories.
"""

import sys
import os
import argparse
import unittest

# Add the project root and test directory to Python path  
project_root = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(project_root, 'test')
sys.path.insert(0, project_root)
sys.path.insert(0, test_dir)

# Import test modules directly
from test_game_logic import (
    TestMahjongGameLogic,
    TestPlayerActions, 
    TestWinConditions,
    TestTurnLogic,
    TestGameStateManagement,
    run_specific_test_suite
)


def run_category_tests(category):
    """Run tests for a specific category."""
    test_classes = {
        'game': TestMahjongGameLogic,
        'actions': TestPlayerActions,
        'win': TestWinConditions, 
        'turns': TestTurnLogic,
        'state': TestGameStateManagement
    }
    
    if category not in test_classes:
        print(f"Unknown category: {category}")
        print(f"Available categories: {', '.join(test_classes.keys())}")
        return False
    
    print(f"🀄 Running {category.title()} Tests 🀄")
    print("=" * 40)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(test_classes[category])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Run Mahjong game logic tests')
    parser.add_argument(
        '--category', '-c',
        choices=['all', 'game', 'actions', 'win', 'turns', 'state'],
        default='all',
        help='Test category to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.category == 'all':
        success = run_specific_test_suite()
    else:
        success = run_category_tests(args.category)
    
    if success:
        print(f"\n✅ {args.category.title()} tests completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ {args.category.title()} tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()