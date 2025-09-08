# Mahjong Game Logic Tests

This directory contains comprehensive tests for the Mahjong game logic implementation.

## Test Files

- **`test/test_game_logic.py`** - Main test suite with all test cases
- **`run_tests.py`** - Test runner with category support (in project root)
- **`TEST_README.md`** - This documentation file

## Test Categories

### 1. Game Logic Tests (`game`)
Tests core game initialization and basic mechanics:
- Game setup and initialization
- Initial hand dealing (13 tiles per player)
- Drawing tiles from wall
- Discarding tiles
- Empty wall handling

### 2. Player Actions Tests (`actions`) 
Tests chi/pong/kong action mechanics:
- Action detection (can player perform action?)
- Action execution (properly modify hand and sets)
- Tile count validation after actions
- Priority handling

### 3. Win Conditions Tests (`win`)
Tests winning logic and priority:
- Win detection with specific tiles
- Win priority over other actions
- Proper game ending on win declaration

### 4. Turn Logic Tests (`turns`)
Tests turn progression and tile counts:
- Normal turn progression
- Correct tile counts after actions
- Turn skipping logic after chi/pong
- Kong replacement tile handling

### 5. Game State Management Tests (`state`)
Tests AI integration and game state:
- AI player setup
- Game state generation for AI
- AI decision making interfaces

## Running Tests

### Run All Tests
```bash
python run_tests.py
# or
python run_tests.py --category all
```

### Run Specific Category
```bash
python run_tests.py --category game      # Basic game logic
python run_tests.py --category actions   # Player actions
python run_tests.py --category win       # Win conditions
python run_tests.py --category turns     # Turn logic
python run_tests.py --category state     # Game state management
```

### Run Tests Directly
```bash
python test/test_game_logic.py         # Run all tests
python -m unittest test.test_game_logic # Alternative method
```

## Test Coverage

The test suite covers:

✅ **Core Game Mechanics**
- Game initialization and setup
- Tile drawing and discarding
- Hand management
- Wall management

✅ **Player Actions**
- Chi/Pong/Kong detection
- Action execution
- Tile count validation
- Set formation

✅ **Win Conditions**
- Win detection algorithms
- Priority over other actions
- Game ending logic

✅ **Turn Management**
- Turn progression rules
- Correct tile counts after actions
- Special turn handling (chi/pong/kong)

✅ **AI Integration**
- Game state generation
- AI decision interfaces
- AI player setup

## Example Output

```
🀄 Running Mahjong Game Logic Tests 🀄
==================================================
test_discard_tile ... ok
test_draw_tile ... ok
test_empty_wall_handling ... ok
test_game_initialization ... ok
test_initial_hand_dealing ... ok

Tests run: 25
Failures: 0
Errors: 0
Success Rate: 100.0%

✅ All tests passed! Game logic is working correctly.
```

## Adding New Tests

To add new tests:

1. **Add test methods** to existing test classes in `test_game_logic.py`
2. **Follow naming convention**: `test_<functionality_being_tested>`
3. **Use descriptive docstrings** explaining what the test validates
4. **Include setup/teardown** if needed in `setUp()` method
5. **Test both success and failure cases** where applicable

### Example Test Structure

```python
def test_new_functionality(self):
    """Test description of what this validates."""
    # Arrange
    setup_data = self.create_test_scenario()
    
    # Act  
    result = self.game.some_method(setup_data)
    
    # Assert
    self.assertEqual(result, expected_value)
    self.assertTrue(some_condition)
```

## Debugging Failed Tests

When tests fail:

1. **Check the error message** - shows which assertion failed
2. **Review test setup** - ensure test data is correct
3. **Add debug prints** if needed to understand state
4. **Run single test** to isolate the issue:
   ```bash
   python -m unittest test_game_logic.TestClassName.test_method_name
   ```

## Integration with Development

- **Run tests before commits** to ensure no regressions
- **Add tests for new features** as they're developed
- **Update tests** when game rules change
- **Use tests to validate bug fixes**

The test suite ensures the Mahjong game logic remains correct and reliable as the codebase evolves.