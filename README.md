# AlphaMJ - Mahjong AI Training Environment

A comprehensive Deep Q-Learning training environment for Mahjong AI with multi-stage learning pipeline and flexible rule systems.

## Features

- **Deep Q-Learning AI**: Two-stage training system with DQN agents
  - Stage 1: Basic win/loss learning 
  - Stage 2: Advanced scoring system integration
- **Training Environment**: Complete game simulation with state encoding
- **Multiple Rule Systems**: 
  - Standard (14 tiles: 4 melds + 1 pair)
<<<<<<< HEAD
  - Taiwan (17 tiles: 5 melds + 1 pair)
- **Hardware Acceleration**: Automatic device selection (CUDA > MPS > CPU)
- **Interactive Training**: Jupyter notebook with validation between stages
- **Comprehensive Evaluation**: Performance analysis and model comparison tools
- **DFS Win Detection**: Uses depth-first search with backtracking for efficient win detection
=======
  - Taiwan (17 tiles: 5 melds + 1 pair) 
- **Kong Support**: Handles kong sets (4 identical tiles) automatically
- **Comprehensive Testing**: Full unittest suite with 15 test cases
>>>>>>> b57a1821bed6780b641f875c11a8cf4f6db1c526

## Quick Start

### Interactive Training (Recommended)
```bash
# Start interactive training notebook
jupyter notebook train.ipynb
```

### Command Line Training
```bash
# Install dependencies
uv sync

# Full training pipeline (Stage 1 → Stage 2)
python train.py --stage full

# Stage 1 only (basic win/loss)
python ai/train_stage1.py

# Stage 2 only (scoring system)
python ai/train_stage2.py --pretrained ai/models/stage1

# Play against trained AI
python ai/play_vs_ai.py --models ai/models/stage2/best_model.pth
```

### Game Engine Usage
```python
from modules.winChecker import is_winning_hand
from modules.tile import mj_tile

# Standard 14-tile winning hand
standard_hand = [
    mj_tile(1, ","), mj_tile(1, ","), mj_tile(1, ","),  # pung
    mj_tile(2, ","), mj_tile(3, ","), mj_tile(4, ","),  # chow
    mj_tile(5, ","), mj_tile(6, ","), mj_tile(7, ","),  # chow
    mj_tile(1, "R"), mj_tile(2, "R"), mj_tile(3, "R"),  # chow
    mj_tile(4, "R"), mj_tile(4, "R")                     # pair
]

# Check winning hand
print(is_winning_hand(standard_hand, 'standard'))  # True
<<<<<<< HEAD
=======

# Taiwan 17-tile hand
taiwan_hand = [
    # ... 5 melds + 1 pair = 17 tiles
]
print(is_winning_hand(taiwan_hand, 'taiwan'))     # True

# Hand with kong sets
kong_hand = [
    mj_tile(1, ","), mj_tile(1, ","), mj_tile(1, ","), mj_tile(1, ","),  # kong
    # ... other melds + pair = 15 tiles total
]
print(is_winning_hand(kong_hand))     # True
>>>>>>> b57a1821bed6780b641f875c11a8cf4f6db1c526
```

## AI Training Pipeline

### Stage 1: Basic Game Knowledge
- **Objective**: Learn fundamental Mahjong rules and winning patterns
- **Rewards**: Winner +1, Loser -0.5, small negative per turn
- **Training Method**: Self-play with epsilon-greedy exploration
- **Output**: Base model understanding basic game mechanics

### Stage 2: Scoring System Integration
- **Objective**: Learn to maximize hand values and scoring combinations
- **Rewards**: Win rewards scaled by hand value, risk-reward balance
- **Training Method**: Transfer learning from Stage 1 model
- **Output**: Advanced model optimizing for realistic scoring scenarios

## Rule Systems

### Standard Rule (`'standard'`)
- **Tile Count**: 14 tiles
- **Structure**: 4 melds (pung/chow/kong) + 1 pair
- **Usage**: Most common mahjong variant

### Taiwan Rule (`'taiwan'`)
- **Tile Count**: 17 tiles  
- **Structure**: 5 melds (pung/chow/kong) + 1 pair
- **Usage**: Popular in Taiwan mahjong (3×5+2=17)


## Meld Types

<<<<<<< HEAD
1. **Pung**: 3 identical tiles (e.g., ,,,)
2. **Chow**: 3 consecutive tiles of same suit (e.g., 1,2,3,)
3. **Kong**: 4 identical tiles (e.g., ,,,,)
4. **Pair**: 2 identical tiles (e.g., RR)

## Hardware Acceleration

The training system automatically selects the best available computing device:
1. **CUDA** (NVIDIA GPUs) - Fastest option for training
2. **MPS** (Apple Silicon M1/M2/M3) - GPU acceleration on Mac
3. **CPU** - Fallback for compatibility

Device selection is automatic but can be overridden with `--device` parameter.
=======
1. **Pung (;P)**: 3 identical tiles (e.g., ,,,)
2. **Chow (P)**: 3 consecutive tiles of same suit (e.g., 1,2,3,)
3. **Kong (ÓP)**: 4 identical tiles (e.g., ,,,,)
4. **Pair (
P)**: 2 identical tiles (e.g., RR)
>>>>>>> b57a1821bed6780b641f875c11a8cf4f6db1c526

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --category game      # Basic game logic
python run_tests.py --category actions   # Player actions (chi/pong/kong)
python run_tests.py --category win       # Win conditions
python run_tests.py --category turns     # Turn logic
python run_tests.py --category state     # Game state management
```

**Test Coverage**:
- ✓ Standard 14-tile hands
- ✓ Taiwan 17-tile hands  
- ✓ Kong sets (single and multiple)
- ✓ Edge cases and boundary conditions
- ✓ AI integration and game state management

## Model Evaluation

```bash
# Evaluate trained models
python ai/evaluate.py --model ai/models/stage2/best_model.pth --games 1000

# Play against trained AI
python ai/play_vs_ai.py --models ai/models/stage2/best_model.pth
```

---

**Algorithm**: DFS with backtracking  
**Time Complexity**: O(3^n) where n is number of tile groups  
<<<<<<< HEAD
**Space Complexity**: O(n) for recursion stack
=======
**Space Complexity**: O(n) for recursion stack

Built with d for flexible mahjong win detection.
>>>>>>> b57a1821bed6780b641f875c11a8cf4f6db1c526
