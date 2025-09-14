# AlphaMJ - Flexible Mahjong Win Checker

A flexible DFS-based mahjong win checker that supports multiple rule systems including standard, Taiwan, and kong variations.

## Features

- **DFS Algorithm**: Uses depth-first search with backtracking for efficient win detection
- **Multiple Rule Systems**: 
  - Standard (14 tiles: 4 melds + 1 pair)
  - Taiwan (17 tiles: 5 melds + 1 pair) 
- **Kong Support**: Handles kong sets (4 identical tiles) automatically
- **Comprehensive Testing**: Full unittest suite with 15 test cases

## Quick Start

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

# Check with different rules
print(is_winning_hand(standard_hand, 'standard'))  # True

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
```

## Rule Systems

### Standard Rule (`'standard'`)
- **Tile Count**: 14 tiles
- **Structure**: 4 melds (pung/chow/kong) + 1 pair
- **Typical**: Most common mahjong variant

### Taiwan Rule (`'taiwan'`)
- **Tile Count**: 17 tiles  
- **Structure**: 5 melds (pung/chow/kong) + 1 pair
- **Usage**: Popular in Taiwan mahjong (3×5+2=17)


## Meld Types

1. **Pung (;P)**: 3 identical tiles (e.g., ,,,)
2. **Chow (P)**: 3 consecutive tiles of same suit (e.g., 1,2,3,)
3. **Kong (ÓP)**: 4 identical tiles (e.g., ,,,,)
4. **Pair (
P)**: 2 identical tiles (e.g., RR)

## Testing

Run the comprehensive test suite:

```bash
python test/test_winChecker.py
```

**Test Coverage**:
-  Standard 14-tile hands
-  Taiwan 17-tile hands  
-  Kong sets (single and multiple)
-  Flexible rule auto-detection
-  Various tile counts (11, 15, 16, 20 tiles)
-  Edge cases and boundary conditions
-  Invalid rule handling

## Examples

See `python modules/winChecker.py` for interactive examples of all rule systems.

---

**Algorithm**: DFS with backtracking  
**Time Complexity**: O(3^n) where n is number of tile groups  
**Space Complexity**: O(n) for recursion stack

Built with d for flexible mahjong win detection.
