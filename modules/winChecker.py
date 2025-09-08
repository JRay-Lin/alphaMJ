try:
    from tile import MahjongTile
except ImportError:
    from .tile import MahjongTile
from collections import Counter
from typing import List, Tuple


class win_checker:
    def __init__(self, rule: str = "standard") -> None:
        self.rule = rule
        pass

    def is_winning_hand(self, tiles: List[MahjongTile]) -> bool:
        """
        Determine if a mahjong hand is a winning hand using DFS.

        Args:
            tiles: List of mahjong tiles
            rule: Winning rule to use
                - "standard": 4 melds + 1 pair = 14 tiles
                - "taiwan": 5 melds + 1 pair = 17 tiles

        Returns:
            bool: True if the hand is winning according to the specified rule
        """
        tile_count = len(tiles)

        # Determine expected melds and pairs based on rule
        if self.rule == "standard":
            expected_melds, expected_pairs = 4, 1
            expected_total = 14
        elif self.rule == "taiwan":
            expected_melds, expected_pairs = 5, 1
            expected_total = 17
        else:
            raise ValueError(f"Unknown rule: {self.rule}. Use 'standard', 'taiwan'")

        if tile_count != expected_total:
            return False

        # Count tiles for easier processing
        tile_counts = Counter()
        for tile in tiles:
            tile_counts[tile.name] += 1

        # Try to find winning combination
        return self.dfs_check_win(tile_counts, 0, False, expected_melds, expected_pairs)

    def dfs_check_win(
        self,
        tile_counts: Counter,
        melds_formed: int,
        has_pair: bool,
        required_melds: int = 4,
        required_pairs: int = 1,
    ) -> bool:
        """
        DFS function to check if the remaining tiles can form a winning combination.

        Args:
            tile_counts: Counter of remaining tiles
            melds_formed: Number of melds (pung/chow/kong) formed so far
            has_pair: Whether a pair has been formed
            required_melds: Number of melds required for winning
            required_pairs: Number of pairs required for winning
        """
        # Base case: if we have required melds and pairs, we win
        if melds_formed == required_melds and has_pair:
            return sum(tile_counts.values()) == 0

        # If no tiles left but not enough melds/pairs, fail
        if sum(tile_counts.values()) == 0:
            return False

        # Get the first available tile (lexicographically first)
        first_tile_name = min(tile_counts.keys())
        first_tile = self.parse_tile_name(first_tile_name)

        # Try to form a pair (only if we don't have required pairs yet)
        if not has_pair and tile_counts[first_tile_name] >= 2:
            tile_counts[first_tile_name] -= 2
            if tile_counts[first_tile_name] == 0:
                del tile_counts[first_tile_name]
            if self.dfs_check_win(
                tile_counts, melds_formed, True, required_melds, required_pairs
            ):
                tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 2
                return True
            tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 2

        # Try to form a kong (4 identical tiles)
        if tile_counts[first_tile_name] >= 4:
            tile_counts[first_tile_name] -= 4
            if tile_counts[first_tile_name] == 0:
                del tile_counts[first_tile_name]
            if self.dfs_check_win(
                tile_counts, melds_formed + 1, has_pair, required_melds, required_pairs
            ):
                tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 4
                return True
            tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 4

        # Try to form a pung (3 identical tiles)
        if tile_counts[first_tile_name] >= 3:
            tile_counts[first_tile_name] -= 3
            if tile_counts[first_tile_name] == 0:
                del tile_counts[first_tile_name]
            if self.dfs_check_win(
                tile_counts, melds_formed + 1, has_pair, required_melds, required_pairs
            ):
                tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 3
                return True
            tile_counts[first_tile_name] = tile_counts.get(first_tile_name, 0) + 3

        # Try to form a chow (3 consecutive tiles of same suit)
        if self.can_form_chow(tile_counts, first_tile):
            self.remove_chow(tile_counts, first_tile)
            if self.dfs_check_win(
                tile_counts, melds_formed + 1, has_pair, required_melds, required_pairs
            ):
                self.add_chow(tile_counts, first_tile)
                return True
            self.add_chow(tile_counts, first_tile)

        return False

    def parse_tile_name(self, tile_name: str) -> Tuple[int, str]:
        """Parse tile name into number and class_name."""
        if "_" in tile_name:
            number_str, class_name = tile_name.split("_", 1)
            return int(number_str), class_name
        else:
            return 0, tile_name

    def can_form_chow(self, tile_counts: Counter, tile: Tuple[int, str]) -> bool:
        """Check if we can form a chow starting from this tile."""
        number, class_name = tile

        # Only numbered tiles (萬, 筒, 條) can form chows
        if class_name not in ["萬", "筒", "條"] or number == 0:
            return False

        # Need consecutive numbers (n, n+1, n+2)
        if number > 7:  # Can't form chow starting from 8 or 9
            return False

        tile1_name = f"{number}_{class_name}"
        tile2_name = f"{number + 1}_{class_name}"
        tile3_name = f"{number + 2}_{class_name}"

        return (
            tile_counts[tile1_name] >= 1
            and tile_counts[tile2_name] >= 1
            and tile_counts[tile3_name] >= 1
        )

    def remove_chow(self, tile_counts: Counter, tile: Tuple[int, str]):
        """Remove a chow starting from this tile."""
        number, class_name = tile

        tile1_name = f"{number}_{class_name}"
        tile2_name = f"{number + 1}_{class_name}"
        tile3_name = f"{number + 2}_{class_name}"

        tile_counts[tile1_name] -= 1
        tile_counts[tile2_name] -= 1
        tile_counts[tile3_name] -= 1

        # Remove entries with 0 count to keep counter clean
        if tile_counts[tile1_name] == 0:
            del tile_counts[tile1_name]
        if tile_counts[tile2_name] == 0:
            del tile_counts[tile2_name]
        if tile_counts[tile3_name] == 0:
            del tile_counts[tile3_name]

    def add_chow(self, tile_counts: Counter, tile: Tuple[int, str]):
        """Add back a chow starting from this tile."""
        number, class_name = tile

        tile1_name = f"{number}_{class_name}"
        tile2_name = f"{number + 1}_{class_name}"
        tile3_name = f"{number + 2}_{class_name}"

        tile_counts[tile1_name] += 1
        tile_counts[tile2_name] += 1
        tile_counts[tile3_name] += 1

    def validate_set(self, tile_set: List[MahjongTile]) -> bool:
        """
        Validate if a set of tiles forms a valid meld (pung, chow, or kong).

        Args:
            tile_set: List of tiles forming a set

        Returns:
            bool: True if the set is a valid meld
        """
        if len(tile_set) not in [3, 4]:
            return False

        # Sort tiles for easier comparison
        sorted_tiles = sorted(tile_set, key=lambda t: (t.class_name, t.number))

        # Check for pung or kong (identical tiles)
        if all(tile.name == sorted_tiles[0].name for tile in sorted_tiles):
            return True

        # Check for chow (only valid for 3-tile sets)
        if len(tile_set) == 3:
            return self._is_valid_chow(sorted_tiles)

        return False

    def _is_valid_chow(self, sorted_tiles: List[MahjongTile]) -> bool:
        """Check if 3 tiles form a valid chow (consecutive sequence)."""
        # Must be same suit and numbered tiles
        if not all(
            tile.class_name == sorted_tiles[0].class_name for tile in sorted_tiles
        ):
            return False

        if sorted_tiles[0].class_name not in ["萬", "筒", "條"]:
            return False

        # Check consecutive numbers
        numbers = [tile.number for tile in sorted_tiles]
        return numbers == [numbers[0], numbers[0] + 1, numbers[0] + 2]

    def count_valid_sets(self, sets: List[List[MahjongTile]]) -> int:
        """
        Count the number of valid meld sets.

        Args:
            sets: List of tile sets to validate

        Returns:
            int: Number of valid meld sets
        """
        return sum(1 for tile_set in sets if self.validate_set(tile_set))

    def is_winning_hand_with_sets(
        self,
        hands: List[MahjongTile],
        sets: List[List[MahjongTile]],
        rule: str = "standard",
    ) -> bool:
        """
        Determine if a mahjong hand is winning when considering both hidden hands and exposed sets.

        Args:
            hands: List of tiles in hand (hidden tiles)
            sets: List of already formed meld sets (exposed melds)
            rule: Winning rule to use ("standard" or "taiwan")

        Returns:
            bool: True if the combination is winning
        """
        # Validate all existing sets
        valid_sets = self.count_valid_sets(sets)
        if valid_sets != len(sets):
            return False  # Invalid sets found

        # Calculate total tiles
        total_tiles_in_sets = sum(len(tile_set) for tile_set in sets)
        total_tiles = len(hands) + total_tiles_in_sets

        # Determine expected totals based on rule
        if rule == "standard":
            expected_total = 14
            expected_melds = 4
        elif rule == "taiwan":
            expected_total = 17
            expected_melds = 5
        else:
            raise ValueError(f"Unknown rule: {rule}. Use 'standard' or 'taiwan'")

        # Calculate expected total accounting for kongs (+1 tile each)
        kong_count = sum(1 for tile_set in sets if len(tile_set) == 4)
        adjusted_expected_total = expected_total + kong_count
        
        # Validate total tile count
        if total_tiles != adjusted_expected_total:
            return False

        # Calculate remaining melds needed
        remaining_melds_needed = expected_melds - valid_sets

        # If we already have all melds, we just need a pair in hands
        if remaining_melds_needed == 0:
            return len(hands) == 2 and hands[0].name == hands[1].name

        # If we need exactly one more meld, hands should have 3 tiles (meld) + 2 tiles (pair)
        # If we need more melds, calculate accordingly
        expected_hand_size = (
            remaining_melds_needed * 3 + 2
        )  # Each meld is 3 tiles + 1 pair (2 tiles)

        if len(hands) != expected_hand_size:
            return False

        # Use DFS to check if remaining hands can form the required melds + pair
        tile_counts = Counter()
        for tile in hands:
            tile_counts[tile.name] += 1

        return self.dfs_check_win(tile_counts, 0, False, remaining_melds_needed, 1)


if __name__ == "__main__":
    wc = win_checker(rule="standard")
    print("=== Enhanced Mahjong Win Checker Examples ===")

    # Traditional win checking (all tiles in hand)
    complete_hand = [
        MahjongTile(1, "萬"),
        MahjongTile(1, "萬"),
        MahjongTile(1, "萬"),  # pung
        MahjongTile(2, "萬"),
        MahjongTile(3, "萬"),
        MahjongTile(4, "萬"),  # chow
        MahjongTile(5, "萬"),
        MahjongTile(6, "萬"),
        MahjongTile(7, "萬"),  # chow
        MahjongTile(1, "筒"),
        MahjongTile(2, "筒"),
        MahjongTile(3, "筒"),  # chow
        MahjongTile(4, "筒"),
        MahjongTile(4, "筒"),  # pair
    ]
    print(f"Complete hand (traditional): {wc.is_winning_hand(complete_hand)}")

    # New approach: separate sets and hands
    exposed_sets = [
        [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
        [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
        [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
    ]
    remaining_hand = [
        MahjongTile(1, "筒"),
        MahjongTile(2, "筒"),
        MahjongTile(3, "筒"),  # final meld
        MahjongTile(4, "筒"),
        MahjongTile(4, "筒"),  # pair
    ]

    print(
        f"Separated sets + hands: {wc.is_winning_hand_with_sets(remaining_hand, exposed_sets, 'standard')}"
    )

    # Example with most sets exposed (easier win checking)
    almost_complete_sets = [
        [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
        [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
        [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
        [MahjongTile(1, "筒"), MahjongTile(2, "筒"), MahjongTile(3, "筒")],  # chow
    ]
    just_pair = [MahjongTile(4, "筒"), MahjongTile(4, "筒")]

    print(
        f"Just need pair: {wc.is_winning_hand_with_sets(just_pair, almost_complete_sets, 'standard')}"
    )

    # Taiwan rule example
    taiwan_sets = [
        [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
        [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
        [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
        [
            MahjongTile(1, "筒"),
            MahjongTile(2, "筒"),
            MahjongTile(3, "筒"),
        ],  # chow (4 sets so far)
    ]
    taiwan_hand = [
        MahjongTile(4, "筒"),
        MahjongTile(5, "筒"),
        MahjongTile(6, "筒"),  # 5th meld
        MahjongTile(7, "筒"),
        MahjongTile(7, "筒"),  # pair
    ]

    print(
        f"Taiwan rule (5 melds + pair): {wc.is_winning_hand_with_sets(taiwan_hand, taiwan_sets, 'taiwan')}"
    )

    # Player integration example
    print("\n=== Player Integration Example ===")
    from player import MahjongPlayer

    p = MahjongPlayer("demo_player", 0)
    p.hand.sets = exposed_sets  # 3 sets exposed
    p.hand.hands = remaining_hand  # 5 tiles in hand

    print(f"Player '{p.name}' is winning: {p.is_winning('standard')}")

    print(f"Player wind: {p.self_wind}")
    print(f"Sets: {len(p.hand.sets)} exposed, Hands: {len(p.hand.hands)} tiles")

    print("\nTo run comprehensive tests, use: python test/test_winChecker.py")
