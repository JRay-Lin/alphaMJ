from .tile import MahjongTile
from .winChecker import win_checker
from collections import Counter
from typing import List, Tuple, Optional


class MahjongHand:
    def __init__(self) -> None:
        self.hands: list[MahjongTile] = []
        self.discards: list[MahjongTile] = []
        self.sets: list[list[MahjongTile]] = []

        return None


class MahjongPlayer:
    def __init__(self, name, index: int, is_ai: bool = False) -> None:
        self.name: str = name
        self.index: int = index  # Fixed: was always 0
        self.self_wind: str = ["東", "南", "西", "北"][index % 4]
        self.is_ai: bool = is_ai  # Flag for AI players

        self.hand = MahjongHand()

        self.score = 100
        self._win_checker = win_checker()

        return None

    def is_winning(self, rule: str = "standard") -> bool:
        """
        Check if this player has a winning hand.

        Args:
            rule: Winning rule to use ("standard" or "taiwan")

        Returns:
            bool: True if the player is winning
        """
        return self._win_checker.is_winning_hand_with_sets(
            self.hand.hands, self.hand.sets, rule
        )

    def can_chi(self, discarded_tile: MahjongTile) -> List[Tuple[int, int]]:
        """
        Check if player can chi with the discarded tile.
        
        Args:
            discarded_tile: The tile that was just discarded
            
        Returns:
            List of tuples (tile1_idx, tile2_idx) representing possible chi combinations
        """
        if discarded_tile.class_name not in ["萬", "筒", "條"] or discarded_tile.number == 0:
            return []  # Cannot chi honor tiles or numbered 0 tiles
        
        possible_chis = []
        hand_tiles = self.hand.hands.copy()
        
        # Sort hand for easier searching
        hand_tiles.sort()
        
        # Check for sequences where discarded tile can fit
        target_number = discarded_tile.number
        target_suit = discarded_tile.class_name
        
        # Pattern 1: discarded tile is the first (e.g., discarded=1, need 2,3)
        if target_number <= 7:
            needed1 = MahjongTile(target_number + 1, target_suit)
            needed2 = MahjongTile(target_number + 2, target_suit)
            
            idx1 = self._find_tile_index(needed1)
            idx2 = self._find_tile_index(needed2)
            
            if idx1 != -1 and idx2 != -1 and idx1 != idx2:
                possible_chis.append((idx1, idx2))
        
        # Pattern 2: discarded tile is the middle (e.g., discarded=2, need 1,3)
        if 2 <= target_number <= 8:
            needed1 = MahjongTile(target_number - 1, target_suit)
            needed2 = MahjongTile(target_number + 1, target_suit)
            
            idx1 = self._find_tile_index(needed1)
            idx2 = self._find_tile_index(needed2)
            
            if idx1 != -1 and idx2 != -1 and idx1 != idx2:
                possible_chis.append((idx1, idx2))
        
        # Pattern 3: discarded tile is the last (e.g., discarded=3, need 1,2)
        if target_number >= 3:
            needed1 = MahjongTile(target_number - 2, target_suit)
            needed2 = MahjongTile(target_number - 1, target_suit)
            
            idx1 = self._find_tile_index(needed1)
            idx2 = self._find_tile_index(needed2)
            
            if idx1 != -1 and idx2 != -1 and idx1 != idx2:
                possible_chis.append((idx1, idx2))
        
        return possible_chis

    def can_pong(self, discarded_tile: MahjongTile) -> bool:
        """
        Check if player can pong with the discarded tile.
        
        Args:
            discarded_tile: The tile that was just discarded
            
        Returns:
            bool: True if player has 2 matching tiles for pong
        """
        count = sum(1 for tile in self.hand.hands if tile == discarded_tile)
        return count >= 2

    def can_kong(self, discarded_tile: Optional[MahjongTile] = None) -> dict:
        """
        Check what kong actions are available.
        
        Args:
            discarded_tile: The tile that was just discarded (None for hidden kong check)
            
        Returns:
            dict with 'hidden_kong' and 'normal_kong' lists
        """
        result = {"hidden_kong": [], "normal_kong": False, "promote_kong": []}
        
        # Check for hidden kong (4 identical tiles in hand)
        tile_counts = Counter(tile.name for tile in self.hand.hands)
        for tile_name, count in tile_counts.items():
            if count >= 4:
                # Find the actual tile object
                tile_obj = next(tile for tile in self.hand.hands if tile.name == tile_name)
                result["hidden_kong"].append(tile_obj)
        
        # Check for normal kong (3 in hand + 1 discarded)
        if discarded_tile:
            if sum(1 for tile in self.hand.hands if tile == discarded_tile) >= 3:
                result["normal_kong"] = True
        
        # Check for promote kong (upgrade existing pong to kong)
        if discarded_tile:
            for meld_set in self.hand.sets:
                if (len(meld_set) == 3 and 
                    all(tile == meld_set[0] for tile in meld_set) and  # It's a pong
                    meld_set[0] == discarded_tile):  # Matches discarded tile
                    result["promote_kong"].append(meld_set)
        
        return result

    def execute_chi(self, discarded_tile: MahjongTile, tile_indices: Tuple[int, int]) -> bool:
        """
        Execute a chi action.
        
        Args:
            discarded_tile: The discarded tile to chi
            tile_indices: Indices of the two tiles from hand to complete the chi
            
        Returns:
            bool: True if chi was successful
        """
        try:
            # Remove the two tiles from hand (in reverse order to maintain indices)
            idx1, idx2 = sorted(tile_indices, reverse=True)
            tile1 = self.hand.hands.pop(idx1)
            tile2 = self.hand.hands.pop(idx2)
            
            # Create the chi set and add to sets
            chi_set = [discarded_tile, tile1, tile2]
            chi_set.sort()  # Sort for consistency
            self.hand.sets.append(chi_set)
            
            return True
        except (IndexError, ValueError):
            return False

    def execute_pong(self, discarded_tile: MahjongTile) -> bool:
        """
        Execute a pong action.
        
        Args:
            discarded_tile: The discarded tile to pong
            
        Returns:
            bool: True if pong was successful
        """
        # Find and remove 2 matching tiles from hand
        removed_count = 0
        tiles_to_remove = []
        
        for i, tile in enumerate(self.hand.hands):
            if tile == discarded_tile and removed_count < 2:
                tiles_to_remove.append(i)
                removed_count += 1
        
        if removed_count < 2:
            return False
        
        # Remove tiles in reverse order to maintain indices
        for idx in reversed(tiles_to_remove):
            self.hand.hands.pop(idx)
        
        # Create the pong set
        pong_set = [discarded_tile, discarded_tile, discarded_tile]
        self.hand.sets.append(pong_set)
        
        return True

    def execute_hidden_kong(self, kong_tile: MahjongTile) -> bool:
        """
        Execute a hidden kong action.
        
        Args:
            kong_tile: The tile to form hidden kong with
            
        Returns:
            bool: True if hidden kong was successful
        """
        # Find and remove 4 matching tiles from hand
        tiles_to_remove = []
        
        for i, tile in enumerate(self.hand.hands):
            if tile == kong_tile:
                tiles_to_remove.append(i)
        
        if len(tiles_to_remove) < 4:
            return False
        
        # Remove 4 tiles in reverse order
        for idx in reversed(tiles_to_remove[:4]):
            self.hand.hands.pop(idx)
        
        # Create the hidden kong set
        kong_set = [kong_tile, kong_tile, kong_tile, kong_tile]
        self.hand.sets.append(kong_set)
        
        return True

    def execute_normal_kong(self, discarded_tile: MahjongTile) -> bool:
        """
        Execute a normal kong action.
        
        Args:
            discarded_tile: The discarded tile to kong
            
        Returns:
            bool: True if normal kong was successful
        """
        # Find and remove 3 matching tiles from hand
        removed_count = 0
        tiles_to_remove = []
        
        for i, tile in enumerate(self.hand.hands):
            if tile == discarded_tile and removed_count < 3:
                tiles_to_remove.append(i)
                removed_count += 1
        
        if removed_count < 3:
            return False
        
        # Remove tiles in reverse order
        for idx in reversed(tiles_to_remove):
            self.hand.hands.pop(idx)
        
        # Create the kong set
        kong_set = [discarded_tile, discarded_tile, discarded_tile, discarded_tile]
        self.hand.sets.append(kong_set)
        
        return True

    def execute_promote_kong(self, discarded_tile: MahjongTile) -> bool:
        """
        Execute a promote kong action (upgrade pong to kong).
        
        Args:
            discarded_tile: The discarded tile to complete the kong
            
        Returns:
            bool: True if promote kong was successful
        """
        # Find the matching pong set
        for i, meld_set in enumerate(self.hand.sets):
            if (len(meld_set) == 3 and 
                all(tile == meld_set[0] for tile in meld_set) and 
                meld_set[0] == discarded_tile):
                # Convert pong to kong
                self.hand.sets[i].append(discarded_tile)
                return True
        
        return False

    def _find_tile_index(self, target_tile: MahjongTile) -> int:
        """
        Find the index of the first matching tile in hand.
        
        Args:
            target_tile: Tile to find
            
        Returns:
            int: Index of tile, or -1 if not found
        """
        for i, tile in enumerate(self.hand.hands):
            if tile == target_tile:
                return i
        return -1

    def get_available_actions(self, discarded_tile: MahjongTile, is_next_player: bool) -> dict:
        """
        Get all available actions for this player given a discarded tile.
        
        Args:
            discarded_tile: The tile that was just discarded
            is_next_player: True if this is the player immediately after the discarder
            
        Returns:
            dict: Available actions with their details
        """
        actions = {
            "chi": [],
            "pong": False,
            "kong": {"normal": False, "promote": []},
            "pass": True
        }
        
        # Chi is only available to the next player
        if is_next_player:
            actions["chi"] = self.can_chi(discarded_tile)
        
        # Pong is available to any player
        actions["pong"] = self.can_pong(discarded_tile)
        
        # Kong checks
        kong_info = self.can_kong(discarded_tile)
        actions["kong"]["normal"] = kong_info["normal_kong"]
        actions["kong"]["promote"] = kong_info["promote_kong"]
        
        return actions

    def get_hidden_kong_options(self) -> List[MahjongTile]:
        """
        Get available hidden kong options during player's own turn.
        
        Returns:
            List of tiles that can form hidden kong
        """
        kong_info = self.can_kong()
        return kong_info["hidden_kong"]


if __name__ == "__main__":
    p1 = MahjongPlayer("tester", 0)
    p1.hand.hands = [
        MahjongTile(1, "萬"),
        MahjongTile(2, "萬"),
        MahjongTile(3, "萬"),  # chow
        MahjongTile(4, "萬"),
        MahjongTile(5, "萬"),
        MahjongTile(6, "萬"),  # chow
        MahjongTile(7, "萬"),
        MahjongTile(8, "萬"),
        MahjongTile(9, "萬"),  # chow
        MahjongTile(1, "筒"),
        MahjongTile(2, "筒"),
        MahjongTile(3, "筒"),  # chow
        MahjongTile(4, "筒"),
        MahjongTile(4, "筒"),  # pair
    ]

    print(p1.is_winning())
