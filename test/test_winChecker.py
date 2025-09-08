import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.winChecker import win_checker
from modules.tile import MahjongTile
from collections import Counter


class TestMahjongWinChecker(unittest.TestCase):
    """Test cases for the DFS-based mahjong win checker."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.wc_std = win_checker(rule="standard")
        self.wc_tw = win_checker(rule="taiwan")

    def test_all_chows_hand(self):
        """Test hand with all chows (sequences) + pair."""
        chow_hand = [
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
        self.assertTrue(self.wc_std.is_winning_hand(chow_hand))

    def test_all_pungs_hand(self):
        """Test hand with all pungs (triplets) + pair."""
        pung_hand = [
            MahjongTile(1, "萬"),
            MahjongTile(1, "萬"),
            MahjongTile(1, "萬"),  # pung
            MahjongTile(2, "萬"),
            MahjongTile(2, "萬"),
            MahjongTile(2, "萬"),  # pung
            MahjongTile(3, "萬"),
            MahjongTile(3, "萬"),
            MahjongTile(3, "萬"),  # pung
            MahjongTile(4, "萬"),
            MahjongTile(4, "萬"),
            MahjongTile(4, "萬"),  # pung
            MahjongTile(5, "萬"),
            MahjongTile(5, "萬"),  # pair
        ]
        self.assertTrue(self.wc_std.is_winning_hand(pung_hand))

    def test_mixed_melds_hand(self):
        """Test hand with mixed pungs and chows + pair."""
        mixed_hand = [
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
        self.assertTrue(self.wc_std.is_winning_hand(mixed_hand))

    def test_honor_tiles_hand(self):
        """Test hand with honor tiles (風牌/字牌)."""
        honor_hand = [
            MahjongTile(0, "東"),
            MahjongTile(0, "東"),
            MahjongTile(0, "東"),  # pung
            MahjongTile(0, "南"),
            MahjongTile(0, "南"),
            MahjongTile(0, "南"),  # pung
            MahjongTile(0, "中"),
            MahjongTile(0, "中"),
            MahjongTile(0, "中"),  # pung
            MahjongTile(1, "萬"),
            MahjongTile(2, "萬"),
            MahjongTile(3, "萬"),  # chow
            MahjongTile(4, "萬"),
            MahjongTile(4, "萬"),  # pair
        ]
        self.assertTrue(self.wc_std.is_winning_hand(honor_hand))

    def test_non_winning_hand(self):
        """Test hand that cannot form winning combination."""
        non_winning = [
            MahjongTile(1, "萬"),
            MahjongTile(2, "萬"),
            MahjongTile(4, "萬"),
            MahjongTile(5, "萬"),
            MahjongTile(7, "萬"),
            MahjongTile(8, "萬"),
            MahjongTile(1, "筒"),
            MahjongTile(3, "筒"),
            MahjongTile(5, "筒"),
            MahjongTile(7, "筒"),
            MahjongTile(9, "筒"),
            MahjongTile(1, "條"),
            MahjongTile(3, "條"),
            MahjongTile(5, "條"),
        ]
        self.assertFalse(self.wc_std.is_winning_hand(non_winning))

    def test_wrong_tile_count(self):
        """Test hands with incorrect number of tiles using separated sets/hands approach."""
        # Too few tiles total (13 instead of 14) - sets + hands
        too_few_sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        ]  # 3 tiles
        too_few_hands = [MahjongTile(2, "萬")] * 10  # 10 tiles, total = 13
        self.assertFalse(
            self.wc_std.is_winning_hand_with_sets(
                too_few_hands, too_few_sets, "standard"
            )
        )

        # Too many tiles total (15 instead of 14) - sets + hands
        too_many_sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        ]  # 3 tiles
        too_many_hands = [MahjongTile(2, "萬")] * 12  # 12 tiles, total = 15
        self.assertFalse(
            self.wc_std.is_winning_hand_with_sets(
                too_many_hands, too_many_sets, "standard"
            )
        )

        # Empty hands and sets
        self.assertFalse(self.wc_std.is_winning_hand_with_sets([], [], "standard"))

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Hand with scattered tiles that cannot form melds
        scattered_tiles = [
            MahjongTile(1, "萬"),
            MahjongTile(3, "萬"),
            MahjongTile(5, "萬"),
            MahjongTile(7, "萬"),
            MahjongTile(9, "萬"),
            MahjongTile(1, "筒"),
            MahjongTile(3, "筒"),
            MahjongTile(5, "筒"),
            MahjongTile(7, "筒"),
            MahjongTile(9, "筒"),
            MahjongTile(1, "條"),
            MahjongTile(3, "條"),
            MahjongTile(0, "東"),
            MahjongTile(0, "南"),
        ]
        self.assertFalse(self.wc_std.is_winning_hand(scattered_tiles))

        # Note: Seven pairs pattern (like 1,1,2,2,3,3,4,4,5,5,6,6,7,7)
        # can actually be winning in standard mahjong rules as it can form pungs.
        # For example: 1,1,1 + 2,2,2 + 3,3,3 + 4,4,4 + 5,5 would be valid.
        # So the current implementation correctly identifies some "seven pairs" as winning.

    def test_chow_formation(self):
        """Test chow (sequence) formation logic."""
        # Test can_form_chow function directly
        test_counts = Counter({"1_萬": 1, "2_萬": 1, "3_萬": 1})
        first_tile = self.wc_std.parse_tile_name("1_萬")
        self.assertTrue(self.wc_std.can_form_chow(test_counts, first_tile))

        # Test with missing middle tile
        incomplete_counts = Counter({"1_萬": 1, "3_萬": 1})
        self.assertFalse(self.wc_std.can_form_chow(incomplete_counts, first_tile))

        # Test with honor tiles (should not form chow)
        honor_counts = Counter({"東": 3})
        honor_tile = self.wc_std.parse_tile_name("東")
        self.assertFalse(self.wc_std.can_form_chow(honor_counts, honor_tile))

    def test_boundary_chows(self):
        """Test chows at boundaries (7,8,9) and invalid cases."""
        # Valid boundary chow (7,8,9)
        boundary_counts = Counter({"7_萬": 1, "8_萬": 1, "9_萬": 1})
        boundary_tile = self.wc_std.parse_tile_name("7_萬")
        self.assertTrue(self.wc_std.can_form_chow(boundary_counts, boundary_tile))

        # Invalid - trying to start chow from 8 (would need 8,9,10)
        invalid_counts = Counter({"8_萬": 1, "9_萬": 1})
        invalid_tile = self.wc_std.parse_tile_name("8_萬")
        self.assertFalse(self.wc_std.can_form_chow(invalid_counts, invalid_tile))

        # Invalid - trying to start chow from 9 (would need 9,10,11)
        invalid_nine = Counter({"9_萬": 1})
        invalid_nine_tile = self.wc_std.parse_tile_name("9_萬")
        self.assertFalse(self.wc_std.can_form_chow(invalid_nine, invalid_nine_tile))

    def test_flexible_rules(self):
        """Test flexible rule system for different tile counts."""
        # Standard 14-tile hand
        standard_hand = [
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
        self.assertTrue(self.wc_std.is_winning_hand(standard_hand))

    def test_taiwan_17_tiles(self):
        """Test Taiwan 17-tile rule (5 melds + 1 pair)."""
        taiwan_hand = [
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
            MahjongTile(5, "筒"),
            MahjongTile(6, "筒"),  # chow (5th meld)
            MahjongTile(7, "筒"),
            MahjongTile(7, "筒"),  # pair
        ]
        self.assertTrue(self.wc_tw.is_winning_hand(taiwan_hand))

        # Should fail with standard rule
        self.assertFalse(self.wc_std.is_winning_hand(taiwan_hand))

    def test_invalid_rules(self):
        """Test invalid rule parameters."""
        hand = [MahjongTile(1, "萬")] * 14

        # Test invalid rule name - validation happens when is_winning_hand is called
        invalid_wc = win_checker(rule="invalid_rule")
        with self.assertRaises(ValueError):
            invalid_wc.is_winning_hand(hand)

    def test_rule_specific_validation(self):
        """Test that rules properly validate tile counts using separated sets/hands approach."""
        # 17 tiles total should fail standard rule (expects 14)
        taiwan_sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        ]  # 3 tiles
        taiwan_hands = [MahjongTile(2, "萬")] * 14  # 14 tiles, total = 17
        self.assertFalse(
            self.wc_std.is_winning_hand_with_sets(taiwan_hands, taiwan_sets, "standard")
        )

        # 14 tiles total should fail taiwan rule (expects 17)
        standard_sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        ]  # 3 tiles
        standard_hands = [MahjongTile(2, "萬")] * 11  # 11 tiles, total = 14
        self.assertFalse(
            self.wc_tw.is_winning_hand_with_sets(
                standard_hands, standard_sets, "taiwan"
            )
        )

    def test_set_validation(self):
        """Test validation of individual meld sets."""

        # Valid pung (3 identical)
        valid_pung = [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        self.assertTrue(self.wc_std.validate_set(valid_pung))

        # Valid chow (3 consecutive)
        valid_chow = [MahjongTile(1, "萬"), MahjongTile(2, "萬"), MahjongTile(3, "萬")]
        self.assertTrue(self.wc_std.validate_set(valid_chow))

        # Valid kong (4 identical)
        valid_kong = [
            MahjongTile(1, "萬"),
            MahjongTile(1, "萬"),
            MahjongTile(1, "萬"),
            MahjongTile(1, "萬"),
        ]
        self.assertTrue(self.wc_std.validate_set(valid_kong))

        # Invalid set - mixed tiles
        invalid_mixed = [
            MahjongTile(1, "萬"),
            MahjongTile(2, "筒"),
            MahjongTile(3, "條"),
        ]
        self.assertFalse(self.wc_std.validate_set(invalid_mixed))

        # Invalid chow - non-consecutive
        invalid_chow = [
            MahjongTile(1, "萬"),
            MahjongTile(3, "萬"),
            MahjongTile(5, "萬"),
        ]
        self.assertFalse(self.wc_std.validate_set(invalid_chow))

        # Invalid set - wrong count
        invalid_count = [MahjongTile(1, "萬"), MahjongTile(1, "萬")]
        self.assertFalse(self.wc_std.validate_set(invalid_count))

    def test_winning_with_sets(self):
        """Test win checking with separate hands and sets."""
        # Standard win: 3 exposed sets + remaining hand forms 1 meld + pair
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

        self.assertTrue(
            self.wc_std.is_winning_hand_with_sets(
                remaining_hand, exposed_sets, "standard"
            )
        )

        # All sets exposed, only pair in hand
        all_sets_exposed = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
            [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
            [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
            [MahjongTile(1, "筒"), MahjongTile(2, "筒"), MahjongTile(3, "筒")],  # chow
        ]
        only_pair = [MahjongTile(4, "筒"), MahjongTile(4, "筒")]

        self.assertTrue(
            self.wc_std.is_winning_hand_with_sets(
                only_pair, all_sets_exposed, "standard"
            )
        )

        # Taiwan rule with 5 melds
        taiwan_sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
            [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
            [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
            [MahjongTile(1, "筒"), MahjongTile(2, "筒"), MahjongTile(3, "筒")],  # chow
        ]
        taiwan_hand = [
            MahjongTile(4, "筒"),
            MahjongTile(5, "筒"),
            MahjongTile(6, "筒"),  # 5th meld
            MahjongTile(7, "筒"),
            MahjongTile(7, "筒"),  # pair
        ]

        self.assertTrue(
            self.wc_tw.is_winning_hand_with_sets(taiwan_hand, taiwan_sets, "taiwan")
        )

    def test_player_winning(self):
        """Test player win detection functionality."""
        from modules.player import MahjongPlayer

        p = MahjongPlayer("test_player", 0)

        # Set up a winning configuration
        p.hand.sets = [
            [MahjongTile(1, "萬"), MahjongTile(1, "萬"), MahjongTile(1, "萬")],  # pung
            [MahjongTile(2, "萬"), MahjongTile(3, "萬"), MahjongTile(4, "萬")],  # chow
            [MahjongTile(5, "萬"), MahjongTile(6, "萬"), MahjongTile(7, "萬")],  # chow
        ]
        p.hand.hands = [
            MahjongTile(1, "筒"),
            MahjongTile(2, "筒"),
            MahjongTile(3, "筒"),  # final meld
            MahjongTile(4, "筒"),
            MahjongTile(4, "筒"),  # pair
        ]

        self.assertTrue(p.is_winning("standard"))

        # Test non-winning configuration
        p.hand.hands = [
            MahjongTile(1, "筒"),
            MahjongTile(3, "筒"),
            MahjongTile(5, "筒"),  # invalid meld
            MahjongTile(4, "筒"),
            MahjongTile(4, "筒"),  # pair
        ]

        self.assertFalse(p.is_winning("standard"))

    def test_set_validation_edge_cases(self):
        """Test edge cases in set validation."""
        # Honor tile pung
        honor_pung = [MahjongTile(0, "東"), MahjongTile(0, "東"), MahjongTile(0, "東")]
        self.assertTrue(self.wc_std.validate_set(honor_pung))

        # Honor tiles cannot form chow
        invalid_honor_chow = [
            MahjongTile(0, "東"),
            MahjongTile(0, "南"),
            MahjongTile(0, "西"),
        ]
        self.assertFalse(self.wc_std.validate_set(invalid_honor_chow))

        # Boundary chow (7,8,9)
        boundary_chow = [
            MahjongTile(7, "萬"),
            MahjongTile(8, "萬"),
            MahjongTile(9, "萬"),
        ]
        self.assertTrue(self.wc_std.validate_set(boundary_chow))

        # Mixed suit chow (invalid)
        mixed_suit = [MahjongTile(1, "萬"), MahjongTile(2, "筒"), MahjongTile(3, "條")]
        self.assertFalse(self.wc_std.validate_set(mixed_suit))

    def test_invalid_sets_rejection(self):
        """Test that hands with invalid sets are rejected."""
        # Invalid set in exposed sets
        invalid_sets = [
            [
                MahjongTile(1, "萬"),
                MahjongTile(3, "萬"),
                MahjongTile(5, "萬"),
            ]  # invalid chow
        ]
        remaining_hand = [
            MahjongTile(1, "筒")
        ] * 11  # doesn't matter, invalid sets should fail first

        self.assertFalse(
            self.wc_std.is_winning_hand_with_sets(
                remaining_hand, invalid_sets, "standard"
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
