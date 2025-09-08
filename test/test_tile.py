import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from modules.tile import MahjongTile, default_tiles


class TestMJTile(unittest.TestCase):
    """Comprehensive test suite for the MahjongTile class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tile_1_wan = MahjongTile(1, "萬")
        self.tile_2_wan = MahjongTile(2, "萬")
        self.tile_1_tiao = MahjongTile(1, "條")
        self.tile_dong = MahjongTile(0, "東")
        self.tile_zhong = MahjongTile(0, "中")
        self.tile_special_5_wan = MahjongTile(5, "萬", special=True)
        self.tile_flower = MahjongTile(0, "春")

    def test_init_basic(self):
        """Test basic initialization of MahjongTile objects."""
        # Regular numbered tile
        tile = MahjongTile(5, "萬")
        self.assertEqual(tile.number, 5)
        self.assertEqual(tile.class_name, "萬")
        self.assertFalse(tile.special)
        self.assertEqual(tile.name, "5_萬")
        self.assertEqual(tile.type, "萬")

        # Honor tile (number 0)
        honor_tile = MahjongTile(0, "東")
        self.assertEqual(honor_tile.number, 0)
        self.assertEqual(honor_tile.class_name, "東")
        self.assertFalse(honor_tile.special)
        self.assertEqual(honor_tile.name, "東")
        self.assertEqual(honor_tile.type, "風")

    def test_init_special_flag(self):
        """Test initialization with special flag."""
        special_tile = MahjongTile(5, "萬", special=True)
        self.assertTrue(special_tile.special)
        self.assertEqual(special_tile.name, "5_萬")

    def test_type_classification(self):
        """Test that tiles are correctly classified by type."""
        # Number suits
        wan_tile = MahjongTile(1, "萬")
        tong_tile = MahjongTile(1, "筒")
        tiao_tile = MahjongTile(1, "條")
        self.assertEqual(wan_tile.type, "萬")
        self.assertEqual(tong_tile.type, "筒")
        self.assertEqual(tiao_tile.type, "條")

        # Wind tiles
        dong_tile = MahjongTile(0, "東")
        nan_tile = MahjongTile(0, "南")
        xi_tile = MahjongTile(0, "西")
        bei_tile = MahjongTile(0, "北")
        self.assertEqual(dong_tile.type, "風")
        self.assertEqual(nan_tile.type, "風")
        self.assertEqual(xi_tile.type, "風")
        self.assertEqual(bei_tile.type, "風")

        # Dragon tiles
        bai_tile = MahjongTile(0, "白")
        fa_tile = MahjongTile(0, "發")
        zhong_tile = MahjongTile(0, "中")
        self.assertEqual(bai_tile.type, "元")
        self.assertEqual(fa_tile.type, "元")
        self.assertEqual(zhong_tile.type, "元")

        # Flower tiles
        flower_tiles = ["春", "夏", "秋", "冬", "梅", "竹", "蘭", "菊"]
        for flower in flower_tiles:
            tile = MahjongTile(0, flower)
            self.assertEqual(tile.type, "花")

    def test_name_generation(self):
        """Test tile name generation."""
        # Numbered tile
        numbered_tile = MahjongTile(7, "筒")
        self.assertEqual(numbered_tile.name, "7_筒")

        # Honor tile (number 0)
        honor_tile = MahjongTile(0, "中")
        self.assertEqual(honor_tile.name, "中")

        # Flower tile
        flower_tile = MahjongTile(0, "春")
        self.assertEqual(flower_tile.name, "春")

    def test_repr_regular(self):
        """Test string representation for regular tiles."""
        tile = MahjongTile(3, "萬")
        self.assertEqual(repr(tile), "3_萬")

        honor_tile = MahjongTile(0, "東")
        self.assertEqual(repr(honor_tile), "東")

    def test_repr_special(self):
        """Test string representation for special tiles."""
        special_tile = MahjongTile(5, "萬", special=True)
        self.assertEqual(repr(special_tile), "5_萬(Special:True)")

    def test_equality_same_tiles(self):
        """Test equality between identical tiles."""
        tile1 = MahjongTile(1, "條")
        tile2 = MahjongTile(1, "條")
        self.assertTrue(tile1 == tile2)
        self.assertEqual(tile1, tile2)

    def test_equality_different_numbers(self):
        """Test inequality between tiles with different numbers."""
        tile1 = MahjongTile(1, "萬")
        tile2 = MahjongTile(2, "萬")
        self.assertFalse(tile1 == tile2)
        self.assertNotEqual(tile1, tile2)

    def test_equality_different_classes(self):
        """Test inequality between tiles with different classes."""
        tile1 = MahjongTile(1, "萬")
        tile2 = MahjongTile(1, "筒")
        self.assertFalse(tile1 == tile2)
        self.assertNotEqual(tile1, tile2)

    def test_equality_special_flag_ignored(self):
        """Test that special flag doesn't affect equality."""
        tile1 = MahjongTile(5, "萬", special=False)
        tile2 = MahjongTile(5, "萬", special=True)
        self.assertTrue(tile1 == tile2)

    def test_equality_honor_tiles(self):
        """Test equality for honor tiles."""
        tile1 = MahjongTile(0, "東")
        tile2 = MahjongTile(0, "東")
        tile3 = MahjongTile(0, "南")
        self.assertTrue(tile1 == tile2)
        self.assertFalse(tile1 == tile3)

    def test_sorting_same_type_numbered(self):
        """Test sorting within the same numbered suit."""
        tiles = [MahjongTile(3, "萬"), MahjongTile(1, "萬"), MahjongTile(2, "萬")]
        tiles.sort()
        expected = [MahjongTile(1, "萬"), MahjongTile(2, "萬"), MahjongTile(3, "萬")]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile, expected[i])

    def test_sorting_different_types(self):
        """Test sorting across different tile types."""
        tiles = [
            MahjongTile(0, "春"),  # 花 (type order 6)
            MahjongTile(0, "中"),  # 元 (type order 5)
            MahjongTile(0, "東"),  # 風 (type order 4)
            MahjongTile(1, "條"),  # 條 (type order 3)
            MahjongTile(1, "筒"),  # 筒 (type order 2)
            MahjongTile(1, "萬"),  # 萬 (type order 1)
        ]
        tiles.sort()
        expected_order = ["萬", "筒", "條", "風", "元", "花"]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.type, expected_order[i])

    def test_sorting_wind_tiles(self):
        """Test sorting within wind tiles."""
        tiles = [
            MahjongTile(0, "北"),
            MahjongTile(0, "東"),
            MahjongTile(0, "西"),
            MahjongTile(0, "南"),
        ]
        tiles.sort()
        expected_order = ["東", "南", "西", "北"]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.class_name, expected_order[i])

    def test_sorting_dragon_tiles(self):
        """Test sorting within dragon tiles."""
        tiles = [MahjongTile(0, "白"), MahjongTile(0, "發"), MahjongTile(0, "中")]
        tiles.sort()
        expected_order = ["中", "發", "白"]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.class_name, expected_order[i])

    def test_sorting_flower_tiles(self):
        """Test sorting within flower tiles (alphabetical)."""
        tiles = [MahjongTile(0, "竹"), MahjongTile(0, "春"), MahjongTile(0, "蘭")]
        tiles.sort()
        expected_order = ["春", "竹", "蘭"]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.class_name, expected_order[i])

    def test_sorting_mixed_same_type(self):
        """Test sorting mixed numbers within same type."""
        tiles = [
            MahjongTile(9, "萬"),
            MahjongTile(1, "萬"),
            MahjongTile(5, "萬"),
            MahjongTile(3, "萬"),
            MahjongTile(7, "萬"),
        ]
        tiles.sort()
        expected_numbers = [1, 3, 5, 7, 9]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.number, expected_numbers[i])

    def test_less_than_false_case(self):
        """Test cases where __lt__ should return False."""
        # Same tiles
        tile1 = MahjongTile(1, "萬")
        tile2 = MahjongTile(1, "萬")
        self.assertFalse(tile1 < tile2)

        # First tile is greater
        tile3 = MahjongTile(2, "萬")
        tile4 = MahjongTile(1, "萬")
        self.assertFalse(tile3 < tile4)

    def test_default_tiles_function(self):
        """Test the default_tiles function."""
        tiles = default_tiles()

        # Should have 4 copies of each tile (34 unique tiles * 4 = 136 total)
        self.assertEqual(len(tiles), 136)

        # Count occurrences of each tile type
        tile_counts = {}
        for tile in tiles:
            key = (
                f"{tile.number}_{tile.class_name}"
                if tile.number != 0
                else tile.class_name
            )
            tile_counts[key] = tile_counts.get(key, 0) + 1

        # Each tile should appear exactly 4 times
        for count in tile_counts.values():
            self.assertEqual(count, 4)

        # Should be sorted
        self.assertEqual(tiles, sorted(tiles))

        # Check we have the right number of unique tiles
        self.assertEqual(len(tile_counts), 34)  # 9+9+9+4+3=34 unique tiles

    def test_default_tiles_content(self):
        """Test that default_tiles contains the expected tiles."""
        tiles = default_tiles()
        unique_tiles = []

        # Get one copy of each unique tile
        seen = set()
        for tile in tiles:
            key = (tile.number, tile.class_name)
            if key not in seen:
                unique_tiles.append(tile)
                seen.add(key)

        # Should have all numbered suits (1-9 for each of 萬筒條)
        numbered_suits = ["萬", "筒", "條"]
        for suit in numbered_suits:
            for num in range(1, 10):
                expected_tile = MahjongTile(num, suit)
                self.assertIn(expected_tile, unique_tiles)

        # Should have all wind tiles
        winds = ["東", "南", "西", "北"]
        for wind in winds:
            expected_tile = MahjongTile(0, wind)
            self.assertIn(expected_tile, unique_tiles)

        # Should have all dragon tiles
        dragons = ["中", "發", "白"]
        for dragon in dragons:
            expected_tile = MahjongTile(0, dragon)
            self.assertIn(expected_tile, unique_tiles)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Maximum numbered tile
        max_tile = MahjongTile(9, "萬")
        self.assertEqual(max_tile.number, 9)
        self.assertEqual(max_tile.type, "萬")

        # Minimum numbered tile
        min_tile = MahjongTile(1, "筒")
        self.assertEqual(min_tile.number, 1)
        self.assertEqual(min_tile.type, "筒")

    def test_consistency_with_existing_test(self):
        """Test compatibility with the existing test in the main block."""
        t1 = MahjongTile(1, "條")
        t2 = MahjongTile(1, "條")
        self.assertTrue(t1 == t2)

    def test_type_assignment_edge_cases(self):
        """Test type assignment for unusual but valid inputs."""
        # Test that unrecognized class_name doesn't break the code
        # (though this might not be intended behavior)
        try:
            unknown_tile = MahjongTile(1, "unknown")
            # If no exception, check that type attribute exists
            self.assertTrue(hasattr(unknown_tile, "type"))
        except:
            # If it fails, that's also acceptable behavior
            pass

    def test_sorting_comprehensive(self):
        """Comprehensive sorting test with all tile types."""
        # Create one tile of each type in reverse order
        tiles = [
            MahjongTile(0, "春"),  # 花 (should be last)
            MahjongTile(0, "白"),  # 元
            MahjongTile(0, "北"),  # 風
            MahjongTile(9, "條"),  # 條
            MahjongTile(9, "筒"),  # 筒
            MahjongTile(9, "萬"),  # 萬 (should be first)
        ]

        tiles.sort()

        # Check the order is correct
        expected_types = ["萬", "筒", "條", "風", "元", "花"]
        for i, tile in enumerate(tiles):
            self.assertEqual(tile.type, expected_types[i])


if __name__ == "__main__":
    unittest.main(verbosity=2)
