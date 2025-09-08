from .tile import MahjongTile


class MahjongWall:
    def __init__(self, tiles, random_seed) -> None:
        self.tiles: list[MahjongTile] = tiles
        self.random_seed = random_seed

        return None

    def shuffle(self):
        import random

        random.seed(self.random_seed)
        random.shuffle(self.tiles)
