class MahjongTile:
    def __init__(self, number: int, class_name: str, special: bool = False) -> None:
        self.number = number
        self.class_name = class_name

        self.special = special  # Deal with special card in JP_MJ(e.g. red 5萬)
        self.name = f"{str(number)}_{class_name}" if number != 0 else f"{class_name}"

        if self.class_name in ["萬", "筒", "條"]:
            self.type = self.class_name
        elif self.class_name in ["東", "南", "西", "北"]:
            self.type = "風"
        elif self.class_name in ["白", "發", "中"]:
            self.type = "元"
        elif self.class_name in ["春", "夏", "秋", "冬", "梅", "竹", "蘭", "菊"]:
            self.type = "花"

    def __repr__(self) -> str:
        return_str = (
            f"{self.name}(Special:{self.special})" if self.special else f"{self.name}"
        )
        return return_str

    def __lt__(self, other):
        """This method allow mj_tile to be sorted by type and number."""

        # Order of types from smallest to largest
        type_order = {"萬": 1, "筒": 2, "條": 3, "風": 4, "元": 5, "花": 6}

        # Order within 風 tiles
        wind_order = {"東": 1, "南": 2, "西": 3, "北": 4}

        # Order within 元 tiles
        dragon_order = {"中": 1, "發": 2, "白": 3}

        # Compare type first
        if self.type != other.type:
            return type_order.get(self.type, 7) < type_order.get(other.type, 7)

        # If same type, compare by number (for 萬筒條) or by specific order (for 風元)
        if self.type in ["萬", "筒", "條"]:
            return self.number < other.number
        elif self.type == "風":
            return wind_order.get(self.class_name, 5) < wind_order.get(
                other.class_name, 5
            )
        elif self.type == "元":
            return dragon_order.get(self.class_name, 4) < dragon_order.get(
                other.class_name, 4
            )
        elif self.type == "花":
            return self.class_name < other.class_name

        return False

    def __eq__(self, other) -> bool:
        number_eq = self.number == other.number
        class_name_eq = self.class_name == other.class_name

        if number_eq and class_name_eq == True:
            return True
        else:
            return False


def default_tiles() -> list[MahjongTile]:
    tiles = [
        MahjongTile(1, "條"),
        MahjongTile(2, "條"),
        MahjongTile(3, "條"),
        MahjongTile(4, "條"),
        MahjongTile(5, "條"),
        MahjongTile(6, "條"),
        MahjongTile(7, "條"),
        MahjongTile(8, "條"),
        MahjongTile(9, "條"),
        MahjongTile(1, "萬"),
        MahjongTile(2, "萬"),
        MahjongTile(3, "萬"),
        MahjongTile(4, "萬"),
        MahjongTile(5, "萬"),
        MahjongTile(6, "萬"),
        MahjongTile(7, "萬"),
        MahjongTile(8, "萬"),
        MahjongTile(9, "萬"),
        MahjongTile(1, "筒"),
        MahjongTile(2, "筒"),
        MahjongTile(3, "筒"),
        MahjongTile(4, "筒"),
        MahjongTile(5, "筒"),
        MahjongTile(6, "筒"),
        MahjongTile(7, "筒"),
        MahjongTile(8, "筒"),
        MahjongTile(9, "筒"),
        MahjongTile(0, "中"),
        MahjongTile(0, "發"),
        MahjongTile(0, "白"),
        MahjongTile(0, "東"),
        MahjongTile(0, "西"),
        MahjongTile(0, "南"),
        MahjongTile(0, "北"),
    ]
    tiles = tiles * 4

    tiles.sort()
    return tiles


def flower_tiles() -> list[MahjongTile]:
    flowers = [
        MahjongTile(0, "春"),
        MahjongTile(0, "夏"),
        MahjongTile(0, "秋"),
        MahjongTile(0, "東"),
        MahjongTile(0, "梅"),
        MahjongTile(0, "竹"),
        MahjongTile(0, "蘭"),
        MahjongTile(0, "菊"),
    ]
    return flowers
