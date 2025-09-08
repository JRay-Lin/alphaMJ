from modules.tile import default_tiles, MahjongTile
from modules.wall import MahjongWall
from modules.player import MahjongPlayer


class MahjongRule:
    def __init__(self, name, nums_player) -> None:
        if name not in ["standard", "taiwan"]:
            raise ValueError()

        self.rule_name = name
        self.nums_player = nums_player

        return None


class MahjongGame:
    def __init__(self, rule: MahjongRule):
        self.rule = rule.rule_name
        self.num_players = rule.nums_player
        self.players = []
        self.wall = MahjongWall(tiles=default_tiles(), random_seed=42)
        self.current_player = 0
        self.turn_count = 0
        self.game_over = False
        self.winner = None

    def setup_game(self, ai_players: list = None):  # type: ignore
        """Initialize the game by creating players and dealing initial hands.

        Args:
            ai_players: List of indices for AI players (e.g. [1, 3] for players 2 and 4)
        """
        print("Setting up Mahjong game...")

        # Create players
        player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
        ai_players = ai_players or []

        for i in range(self.num_players):
            is_ai = i in ai_players
            player_type = "AI" if is_ai else "Human"
            self.players.append(
                MahjongPlayer(f"{player_names[i]} ({player_type})", i, is_ai)
            )
            print(
                f"Created {self.players[i].name} with wind: {self.players[i].self_wind}"
            )

        # Shuffle the wall
        self.wall.shuffle()
        print("Wall shuffled.")

        # Deal initial hands (13 tiles each)
        self.deal_initial_hands()

        print(f"Game setup complete. Rule: {self.rule}")
        print(f"Starting player: {self.players[self.current_player].name}")

    def deal_initial_hands(self):
        """Deal 13 tiles to each player."""
        for round_num in range(13):
            for player_idx in range(self.num_players):
                if self.wall.tiles:
                    tile = self.wall.tiles.pop(0)
                    self.players[player_idx].hand.hands.append(tile)

        # Sort each player's hand
        for player in self.players:
            player.hand.hands.sort()

        print("Initial hands dealt to all players.")

    def draw_tile(self, player_obj: MahjongPlayer) -> bool:
        """Draw a tile from the wall for the given player."""
        if not self.wall.tiles:
            print("Wall is empty! Game ends in draw.")
            self.game_over = True
            return False

        tile = self.wall.tiles.pop(0)
        player_obj.hand.hands.append(tile)
        player_obj.hand.hands.sort()
        return True

    def discard_tile(self, player_obj: MahjongPlayer, tile_index: int) -> MahjongTile:
        """Player discards a tile from their hand."""
        if 0 <= tile_index < len(player_obj.hand.hands):
            discarded_tile = player_obj.hand.hands.pop(tile_index)
            player_obj.hand.discards.append(discarded_tile)
            return discarded_tile
        return None  # type: ignore

    def check_win(self, player_obj: MahjongPlayer) -> bool:
        """Check if the player has a winning hand."""
        return player_obj.is_winning(self.rule)

    def display_player_hand(self, player_obj: MahjongPlayer):
        """Display a player's current hand."""
        print(f"\n{player_obj.name}'s hand ({len(player_obj.hand.hands)} tiles):")
        for i, tile in enumerate(player_obj.hand.hands):
            print(f"  {i}: {tile}")

        if player_obj.hand.discards:
            print(f"Discards: {player_obj.hand.discards[-3:]}")  # Show last 3 discards

    def display_player_sets(self, player_obj: MahjongPlayer):
        """Display a player's exposed sets."""
        if player_obj.hand.sets:
            print(f"\n{player_obj.name}'s exposed sets:")
            for i, meld_set in enumerate(player_obj.hand.sets):
                set_type = self.get_set_type_name(meld_set)
                tiles_str = ", ".join(str(tile) for tile in meld_set)
                print(f"  Set {i+1} ({set_type}): [{tiles_str}]")

    def get_set_type_name(self, meld_set: list) -> str:
        """Get the name of a meld set type."""
        if len(meld_set) == 4:
            return "Kong"
        elif len(meld_set) == 3:
            if all(tile.name == meld_set[0].name for tile in meld_set):
                return "Pong"
            else:
                return "Chi"
        elif len(meld_set) == 2:
            return "Pair"
        return "Unknown"

    def get_player_input(self, player_obj: MahjongPlayer) -> int:
        """Get player's choice of tile to discard."""
        # Check if player is AI
        if hasattr(player_obj, "is_ai") and player_obj.is_ai:
            return self.get_ai_discard_choice(player_obj)

        # Human player input
        while True:
            try:
                choice = input(
                    f"Choose tile to discard (0-{len(player_obj.hand.hands)-1}): "
                )
                index = int(choice)
                if 0 <= index < len(player_obj.hand.hands):
                    return index
                else:
                    print(
                        f"Please enter a number between 0 and {len(player_obj.hand.hands)-1}"
                    )
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                self.game_over = True
                return 0

    def get_ai_discard_choice(self, player_obj) -> int:
        """Get AI choice for tile to discard (placeholder for Q-learning)."""
        # Get game state for decision
        state = self.get_game_state_for_ai(player_obj.index)

        # Simple heuristic: discard randomly for now
        # In Q-learning, this would be where the AI model makes the decision
        import random

        choice = random.randint(0, len(player_obj.hand.hands) - 1)
        print(
            f"🤖 {player_obj.name} (AI) chooses to discard tile {choice}: {player_obj.hand.hands[choice]}"
        )
        return choice

    def play_turn(self):
        """Execute one turn for the current player."""
        current_player_obj = self.players[self.current_player]
        print(f"\n{'='*50}")
        print(f"Turn {self.turn_count + 1}: {current_player_obj.name}'s turn")
        print(f"{'='*50}")

        # Check for hidden kong options before drawing
        hidden_kong_options = current_player_obj.get_hidden_kong_options()
        kong_performed = False
        if hidden_kong_options:
            if self.ask_hidden_kong(current_player_obj, hidden_kong_options):
                # Player chose to do hidden kong, draw replacement tile
                if not self.draw_replacement_tile(current_player_obj):
                    return  # Game ends due to empty wall
                print(
                    f"{current_player_obj.name} draws a replacement tile after hidden kong."
                )
                kong_performed = True

        # Draw a tile (normal turn draw)
        if not self.draw_tile(current_player_obj):
            return  # Game ends due to empty wall

        print(f"{current_player_obj.name} draws a tile.")

        # Check for immediate win after drawing
        if self.check_win(current_player_obj):
            self.winner = current_player_obj
            self.game_over = True
            print(f"🎉 {current_player_obj.name} wins by self-draw!")
            return

        # Display current hand
        self.display_player_hand(current_player_obj)
        self.display_player_sets(current_player_obj)

        # Player chooses tile to discard
        tile_index = self.get_player_input(current_player_obj)
        discarded_tile = self.discard_tile(current_player_obj, tile_index)

        if discarded_tile:
            print(f"{current_player_obj.name} discards: {discarded_tile}")

            # Check for actions from other players
            action_taken = self.handle_discard_actions(
                discarded_tile, self.current_player
            )

            if not action_taken:
                # Normal turn progression if no actions taken
                self.current_player = (self.current_player + 1) % self.num_players
            # If action was taken, current_player is already updated

        self.turn_count += 1

    def display_game_state(self):
        """Display current game state."""
        print(f"\n--- Game State (Turn {self.turn_count}) ---")
        print(f"Tiles remaining in wall: {len(self.wall.tiles)}")
        print(f"Current player: {self.players[self.current_player].name}")

        for i, player_obj in enumerate(self.players):
            hand_size = len(player_obj.hand.hands)
            discard_size = len(player_obj.hand.discards)
            sets_size = len(player_obj.hand.sets)
            print(
                f"{player_obj.name}: {hand_size} tiles, {discard_size} discards, {sets_size} sets"
            )

    def handle_discard_actions(
        self, discarded_tile: MahjongTile, discarder_index: int
    ) -> bool:
        """
        Handle potential actions from other players after a discard.

        Args:
            discarded_tile: The tile that was discarded
            discarder_index: Index of the player who discarded

        Returns:
            bool: True if an action was taken, False otherwise
        """
        # First check for win conditions - highest priority
        for i in range(self.num_players):
            if i == discarder_index:
                continue  # Skip the player who discarded

            player_obj = self.players[i]

            # Check if this player can win with the discarded tile
            if self.can_win_with_tile(player_obj, discarded_tile):
                if self.ask_player_win(player_obj, discarded_tile):
                    # Player declares win
                    player_obj.hand.hands.append(discarded_tile)  # Add the winning tile
                    self.winner = player_obj
                    self.game_over = True
                    print(
                        f"🎉 {player_obj.name} wins by declaring Mahjong with {discarded_tile}!"
                    )
                    return True

        # Collect all possible actions from all players
        possible_actions = []

        for i in range(self.num_players):
            if i == discarder_index:
                continue  # Skip the player who discarded

            player_obj = self.players[i]
            is_next_player = i == (discarder_index + 1) % self.num_players

            actions = player_obj.get_available_actions(discarded_tile, is_next_player)

            # Add actions to the list with priorities
            if actions["kong"]["normal"] or actions["kong"]["promote"]:
                possible_actions.append(("kong", i, actions["kong"]))
            if actions["pong"]:
                possible_actions.append(("pong", i, None))
            if actions["chi"] and is_next_player:
                possible_actions.append(("chi", i, actions["chi"]))

        if not possible_actions:
            return False  # No actions available

        # Apply priority system: Kong > Pong > Chi
        priority_order = {"kong": 3, "pong": 2, "chi": 1}
        possible_actions.sort(key=lambda x: priority_order[x[0]], reverse=True)

        # If multiple actions of same priority, ask all players
        highest_priority = priority_order[possible_actions[0][0]]
        candidates = [
            action
            for action in possible_actions
            if priority_order[action[0]] == highest_priority
        ]

        # Always ask players for their choice, regardless of how many actions are possible
        for action_type, player_idx, action_data in candidates:
            if self.ask_player_action(
                action_type, player_idx, discarded_tile, action_data
            ):
                return self.execute_player_action(
                    action_type, player_idx, discarded_tile, action_data
                )

        # If all highest priority actions were declined, check lower priority ones
        remaining_actions = [
            action for action in possible_actions if action not in candidates
        ]
        for action_type, player_idx, action_data in remaining_actions:
            if self.ask_player_action(
                action_type, player_idx, discarded_tile, action_data
            ):
                return self.execute_player_action(
                    action_type, player_idx, discarded_tile, action_data
                )

        return False

    def ask_player_action(
        self,
        action_type: str,
        player_idx: int,
        discarded_tile: MahjongTile,
        action_data,
    ) -> bool:
        """Ask a player if they want to perform an action."""
        player_obj = self.players[player_idx]

        # Check if player is AI (for future Q-learning integration)
        if hasattr(player_obj, "is_ai") and player_obj.is_ai:
            return self.get_ai_action_decision(
                player_obj, action_type, discarded_tile, action_data
            )

        # Human player decision
        print(f"\n🔔 {player_obj.name}, you can {action_type} with {discarded_tile}!")

        if action_type == "chi":
            if not action_data:
                return False
            print("Available chi options:")
            for i, (idx1, idx2) in enumerate(action_data):
                tile1, tile2 = player_obj.hand.hands[idx1], player_obj.hand.hands[idx2]
                print(f"  Option {i}: Use {tile1} and {tile2} to form sequence")

        elif action_type == "pong":
            matching_tiles = [
                tile for tile in player_obj.hand.hands if tile == discarded_tile
            ]
            print(
                f"You have {len(matching_tiles)} matching {discarded_tile} tiles for pong"
            )

        elif action_type == "kong":
            if action_data and action_data.get("normal"):
                print(f"You can form a normal kong with your 3 {discarded_tile} tiles")
            if action_data and action_data.get("promote"):
                print(f"You can promote an existing pong to kong")

        # Show current hand for decision making
        self.display_player_hand_compact(player_obj)

        while True:
            try:
                choice = input(f"Do you want to {action_type}? (y/n): ").lower().strip()
                if choice in ["y", "yes"]:
                    return True
                elif choice in ["n", "no"]:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nSkipping action.")
                return False

    def execute_player_action(
        self,
        action_type: str,
        player_idx: int,
        discarded_tile: MahjongTile,
        action_data,
    ) -> bool:
        """Execute a player's action."""
        player_obj = self.players[player_idx]

        success = False

        if action_type == "chi":
            if action_data:
                # If multiple chi options, let player choose
                if len(action_data) > 1:
                    if player_obj.is_ai:
                        # AI automatically chooses first option
                        choice = 0
                        print(f"🤖 {player_obj.name} (AI) automatically chooses chi option {choice}")
                    else:
                        # Human player chooses
                        print("Choose chi option:")
                        for i, (idx1, idx2) in enumerate(action_data):
                            tile1, tile2 = (
                                player_obj.hand.hands[idx1],
                                player_obj.hand.hands[idx2],
                            )
                            print(f"  {i}: Use {tile1} and {tile2}")
                        try:
                            choice = int(input("Enter option number: "))
                        except (ValueError, EOFError):
                            choice = 0  # Default to first option if input fails
                    
                    if 0 <= choice < len(action_data):
                        success = player_obj.execute_chi(
                            discarded_tile, action_data[choice]
                        )
                else:
                    success = player_obj.execute_chi(discarded_tile, action_data[0])

        elif action_type == "pong":
            success = player_obj.execute_pong(discarded_tile)

        elif action_type == "kong":
            if action_data["normal"]:
                success = player_obj.execute_normal_kong(discarded_tile)
            elif action_data["promote"]:
                success = player_obj.execute_promote_kong(discarded_tile)

        if success:
            print(
                f"✓ {player_obj.name} successfully performed {action_type} with {discarded_tile}"
            )

            # For kong, draw replacement tile
            if action_type == "kong":
                if self.wall.tiles:
                    replacement_tile = self.wall.tiles.pop(-1)  # Draw from end for kong
                    player_obj.hand.hands.append(replacement_tile)
                    player_obj.hand.hands.sort()
                    print(f"{player_obj.name} draws a replacement tile.")

            # Update turn order - player who took action gets the turn
            self.current_player = player_idx

            # For chi/pong/kong, handle immediate discard without normal turn flow
            if action_type in ["chi", "pong", "kong"]:
                self.handle_immediate_discard(player_obj)

            # Check for win after action
            if self.check_win(player_obj):
                self.winner = player_obj
                self.game_over = True
                print(f"🎉 {player_obj.name} wins after {action_type}!")

            return True
        else:
            print(f"✗ {action_type} failed for {player_obj.name}")
            return False

    def ask_hidden_kong(self, player_obj: MahjongPlayer, kong_options: list) -> bool:
        """Ask player if they want to do hidden kong."""
        if not kong_options:
            return False

        print(f"\n{player_obj.name}, you can perform hidden kong with:")
        for i, tile in enumerate(kong_options):
            print(f"  {i}: {tile} (hidden kong)")

        choice = input(
            "Do you want to perform hidden kong? Enter tile number or 'n' to skip: "
        ).strip()

        if choice.lower() in ["n", "no", "skip"]:
            return False

        try:
            tile_choice = int(choice)
            if 0 <= tile_choice < len(kong_options):
                kong_tile = kong_options[tile_choice]
                if player_obj.execute_hidden_kong(kong_tile):
                    print(f"✓ {player_obj.name} performs hidden kong with {kong_tile}")
                    return True
                else:
                    print(f"✗ Hidden kong failed")
                    return False
        except ValueError:
            pass

        print("Invalid choice, skipping hidden kong.")
        return False

    def draw_replacement_tile(self, player_obj: MahjongPlayer) -> bool:
        """Draw replacement tile from end of wall (for kong)."""
        if not self.wall.tiles:
            print("No replacement tiles available!")
            return False

        tile = self.wall.tiles.pop(-1)  # Draw from end
        player_obj.hand.hands.append(tile)
        player_obj.hand.hands.sort()
        return True

    def display_player_hand_compact(self, player_obj: MahjongPlayer):
        """Display player's hand in compact format for action decisions."""
        tiles_str = ", ".join(str(tile) for tile in player_obj.hand.hands)
        print(f"Your hand: [{tiles_str}]")
        if player_obj.hand.sets:
            sets_str = " | ".join(
                f"{self.get_set_type_name(s)}: {s}" for s in player_obj.hand.sets
            )
            print(f"Your sets: {sets_str}")

    def get_ai_action_decision(
        self, player_obj, action_type: str, discarded_tile: MahjongTile, action_data
    ) -> bool:
        """
        Get AI decision for action (placeholder for Q-learning integration).

        Args:
            player_obj: AI player object
            action_type: Type of action (chi/pong/kong)
            discarded_tile: The discarded tile
            action_data: Additional action data

        Returns:
            bool: AI's decision to take action or not
        """
        # Get current game state for AI decision
        state = self.get_game_state_for_ai(player_obj.index)

        # Simple heuristic for now (replace with Q-learning model)
        import random

        action_probabilities = {
            "kong": 0.9,  # Almost always take kong (strongest)
            "pong": 0.7,  # Often take pong
            "chi": 0.5,  # Sometimes take chi
        }

        probability = action_probabilities.get(action_type, 0.5)
        decision = random.random() < probability

        print(
            f"🤖 {player_obj.name} (AI) {'accepts' if decision else 'declines'} {action_type}"
        )
        return decision

    def get_ai_hidden_kong_decision(self, player_obj, kong_options: list) -> bool:
        """Get AI decision for hidden kong (placeholder for Q-learning)."""
        if not kong_options:
            return False

        # Get game state for decision
        state = self.get_game_state_for_ai(player_obj.index)

        # Simple AI heuristic: usually do hidden kong
        import random

        if random.random() < 0.8:  # 80% chance
            kong_tile = kong_options[0]
            if player_obj.execute_hidden_kong(kong_tile):
                print(
                    f"🤖 {player_obj.name} (AI) performs hidden kong with {kong_tile}"
                )
                return True

        print(f"🤖 {player_obj.name} (AI) declines hidden kong")
        return False

    def can_win_with_tile(self, player_obj: MahjongPlayer, tile: MahjongTile) -> bool:
        """Check if player can win by taking the discarded tile."""
        # Temporarily add the tile to check for winning condition
        temp_hand = player_obj.hand.hands[:]
        temp_hand.append(tile)
        temp_hand.sort()

        # Create a temporary hand object to test winning condition
        original_hands = player_obj.hand.hands
        player_obj.hand.hands = temp_hand

        # Check if this creates a winning hand
        can_win = player_obj.is_winning(self.rule)

        # Restore original hand
        player_obj.hand.hands = original_hands

        return can_win

    def ask_player_win(self, player_obj: MahjongPlayer, tile: MahjongTile) -> bool:
        """Ask player if they want to declare win with the discarded tile."""
        # Check if player is AI
        if hasattr(player_obj, "is_ai") and player_obj.is_ai:
            print(f"🤖 {player_obj.name} (AI) declares Mahjong!")
            return True

        # Human player decision
        print(f"\n🎉 {player_obj.name}, you can win with {tile}!")
        print("Your current hand and sets:")
        self.display_player_hand_compact(player_obj)

        while True:
            try:
                choice = (
                    input("Do you want to declare Mahjong and win? (y/n): ")
                    .lower()
                    .strip()
                )
                if choice in ["y", "yes"]:
                    return True
                elif choice in ["n", "no"]:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nSkipping win declaration.")
                return False

    def handle_immediate_discard(self, player_obj: MahjongPlayer):
        """Handle immediate discard after chi/pong without going through normal turn flow."""
        print(f"\n{player_obj.name} must now discard a tile.")

        # Display current hand and sets
        self.display_player_hand(player_obj)
        self.display_player_sets(player_obj)

        # Player chooses tile to discard
        tile_index = self.get_player_input(player_obj)
        discarded_tile = self.discard_tile(player_obj, tile_index)

        if discarded_tile:
            print(f"{player_obj.name} discards: {discarded_tile}")

            # Check for actions from other players on this new discard
            action_taken = self.handle_discard_actions(
                discarded_tile, self.current_player
            )

            if not action_taken:
                # Normal turn progression if no actions taken
                self.current_player = (self.current_player + 1) % self.num_players

        # Increment turn count
        self.turn_count += 1

    def get_game_state_for_ai(self, player_idx: int) -> dict:
        """
        Get current game state for AI decision making.
        This is crucial for Q-learning integration.

        Returns:
            dict: Comprehensive game state for AI
        """
        player_obj = self.players[player_idx]

        state = {
            # Player's own state
            "hand": [tile.name for tile in player_obj.hand.hands],
            "hand_size": len(player_obj.hand.hands),
            "sets": [[tile.name for tile in meld] for meld in player_obj.hand.sets],
            "discards": [tile.name for tile in player_obj.hand.discards],
            "wind": player_obj.self_wind,
            # Game state
            "current_player": self.current_player,
            "turn_count": self.turn_count,
            "wall_remaining": len(self.wall.tiles),
            "rule": self.rule,
            # Other players (limited info like real mahjong)
            "other_players": [],
        }

        for i, other_player in enumerate(self.players):
            if i != player_idx:
                state["other_players"].append(
                    {
                        "index": i,
                        "hand_size": len(other_player.hand.hands),
                        "sets": [
                            [tile.name for tile in meld]
                            for meld in other_player.hand.sets
                        ],
                        "discards": [
                            tile.name for tile in other_player.hand.discards[-5:]
                        ],
                        "wind": other_player.self_wind,
                    }
                )

        return state

    def run_game(self):
        """Main game loop."""
        self.setup_game()

        print(f"\n🀄 Starting Mahjong Game! 🀄")
        print(f"Rule: {self.rule.title()}")

        # Main game loop
        while not self.game_over:
            try:
                self.play_turn()

                # Display game state every 4 turns (one round)
                if self.turn_count % 4 == 0:
                    self.display_game_state()

                # Safety check to prevent infinite games
                if self.turn_count > 200:
                    print("Game has gone on too long. Ending in draw.")
                    self.game_over = True

            except KeyboardInterrupt:
                print("\n\nGame interrupted by user. Goodbye!")
                break

        # Game end
        if self.winner:
            print(f"\n🎉 Game Over! {self.winner.name} wins! 🎉")
            self.display_player_hand(self.winner)
        else:
            print("\n Game ended in a draw.")

        print("Thanks for playing!")


def main():
    print("Welcome to Mahjong!")

    # Game configuration
    try:
        rule = MahjongRule("standard", 4)
        game = MahjongGame(rule)

        game.run_game()

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
