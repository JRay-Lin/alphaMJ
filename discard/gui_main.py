"""
PySide6 GUI for AlphaMJ Mahjong Game
Provides a visual interface for better user interaction
"""

import sys
import json
from typing import List, Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QPushButton, QLabel, QTextEdit, QFrame, QGroupBox,
    QMessageBox, QDialog, QDialogButtonBox, QListWidget, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QTimer, QThread
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QIcon

# Import game logic
from main import MahjongGame, MahjongRule
from modules.tile import MahjongTile
from modules.player import MahjongPlayer


class TileWidget(QPushButton):
    """Widget representing a single mahjong tile."""
    
    def __init__(self, tile: MahjongTile = None, clickable: bool = True, parent=None):
        super().__init__(parent)
        self.tile = tile
        self.clickable = clickable
        self.selected = False
        
        self.setFixedSize(60, 80)
        self.setStyleSheet(self.get_tile_style())
        
        if tile:
            self.setText(str(tile))
            self.setToolTip(f"{tile.name} ({tile.type})")
        
        if not clickable:
            self.setEnabled(False)
    
    def get_tile_style(self) -> str:
        """Get CSS styling for the tile based on its properties."""
        if not self.tile:
            return """
                QPushButton {
                    background-color: #f0f0f0;
                    border: 2px solid #ccc;
                    border-radius: 5px;
                    font-size: 10px;
                    color: #333;
                }
            """
        
        # Different colors for different tile types
        type_colors = {
            "萬": "#FFE4E1",  # Light pink
            "筒": "#E1F5FE",  # Light blue  
            "條": "#F1F8E9",  # Light green
            "風": "#FFF3E0",  # Light orange
            "元": "#F3E5F5",  # Light purple
            "花": "#FFEBEE"   # Light red
        }
        
        color = type_colors.get(self.tile.type, "#f0f0f0")
        border_color = "#FF6B6B" if self.selected else "#666"
        border_width = "3px" if self.selected else "1px"
        
        return f"""
            QPushButton {{
                background-color: {color};
                border: {border_width} solid {border_color};
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
                color: #333;
            }}
            QPushButton:hover {{
                border: 2px solid #FF6B6B;
                background-color: {self.lighten_color(color)};
                color: #333;
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color)};
                color: #333;
            }}
        """
    
    def lighten_color(self, color: str) -> str:
        """Lighten a color for hover effect."""
        return color.replace("E1", "F0").replace("F1", "F8").replace("F3", "F9")
    
    def darken_color(self, color: str) -> str:
        """Darken a color for pressed effect."""
        return color.replace("F0", "E1").replace("F8", "F1").replace("F9", "F3")
    
    def set_selected(self, selected: bool):
        """Set tile selection state."""
        self.selected = selected
        self.setStyleSheet(self.get_tile_style())
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.clickable and self.tile:
            super().mousePressEvent(event)


class PlayerHandWidget(QFrame):
    """Widget displaying a player's hand and sets."""
    
    tile_clicked = Signal(int)  # Emits tile index when clicked
    
    def __init__(self, player_name: str, is_current: bool = False, parent=None):
        super().__init__(parent)
        self.player_name = player_name
        self.is_current = is_current
        self.tile_widgets = []
        self.set_widgets = []
        
        self.setup_ui()
        self.update_style()
    
    def setup_ui(self):
        """Set up the player hand UI."""
        layout = QVBoxLayout()
        
        # Player name and info
        self.name_label = QLabel(self.player_name)
        self.name_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(self.name_label)
        
        # Hand tiles
        self.hand_layout = QHBoxLayout()
        self.hand_widget = QWidget()
        self.hand_widget.setLayout(self.hand_layout)
        layout.addWidget(self.hand_widget)
        
        # Exposed sets
        self.sets_label = QLabel("Exposed Sets:")
        self.sets_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.sets_label)
        
        self.sets_layout = QVBoxLayout()
        self.sets_widget = QWidget()
        self.sets_widget.setLayout(self.sets_layout)
        layout.addWidget(self.sets_widget)
        
        # Discard info
        self.discard_label = QLabel("Recent discards: ")
        self.discard_label.setFont(QFont("Arial", 9))
        layout.addWidget(self.discard_label)
        
        self.setLayout(layout)
    
    def update_style(self):
        """Update styling based on current player status."""
        if self.is_current:
            self.setStyleSheet("""
                QFrame {
                    border: 3px solid #4CAF50;
                    border-radius: 10px;
                    background-color: #F8FFF8;
                    padding: 5px;
                }
            """)
            self.name_label.setStyleSheet("color: #4CAF50;")
        else:
            self.setStyleSheet("""
                QFrame {
                    border: 1px solid #ccc;
                    border-radius: 10px;
                    background-color: #fafafa;
                    padding: 5px;
                }
            """)
            self.name_label.setStyleSheet("color: #333;")
    
    def update_hand(self, tiles: List[MahjongTile], clickable: bool = True):
        """Update the displayed hand tiles."""
        # Clear existing tiles
        for widget in self.tile_widgets:
            self.hand_layout.removeWidget(widget)
            widget.deleteLater()
        self.tile_widgets.clear()
        
        # Add new tiles
        for i, tile in enumerate(tiles):
            tile_widget = TileWidget(tile, clickable)
            tile_widget.clicked.connect(lambda checked, idx=i: self.tile_clicked.emit(idx))
            self.hand_layout.addWidget(tile_widget)
            self.tile_widgets.append(tile_widget)
        
        # Add stretch to left-align tiles
        self.hand_layout.addStretch()
    
    def update_sets(self, sets: List[List[MahjongTile]]):
        """Update the displayed exposed sets."""
        # Clear existing sets
        for widget in self.set_widgets:
            self.sets_layout.removeWidget(widget)
            widget.deleteLater()
        self.set_widgets.clear()
        
        if not sets:
            self.sets_label.setText("Exposed Sets: None")
            return
        
        self.sets_label.setText(f"Exposed Sets ({len(sets)}):")
        
        for i, meld_set in enumerate(sets):
            set_layout = QHBoxLayout()
            set_widget = QWidget()
            
            # Determine set type
            set_type = self.get_set_type(meld_set)
            type_label = QLabel(f"{set_type}:")
            type_label.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            set_layout.addWidget(type_label)
            
            # Add tiles in set
            for tile in meld_set:
                tile_widget = TileWidget(tile, clickable=False)
                tile_widget.setFixedSize(40, 50)  # Smaller for sets
                set_layout.addWidget(tile_widget)
            
            set_layout.addStretch()
            set_widget.setLayout(set_layout)
            self.sets_layout.addWidget(set_widget)
            self.set_widgets.append(set_widget)
    
    def get_set_type(self, meld_set: List[MahjongTile]) -> str:
        """Get the type name of a meld set."""
        if len(meld_set) == 4:
            return "Kong"
        elif len(meld_set) == 3:
            if all(tile.name == meld_set[0].name for tile in meld_set):
                return "Pong"
            else:
                return "Chi"
        return "Unknown"
    
    def update_discards(self, discards: List[MahjongTile]):
        """Update the recent discards display."""
        if not discards:
            self.discard_label.setText("Recent discards: None")
        else:
            recent = discards[-3:]  # Last 3 discards
            discard_str = ", ".join(str(tile) for tile in recent)
            self.discard_label.setText(f"Recent discards: {discard_str}")
    
    def set_current(self, is_current: bool):
        """Set whether this is the current player."""
        self.is_current = is_current
        self.update_style()
    
    def set_tile_selection(self, tile_index: int, selected: bool):
        """Set selection state of a specific tile."""
        if 0 <= tile_index < len(self.tile_widgets):
            self.tile_widgets[tile_index].set_selected(selected)


class ActionDialog(QDialog):
    """Dialog for player action decisions (chi/pong/kong)."""
    
    def __init__(self, action_type: str, discarded_tile: MahjongTile, 
                 action_data=None, player_hand=None, parent=None):
        super().__init__(parent)
        self.action_type = action_type
        self.discarded_tile = discarded_tile
        self.action_data = action_data
        self.player_hand = player_hand
        self.selected_option = None
        
        self.setup_ui()
        self.setModal(True)
    
    def setup_ui(self):
        """Set up the action dialog UI."""
        self.setWindowTitle(f"{self.action_type.title()} Action")
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"You can {self.action_type} with {self.discarded_tile}!")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Show discarded tile
        tile_layout = QHBoxLayout()
        tile_layout.addStretch()
        discarded_widget = TileWidget(self.discarded_tile, False)
        tile_layout.addWidget(discarded_widget)
        tile_layout.addStretch()
        layout.addLayout(tile_layout)
        
        # Action-specific options
        if self.action_type == "chi" and self.action_data:
            self.setup_chi_options(layout)
        elif self.action_type == "pong":
            self.setup_pong_info(layout)
        elif self.action_type == "kong":
            self.setup_kong_info(layout)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        button_box.button(QDialogButtonBox.StandardButton.Yes).setText(f"Take {self.action_type.title()}")
        button_box.button(QDialogButtonBox.StandardButton.No).setText("Pass")
        
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def setup_chi_options(self, layout):
        """Set up chi-specific options."""
        options_label = QLabel("Available sequences:")
        options_label.setFont(QFont("Arial", 12))
        layout.addWidget(options_label)
        
        self.chi_list = QListWidget()
        for i, (idx1, idx2) in enumerate(self.action_data):
            if self.player_hand and idx1 < len(self.player_hand) and idx2 < len(self.player_hand):
                tile1, tile2 = self.player_hand[idx1], self.player_hand[idx2]
                option_text = f"Option {i}: Use {tile1} and {tile2}"
                self.chi_list.addItem(option_text)
        
        self.chi_list.setCurrentRow(0)  # Select first option by default
        layout.addWidget(self.chi_list)
    
    def setup_pong_info(self, layout):
        """Set up pong-specific information."""
        info_label = QLabel(f"Form a pong with your matching {self.discarded_tile} tiles")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
    
    def setup_kong_info(self, layout):
        """Set up kong-specific information."""
        info_label = QLabel(f"Form a kong with {self.discarded_tile}")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_label)
    
    def get_chi_option(self) -> int:
        """Get selected chi option."""
        if hasattr(self, 'chi_list'):
            return self.chi_list.currentRow()
        return 0


class MahjongGUI(QMainWindow):
    """Main GUI window for the Mahjong game."""
    
    def __init__(self):
        super().__init__()
        self.game = None
        self.current_action_dialog = None
        self.selected_tile_index = -1
        
        self.setup_ui()
        self.setup_game()
    
    def setup_ui(self):
        """Set up the main UI."""
        self.setWindowTitle("AlphaMJ - Mahjong Game")
        self.setGeometry(100, 100, 1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left side: Game board
        self.setup_game_board(main_layout)
        
        # Right side: Controls and log
        self.setup_controls(main_layout)
    
    def setup_game_board(self, main_layout):
        """Set up the game board area."""
        board_widget = QWidget()
        board_layout = QVBoxLayout()
        
        # Player hands (arranged around the table)
        self.player_widgets = []
        
        # Top player (opponent)
        self.player_widgets.append(PlayerHandWidget("Player 2"))
        board_layout.addWidget(self.player_widgets[0])
        
        # Middle row: left and right players
        middle_layout = QHBoxLayout()
        self.player_widgets.append(PlayerHandWidget("Player 4"))  # Left
        middle_layout.addWidget(self.player_widgets[1])
        
        # Center area for discarded tiles and info
        center_widget = QGroupBox("Game Center")
        center_layout = QVBoxLayout()
        
        self.game_info_label = QLabel("Game: Standard Mahjong\nTurn: 1\nWall: 84 tiles")
        self.game_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.game_info_label.setFont(QFont("Arial", 10))
        center_layout.addWidget(self.game_info_label)
        
        self.last_discard_label = QLabel("Last discard: None")
        self.last_discard_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(self.last_discard_label)
        
        center_widget.setLayout(center_layout)
        middle_layout.addWidget(center_widget)
        
        self.player_widgets.append(PlayerHandWidget("Player 3"))  # Right
        middle_layout.addWidget(self.player_widgets[2])
        
        board_layout.addLayout(middle_layout)
        
        # Bottom player (human player)
        self.player_widgets.append(PlayerHandWidget("Player 1 (You)", True))
        board_layout.addWidget(self.player_widgets[3])
        
        # Reorder widgets to match game player order: [0]=Player1, [1]=Player2, [2]=Player3, [3]=Player4
        # Current order is [Player2, Player4, Player3, Player1], need to reorder
        temp_widgets = self.player_widgets[:]
        self.player_widgets = [temp_widgets[3], temp_widgets[0], temp_widgets[2], temp_widgets[1]]
        
        # Connect tile clicks for human player
        self.player_widgets[0].tile_clicked.connect(self.on_tile_clicked)
        
        board_widget.setLayout(board_layout)
        main_layout.addWidget(board_widget, 2)  # 2/3 of space
    
    def setup_controls(self, main_layout):
        """Set up the controls and log area."""
        controls_widget = QWidget()
        controls_layout = QVBoxLayout()
        
        # Action buttons
        action_group = QGroupBox("Actions")
        action_layout = QGridLayout()
        
        self.discard_button = QPushButton("Discard Selected")
        self.discard_button.clicked.connect(self.discard_selected_tile)
        self.discard_button.setEnabled(False)
        action_layout.addWidget(self.discard_button, 0, 0)
        
        self.hidden_kong_button = QPushButton("Hidden Kong")
        self.hidden_kong_button.clicked.connect(self.check_hidden_kong)
        action_layout.addWidget(self.hidden_kong_button, 0, 1)
        
        action_group.setLayout(action_layout)
        controls_layout.addWidget(action_group)
        
        # Game log
        log_group = QGroupBox("Game Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        controls_layout.addWidget(log_group)
        
        # Game controls
        game_group = QGroupBox("Game Controls")
        game_layout = QVBoxLayout()
        
        self.new_game_button = QPushButton("New Game")
        self.new_game_button.clicked.connect(self.new_game)
        game_layout.addWidget(self.new_game_button)
        
        self.ai_setup_button = QPushButton("Setup AI Players")
        self.ai_setup_button.clicked.connect(self.setup_ai_dialog)
        game_layout.addWidget(self.ai_setup_button)
        
        game_group.setLayout(game_layout)
        controls_layout.addWidget(game_group)
        
        controls_layout.addStretch()
        controls_widget.setLayout(controls_layout)
        main_layout.addWidget(controls_widget, 1)  # 1/3 of space
    
    def setup_game(self, ai_players=None):
        """Initialize the game."""
        rule = MahjongRule("standard", 4)
        self.game = MahjongGame(rule)
        self.game.setup_game(ai_players)
        self.update_display()
        if ai_players:
            self.log_message(f"Game started! AI players: {ai_players}")
        else:
            self.log_message("Game started! You are Player 1.")
    
    def update_display(self):
        """Update the entire display."""
        if not self.game:
            return
        
        # Update player hands
        for i, player in enumerate(self.game.players):
            widget = self.player_widgets[i]
            
            # Update current player indicator
            widget.set_current(i == self.game.current_player)
            
            # Update hand (only show human player's tiles)
            if i == 0:  # Human player
                widget.update_hand(player.hand.hands, True)
            else:  # AI/other players - show back of tiles
                hidden_tiles = [None] * len(player.hand.hands)
                widget.update_hand(hidden_tiles, False)
            
            # Update sets and discards
            try:
                sets = getattr(player.hand, 'sets', [])
                discards = getattr(player.hand, 'discards', [])
                widget.update_sets(sets)
                widget.update_discards(discards)
            except AttributeError as e:
                print(f"Error updating sets/discards for {player.name}: {e}")
                widget.update_sets([])
                widget.update_discards([])
        
        # Update game info
        info_text = f"Game: {self.game.rule.title()} Mahjong\n"
        info_text += f"Turn: {self.game.turn_count + 1}\n"
        info_text += f"Wall: {len(self.game.wall.tiles)} tiles"
        self.game_info_label.setText(info_text)
        
        # Update last discard
        try:
            if self.game.players and any(getattr(p.hand, 'discards', []) for p in self.game.players):
                last_discards = []
                for player in self.game.players:
                    discards = getattr(player.hand, 'discards', [])
                    if discards:
                        last_discards.append((player.name, discards[-1]))
                
                if last_discards:
                    last_player, last_tile = last_discards[-1]
                    self.last_discard_label.setText(f"Last discard: {last_tile} by {last_player}")
                else:
                    self.last_discard_label.setText("Last discard: None")
            else:
                self.last_discard_label.setText("Last discard: None")
        except Exception as e:
            print(f"Error updating last discard: {e}")
            self.last_discard_label.setText("Last discard: None")
        
        # Update button states
        self.update_button_states()
    
    def update_button_states(self):
        """Update the enabled state of action buttons."""
        if not self.game:
            return
        
        current_player = self.game.players[self.game.current_player]
        is_human_turn = self.game.current_player == 0 and not getattr(current_player, 'is_ai', False)
        
        self.discard_button.setEnabled(is_human_turn and self.selected_tile_index >= 0)
        self.hidden_kong_button.setEnabled(is_human_turn)
    
    def on_tile_clicked(self, tile_index: int):
        """Handle tile click events."""
        # Clear previous selection
        if self.selected_tile_index >= 0:
            self.player_widgets[0].set_tile_selection(self.selected_tile_index, False)
        
        # Set new selection
        self.selected_tile_index = tile_index
        self.player_widgets[0].set_tile_selection(tile_index, True)
        self.update_button_states()
        
        self.log_message(f"Selected tile {tile_index}: {self.game.players[0].hand.hands[tile_index]}")
    
    def discard_selected_tile(self):
        """Discard the selected tile."""
        if not self.game or self.selected_tile_index < 0:
            return
        
        current_player = self.game.players[self.game.current_player]
        if self.game.current_player != 0:  # Not human player's turn
            return
        
        # Discard tile
        discarded_tile = self.game.discard_tile(current_player, self.selected_tile_index)
        self.log_message(f"You discarded: {discarded_tile}")
        
        # Clear selection
        self.selected_tile_index = -1
        
        # Handle actions from other players
        action_taken = self.handle_discard_actions(discarded_tile, self.game.current_player)
        
        if not action_taken:
            # Normal turn progression
            self.game.current_player = (self.game.current_player + 1) % self.game.num_players
            self.game.turn_count += 1
        
        # Check for game end
        if self.check_game_end():
            return
        
        # Update display
        self.update_display()
        
        # Process AI turns
        self.process_ai_turns()
    
    def handle_discard_actions(self, discarded_tile: MahjongTile, discarder_index: int) -> bool:
        """Handle potential actions from players after a discard."""
        # Get all possible actions
        possible_actions = []
        
        for i in range(self.game.num_players):
            if i == discarder_index:
                continue
            
            player_obj = self.game.players[i]
            is_next_player = i == (discarder_index + 1) % self.game.num_players
            actions = player_obj.get_available_actions(discarded_tile, is_next_player)
            
            # Add actions with priorities
            if actions["kong"]["normal"] or actions["kong"]["promote"]:
                possible_actions.append(("kong", i, actions["kong"]))
            if actions["pong"]:
                possible_actions.append(("pong", i, None))
            if actions["chi"] and is_next_player:
                possible_actions.append(("chi", i, actions["chi"]))
        
        if not possible_actions:
            return False
        
        # Sort by priority
        priority_order = {"kong": 3, "pong": 2, "chi": 1}
        possible_actions.sort(key=lambda x: priority_order[x[0]], reverse=True)
        
        # Ask players in priority order
        for action_type, player_idx, action_data in possible_actions:
            player_obj = self.game.players[player_idx]
            
            # Check if player wants to take action
            if self.ask_player_action(action_type, player_idx, discarded_tile, action_data):
                # Execute action
                success = self.execute_player_action(action_type, player_idx, discarded_tile, action_data)
                if success:
                    return True
        
        return False
    
    def ask_player_action(self, action_type: str, player_idx: int, discarded_tile: MahjongTile, action_data) -> bool:
        """Ask a player if they want to perform an action."""
        player_obj = self.game.players[player_idx]
        
        # AI player decision
        if getattr(player_obj, 'is_ai', False):
            return self.game.get_ai_action_decision(player_obj, action_type, discarded_tile, action_data)
        
        # Human player decision - show dialog
        dialog = ActionDialog(action_type, discarded_tile, action_data, player_obj.hand.hands, self)
        result = dialog.exec()
        
        if result == QDialog.DialogCode.Accepted:
            if action_type == "chi" and action_data:
                self.chi_option = dialog.get_chi_option()
            return True
        
        return False
    
    def execute_player_action(self, action_type: str, player_idx: int, discarded_tile: MahjongTile, action_data) -> bool:
        """Execute a player's action."""
        player_obj = self.game.players[player_idx]
        success = False
        
        if action_type == "chi":
            if action_data:
                if hasattr(self, 'chi_option') and 0 <= self.chi_option < len(action_data):
                    success = player_obj.execute_chi(discarded_tile, action_data[self.chi_option])
                else:
                    success = player_obj.execute_chi(discarded_tile, action_data[0])
        elif action_type == "pong":
            success = player_obj.execute_pong(discarded_tile)
        elif action_type == "kong":
            if action_data and action_data.get("normal"):
                success = player_obj.execute_normal_kong(discarded_tile)
            elif action_data and action_data.get("promote"):
                success = player_obj.execute_promote_kong(discarded_tile)
        
        if success:
            self.log_message(f"{player_obj.name} performed {action_type} with {discarded_tile}")
            
            # Draw replacement tile for kong
            if action_type == "kong":
                if self.game.wall.tiles:
                    replacement_tile = self.game.wall.tiles.pop(-1)
                    player_obj.hand.hands.append(replacement_tile)
                    player_obj.hand.hands.sort()
                    self.log_message(f"{player_obj.name} draws replacement tile")
            
            # Update turn
            self.game.current_player = player_idx
            
            # Check for win
            if self.game.check_win(player_obj):
                self.game.winner = player_obj
                self.game.game_over = True
                self.show_win_message(player_obj)
                return True
        
        return success
    
    def check_hidden_kong(self):
        """Check for hidden kong options."""
        if not self.game or self.game.current_player != 0:
            return
        
        current_player = self.game.players[0]
        kong_options = current_player.get_hidden_kong_options()
        
        if not kong_options:
            QMessageBox.information(self, "Hidden Kong", "No hidden kong options available.")
            return
        
        # Show options dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Hidden Kong")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Available hidden kong options:"))
        
        option_list = QListWidget()
        for i, tile in enumerate(kong_options):
            option_list.addItem(f"Kong with {tile}")
        
        option_list.setCurrentRow(0)
        layout.addWidget(option_list)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_idx = option_list.currentRow()
            if 0 <= selected_idx < len(kong_options):
                kong_tile = kong_options[selected_idx]
                if current_player.execute_hidden_kong(kong_tile):
                    self.log_message(f"You performed hidden kong with {kong_tile}")
                    # Draw replacement tile
                    if self.game.wall.tiles:
                        replacement_tile = self.game.wall.tiles.pop(-1)
                        current_player.hand.hands.append(replacement_tile)
                        current_player.hand.hands.sort()
                        self.log_message("You draw a replacement tile")
                    
                    self.update_display()
    
    def process_ai_turns(self):
        """Process AI player turns."""
        while (not self.game.game_over and 
               self.game.current_player != 0 and 
               getattr(self.game.players[self.game.current_player], 'is_ai', True)):
            
            # AI turn
            current_player = self.game.players[self.game.current_player]
            
            # AI draws tile
            if not self.game.draw_tile(current_player):
                break
            
            self.log_message(f"{current_player.name} draws a tile")
            
            # Check for win after drawing
            if self.game.check_win(current_player):
                self.game.winner = current_player
                self.game.game_over = True
                self.show_win_message(current_player)
                break
            
            # AI discards
            discard_choice = self.game.get_ai_discard_choice(current_player)
            discarded_tile = self.game.discard_tile(current_player, discard_choice)
            
            if discarded_tile:
                self.log_message(f"{current_player.name} discards: {discarded_tile}")
                
                # Handle actions from other players
                action_taken = self.handle_discard_actions(discarded_tile, self.game.current_player)
                
                if not action_taken:
                    # Normal progression
                    self.game.current_player = (self.game.current_player + 1) % self.game.num_players
                    self.game.turn_count += 1
            
            # Check for game end
            if self.check_game_end():
                break
            
            # Update display
            self.update_display()
        
        # Final display update
        self.update_display()
    
    def check_game_end(self) -> bool:
        """Check if the game has ended."""
        if self.game.game_over:
            if self.game.winner:
                self.show_win_message(self.game.winner)
            else:
                QMessageBox.information(self, "Game Over", "Game ended in a draw.")
            return True
        
        if len(self.game.wall.tiles) == 0:
            QMessageBox.information(self, "Game Over", "Wall is empty! Game ends in draw.")
            self.game.game_over = True
            return True
        
        return False
    
    def show_win_message(self, winner: MahjongPlayer):
        """Show win message."""
        QMessageBox.information(self, "Game Over", f"🎉 {winner.name} wins! 🎉")
    
    def log_message(self, message: str):
        """Add message to game log."""
        self.log_text.append(message)
        self.log_text.ensureCursorVisible()
    
    def new_game(self):
        """Start a new game."""
        self.game = None
        self.selected_tile_index = -1
        self.log_text.clear()
        self.setup_game()
    
    def setup_ai_dialog(self):
        """Show AI setup dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("AI Player Setup")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Select which players should be AI:"))
        
        # Checkboxes for each player
        self.ai_checkboxes = []
        for i in range(4):
            checkbox = QPushButton(f"Player {i+1}")
            checkbox.setCheckable(True)
            checkbox.setChecked(i > 0)  # Default: only player 1 is human
            layout.addWidget(checkbox)
            self.ai_checkboxes.append(checkbox)
        
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            ai_players = [i for i, cb in enumerate(self.ai_checkboxes) if cb.isChecked()]
            
            # Start new game with AI setup
            self.setup_game(ai_players)


def main():
    """Main function to run the GUI."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MahjongGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()