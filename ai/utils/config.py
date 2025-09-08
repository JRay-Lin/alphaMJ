"""Configuration settings for AI training."""

class DQNConfig:
    """Deep Q-Network configuration parameters."""
    
    # Network Architecture
    STATE_SIZE = 348  # Total state vector size (measured from actual encoder output)
    HIDDEN_SIZES = [512, 256, 128]  # Hidden layer sizes
    ACTION_SIZE = 14  # Max number of tiles in hand (discard actions)
    
    # Training Parameters
    LEARNING_RATE = 0.0005  # Reduced for more stable learning
    BATCH_SIZE = 64         # Increased for better gradient estimates
    GAMMA = 0.95           # Slightly lower discount for faster learning
    TAU = 0.01             # Faster target network updates
    UPDATE_EVERY = 2       # More frequent updates for faster learning
    
    # Exploration Parameters - Much slower decay
    EPSILON_START = 1.0
    EPSILON_END = 0.15     # High minimum exploration for complex game
    EPSILON_DECAY = 0.9999 # Very slow decay - explore for thousands of episodes
    
    # Memory Parameters
    BUFFER_SIZE = 100000  # Experience replay buffer size
    MIN_BUFFER_SIZE = 1000  # Minimum buffer size before training starts
    
    # Training Parameters
    MAX_EPISODES = 10000
    MAX_STEPS_PER_EPISODE = 200
    TARGET_UPDATE_FREQUENCY = 1000  # Update target network every N steps
    
    # Checkpointing
    SAVE_FREQUENCY = 100  # Save model every N episodes
    
    # Stage 1 Rewards
    STAGE1_WIN_REWARD = 1.0
    STAGE1_LOSS_REWARD = -0.5
    STAGE1_TURN_PENALTY = -0.01
    STAGE1_DRAW_REWARD = 0.0
    
    # Stage 2 Rewards
    STAGE2_SCORE_MULTIPLIER = 0.1
    STAGE2_BASE_WIN_REWARD = 0.5
    STAGE2_OPPONENT_SCORE_PENALTY = -0.05


class StateConfig:
    """State representation configuration."""
    
    # Tile encoding
    TILE_TYPES = 34  # Total unique tile types
    MAX_TILE_COUNT = 4  # Maximum count of each tile type
    
    # Hand representation
    MAX_HAND_SIZE = 14
    MAX_SETS = 5
    SET_SIZE = 4  # Maximum tiles in a set (kong)
    
    # Game context
    NUM_PLAYERS = 4
    MAX_WALL_SIZE = 144
    MAX_DISCARD_HISTORY = 10  # Number of recent discards to track per player
    
    # State vector breakdown
    HAND_FEATURES = TILE_TYPES * MAX_TILE_COUNT  # 136
    SET_FEATURES = MAX_SETS * SET_SIZE * TILE_TYPES  # Variable encoding
    GAME_CONTEXT_FEATURES = 20  # Player index, wall remaining, turn count, etc.
    OTHER_PLAYERS_FEATURES = (NUM_PLAYERS - 1) * 30  # Visible info per opponent
    
    @property
    def total_features(self):
        return (self.HAND_FEATURES + 
                self.GAME_CONTEXT_FEATURES + 
                self.OTHER_PLAYERS_FEATURES + 
                100)  # Buffer for sets and other features


# Global config instances
dqn_config = DQNConfig()
state_config = StateConfig()