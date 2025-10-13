"""
Game Settings and Constants
Window dimensions, gameplay parameters, etc.
"""

# World dimensions
WORLD_WIDTH = 1250
WORLD_HEIGHT = 750
CELL_SIZE = 32  # GBA-style (Pokemon-like)

# Viewport dimensions (GBA-style: only ~12x9 tiles visible like Pokemon!)
VIEWPORT_WIDTH = 384   # 12 cells visible width
VIEWPORT_HEIGHT = 288  # 9 cells visible height

# UI dimensions
PANEL_WIDTH = 450
WINDOW_WIDTH = VIEWPORT_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = max(VIEWPORT_HEIGHT, 800)

# Simulation parameters
DEFAULT_TEMPERATURE = 25
DEFAULT_RAINFALL = 100
DEFAULT_NUM_HERBIVORES = 500
DEFAULT_NUM_CARNIVORES = 200

# Visual effects
TIME_SPEED = 0.001  # Speed of day/night cycle
WATER_ANIMATION_SPEED = 0.5
MAX_VOLCANO_PARTICLES = 100

# Minimap
MINIMAP_WIDTH = 250
MINIMAP_HEIGHT = 150

# Game performance
FPS = 60
DEFAULT_SPEED = 1

# Font sizes
FONT_TITLE_SIZE = 42
FONT_LARGE_SIZE = 32
FONT_MEDIUM_SIZE = 22
FONT_SMALL_SIZE = 16
FONT_COORD_SIZE = 14

# Dinosaur sprite scaling (GBA-style: span multiple tiles)
SPRITE_SCALE_MULTIPLIER = 3.5

# Movement trail
MOVEMENT_TRAIL_LENGTH = 3
MOVEMENT_TRAIL_ALPHA_MIN = 33
MOVEMENT_TRAIL_ALPHA_MAX = 100

# Energy bar
ENERGY_BAR_HEIGHT = 4
ENERGY_BAR_OFFSET = 7

# AI Badge
AI_BADGE_RADIUS_RATIO = 0.18
AI_BADGE_OFFSET_RATIO = 0.2
AI_BADGE_FONT_SIZE_RATIO = 0.8
