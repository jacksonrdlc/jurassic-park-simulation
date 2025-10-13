"""
Game Settings and Constants
Window dimensions, gameplay parameters, etc.
"""

# World dimensions - Costa Rican Island
WORLD_WIDTH = 625
WORLD_HEIGHT = 375
CELL_SIZE = 16  # 16px cells for zoomed-in terrain detail

# Viewport dimensions - Widescreen modern display
VIEWPORT_WIDTH = 1600   # ~100 cells visible width at 16px
VIEWPORT_HEIGHT = 900   # ~56 cells visible height at 16px

# UI dimensions
PANEL_WIDTH = 450
WINDOW_WIDTH = VIEWPORT_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = VIEWPORT_HEIGHT

# Simulation parameters
DEFAULT_TEMPERATURE = 25
DEFAULT_RAINFALL = 100
DEFAULT_NUM_HERBIVORES = 125  # Balanced for 625x375 world
DEFAULT_NUM_CARNIVORES = 50

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

# Dinosaur sprite scaling (proportional to 16px cells)
SPRITE_SCALES = {
    'trex': 1.8,
    'triceratops': 1.5,
    'velociraptor': 1.5,
    'gallimimus': 1.2,
    'stegosaurus': 1.2,
    'pachycephalosaurus': 1.3,
    'brachiosaurus': 1.5,
    'archeopteryx': 1.0,
}

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
