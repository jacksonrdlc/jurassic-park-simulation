"""
Jurassic Park Color Scheme
All colors used in the visualization
"""

# Jurassic Park Color Scheme
COLORS = {
    # Environment
    'background': (15, 25, 15),        # Deep jungle green
    'grass_full': (45, 85, 35),        # Rich grass
    'grass_eaten': (75, 60, 40),       # Dry earth
    'grid_lines': (80, 100, 50),       # Subtle grid overlay

    # Jurassic Park Branding
    'jp_yellow': (255, 210, 0),        # Iconic JP yellow
    'jp_red': (220, 20, 20),           # Danger red
    'jp_black': (20, 20, 20),          # Deep black
    'warning_stripe': (255, 200, 0),   # Warning yellow

    # Dinosaurs - Species-specific colors
    'triceratops': (85, 160, 70),      # Forest green herbivore
    'gallimimus': (200, 180, 90),      # Tan/beige fast herbivore
    'trex': (140, 40, 30),             # Blood red apex predator
    'velociraptor': (100, 60, 120),    # Deep purple pack hunter
    'herbivore': (80, 140, 80),        # Generic herbivore
    'carnivore': (180, 40, 40),        # Generic carnivore

    # UI Elements
    'text': (240, 240, 240),           # Off-white text
    'text_highlight': (255, 210, 0),   # JP yellow for highlights
    'panel': (25, 30, 25),             # Dark panel background
    'panel_border': (255, 210, 0),     # Yellow border
    'containment_active': (50, 200, 80),   # Green = safe
    'containment_warning': (255, 180, 0),  # Yellow = caution
    'containment_breach': (255, 50, 50),   # Red = danger

    # Events
    'event_birth': (80, 220, 100),
    'event_death': (255, 80, 80),
    'event_hunt': (255, 140, 0),
    'event_eat': (120, 200, 120),
    'event_info': (150, 180, 255),

    # Energy bars
    'energy_high': (50, 200, 80),      # Green
    'energy_medium': (255, 180, 0),    # Yellow
    'energy_low': (255, 80, 60),       # Red
    'energy_bg': (40, 40, 40),         # Dark background
}

# AI Badge colors for different species
AI_BADGE_COLORS = {
    'TRex': ('T', (255, 100, 100)),        # Red T for T-Rex
    'Velociraptor': ('V', (200, 150, 255)),  # Purple V for Velociraptor
    'Triceratops': ('3', (150, 255, 150)),   # Green 3 for Triceratops
    'Gallimimus': ('G', (255, 255, 150)),    # Yellow G for Gallimimus
    'CarnivoreAgent': ('C', (255, 100, 100)),
    'HerbivoreAgent': ('H', (150, 255, 150))
}
