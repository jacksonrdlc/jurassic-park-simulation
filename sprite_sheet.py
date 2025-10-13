"""
Sprite Sheet Handler for Jurassic Park Simulation
Handles clipping individual frames from sprite sheets (SNES-style)
"""
import pygame


class SpriteSheet:
    """Load and clip sprites from a sprite sheet"""

    def __init__(self, filename):
        """Load the sprite sheet"""
        try:
            self.sheet = pygame.image.load(filename).convert_alpha()
        except pygame.error as e:
            print(f'Unable to load spritesheet: {filename}')
            raise SystemExit(e)

    def image_at(self, rectangle, colorkey=None):
        """Extract a single image from the sheet at given rectangle"""
        rect = pygame.Rect(rectangle)
        image = pygame.Surface(rect.size, pygame.SRCALPHA).convert_alpha()
        image.blit(self.sheet, (0, 0), rect)

        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)

        return image

    def images_at(self, rects, colorkey=None):
        """Load multiple images from a list of rectangles"""
        return [self.image_at(rect, colorkey) for rect in rects]

    def load_grid(self, frame_width, frame_height, columns, rows, colorkey=None):
        """
        Load a grid of sprites from the sheet
        Returns a 2D list [row][column] of images

        Args:
            frame_width: Width of each frame in pixels
            frame_height: Height of each frame in pixels
            columns: Number of columns in the grid
            rows: Number of rows in the grid
            colorkey: Transparent color key (optional)
        """
        sprites = []
        for row in range(rows):
            row_sprites = []
            for col in range(columns):
                x = col * frame_width
                y = row * frame_height
                rect = (x, y, frame_width, frame_height)
                row_sprites.append(self.image_at(rect, colorkey))
            sprites.append(row_sprites)
        return sprites


class DinosaurSprite:
    """
    Handles directional animated sprites for dinosaurs
    Sprite sheets are organized as:
    - Row 0: Down/South facing (4 frames)
    - Row 1: Left/West facing (4 frames)
    - Row 2: Right/East facing (4 frames)
    - Row 3: Up/North facing (4 frames)
    """

    # Direction mappings
    SOUTH = 0
    WEST = 1
    EAST = 2
    NORTH = 3

    def __init__(self, sprite_sheet_path, frame_size=16, scale=1.0):
        """
        Initialize dinosaur sprite

        Args:
            sprite_sheet_path: Path to sprite sheet image
            frame_size: Size of each frame (assumes square frames)
            scale: Scale factor for rendering
        """
        self.sprite_sheet = SpriteSheet(sprite_sheet_path)
        self.frame_size = frame_size
        self.scale = scale

        # Load all frames as 2D grid [row][column]
        self.frames = self.sprite_sheet.load_grid(
            frame_size, frame_size,
            columns=4, rows=4
        )

        # Animation state
        self.current_frame = 0
        self.animation_speed = 0.15  # Frames per game step
        self.animation_counter = 0

    def get_direction_from_vector(self, dx, dy):
        """Convert movement vector to sprite direction"""
        if dx == 0 and dy == 0:
            return self.SOUTH  # Default facing down when stationary

        # Determine primary direction based on larger component
        if abs(dx) > abs(dy):
            return self.EAST if dx > 0 else self.WEST
        else:
            return self.SOUTH if dy > 0 else self.NORTH

    def get_sprite(self, direction, animate=True):
        """
        Get the current sprite for the given direction

        Args:
            direction: Direction constant (SOUTH, WEST, EAST, NORTH)
            animate: Whether to advance animation frame

        Returns:
            pygame.Surface: The sprite image
        """
        if animate:
            self.animation_counter += self.animation_speed
            if self.animation_counter >= 1.0:
                self.animation_counter = 0
                self.current_frame = (self.current_frame + 1) % 4

        # Get frame from grid
        sprite = self.frames[direction][self.current_frame]

        # Scale if needed
        if self.scale != 1.0:
            new_size = int(self.frame_size * self.scale)
            sprite = pygame.transform.scale(sprite, (new_size, new_size))

        return sprite

    def get_sprite_from_movement(self, dx, dy, animate=True):
        """Get sprite based on movement direction vector"""
        direction = self.get_direction_from_vector(dx, dy)
        return self.get_sprite(direction, animate)
