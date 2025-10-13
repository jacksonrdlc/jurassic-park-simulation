"""
Sprite Loader
Loads and caches dinosaur sprite images
"""

import os
import pygame
from src.config.settings import CELL_SIZE, SPRITE_SCALE_MULTIPLIER


class SpriteLoader:
    """Loads and manages sprite images"""

    # Class-level cache for loaded sprites
    _sprite_cache = {}
    _sprites_loaded = False

    @classmethod
    def load_sprites(cls, sprite_dir="dino_sprites"):
        """
        Load all dinosaur sprites from directory

        Args:
            sprite_dir: Directory containing sprite images

        Returns:
            dict: Mapping of species name to pygame surface
        """
        if cls._sprites_loaded:
            return cls._sprite_cache

        sprite_files = {
            'triceratops': 'triceratops.png',
            'gallimimus': 'gallimimus.png',
            'trex': 'trex.png',
            'velociraptor': 'velociraptor.png',
        }

        for species, filename in sprite_files.items():
            filepath = os.path.join(sprite_dir, filename)
            if os.path.exists(filepath):
                try:
                    # Load image
                    sprite = pygame.image.load(filepath).convert_alpha()
                    cls._sprite_cache[species] = sprite
                    print(f"✅ Loaded sprite: {species}")
                except Exception as e:
                    print(f"⚠️  Failed to load {filename}: {e}")
                    cls._sprite_cache[species] = None
            else:
                print(f"⚠️  Sprite not found: {filepath}")
                cls._sprite_cache[species] = None

        cls._sprites_loaded = True

        # Print summary
        loaded_count = sum(1 for v in cls._sprite_cache.values() if v is not None)
        total_count = len(sprite_files)

        if loaded_count == 0:
            print("ℹ️  No sprites loaded - using polygon rendering")
        elif loaded_count < total_count:
            print(f"ℹ️  Loaded {loaded_count}/{total_count} sprites - mixing sprites and polygons")
        else:
            print(f"✅ All sprites loaded ({loaded_count}/{total_count})")

        return cls._sprite_cache

    @classmethod
    def get_sprite(cls, species_name):
        """
        Get sprite for a species

        Args:
            species_name: Name of species (lowercase)

        Returns:
            pygame.Surface or None if not loaded
        """
        if not cls._sprites_loaded:
            cls.load_sprites()

        return cls._sprite_cache.get(species_name.lower())

    @classmethod
    def has_sprite(cls, species_name):
        """
        Check if sprite is loaded for species

        Args:
            species_name: Name of species

        Returns:
            bool: True if sprite is loaded
        """
        sprite = cls.get_sprite(species_name)
        return sprite is not None

    @classmethod
    def get_scaled_sprite(cls, species_name, size_ratio, direction=(1, 0)):
        """
        Get scaled and rotated sprite for rendering

        Args:
            species_name: Name of species
            size_ratio: Size multiplier (0.3-0.45)
            direction: Movement direction tuple (dx, dy)

        Returns:
            pygame.Surface or None
        """
        import math

        sprite = cls.get_sprite(species_name)
        if sprite is None:
            return None

        # Calculate target size (GBA-style scaling)
        target_size = int(CELL_SIZE * size_ratio * SPRITE_SCALE_MULTIPLIER)

        # Scale sprite maintaining aspect ratio
        sprite_rect = sprite.get_rect()
        scale_factor = target_size / max(sprite_rect.width, sprite_rect.height)
        new_width = int(sprite_rect.width * scale_factor)
        new_height = int(sprite_rect.height * scale_factor)

        scaled_sprite = pygame.transform.scale(sprite, (new_width, new_height))

        # Calculate rotation angle (sprites should face right by default)
        if direction == (0, 0):
            angle = 0
        else:
            # Convert to degrees (pygame rotates counter-clockwise)
            angle = -math.degrees(math.atan2(direction[1], direction[0]))

        # Rotate sprite
        if angle != 0:
            rotated_sprite = pygame.transform.rotate(scaled_sprite, angle)
        else:
            rotated_sprite = scaled_sprite

        return rotated_sprite

    @classmethod
    def clear_cache(cls):
        """Clear sprite cache (useful for reloading)"""
        cls._sprite_cache.clear()
        cls._sprites_loaded = False
