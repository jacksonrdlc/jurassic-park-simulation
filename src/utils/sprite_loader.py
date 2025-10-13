"""
Sprite Loader
Loads and caches dinosaur sprite sheets with animation support
"""

import pygame
from sprite_sheet import DinosaurSprite
from src.config.settings import SPRITE_SCALES


class SpriteLoader:
    """Loads and manages sprite sheet images with animation"""

    # Class-level cache for loaded sprites
    _sprite_cache = {}
    _sprites_loaded = False

    # Color tints for each species (earthy colors)
    _dino_tints = {
        'trex': (140, 40, 30),             # Blood red/brown apex predator
        'triceratops': (120, 90, 60),      # Earthy tan/brown
        'velociraptor': (100, 70, 50),     # Dark brown pack hunter
        'gallimimus': (180, 150, 100),     # Tan/beige fast herbivore
        'stegosaurus': (130, 100, 70),     # Brown plated herbivore
        'pachycephalosaurus': (150, 110, 80),  # Light brown dome-headed
        'brachiosaurus': (160, 140, 100),  # Warm tan massive sauropod
        'archeopteryx': (110, 100, 90),    # Gray-brown proto-bird
    }

    @classmethod
    def load_sprites(cls):
        """
        Load all dinosaur sprite sheets from dino_sprites directory

        Returns:
            dict: Mapping of species name to DinosaurSprite object
        """
        if cls._sprites_loaded:
            return cls._sprite_cache

        try:
            # 16x16 sprites
            cls._sprite_cache['trex'] = DinosaurSprite(
                'dino_sprites/TyrannosaurusRex_16x16.png',
                scale=SPRITE_SCALES['trex']
            )
            cls._sprite_cache['triceratops'] = DinosaurSprite(
                'dino_sprites/Triceratops_16x16.png',
                scale=SPRITE_SCALES['triceratops']
            )
            cls._sprite_cache['velociraptor'] = DinosaurSprite(
                'dino_sprites/Spinosaurus_16x16.png',
                scale=SPRITE_SCALES['velociraptor']
            )
            cls._sprite_cache['gallimimus'] = DinosaurSprite(
                'dino_sprites/Parasaurolophus_16x16.png',
                scale=SPRITE_SCALES['gallimimus']
            )
            cls._sprite_cache['pachycephalosaurus'] = DinosaurSprite(
                'dino_sprites/Pachycephalosaurus_16x16.png',
                scale=SPRITE_SCALES['pachycephalosaurus']
            )
            cls._sprite_cache['archeopteryx'] = DinosaurSprite(
                'dino_sprites/Archeopteryx_16x16.png',
                scale=SPRITE_SCALES['archeopteryx']
            )

            # 32x32 sprites
            cls._sprite_cache['stegosaurus'] = DinosaurSprite(
                'dino_sprites/Stegosaurus_32x32.png',
                frame_size=32,
                scale=SPRITE_SCALES['stegosaurus']
            )
            cls._sprite_cache['brachiosaurus'] = DinosaurSprite(
                'dino_sprites/Brachiosaurus_32x32.png',
                frame_size=32,
                scale=SPRITE_SCALES['brachiosaurus']
            )

            cls._sprites_loaded = True
            print("✅ Loaded dinosaur sprite sheets with directional frames!")

        except Exception as e:
            print(f"⚠️  Warning: Could not load sprite sheets: {e}")
            cls._sprites_loaded = False

        return cls._sprite_cache

    @classmethod
    def get_sprite(cls, species_name, direction=(1, 0), animate=True):
        """
        Get animated sprite for a species

        Args:
            species_name: Name of species (lowercase)
            direction: Movement direction (dx, dy)
            animate: Whether to advance animation frame

        Returns:
            pygame.Surface or None if not loaded
        """
        if not cls._sprites_loaded:
            cls.load_sprites()

        sprite_handler = cls._sprite_cache.get(species_name.lower())
        if sprite_handler is None:
            return None

        # Get sprite based on direction
        sprite = sprite_handler.get_sprite_from_movement(direction[0], direction[1], animate)

        # Apply earthy color tint
        if species_name.lower() in cls._dino_tints:
            tint_color = cls._dino_tints[species_name.lower()]
            tinted_sprite = sprite.copy()
            color_overlay = pygame.Surface(tinted_sprite.get_size(), pygame.SRCALPHA)
            color_overlay.fill((*tint_color, 180))  # Semi-transparent tint
            tinted_sprite.blit(color_overlay, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            sprite = tinted_sprite

        return sprite

    @classmethod
    def has_sprite(cls, species_name):
        """
        Check if sprite is loaded for species

        Args:
            species_name: Name of species

        Returns:
            bool: True if sprite is loaded
        """
        if not cls._sprites_loaded:
            cls.load_sprites()

        return species_name.lower() in cls._sprite_cache

    @classmethod
    def clear_cache(cls):
        """Clear sprite cache (useful for reloading)"""
        cls._sprite_cache.clear()
        cls._sprites_loaded = False
