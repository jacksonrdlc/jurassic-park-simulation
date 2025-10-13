# 🦕 Dinosaur Sprites

This folder contains retro pixel art sprite images for the dinosaurs in the simulation.

## Current Sprites

The simulation uses the following 16x16 pixel art sprites:

### Herbivores
- `Triceratops_16x16.png` - Triceratops sprite (scaled 2.5x in-game)
- `Parasaurolophus_16x16.png` - Used for Gallimimus (scaled 2.0x in-game)

### Carnivores
- `TyrannosaurusRex_16x16.png` - T-Rex sprite (scaled 3.0x in-game)
- `Spinosaurus_16x16.png` - Used for Velociraptor (scaled 2.5x in-game)

### Additional Sprites Available
This folder also contains additional dinosaur sprites that can be used:
- `Amargasaurus_32x32.png`
- `Archeopteryx_16x16.png`
- `Brachiosaurus_32x32.png`
- `Brontosaurus_32x32.png`
- `Diplodocus_32x32.png`
- `Kentrosaurus_32x32.png`
- `Pachycephalosaurus_16x16.png`
- `Saltasaurus_32x32.png`
- `Stegosaurus_32x32.png`
- `Stygimoloch_16x16.png`
- `Styracosaurus_16x16.png`
- `Wuerhosaurus_32x32.png`

## Image Specifications

- **Format**: PNG with alpha transparency
- **Size**: 16x16 or 32x32 pixels (originals)
- **Scaling**: Dynamically scaled 2-3x during gameplay
- **Rotation**: Sprites automatically rotate to face movement direction
- **Style**: Retro pixel art with crisp rendering on modern displays

## How Sprites Are Used

1. **Loading**: Sprites are loaded at startup in `pygame_viz.py`
2. **Scaling**: Each species has a specific scale factor (T-Rex: 3x, Triceratops: 2.5x, etc.)
3. **Rotation**: Sprites dynamically rotate using `pygame.transform.rotate()` based on movement direction
4. **Rendering**: The `draw_sprite()` method handles all transformations

## Fallback Behavior

If sprite images are not found, the game automatically falls back to procedurally-generated dinosaur shapes (polygon rendering) with species-specific colors.

## Code Integration

Sprites are loaded in `pygame_viz.py` at initialization:

```python
self.sprites = {
    'trex': pygame.image.load('dino_sprites/TyrannosaurusRex_16x16.png').convert_alpha(),
    'triceratops': pygame.image.load('dino_sprites/Triceratops_16x16.png').convert_alpha(),
    'velociraptor': pygame.image.load('dino_sprites/Spinosaurus_16x16.png').convert_alpha(),
    'gallimimus': pygame.image.load('dino_sprites/Parasaurolophus_16x16.png').convert_alpha(),
}
```

To use different sprites, simply update the file paths in the sprite loading code!
