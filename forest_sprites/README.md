# 🌲 Forest Sprites

This folder contains tree and vegetation sprites for the Jurassic Park simulation.

## Sprite Specifications

### Tree Sprites
- **Format**: PNG with transparency (alpha channel)
- **Recommended sizes**: 32x32, 48x48, or 64x64 pixels
- **Color palette**: Earthy greens and browns for realistic foliage

### Naming Convention
Use descriptive names for different tree types:
- `tree_oak.png` - Regular forest trees
- `tree_rainforest.png` - Dense rainforest trees
- `tree_palm.png` - Tropical palm trees
- `tree_dead.png` - Dead/barren trees
- `bush.png` - Small bushes/shrubs
- `fern.png` - Ground vegetation

## Usage

The terrain renderer will automatically load sprites from this folder and use them instead of procedural tree rendering when available.

### Current Tree Rendering
- **Forests**: Medium green trees with brown trunks
- **Rainforests**: Dark green trees with dark brown trunks
- **Size**: Trees are 3x larger than terrain cells for visibility
- **Spacing**: Trees appear every 4 cells to avoid overlap

## Tips for Creating Sprites
1. Use transparent backgrounds (PNG alpha)
2. Include shadows at the base for depth
3. Match the color scheme:
   - Regular forest: Medium brown trunk (#654321), medium green leaves (#2E7D32)
   - Rainforest: Dark brown trunk (#462D1E), dark green leaves (#1E501E)
4. Consider pixel art style to match dinosaur sprites
5. Trees should be taller than wide for realistic proportions

## Example Structure
```
forest_sprites/
├── README.md (this file)
├── tree_forest_01.png
├── tree_forest_02.png
├── tree_rainforest_01.png
├── tree_rainforest_02.png
├── palm_01.png
└── bush_01.png
```
