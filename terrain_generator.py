"""
Terrain Generator for Costa Rican Island
Creates diverse landscapes using Perlin noise: beaches, rainforests, mountains, rivers, volcanoes
"""

import numpy as np
import random
from enum import Enum


class TerrainType(Enum):
    """Types of terrain on the island"""
    OCEAN = 0
    BEACH = 1
    SAND = 2
    GRASSLAND = 3
    FOREST = 4
    RAINFOREST = 5
    RIVER = 6
    MOUNTAIN = 7
    VOLCANO = 8


# Costa Rican color palette
TERRAIN_COLORS = {
    TerrainType.OCEAN: (0, 119, 190),      # Deep blue
    TerrainType.BEACH: (244, 228, 193),    # Sandy tan
    TerrainType.SAND: (237, 201, 175),     # Light sand
    TerrainType.GRASSLAND: (102, 187, 106), # Vibrant green
    TerrainType.FOREST: (76, 175, 80),     # Medium green
    TerrainType.RAINFOREST: (46, 125, 50), # Dark rainforest green
    TerrainType.RIVER: (79, 195, 247),     # Light blue
    TerrainType.MOUNTAIN: (141, 110, 99),  # Gray-brown
    TerrainType.VOLCANO: (62, 39, 35),     # Dark volcanic
}

# Terrain properties
TERRAIN_WALKABLE = {
    TerrainType.OCEAN: False,
    TerrainType.BEACH: True,
    TerrainType.SAND: True,
    TerrainType.GRASSLAND: True,
    TerrainType.FOREST: True,
    TerrainType.RAINFOREST: True,
    TerrainType.RIVER: False,  # Can't walk through rivers
    TerrainType.MOUNTAIN: True,  # Can walk but slow
    TerrainType.VOLCANO: True,   # Can walk but slow
}

TERRAIN_SPEED_MULTIPLIER = {
    TerrainType.OCEAN: 0.0,
    TerrainType.BEACH: 0.9,
    TerrainType.SAND: 0.85,
    TerrainType.GRASSLAND: 1.0,  # Normal speed
    TerrainType.FOREST: 0.8,
    TerrainType.RAINFOREST: 0.7,
    TerrainType.RIVER: 0.0,
    TerrainType.MOUNTAIN: 0.5,  # Very slow
    TerrainType.VOLCANO: 0.4,   # Even slower
}


class SimplexNoise:
    """Simple Perlin-like noise generator for terrain"""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Generate permutation table
        self.perm = list(range(256))
        random.shuffle(self.perm)
        self.perm = self.perm * 2

    def noise2d(self, x, y):
        """Generate 2D noise value between -1 and 1"""
        # Simplified noise - blend of sine waves with different frequencies
        value = 0
        amplitude = 1.0
        frequency = 1.0

        for _ in range(4):  # 4 octaves
            value += amplitude * (
                np.sin(x * frequency * 0.1) * np.cos(y * frequency * 0.1) +
                np.sin((x + y) * frequency * 0.05)
            )
            amplitude *= 0.5
            frequency *= 2.0

        return np.clip(value, -1, 1)


class IslandGenerator:
    """Generate Costa Rican-style island terrain"""

    def __init__(self, width, height, seed=None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 999999)
        self.noise = SimplexNoise(self.seed)

        # Terrain maps
        self.heightmap = None
        self.terrain_map = None

    def generate(self):
        """Generate complete island terrain"""
        print(f"🏝️  Generating {self.width}x{self.height} Costa Rican island (seed: {self.seed})...")

        # Step 1: Generate heightmap
        self.heightmap = self._generate_heightmap()

        # Step 2: Apply island shape mask
        self.heightmap = self._apply_island_mask(self.heightmap)

        # Step 3: Add volcanic peak
        self.heightmap = self._add_volcano(self.heightmap)

        # Step 4: Convert heightmap to terrain types
        self.terrain_map = self._heightmap_to_terrain(self.heightmap)

        # Step 5: Add rivers
        self.terrain_map = self._add_rivers(self.terrain_map, self.heightmap)

        # Step 6: Smooth terrain transitions
        self.terrain_map = self._smooth_terrain(self.terrain_map)

        print("✅ Island generation complete!")
        self._print_stats()

        return self.terrain_map

    def _generate_heightmap(self):
        """Generate base heightmap using noise"""
        heightmap = np.zeros((self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                # Multiple octaves of noise for natural terrain
                value = 0.0
                value += 0.5 * self.noise.noise2d(x * 0.5, y * 0.5)
                value += 0.25 * self.noise.noise2d(x * 1.0, y * 1.0)
                value += 0.125 * self.noise.noise2d(x * 2.0, y * 2.0)

                # Normalize to 0-1
                heightmap[y, x] = (value + 1.0) / 2.0

        return heightmap

    def _apply_island_mask(self, heightmap):
        """Shape terrain into island form (higher in center, ocean at edges)"""
        center_x = self.width / 2
        center_y = self.height / 2

        max_dist = np.sqrt(center_x**2 + center_y**2)

        masked = np.copy(heightmap)

        for y in range(self.height):
            for x in range(self.width):
                # Distance from center
                dx = (x - center_x) / center_x
                dy = (y - center_y) / center_y
                dist = np.sqrt(dx**2 + dy**2)

                # Falloff curve - creates island shape
                # Values close to 1.0 at edges become 0 (ocean)
                # Values at center stay high (land)
                # SOFTENED: More gradual falloff for cohesive island
                falloff = 1.0 - np.clip(dist * 0.6, 0, 1)  # Was 0.8, now 0.6 (wider island)
                falloff = falloff ** 1.2  # Was 1.5, now 1.2 (gentler falloff)

                masked[y, x] = heightmap[y, x] * falloff

        return masked

    def _add_volcano(self, heightmap):
        """Add central volcanic peak"""
        center_x = self.width // 2 + random.randint(-20, 20)
        center_y = self.height // 2 + random.randint(-20, 20)

        volcano_radius = 15

        modified = np.copy(heightmap)

        for y in range(self.height):
            for x in range(self.width):
                dx = x - center_x
                dy = y - center_y
                dist = np.sqrt(dx**2 + dy**2)

                if dist < volcano_radius:
                    # Cone shape
                    peak_height = 1.0 - (dist / volcano_radius)
                    peak_height = peak_height ** 0.8  # Gentle slopes
                    modified[y, x] = max(modified[y, x], peak_height * 0.95 + 0.05)

        return modified

    def _heightmap_to_terrain(self, heightmap):
        """Convert heightmap values to terrain types"""
        terrain_map = np.zeros((self.height, self.width), dtype=int)

        for y in range(self.height):
            for x in range(self.width):
                h = heightmap[y, x]

                if h < 0.25:
                    terrain_map[y, x] = TerrainType.OCEAN.value
                elif h < 0.28:
                    terrain_map[y, x] = TerrainType.BEACH.value
                elif h < 0.32:
                    terrain_map[y, x] = TerrainType.SAND.value
                elif h < 0.45:
                    terrain_map[y, x] = TerrainType.GRASSLAND.value
                elif h < 0.60:
                    terrain_map[y, x] = TerrainType.FOREST.value
                elif h < 0.75:
                    terrain_map[y, x] = TerrainType.RAINFOREST.value
                elif h < 0.88:
                    terrain_map[y, x] = TerrainType.MOUNTAIN.value
                else:
                    terrain_map[y, x] = TerrainType.VOLCANO.value

        return terrain_map

    def _add_rivers(self, terrain_map, heightmap):
        """Add rivers flowing from mountains to ocean"""
        modified = np.copy(terrain_map)

        # Find high points (sources)
        sources = []
        for y in range(10, self.height - 10):
            for x in range(10, self.width - 10):
                if heightmap[y, x] > 0.7 and random.random() < 0.02:
                    sources.append((x, y))

        # Flow rivers downhill
        for start_x, start_y in sources:
            x, y = start_x, start_y
            path_length = 0
            max_length = 100

            while path_length < max_length:
                # Current height
                current_h = heightmap[y, x]

                # Find lowest neighbor
                lowest_h = current_h
                lowest_pos = None

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        neighbor_h = heightmap[ny, nx]
                        if neighbor_h < lowest_h:
                            lowest_h = neighbor_h
                            lowest_pos = (nx, ny)

                # No downhill neighbor or reached ocean
                if lowest_pos is None or lowest_h < 0.26:
                    break

                # Move to lowest neighbor and place river
                x, y = lowest_pos
                current_terrain = terrain_map[y, x]

                # Only place river on valid terrain (not ocean/beach)
                if current_terrain >= TerrainType.GRASSLAND.value:
                    modified[y, x] = TerrainType.RIVER.value

                path_length += 1

        return modified

    def _smooth_terrain(self, terrain_map):
        """Smooth terrain transitions (remove single-cell anomalies)"""
        modified = np.copy(terrain_map)

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                current = terrain_map[y, x]

                # Get neighbors
                neighbors = [
                    terrain_map[y-1, x], terrain_map[y+1, x],
                    terrain_map[y, x-1], terrain_map[y, x+1]
                ]

                # If surrounded by same terrain, adopt it (smooths isolated cells)
                if len(set(neighbors)) == 1 and neighbors[0] != current:
                    # Don't smooth rivers or special features
                    if current not in [TerrainType.RIVER.value, TerrainType.VOLCANO.value]:
                        modified[y, x] = neighbors[0]

        return modified

    def _print_stats(self):
        """Print terrain distribution statistics"""
        total_cells = self.width * self.height

        print("\n📊 Terrain Distribution:")
        for terrain_type in TerrainType:
            count = np.sum(self.terrain_map == terrain_type.value)
            percentage = (count / total_cells) * 100
            print(f"  {terrain_type.name:12s}: {count:6d} cells ({percentage:5.1f}%)")

    def get_spawn_locations(self, num_locations, terrain_types):
        """Get valid spawn locations for agents"""
        valid_locations = []

        for y in range(self.height):
            for x in range(self.width):
                terrain = TerrainType(self.terrain_map[y, x])
                if terrain in terrain_types:
                    valid_locations.append((x, y))

        if len(valid_locations) < num_locations:
            return valid_locations

        # Random sample
        indices = random.sample(range(len(valid_locations)), num_locations)
        return [valid_locations[i] for i in indices]


def generate_island(width=250, height=150, seed=None):
    """Quick function to generate an island"""
    generator = IslandGenerator(width, height, seed)
    return generator.generate(), generator


if __name__ == "__main__":
    # Test generation
    import time
    start = time.time()
    terrain_map, generator = generate_island(250, 150, seed=42)
    elapsed = time.time() - start
    print(f"\n⏱️  Generation took {elapsed:.2f} seconds")

    # Get some spawn locations
    herbivore_spawns = generator.get_spawn_locations(20, [TerrainType.GRASSLAND, TerrainType.FOREST])
    print(f"\n🦕 Found {len(herbivore_spawns)} herbivore spawn locations")
