# 🏗️ Project Architecture

This document describes the modular architecture of the Jurassic Park Ecosystem Simulation, following pygame best practices.

## 📁 Directory Structure

```
jurassic-park-simulation/
├── src/                          # Main source code (NEW!)
│   ├── config/                   # Configuration and constants
│   │   ├── __init__.py
│   │   ├── colors.py            # All color definitions
│   │   └── settings.py          # Game settings & constants
│   │
│   ├── entities/                 # Game entities
│   │   ├── __init__.py
│   │   ├── dinosaur_sprite.py   # Dinosaur rendering functions
│   │   └── particle.py          # Particle system (volcano smoke)
│   │
│   ├── rendering/                # Rendering modules
│   │   ├── __init__.py
│   │   ├── terrain_renderer.py  # Terrain & agent rendering
│   │   ├── ui_renderer.py       # UI panels & minimap
│   │   └── effects_renderer.py  # Weather, day/night, particles
│   │
│   ├── managers/                 # Game management
│   │   ├── __init__.py
│   │   ├── game_manager.py      # Main game loop & state
│   │   └── event_manager.py     # Input handling
│   │
│   └── utils/                    # Utilities (future use)
│       └── __init__.py
│
├── run_game.py                   # NEW: Clean entry point
├── pygame_viz.py                 # Legacy: Original monolithic version
├── camera.py                     # Camera/viewport system
├── terrain_generator.py          # Procedural terrain generation
├── my_first_island.py            # Simulation model & agents
├── learning_agents.py            # Q-learning AI agents
├── ppo_agents.py                 # PPO RL agents
├── train_agents.py               # Training scripts
├── train_ppo.py
├── compare_agents.py
└── models/                       # Saved AI models
    ├── herbivore_final.pth
    ├── carnivore_final.pth
    ├── ppo_herbivore_best.zip
    └── ppo_carnivore_best.zip
```

## 🎯 Design Principles

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility:
- **Config**: Static configuration data
- **Entities**: Data structures and visual representations
- **Rendering**: All drawing logic
- **Managers**: Game state and user input

### 2. **Modularity**
- Components can be tested independently
- Easy to swap implementations (e.g., different renderers)
- Clear interfaces between modules

### 3. **Maintainability**
- Small, focused files (<300 lines each)
- Clear naming conventions
- Documented public interfaces

### 4. **Performance**
- Viewport culling for efficient rendering
- Particle pooling to avoid allocation overhead
- Optimized rendering pipeline

## 📦 Module Descriptions

### Config Module (`src/config/`)

**colors.py**
- Defines all color constants (COLORS dict)
- AI badge color mappings
- Centralizes visual theming

**settings.py**
- Game constants (world size, viewport, FPS)
- Visual effect parameters
- Font sizes and UI dimensions

### Entities Module (`src/entities/`)

**dinosaur_sprite.py**
- `calculate_sprite_points()`: Generate dinosaur geometry
- `rotate_points()`: Transform coordinates
- `draw_dinosaur()`: Render dinosaur sprite
- `draw_movement_trail()`: Render position history

**particle.py**
- `VolcanoParticle`: Smoke/steam particle class
- Handles particle lifecycle (spawn, update, death)
- Alpha fade animation

### Rendering Module (`src/rendering/`)

**terrain_renderer.py**
- Draws terrain tiles with procedural variation
- Animated water with sine wave effects
- Renders all agents (dinosaurs, grass)
- Energy bars and AI indicators
- Grid overlay and coordinates

**ui_renderer.py**
- Info panel (population, stats, events)
- Minimap with viewport indicator
- Controls legend
- Jurassic Park branding

**effects_renderer.py**
- Day/night cycle lighting
- Weather effects (rain, heat shimmer)
- Volcano smoke particles
- Environmental overlays

### Managers Module (`src/managers/`)

**game_manager.py**
- **Main game loop**: Update → Render → Repeat
- **State management**: Pause, reset, speed control
- **World initialization**: Terrain generation, model setup
- **AI model loading**: Q-learning and PPO agents
- **Coordinates all systems**: Renderers, event manager, camera

**event_manager.py**
- Keyboard input (SPACE, R, M, ESC, arrows)
- Mouse input (drag, click to center)
- Event→Action mapping
- State flags for game manager

## 🔄 Data Flow

```
User Input → EventManager → GameManager
                                ↓
                          Update State
                          (Model.step())
                                ↓
                          Render Pipeline:
                          1. TerrainRenderer
                          2. EffectsRenderer
                          3. UIRenderer
                                ↓
                          Display.flip()
```

## 🚀 Running the Game

### New Modular Version (Recommended)
```bash
# Traditional agents
python run_game.py

# Q-learning agents
python run_game.py --ai

# PPO agents
python run_game.py --ppo
```

### Legacy Version (Backwards Compatible)
```bash
python pygame_viz.py [--ai | --ppo]
```

## 🧪 Testing

The modular design makes testing easier:

```python
# Test individual renderers
from src.rendering.terrain_renderer import TerrainRenderer
renderer = TerrainRenderer(screen, camera)
renderer.draw_terrain(terrain_map, noise)

# Test game manager
from src.managers.game_manager import GameManager
game = GameManager(use_learning_agents=False)
game.update()  # Single update step
```

## 🛠️ Extending the Game

### Adding a New Visual Effect
1. Add effect code to `effects_renderer.py`
2. Call from `draw_all_effects()`
3. Add config to `settings.py` if needed

### Adding a New UI Panel
1. Add method to `ui_renderer.py`
2. Call from `draw_panel()` or create new method
3. Add colors to `colors.py`

### Adding a New Agent Type
1. Define agent in `my_first_island.py`
2. Add rendering logic in `terrain_renderer.py`
3. Add color to `colors.py`

## 📊 Benefits of Refactoring

### Before (pygame_viz.py)
- ❌ 1065 lines in single file
- ❌ All logic mixed together
- ❌ Hard to test individual components
- ❌ Difficult to navigate code
- ❌ Constants scattered throughout

### After (Modular Architecture)
- ✅ Max 300 lines per file
- ✅ Clear separation of concerns
- ✅ Easy to test and maintain
- ✅ Logical organization
- ✅ Centralized configuration

## 🎓 Learning Pygame Best Practices

This refactoring demonstrates:
1. **Game Loop Pattern**: Clean separation of update/render
2. **State Management**: Centralized game state
3. **Event Handling**: Dedicated input manager
4. **Rendering Pipeline**: Layered rendering (terrain → effects → UI)
5. **Resource Management**: Organized asset loading
6. **Configuration**: Constants in dedicated files
7. **Modularity**: Single Responsibility Principle

## 🔗 Related Documentation

- [README.md](README.md) - Main project documentation
- [PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md) - AI/ML implementation
- [AI_SETUP.md](AI_SETUP.md) - Training guide
