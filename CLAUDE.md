# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jurassic Park themed agent-based simulation built with Mesa (Python agent-based modeling framework). The simulation models dinosaur behavior, park dynamics, and ecosystem interactions.

## Environment Setup

**Virtual Environment**: `jurassic_env`
- Python 3.13
- Activate: `source jurassic_env/bin/activate`

**Key Dependencies**:
- Mesa 3.3.0 - Agent-based modeling framework
- Pygame 2.6.1 - Real-time visualization (60 FPS, zero lag!)
- NumPy, Pandas - Data manipulation
- Matplotlib - Static visualization for experiments

**Development Commands**:
```bash
# Activate virtual environment
source jurassic_env/bin/activate

# Run real-time visualization (RECOMMENDED)
python pygame_viz.py

# Run experiments with charts
python my_first_island.py
```

## Architecture (To Be Implemented)

**Expected Structure**:
- **Model** (`model.py`): Main simulation model defining grid, schedule, and data collection
- **Agents** (`agents.py` or `agents/`): Dinosaur agents (herbivores, carnivores), park staff, visitors
- **Server/App** (`server.py` or `app.py`): Visualization setup using Solara or Mesa's built-in visualization
- **Config**: Parameters for simulation (population sizes, energy levels, reproduction rates, etc.)

**Mesa 3.x Framework (Important API Changes)**:
- **Model**: Automatically manages agent registration
  - ❌ No `mesa.time.RandomActivation` - use `self.agents.shuffle_do("step")` instead
  - ✅ Agents auto-register when created, access via `self.agents`
  - Initialize with `super().__init__()` (optionally pass `seed=` for reproducibility)

- **Space Options**:
  - **Legacy (still supported)**: `from mesa.space import MultiGrid` - coordinate-based
  - **New (recommended)**: `from mesa.discrete_space import OrthogonalMooreGrid` - cell-based
  - Choose based on your needs; both work in Mesa 3.x

- **Agents**:
  - **Coordinate-based** (with MultiGrid): Inherit from `Agent`
    - Agents have `self.pos` tuple coordinates
    - Move with `self.model.grid.move_agent(self, new_pos)`
    - Access cell contents: `self.model.grid.get_cell_list_contents([self.pos])`
  - **Cell-based** (with OrthogonalMooreGrid): Inherit from `CellAgent`
    - Agents have `self.cell` reference
    - Move with `self.cell = self.cell.neighborhood.select_random_cell()`
    - Access cell contents: `self.cell.agents`
  - Use `self.remove()` to remove agent from simulation (replaces `scheduler.remove()`)

- **Visualization** (Pygame):
  - **Native window** - No browser, no web server needed
  - **60 FPS** - Smooth, real-time rendering with zero lag
  - Event loop: `pygame.event.get()` for keyboard input
  - Drawing: `pygame.draw.circle()`, `pygame.draw.rect()`
  - Run with: `python pygame_viz.py`

**Agent Design Patterns**:
- Implement `step()` method for agent behavior each timestep
- Access model random generator: `self.random` or `self.model.random`
- Implement energy/health systems for survival mechanics
- Add breeding mechanics with cooldown periods
- Track agent states (hunting, grazing, fleeing, resting)

## Testing

When tests are added:
```bash
pytest tests/
pytest tests/test_agents.py -v  # Single test file
pytest -k "test_name"           # Specific test
```

## Visualization with Pygame

Pygame provides smooth, real-time visualization:
- Game loop running at 60 FPS: `clock.tick(60)`
- Event handling: `pygame.event.get()` for keyboard input
- Draw agents: `pygame.draw.circle()` for dinos, `pygame.draw.rect()` for grass
- Render text: `font.render()` for stats and event log
- No flickering, no lag, instant updates

**Why Pygame:**
- ✅ 60 FPS smooth rendering
- ✅ Zero flickering or dimming
- ✅ Native performance
- ✅ Game-like feel
- ✅ Perfect for real-time agent simulations
