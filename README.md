# 🦕 Jurassic Park Ecosystem Simulation

A real-time agent-based ecosystem simulation featuring diverse dinosaur species with unique behaviors, environmental dynamics, and predator-prey interactions.

## 🚀 Quick Start

**Activate environment and run:**
```bash
source jurassic_env/bin/activate
python pygame_viz.py
```

## 🦖 Dinosaur Species

### Herbivores
- **Triceratops** - Heavily armored (70% defense), slow but tough
- **Gallimimus** - Fast runner (3x speed), fragile but quick

### Carnivores
- **T-Rex** - Apex predator with maximum attack power
- **Velociraptor** - Pack hunter with 1.5x group bonus

## 🎮 Controls

- **SPACE** - Pause/Play
- **R** - Reset simulation
- **↑/↓** - Adjust speed (1-60 steps/sec)
- **ESC** - Quit

## 🌍 Ecosystem Mechanics

- **Defense System** - Herbivores can survive attacks based on defense stat
- **Speed-Based Movement** - Faster species move multiple cells per turn
- **Pack Hunting** - Velociraptors get bonus when hunting together
- **Environmental Effects** - Temperature affects metabolism, rainfall affects grass growth
- **Energy System** - Agents must eat to survive and reproduce

## 📊 Visualization

- **60 FPS real-time rendering** - Smooth, no lag
- **Color-coded species** with live legend
- **Event log** tracking births, deaths, hunts, and defenses
- **Population stats** for each species

## 🔬 Run Experiments

Generate graphs analyzing different environmental scenarios:
```bash
python my_first_island.py
```

Includes heat wave, drought, and multi-scenario comparisons.

## 📚 Tech Stack

- **Mesa 3.3.0** - Agent-based modeling
- **Pygame 2.6.1** - 60 FPS visualization
- **Matplotlib** - Data analysis plots
