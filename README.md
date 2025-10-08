# 🦕 Jurassic Park Ecosystem Simulation

An agent-based model simulating a Jurassic Park ecosystem with herbivores, carnivores, and dynamic environmental conditions.

## 🚀 Quick Start

### Activate the environment
```bash
source jurassic_env/bin/activate
```

### Run the interactive visualization
```bash
python pygame_viz.py
```
**Runs at 60 FPS - smooth, real-time, game-like visualization!**

### Run experiments (generates graphs)
```bash
python my_first_island.py
```

## 📊 Features

### 🎮 Real-Time Pygame Visualization
- **Smooth 60 FPS rendering** - No dimming, no lag, no flickering!
- **Real-time updates** - Watch dinosaurs move in real-time
- **Color-coded agents**:
  - 🟢 Green squares = Grass (full energy)
  - 🟤 Brown squares = Eaten grass
  - 🔵 Blue circles = Herbivores (plant eaters)
  - 🔴 Red circles = Carnivores (meat eaters)
- **Live event log** - Scrolls smoothly with color-coded events
- **Live stats panel** - Population counts, temperature, rainfall
- **Keyboard controls**:
  - SPACE - Pause/Play
  - R - Reset simulation
  - ↑/↓ - Adjust speed (1-60 steps/sec)
  - ESC - Quit

### Environmental Dynamics
- **Temperature** affects metabolism (higher temp = more energy consumption)
- **Rainfall** affects grass growth rate
- **Energy system**: Agents must eat to survive and reproduce

### Agent Behaviors
- **Herbivores**: Move randomly, eat grass, reproduce when energy > 100
- **Carnivores**: Hunt herbivores, reproduce when energy > 150
- **Grass**: Regrows over time based on rainfall

## 📁 Files

- `my_first_island.py` - Main simulation model with agents and environment
- `pygame_viz.py` - Real-time Pygame visualization (60 FPS!)
- `CLAUDE.md` - Development guide for Mesa 3.x API
- `heat_wave_experiment.png` - Generated graph from experiments
- `scenario_comparison.png` - Generated comparison of different conditions

## 🎮 How to Use the Visualization

1. **Start**: Click the ▶️ play button or ⏭️ step button
2. **Adjust parameters**: Use the sliders on the left
3. **Watch the legend**: Shows what each color means
4. **Monitor events**: Real-time log of births (green), deaths (red), hunts (yellow)
5. **View trends**: Charts show population dynamics over time

## 🔬 Experiments

The `my_first_island.py` script runs several experiments:
- **Normal conditions**: Baseline ecosystem
- **Heat wave**: Temperature increase from 25°C → 35°C
- **Drought**: Low rainfall conditions
- **Comparison**: Multiple scenarios side-by-side

## 🐛 Troubleshooting

### Pygame window not opening
- Make sure you have display access
- On macOS: Grant terminal display permissions in System Preferences

### Clear Python cache
```bash
rm -rf __pycache__
```

## 📚 Tech Stack

- **Mesa 3.3.0** - Agent-based modeling framework
- **Pygame 2.6.1** - Real-time 60 FPS visualization
- **Matplotlib** - Static plots for experiments
- **NumPy/Pandas** - Data analysis

## 🎯 Next Steps

Try modifying the simulation:
- Adjust energy values in agent classes
- Add new agent types (scavengers, plants)
- Implement terrain features (water, forests)
- Add disease or aging mechanics
- Create seasonal weather changes
