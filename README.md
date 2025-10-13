# 🦕 Jurassic Park Ecosystem Simulation

A real-time agent-based ecosystem simulation featuring diverse dinosaur species with unique behaviors, environmental dynamics, predator-prey interactions, and **AI learning agents powered by PyTorch**. Complete with Jurassic Park-themed UI, **authentic pixel art sprites**, movement trails, widescreen viewport with camera exploration, and dynamic weather effects on a massive procedurally-generated island.

## 🚀 Quick Start

### ✅ Run the Simulation

**Virtual Environment:** `jurassic_env` (Python 3.13)

```bash
source jurassic_env/bin/activate
python run_game.py              # Traditional rule-based agents
python run_game.py --ai         # Q-learning neural network agents
python run_game.py --ppo        # PPO reinforcement learning agents
```

**Features:**
- ✅ Authentic pixel art dinosaur sprites with 4-directional animation
- ✅ Tree sprites on forests and rainforests
- ✅ Goal-oriented AI (herbivores seek grass, carnivores hunt prey)
- ✅ Animation only when moving (no idle animation)
- ✅ 16px ultra-fine terrain detail with proper proportions
- ✅ Widescreen 1600x900 viewport with smooth camera exploration
- ✅ Cohesive procedurally-generated Costa Rican island
- ✅ Movement trails showing last 3 positions
- ✅ Modular architecture with clean separation of concerns

**Note:** PPO agents require training first (see PPO Training section below)

## 🦖 Dinosaur Species

All species feature **size-based metabolism** - larger dinosaurs consume more energy!

### Herbivores
- **Triceratops** 🛡️
  - Defense: 50% (heavily armored)
  - Speed: 1 cell/step (slow)
  - Metabolism: 1.5x (LARGE - burns more energy)
  - Starting Energy: 80

- **Gallimimus** 💨
  - Defense: 30% (quick escape artist)
  - Speed: 3 cells/step (fast runner)
  - Metabolism: 0.7x (SMALL - energy efficient)
  - Starting Energy: 40

### Carnivores
- **T-Rex** 👑
  - Attack: 100% (apex predator)
  - Speed: 1 cell/step (slow stalker)
  - Metabolism: 1.8x (MASSIVE - very hungry)
  - Starting Energy: 150/250 (Traditional/AI)

- **Velociraptor** 🦖
  - Attack: 50% (pack hunter)
  - Speed: 2 cells/step (fast hunter)
  - Metabolism: 0.8x (MEDIUM - efficient)
  - Pack Bonus: 1.5x when hunting together
  - Starting Energy: 60/120 (Traditional/AI)

## 🎮 Controls

### Navigation
- **Left/Right/Up/Down Arrows** - Pan camera around the vast island
- **Left Click + Drag** - Pan camera by dragging the view
- **Right Click** - Center camera on clicked location
- **M** - Toggle minimap on/off

### Simulation
- **SPACE** - Pause/Play simulation
- **R** - Reset simulation (generates new random island)
- **ESC** - Quit application

## 🏝️ Island World

Experience a **625x375 cell Costa Rican island** with diverse terrain, animated water, day/night cycles, and smooth camera exploration!

### Terrain Types
- **Ocean** 🌊 - Deep blue waters surrounding the island (impassable)
- **Beach** 🏖️ - Sandy tan shores at the water's edge
- **Sand** 🏝️ - Light sand transition zones
- **Grassland** 🌿 - Vibrant green plains (optimal for dinosaurs)
- **Forest** 🌲 - Medium green wooded areas
- **Rainforest** 🌴 - Dense dark green jungle
- **River** 💧 - Light blue flowing water (impassable)
- **Mountain** ⛰️ - Gray-brown elevated terrain (slows movement)
- **Volcano** 🌋 - Dark volcanic peak at island center

### Procedural Generation
- **Perlin noise terrain** - Natural-looking heightmaps with 4 octaves
- **Island shape masking** - Falloff curves create realistic coastlines
- **Volcanic peak** - Central mountain with gentle slopes
- **River systems** - Water flows downhill from mountains to ocean
- **Terrain smoothing** - Removes single-cell anomalies for realistic transitions
- **Unique every run** - Each reset generates a new random island

### Camera System
- **Widescreen viewport** - 1600x900 pixels (~50x28 cells visible at once)
- **Large view area** - See most of the 625x375 cell island at once
- **Smooth panning** - Arrow keys or drag to explore the island
- **Minimap** - Bottom-left overlay shows full island with yellow viewport indicator
- **Coordinate grid** - A-Z columns, numbered rows for sector tracking
- **Follow action** - Right-click to center camera on any location

### Terrain-Based Movement
- **Walkable/Impassable** - Dinosaurs avoid ocean and rivers
- **Movement costs** - Mountains slow down movement (future enhancement)
- **Strategic spawning** - Dinosaurs spawn on grassland/forest areas
- **Grass placement** - Vegetation only grows on suitable land

### ✨ Visual Polish

**Animated Water** 🌊
- Sine wave effects create realistic ocean movement
- Shimmer highlights on wave peaks
- Rivers flow with gentle animation
- Dynamic brightness changes based on waves

**Day/Night Cycle** 🌙☀️
- Full 24-hour light cycle (sped up for viewing)
- Midnight → Sunrise → Noon → Sunset → Midnight
- Dark blue tint increases at night
- Automatic time progression

**Volcano Effects** 🌋
- Rising smoke/steam particles from volcanic peaks
- Gray smoke with fading alpha
- Particles drift upward and dissipate
- Only visible volcanoes spawn smoke (optimization)

**Authentic Pixel Art Sprites** 🦖
- Custom 16x16 dinosaur sprites from retro sprite sheet
- Dynamic rotation based on movement direction
- Species-specific scaling (T-Rex 3.0x, Triceratops 2.5x, Velociraptor 2.5x, Gallimimus 2.0x, Brachiosaurus 2.5x)
- **Earthy color tints** - Browns, tans, and natural colors (not green!)
- Smooth movement trails showing last 3 positions
- High-quality pixel art spanning multiple cells for visibility

**Balanced Population** 🎯
- **125 herbivores** across the island
- **50 carnivores** hunting territory
- **~25,000 grass tiles** on suitable terrain
- **~234,000 total cells** in the world

## 🌍 Ecosystem Mechanics

- **Goal-Oriented Behavior** - NEW! Dinosaurs actively seek food and prey
  - **Herbivores** search for grass within 10-cell vision radius
  - **Carnivores** hunt prey within 15-cell vision radius
  - Random exploration when no targets visible
- **Size-Based Metabolism** - Larger dinosaurs burn more energy per step
- **Defense System** - Herbivores can survive attacks (Triceratops 50%, Gallimimus 30%)
- **Speed-Based Movement** - Gallimimus moves 3 cells/step, others 1-2
- **Pack Hunting** - Velociraptors get 1.5x bonus when hunting in groups
- **Environmental Effects**:
  - **Temperature** - Affects all agent metabolism (higher temp = more energy loss)
  - **Rainfall** - Affects grass regrowth rate (more rain = faster growth)
- **Energy System** - Agents must eat to survive, maintain energy above 0
- **Reproduction** - Agents reproduce when energy is high (thresholds vary by species)
- **Balanced Ecosystem** - 125 herbivores, 50 carnivores spread across the island

## 🎨 Jurassic Park UI Features

### Visual Elements
- **2050x900 widescreen window** - Large viewport (1600x900) + info panel (450px)
- **625x375 cell world** - Explore via smooth camera scrolling
- **16px cell rendering** - ~100x56 cells visible at once (zoomed-in terrain detail!)
- **Tree sprites** - Forests and rainforests feature procedural tree sprites
- **Jurassic Park branding** - Iconic yellow/black color scheme
- **"ISLA NUBLAR - SECTOR 7G"** subtitle with warning stripes
- **Containment status** - Green (safe), Yellow (overpopulation), Red (critical)

### Authentic Dinosaur Sprites
- **Pixel art sprites** from custom sprite sheet (`dino_sprites/` folder)
- **Dynamic rotation** - Sprites automatically rotate to face movement direction
- **Animation on movement** - Walking animations play ONLY when dinosaurs move
- **Earthy color tints** - Browns, tans, and natural colors applied to each species
- **Species-specific artwork**:
  - T-Rex: TyrannosaurusRex_16x16.png (scaled 1.8x, blood red-brown tint)
  - Triceratops: Triceratops_16x16.png (scaled 1.5x, earthy tan tint)
  - Velociraptor: Spinosaurus_16x16.png (scaled 1.5x, dark brown tint)
  - Gallimimus: Parasaurolophus_16x16.png (scaled 1.2x, tan/beige tint)
  - Brachiosaurus: Brachiosaurus_32x32.png (scaled 1.5x, warm tan tint)
  - Stegosaurus: Stegosaurus_32x32.png (scaled 1.2x, brown tint)
- **Movement trails** - Fading ghost images showing last 3 positions
- **Smooth rendering** with pygame transform operations
- **Proportional sizing** - Dinosaurs properly scaled relative to 16px terrain cells

### Environmental Effects
- **Procedural terrain** - Each grass tile has unique color variation
- **Rain overlay** - Animated raindrops when rainfall > 120mm
- **Heat shimmer** - Orange overlay when temperature > 32°C
- **Grid coordinates** - A-Z columns, numbered rows for sector tracking

### Real-Time Monitoring
- **Energy bars** - Color-coded health above each dinosaur (green/yellow/red)
- **Population tracking** - Live counts with colored indicators
- **Event log** - Scrolling feed of births, deaths, hunts
- **System status** - Step count, speed, mode (AI/Traditional)
- **60 FPS rendering** - Buttery smooth animation

### AI Visual Indicators
- **Colored badge** in top-left corner = AI-powered learning agents
  - **T** (red) = T-Rex
  - **V** (purple) = Velociraptor
  - **3** (green) = Triceratops
  - **G** (yellow) = Gallimimus
- No badge = Traditional rule-based agents

## 🔬 Run Experiments

Generate graphs analyzing different environmental scenarios:
```bash
source jurassic_env/bin/activate
python my_first_island.py
```

Includes heat wave, drought, and multi-scenario comparisons with matplotlib plots.

## 🚀 PPO Training (State-of-the-Art RL)

### What is PPO?

**PPO (Proximal Policy Optimization)** is a modern, state-of-the-art reinforcement learning algorithm that significantly outperforms traditional Q-learning:

**Advantages over Q-Learning:**
- ✅ **2-3x better performance** - achieves higher rewards and better strategies
- ✅ **More stable training** - smoother learning curves, less variance
- ✅ **Policy gradient method** - learns actions directly (not Q-values)
- ✅ **Industry standard** - used by OpenAI, DeepMind, and researchers worldwide
- ✅ **Better sample efficiency** - learns faster from fewer experiences

### Training PPO Agents

Train both herbivore and carnivore agents:
```bash
source jurassic_ml_env/bin/activate
python train_ppo.py --agent both --timesteps 500000
```

Train specific agent type:
```bash
# Train only herbivores
python train_ppo.py --agent herbivore --timesteps 500000

# Train only carnivores
python train_ppo.py --agent carnivore --timesteps 500000
```

**Training Arguments:**
- `--agent` - Which agents to train: `herbivore`, `carnivore`, or `both` (default: both)
- `--timesteps` - Total training timesteps (default: 500,000)
- `--n-envs` - Parallel environments for faster training (default: 4)
- `--save-freq` - Checkpoint frequency (default: 50,000 steps)

**Training Features:**
- **Parallel training** - 4 environments run simultaneously for 4x faster training
- **Automatic checkpoints** - Models saved every 50,000 steps
- **Best model tracking** - Best performing model saved separately
- **TensorBoard logging** - Real-time training metrics visualization
- **Progress bars** - Visual feedback during training

**View Training Progress:**
```bash
tensorboard --logdir ./logs_ppo
```
Open http://localhost:6006 to see live training graphs!

**Training Time:**
- **500,000 timesteps** ≈ 30-60 minutes (with 4 parallel environments)
- **1,000,000 timesteps** ≈ 1-2 hours (recommended for best results)

### Visualize PPO Agents

After training, run the simulation with PPO agents:
```bash
source jurassic_env/bin/activate
python run_game.py --ppo
```

**What to Expect:**
- Herbivores will demonstrate learned evasion strategies
- Carnivores will show coordinated hunting behaviors
- Both will optimize energy management and reproduction timing
- Better performance than both traditional and Q-learning agents

## 🧠 Q-Learning Agents (Neural Network DQN)

### Training Agents
Train dinosaurs to learn optimal survival strategies using **PyTorch neural networks**:

```bash
source jurassic_ml_env/bin/activate  # Python 3.11 environment with PyTorch
python train_agents.py --episodes 100 --steps 200 --save-every 10
```

**Arguments:**
- `--episodes` - Number of training episodes (default: 100)
- `--steps` - Simulation steps per episode (default: 200)
- `--save-every` - Save model checkpoint frequency (default: 10)

**What they learn:**
- **Herbivores**: Find grass efficiently, avoid predators, manage energy reserves
- **Carnivores**: Hunt prey using proximity rewards, coordinate with allies
- **All agents**: Optimize movement patterns, reproduction timing, survival strategies

**Training improvements:**
- **Size-based metabolism** incorporated into reward calculations
- **Proximity rewards** for carnivores (approach prey = +1.0, flee = -0.5)
- **Hunting attempt rewards** (+2.0 for trying, +30.0 for success)
- **Energy efficiency** rewards for maintaining optimal energy levels

### Compare Performance
Benchmark AI agents vs traditional rule-based agents:

```bash
source jurassic_ml_env/bin/activate
python compare_agents.py --runs 10 --steps 200
```

**Arguments:**
- `--runs` - Number of simulation runs to average (default: 10)
- `--steps` - Steps per simulation (default: 200)

Generates comparison graphs showing:
- Population dynamics over time
- Survival rates by species
- Average steps alive
- Learning progress (moving average)

### Trained Models

Pre-trained models saved in `models/`:
- `herbivore_final.pth` - Trained herbivore neural network
- `carnivore_final.pth` - Trained carnivore neural network
- `*_episode_N.pth` - Checkpoint models from training

Load automatically when running `python run_game.py --ai`

## 📚 Tech Stack

- **Mesa 3.3.0** - Agent-based modeling framework
- **Pygame 2.6.1** - 60 FPS real-time visualization
- **PyTorch 2.2.2** - Neural networks & deep learning (CPU-compatible)
- **Stable-Baselines3 2.4.1** - State-of-the-art RL algorithms (PPO, A3C, SAC)
- **Gymnasium 1.0.0** - OpenAI Gym API for RL environments
- **Matplotlib 3.x** - Data visualization and training graphs
- **NumPy** - Numerical computations
- **TensorBoard** - Real-time training metrics visualization

## 🎨 Sprite Attribution

Dinosaur sprites located in `dino_sprites/` folder:
- 16x16 pixel art dinosaur sprites
- Dynamically scaled and rotated during gameplay
- See `dino_sprites/README.md` for sprite specifications

## 📁 Project Structure

```
jurassic-park-simulation/
├── src/                        # Modular source code
│   ├── config/                 # Configuration
│   │   ├── colors.py           # Color scheme (Jurassic Park theme)
│   │   └── settings.py         # Game constants (viewport, cell size, etc.)
│   ├── entities/               # Game entities
│   │   ├── dinosaur_sprite.py  # Sprite rendering helpers
│   │   └── particle.py         # Particle system (volcano smoke)
│   ├── rendering/              # Rendering modules
│   │   ├── terrain_renderer.py # Terrain, agents, sprites, trees
│   │   ├── ui_renderer.py      # UI panels & minimap
│   │   └── effects_renderer.py # Weather & environmental effects
│   ├── managers/               # Game management
│   │   ├── game_manager.py     # Main game loop
│   │   └── event_manager.py    # Input handling (keyboard, mouse)
│   └── utils/                  # Utilities
│       └── sprite_loader.py    # Sprite sheet loading & animation
│
├── run_game.py                 # Main entry point (start here!)
├── my_first_island.py          # Core ecosystem model (Mesa agents)
├── terrain_generator.py        # Procedural island generation (Perlin noise)
├── camera.py                   # Viewport/camera system
├── sprite_sheet.py             # Sprite sheet parser (4x4 directional frames)
├── dino_sprites/               # Dinosaur sprite artwork
│   ├── TyrannosaurusRex_16x16.png
│   ├── Triceratops_16x16.png
│   ├── Spinosaurus_16x16.png (used for Velociraptor)
│   ├── Parasaurolophus_16x16.png (used for Gallimimus)
│   ├── Brachiosaurus_32x32.png
│   ├── Stegosaurus_32x32.png
│   ├── Archeopteryx_16x16.png
│   ├── Pachycephalosaurus_16x16.png
│   └── README.md               # Sprite specifications
├── learning_agents.py          # PyTorch Q-learning agents (DQN)
├── ppo_agents.py               # PPO reinforcement learning agents
├── ppo_env.py                  # Gymnasium environments for PPO
├── train_agents.py             # Q-learning training script
├── train_ppo.py                # PPO training script
├── compare_agents.py           # Performance comparison tool
├── models/                     # Saved neural network weights
│   ├── herbivore_final.pth     # Q-learning herbivore
│   ├── carnivore_final.pth     # Q-learning carnivore
│   ├── ppo_herbivore_best.zip  # PPO herbivore
│   └── ppo_carnivore_best.zip  # PPO carnivore
├── jurassic_env/              # Python 3.13 environment (main)
├── jurassic_ml_env/           # Python 3.11 ML environment (training)
├── README.md                  # This file
└── ARCHITECTURE.md            # Architecture documentation
```

**See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed module descriptions and design principles.**

## 🎯 Learning Objectives

This simulation teaches:
- **Agent-based modeling** - Complex systems emerge from simple rules
- **Reinforcement learning** - PPO (state-of-the-art), Q-learning, experience replay
- **Neural networks** - PyTorch implementation, training loops, model evaluation
- **Procedural generation** - Perlin noise, heightmaps, terrain systems
- **Game development** - Camera systems, viewport rendering, minimap design
- **Ecosystem dynamics** - Predator-prey balance, population oscillations
- **Emergent behavior** - Strategies develop from environmental constraints
- **Spatial optimization** - Culling, coordinate transformation, efficient rendering
- **System design** - Modular architecture, backward compatibility
