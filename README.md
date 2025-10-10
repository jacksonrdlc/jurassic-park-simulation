# 🦕 Jurassic Park Ecosystem Simulation

A real-time agent-based ecosystem simulation featuring diverse dinosaur species with unique behaviors, environmental dynamics, predator-prey interactions, and **AI learning agents powered by PyTorch**. Complete with Jurassic Park-themed UI, directional sprites, movement trails, and dynamic weather effects.

## 🚀 Quick Start

### Traditional Rule-Based Agents
```bash
source jurassic_env/bin/activate
python pygame_viz.py
```

### Q-Learning Agents 🧠
```bash
source jurassic_ml_env/bin/activate
python pygame_viz.py --ai  # or --learning-agents
```

### PPO Agents 🚀 (State-of-the-Art!)
```bash
source jurassic_ml_env/bin/activate
python pygame_viz.py --ppo
```
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

- **SPACE** - Pause/Play simulation
- **R** - Reset simulation (new random world)
- **UP Arrow** - Increase speed (max 60x)
- **DOWN Arrow** - Decrease speed (min 1x)
- **ESC** - Quit application

## 🌍 Ecosystem Mechanics

- **Size-Based Metabolism** - Larger dinosaurs burn more energy per step
- **Defense System** - Herbivores can survive attacks (Triceratops 50%, Gallimimus 30%)
- **Speed-Based Movement** - Gallimimus moves 3 cells/step, others 1-2
- **Pack Hunting** - Velociraptors get 1.5x bonus when hunting in groups
- **Environmental Effects**:
  - **Temperature** - Affects all agent metabolism (higher temp = more energy loss)
  - **Rainfall** - Affects grass regrowth rate (more rain = faster growth)
- **Energy System** - Agents must eat to survive, maintain energy above 0
- **Reproduction** - Agents reproduce when energy is high (thresholds vary by species)
- **Balanced Ecosystem** - 20 herbivores, 10 carnivores for sustainable populations

## 🎨 Jurassic Park UI Features

### Visual Elements
- **1250x800 HD window** - Large, detailed view (16px cells)
- **Jurassic Park branding** - Iconic yellow/black color scheme
- **"ISLA NUBLAR - SECTOR 7G"** subtitle with warning stripes
- **Containment status** - Green (safe), Yellow (overpopulation), Red (critical)

### Directional Sprites
- **Triangle sprites** pointing in movement direction
- **Species-specific sizes** - T-Rex largest, Gallimimus smallest
- **Movement trails** - Fading ghost images showing last 3 positions
- **Dark outlines** for better visibility

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
python pygame_viz.py --ppo
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

Load automatically when running `python pygame_viz.py --ai`

## 📚 Tech Stack

- **Mesa 3.3.0** - Agent-based modeling framework
- **Pygame 2.6.1** - 60 FPS real-time visualization
- **PyTorch 2.2.2** - Neural networks & deep learning (CPU-compatible)
- **Stable-Baselines3 2.4.1** - State-of-the-art RL algorithms (PPO, A3C, SAC)
- **Gymnasium 1.0.0** - OpenAI Gym API for RL environments
- **Matplotlib 3.x** - Data visualization and training graphs
- **NumPy** - Numerical computations
- **TensorBoard** - Real-time training metrics visualization

## 📁 Project Structure

```
jurassic-park-simulation/
├── my_first_island.py          # Core ecosystem model
├── pygame_viz.py               # Jurassic Park UI visualization
├── learning_agents.py          # PyTorch neural network agents
├── train_agents.py             # Training script for AI agents
├── compare_agents.py           # Performance comparison tool
├── models/                     # Saved neural network weights
│   ├── herbivore_final.pth
│   └── carnivore_final.pth
├── jurassic_env/              # Python 3.13 environment
├── jurassic_ml_env/           # Python 3.11 ML environment
└── README.md                  # This file
```

## 🎯 Learning Objectives

This simulation teaches:
- **Agent-based modeling** - Complex systems from simple rules
- **Reinforcement learning** - Q-learning, experience replay, epsilon-greedy
- **Neural networks** - PyTorch implementation, training loops
- **Ecosystem dynamics** - Predator-prey balance, population oscillations
- **Emergent behavior** - Strategies develop from reward signals
- **System optimization** - Debugging, performance, hardware compatibility
