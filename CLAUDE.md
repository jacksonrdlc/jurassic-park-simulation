# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 🦕 Project Vision

A living Jurassic Park ecosystem simulation — not just a tech demo, but a window into emergent behavior. The goal: open it and the island is *alive*. Herds moving. Pack hunts in progress. Velociraptors developing flanking strategies. Watch chaos theory unfold. A dinosaur ant farm with unlimited possibilities.

This is simultaneously:
- A real-time agent-based ecosystem with predator/prey dynamics
- A machine learning playground (Q-learning DQN + PPO reinforcement learning)
- A visual spectacle (pixel art sprites, procedural terrain, weather, day/night cycle)
- A living experiment where complex behaviors emerge from simple rules

**The north star:** Run the simulation and be surprised. That surprise = emergence = success.

---

## 📊 Current State (as of 2026-03-07)

### ✅ What Works
- Full Pygame simulation at 60fps with clean modular `src/` architecture
- Procedurally generated island (625×375 cells, Perlin noise, volcanic peak, rivers, coastline)
- 4 dinosaur species: T-Rex, Velociraptor, Triceratops, Gallimimus
- Three agent modes: traditional rule-based, Q-learning (DQN/PyTorch), PPO (Stable-Baselines3)
- Pre-trained models in `models/` and `models_ppo/`
- Custom pixel art sprites, camera system, minimap, day/night cycle, weather, volcano smoke
- DINOSAUR_PROFILES.md with deep behavioral research for 10+ species
- Web viewer: `python server.py` → localhost:8000 (FastAPI + WebSocket + Canvas)

### 🔴 Known Problems

1. **Dual environment split** — `jurassic_env` (Python 3.13, game only) and `jurassic_ml_env` (Python 3.11, ML training). Constant context switching kills momentum. Goal: single unified env.

2. **Training/simulation disconnect** — `ppo_env.py` trains PPO agents in an *isolated* simplified Gymnasium environment, NOT in the real `IslandModel`. Agents learn strategies that don't transfer — they behave dumbly in the real sim because the state space doesn't match.

3. **Undertrained agents** — 500k PPO steps in a disconnected env means emergent behavior never develops. Needs: train IN the world, not beside it.

4. **Species gap** — DINOSAUR_PROFILES.md has 10+ researched species; only 4 are implemented. All that research is waiting.

5. **Legacy file confusion** — `pygame_viz.py` is the old monolith. `run_game.py` is the real entry point. Ignore `pygame_viz.py` for new development.

---

## 🚀 Entry Points

```bash
# CURRENT: Use jurassic_env (Python 3.13)
source jurassic_env/bin/activate

python run_game.py              # Traditional rule-based agents
python run_game.py --ai         # Q-learning neural network agents
python run_game.py --ppo        # PPO reinforcement learning agents (needs training first)
```

**⚠️ Note:** `pygame_viz.py` is legacy. `run_game.py` is canonical. Use `run_game.py`.

Training (currently needs different env):
```bash
source jurassic_ml_env/bin/activate
python train_ppo.py --agent both --timesteps 500000
python train_agents.py --episodes 100
```

---

## 🏗️ Architecture

```
src/
├── config/
│   ├── colors.py             # All color constants (Jurassic Park theme)
│   └── settings.py           # Game constants (world size, viewport, FPS, cell size)
├── entities/
│   ├── dinosaur_sprite.py    # Sprite rendering helpers
│   └── particle.py           # Volcano smoke particle system
├── rendering/
│   ├── terrain_renderer.py   # Terrain, agents, sprites, trees, energy bars
│   ├── ui_renderer.py        # Info panel, minimap, Jurassic Park branding
│   └── effects_renderer.py   # Day/night, weather, rain, heat shimmer
├── managers/
│   ├── game_manager.py       # Main game loop, state management, AI model loading
│   └── event_manager.py      # Keyboard/mouse input handling
└── utils/
    └── sprite_loader.py      # Sprite sheet loading and animation

Core files (root level):
├── run_game.py               # Entry point — parses args, boots GameManager
├── my_first_island.py        # IslandModel + all agent classes (the brain of the sim)
├── terrain_generator.py      # Procedural island generation (Perlin noise)
├── camera.py                 # Viewport/camera system
├── sprite_sheet.py           # Sprite sheet parser (4x4 directional frames)
├── learning_agents.py        # Q-learning DQN agents (PyTorch)
├── ppo_agents.py             # PPO agent wrappers
├── ppo_env.py                # Gymnasium envs for PPO training (DISCONNECTED from real sim — known issue)
├── train_agents.py           # Q-learning training script
├── train_ppo.py              # PPO training script
└── compare_agents.py         # Performance comparison tool
```

---

## 🧠 Key Design Decisions

### Mesa Framework
Using `mesa.space.MultiGrid` (coordinate-based, not the newer `OrthogonalMooreGrid`).

```python
# Move agent
self.model.grid.move_agent(self, new_pos)

# Access cell contents
self.model.grid.get_cell_list_contents([self.pos])

# Remove agent (Mesa 3.x)
self.remove()  # from within agent's step()

# Step all agents (Mesa 3.x — NOT RandomActivation)
self.agents.shuffle_do("step")

# Model init
class IslandModel(Model):
    def __init__(self, ...):
        super().__init__()  # optionally: super().__init__(seed=42)
```

### Agent Hierarchy
- `GrassAgent` — passive patch, regrows based on rainfall
- `HerbivoreAgent` — base herbivore: vision radius, goal-seeking toward grass, terrain-aware movement
- `CarnivoreAgent` — base carnivore: hunting, pack bonus for Velociraptors
- Species override base classes: `TRexAgent`, `VelociraptorAgent`, `TriceratopsAgent`, `GallimimusAgent`

### Terrain System
- `terrain_generator.py` → NumPy 2D array of `TerrainType` enum values
- `TERRAIN_WALKABLE` dict controls movement (ocean, rivers = impassable)
- Mountains apply movement cost (slow but passable)

### AI Models
- **Q-learning**: `learning_agents.py` — 19-feature state observation, 9-action output, experience replay buffer
- **PPO**: `ppo_env.py` wraps ecosystem as Gymnasium env, trains with stable-baselines3
- Saved: `models/*.pth` (Q-learning), `models_ppo/*.zip` (PPO)

### Self.model Conflict
In learning agents, `self.model` is the Mesa simulation model (inherited from Agent). The PyTorch neural network is stored as `self.neural_net` to avoid collision. **Do not rename this back.**

---

## 🎯 Development Priorities

### Phase 1 — Fix the Foundation
- [ ] Unify `jurassic_env` + `jurassic_ml_env` → single `requirements.txt` (Python 3.12)
- [ ] Bridge `ppo_env.py` to train directly inside `IslandModel` (real terrain, real agents)
- [ ] Add `Makefile` or `setup.sh` for one-command environment setup

### Phase 2 — AI That Surprises You
- [ ] Longer training runs (2M+ steps) with curriculum learning
- [ ] Agent memory: last N positions, hunger trend, threat awareness
- [ ] Social behaviors: herd cohesion, alarm calls, coordinated fleeing

### Phase 3 — Fill the Species Gap
- [ ] Implement Stegosaurus (tank herbivore, tail spike counterattack)
- [ ] Implement Pachycephalosaurus (dome charge ability)
- [ ] Brachiosaurus (herd, tall, can browse forest canopy)
- [ ] Reference DINOSAUR_PROFILES.md for behavioral specs

### Phase 4 — The Jurassic Park Part
- [ ] Containment fences (degrade over time, carnivores test them)
- [ ] Park visitors + jeeps (objectives: safety, revenue, wonder)
- [ ] Containment breach events — the chaos theory payoff
- [ ] Scenario editor: trigger drought, volcano eruption, introduce new species

---

## ⚠️ Common Pitfalls

1. **Wrong Python env** — Game runs in `jurassic_env` (3.13), training currently needs `jurassic_ml_env` (3.11). This is the known issue Phase 1 fixes.

2. **`self.model` collision** — In learning agents, `self.model` = Mesa sim, `self.neural_net` = PyTorch network. Don't rename.

3. **Training/sim state mismatch** — PPO agents trained in `ppo_env.py` use a simplified observation/action space. Until Phase 1 bridges these, expect awkward behavior in the real sim.

4. **Large files in git** — `models/` and `models_ppo/` are committed. Consider .gitignoring `*.zip` and `*.pth` checkpoint files (keep `*_final.*`).

5. **pygame_viz.py is dead** — It still works but is 1065 lines and not maintained. All new work goes into `src/` modules.

---

## 🔬 Quick Test Commands

```bash
# Smoke test — headless experiment run
source jurassic_env/bin/activate
python my_first_island.py

# Q-learning training smoke test
source jurassic_ml_env/bin/activate
python train_agents.py --episodes 2 --steps 50

# PPO training smoke test
source jurassic_ml_env/bin/activate
python train_ppo.py --agent both --timesteps 10000

# Full game
source jurassic_env/bin/activate
python run_game.py
```

---

## 🦖 Species Roster

| Species | Type | Speed | Metabolism | Defense/Attack | Implemented |
|---------|------|-------|------------|----------------|-------------|
| T-Rex | Carnivore | 1 | 1.8x | 100% attack | ✅ |
| Velociraptor | Carnivore | 2 | 0.8x | 50% + pack 1.5x | ✅ |
| Triceratops | Herbivore | 1 | 1.5x | 50% defense | ✅ |
| Gallimimus | Herbivore | 3 | 0.7x | 30% defense | ✅ |
| Stegosaurus | Herbivore | 1 | 1.6x | 70% defense + spike | 🔲 |
| Pachycephalosaurus | Herbivore | 2 | 1.0x | Dome charge | 🔲 |
| Brachiosaurus | Herbivore | 1 | 2.0x | Herd protection | 🔲 |
| Spinosaurus | Carnivore | 2 | 1.4x | Semiaquatic | 🔲 |

---

## 📚 Related Docs

- [README.md](README.md) — Full feature documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) — Rendering pipeline and module design
- [DINOSAUR_PROFILES.md](DINOSAUR_PROFILES.md) — Paleontological behavioral specs for all species
- [AI_SETUP.md](AI_SETUP.md) — ML training guide
- [PYTORCH_MIGRATION.md](PYTORCH_MIGRATION.md) — Why PyTorch (TF had AVX issues)
