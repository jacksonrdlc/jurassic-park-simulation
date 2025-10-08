# 🧠 PyTorch Learning Agents - Setup Complete!

## ✅ What's Been Added

### New Files Created

1. **`learning_agents.py`** - Neural network learning agents
   - `LearningHerbivoreAgent` - AI herbivore with neural decision-making
   - `LearningCarnivoreAgent` - AI carnivore with neural decision-making
   - Experience replay buffer for training
   - Observation/action/reward systems
   - Support for all 4 species (Triceratops, Gallimimus, T-Rex, Velociraptor)

2. **`train_agents.py`** - Training script
   - Train agents over multiple episodes
   - Automatic model saving
   - Training metrics and graphs
   - Adjustable exploration rate (epsilon decay)

3. **`compare_agents.py`** - Evaluation script
   - Compare AI vs traditional agents
   - Statistical analysis
   - Performance graphs

### Updated Files

1. **`my_first_island.py`**
   - Added `use_learning_agents` parameter to `IslandModel`
   - Conditional agent spawning (AI or traditional)

2. **`pygame_viz.py`**
   - Support for visualizing learning agents
   - White ring indicator for AI agents
   - Command-line flag: `--ai` or `--learning-agents`
   - Model loading for trained agents

3. **`README.md`**
   - New AI Learning Agents section
   - Training instructions
   - Comparison instructions

### New Environment

**`jurassic_ml_env/`** - Python 3.11 virtual environment
- PyTorch 2.2.2 (CPU version - compatible with older Intel Macs)
- All required dependencies (Mesa, Pygame, scikit-learn, etc.)

---

## 🚀 How to Use

### 1. Train Learning Agents

```bash
# Activate ML environment (Python 3.11 + TensorFlow)
source jurassic_ml_env/bin/activate

# Train for 100 episodes
python train_agents.py --episodes 100 --steps 200

# Shorter training (50 episodes)
python train_agents.py --episodes 50
```

**Output:**
- Trained models saved to `models/` directory
- Training graphs: `training_results.png`
- Progress printed to console

### 2. Visualize Learning Agents

```bash
# Activate ML environment
source jurassic_ml_env/bin/activate

# Run with AI agents
python pygame_viz.py --ai
```

**What you'll see:**
- AI agents have **white rings** around them
- Status panel shows "Agents: AI (Learning)"
- Legend includes "AI Agent (ring)" entry
- Agents use trained neural networks for decisions

### 3. Compare Performance

```bash
# Activate ML environment
source jurassic_ml_env/bin/activate

# Compare AI vs traditional (10 runs each)
python compare_agents.py --runs 10
```

**Output:**
- Comparison graph: `agent_comparison.png`
- Statistical summary in console
- Performance improvement percentages

---

## 🎯 How Learning Works

### State Observation (19 features)
- Own energy level (normalized)
- Distance to grass/prey in 8 directions
- Distance to predators/allies in 8 directions
- Environmental factors (temperature, rainfall)

### Action Space (9 actions)
- Move in 8 directions (N, NE, E, SE, S, SW, W, NW)
- Stay in place

### Rewards
**Herbivores:**
- +10 for eating grass
- +0.1 per survival step
- -50 for death
- +100 for reproduction

**Carnivores:**
- +20 for successful hunt
- +0.1 per survival step
- -50 for death
- +150 for reproduction

### Neural Network
- Input: 19 neurons (state observation)
- Hidden Layer 1: 64 neurons (ReLU)
- Hidden Layer 2: 32 neurons (ReLU)
- Output: 9 neurons (Softmax - action probabilities)

### Training Algorithm
- **Experience Replay**: Stores (state, action, reward, next_state) tuples
- **Batch Training**: Trains on random batches of 32 experiences
- **Q-Learning Update**: Target = reward + 0.95 * max(next_Q)
- **Epsilon-Greedy**: Starts at 20% exploration, decays to 5%

---

## 📁 File Structure

```
jurassic-park-simulation/
├── learning_agents.py          # Neural network agents
├── train_agents.py             # Training script
├── compare_agents.py           # Comparison script
├── my_first_island.py          # Main simulation (updated)
├── pygame_viz.py               # Visualization (updated)
├── README.md                   # Updated docs
├── jurassic_env/               # Original Python 3.13 env
├── jurassic_ml_env/            # NEW: Python 3.11 + PyTorch
└── models/                     # Trained neural networks (created after training)
    ├── herbivore_final.pth
    └── carnivore_final.pth
```

---

## 🔬 Expected Behavior

After training, AI agents should demonstrate:

1. **Herbivores:**
   - Move toward grass patches
   - Flee from carnivores
   - Better energy management
   - Higher survival rates

2. **Carnivores:**
   - Chase prey effectively
   - Coordinate pack hunts (Velociraptors)
   - Avoid energy waste
   - Efficient hunting patterns

3. **Overall:**
   - More stable population dynamics
   - Better adaptation to environmental changes
   - Emergent strategies and behaviors

---

## 🛠️ Troubleshooting

### Issue: PyTorch import error
**Solution:** Make sure you're using `jurassic_ml_env` (Python 3.11), not `jurassic_env` (Python 3.13)
```bash
source jurassic_ml_env/bin/activate
```

### Issue: No trained models found
**Solution:** Train agents first before running visualization or comparison
```bash
python train_agents.py --episodes 50
```

### Issue: Agents behaving randomly
**Solution:**
- Ensure models are loaded (check console output)
- Training mode should be OFF for visualization
- Epsilon should be 0.0 during evaluation

### Issue: Low performance improvement
**Solution:**
- Train for more episodes (try 200-500)
- Adjust reward values in `learning_agents.py`
- Increase network size or learning rate

---

## 🎮 Quick Commands

```bash
# Train agents (recommended first step)
source jurassic_ml_env/bin/activate && python train_agents.py --episodes 100

# Watch AI agents in action
source jurassic_ml_env/bin/activate && python pygame_viz.py --ai

# Compare performance
source jurassic_ml_env/bin/activate && python compare_agents.py --runs 10

# Watch traditional agents (no AI)
source jurassic_env/bin/activate && python pygame_viz.py
```

---

## 📊 Metrics to Track

1. **Survival Rate**: % of agents alive at end of episode
2. **Average Steps Alive**: How long agents survive
3. **Total Reward**: Cumulative reward per episode
4. **Population Stability**: Variance in population over time

Look for:
- ✅ Increasing rewards over episodes
- ✅ Higher survival rates compared to traditional agents
- ✅ More stable population dynamics
- ✅ Adaptive behaviors emerging

---

**Next Steps:**
1. Run `train_agents.py` to create your first trained models
2. Visualize with `pygame_viz.py --ai` to see them in action
3. Compare with `compare_agents.py` to measure improvement
4. Experiment with different reward structures and network architectures!

Happy Training! 🦕🧠
