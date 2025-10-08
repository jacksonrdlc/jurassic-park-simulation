#!/usr/bin/env python3
"""
Agent Comparison Script
Compare performance of learning agents vs traditional rule-based agents
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from my_first_island import IslandModel
from learning_agents import LearningHerbivoreAgent, LearningCarnivoreAgent
import numpy as np
import os


def run_simulation(use_learning_agents, num_steps=200, num_runs=10):
    """Run simulation and collect metrics"""

    all_herbivore_counts = []
    all_carnivore_counts = []
    all_grass_counts = []

    for run in range(num_runs):
        model = IslandModel(
            width=50,
            height=50,
            num_herbivores=20,
            num_carnivores=5,
            temperature=25,
            rainfall=100,
            use_learning_agents=use_learning_agents
        )

        # If using learning agents, disable training mode (use learned policy)
        if use_learning_agents:
            # Load trained models if they exist
            if os.path.exists('models/herbivore_final.pth'):
                import torch
                from learning_agents import NeuralNetwork
                # Initialize models if not already done
                if LearningHerbivoreAgent.neural_net is None:
                    LearningHerbivoreAgent.neural_net = NeuralNetwork(19, 9)
                if LearningCarnivoreAgent.neural_net is None:
                    LearningCarnivoreAgent.neural_net = NeuralNetwork(19, 9)
                # Load saved weights
                LearningHerbivoreAgent.neural_net.load_state_dict(torch.load('models/herbivore_final.pth'))
                LearningCarnivoreAgent.neural_net.load_state_dict(torch.load('models/carnivore_final.pth'))

            LearningHerbivoreAgent.training_mode = False
            LearningCarnivoreAgent.training_mode = False
            LearningHerbivoreAgent.epsilon = 0.0  # No exploration
            LearningCarnivoreAgent.epsilon = 0.0

        herbivore_counts = []
        carnivore_counts = []
        grass_counts = []

        for step in range(num_steps):
            model.step()

            h_count = len([a for a in model.agents if hasattr(a, 'species_name') and
                          ('Triceratops' in a.species_name or 'Gallimimus' in a.species_name)])
            c_count = len([a for a in model.agents if hasattr(a, 'species_name') and
                          ('Rex' in a.species_name or 'Velociraptor' in a.species_name)])
            from my_first_island import GrassAgent
            g_count = len([a for a in model.agents if isinstance(a, GrassAgent) and a.energy > 0])

            herbivore_counts.append(h_count)
            carnivore_counts.append(c_count)
            grass_counts.append(g_count)

        all_herbivore_counts.append(herbivore_counts)
        all_carnivore_counts.append(carnivore_counts)
        all_grass_counts.append(grass_counts)

    # Calculate means and stds
    herb_mean = np.mean(all_herbivore_counts, axis=0)
    herb_std = np.std(all_herbivore_counts, axis=0)
    carn_mean = np.mean(all_carnivore_counts, axis=0)
    carn_std = np.std(all_carnivore_counts, axis=0)
    grass_mean = np.mean(all_grass_counts, axis=0)
    grass_std = np.std(all_grass_counts, axis=0)

    return {
        'herbivore_mean': herb_mean,
        'herbivore_std': herb_std,
        'carnivore_mean': carn_mean,
        'carnivore_std': carn_std,
        'grass_mean': grass_mean,
        'grass_std': grass_std
    }


def compare_agents(num_steps=200, num_runs=10):
    """Compare learning agents vs traditional agents"""

    print("=" * 60)
    print("🔬 COMPARING LEARNING AGENTS VS TRADITIONAL AGENTS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Steps per simulation: {num_steps}")
    print(f"  Number of runs: {num_runs}")
    print()

    print("Running traditional (rule-based) agents...")
    traditional_results = run_simulation(use_learning_agents=False, num_steps=num_steps, num_runs=num_runs)

    print("Running learning (AI) agents...")
    learning_results = run_simulation(use_learning_agents=True, num_steps=num_steps, num_runs=num_runs)

    # Create comparison plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    steps = np.arange(num_steps)

    # Herbivore comparison
    axes[0].plot(steps, traditional_results['herbivore_mean'], label='Traditional', color='blue', linewidth=2)
    axes[0].fill_between(steps,
                         traditional_results['herbivore_mean'] - traditional_results['herbivore_std'],
                         traditional_results['herbivore_mean'] + traditional_results['herbivore_std'],
                         color='blue', alpha=0.2)

    axes[0].plot(steps, learning_results['herbivore_mean'], label='Learning (AI)', color='green', linewidth=2)
    axes[0].fill_between(steps,
                         learning_results['herbivore_mean'] - learning_results['herbivore_std'],
                         learning_results['herbivore_mean'] + learning_results['herbivore_std'],
                         color='green', alpha=0.2)

    axes[0].set_title('Herbivore Population: Learning vs Traditional Agents', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Population')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Carnivore comparison
    axes[1].plot(steps, traditional_results['carnivore_mean'], label='Traditional', color='darkred', linewidth=2)
    axes[1].fill_between(steps,
                         traditional_results['carnivore_mean'] - traditional_results['carnivore_std'],
                         traditional_results['carnivore_mean'] + traditional_results['carnivore_std'],
                         color='darkred', alpha=0.2)

    axes[1].plot(steps, learning_results['carnivore_mean'], label='Learning (AI)', color='orange', linewidth=2)
    axes[1].fill_between(steps,
                         learning_results['carnivore_mean'] - learning_results['carnivore_std'],
                         learning_results['carnivore_mean'] + learning_results['carnivore_std'],
                         color='orange', alpha=0.2)

    axes[1].set_title('Carnivore Population: Learning vs Traditional Agents', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Population')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Grass comparison
    axes[2].plot(steps, traditional_results['grass_mean'], label='Traditional', color='brown', linewidth=2)
    axes[2].fill_between(steps,
                         traditional_results['grass_mean'] - traditional_results['grass_std'],
                         traditional_results['grass_mean'] + traditional_results['grass_std'],
                         color='brown', alpha=0.2)

    axes[2].plot(steps, learning_results['grass_mean'], label='Learning (AI)', color='green', linewidth=2)
    axes[2].fill_between(steps,
                         learning_results['grass_mean'] - learning_results['grass_std'],
                         learning_results['grass_mean'] + learning_results['grass_std'],
                         color='green', alpha=0.2)

    axes[2].set_title('Grass Availability: Learning vs Traditional Agents', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Simulation Steps')
    axes[2].set_ylabel('Grass Patches')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('agent_comparison.png', dpi=150)
    print("\n✅ Comparison graph saved as 'agent_comparison.png'")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("📊 SUMMARY STATISTICS")
    print("=" * 60)

    print("\nTraditional Agents:")
    print(f"  Final Herbivore Pop: {traditional_results['herbivore_mean'][-1]:.1f} ± {traditional_results['herbivore_std'][-1]:.1f}")
    print(f"  Final Carnivore Pop: {traditional_results['carnivore_mean'][-1]:.1f} ± {traditional_results['carnivore_std'][-1]:.1f}")
    print(f"  Avg Grass Availability: {np.mean(traditional_results['grass_mean']):.1f}")

    print("\nLearning Agents (AI):")
    print(f"  Final Herbivore Pop: {learning_results['herbivore_mean'][-1]:.1f} ± {learning_results['herbivore_std'][-1]:.1f}")
    print(f"  Final Carnivore Pop: {learning_results['carnivore_mean'][-1]:.1f} ± {learning_results['carnivore_std'][-1]:.1f}")
    print(f"  Avg Grass Availability: {np.mean(learning_results['grass_mean']):.1f}")

    # Calculate improvement
    herb_improvement = ((learning_results['herbivore_mean'][-1] - traditional_results['herbivore_mean'][-1]) /
                        traditional_results['herbivore_mean'][-1] * 100)
    carn_improvement = ((learning_results['carnivore_mean'][-1] - traditional_results['carnivore_mean'][-1]) /
                        traditional_results['carnivore_mean'][-1] * 100)

    print("\nPerformance Improvement:")
    print(f"  Herbivore Survival: {herb_improvement:+.1f}%")
    print(f"  Carnivore Survival: {carn_improvement:+.1f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare learning vs traditional agents')
    parser.add_argument('--steps', type=int, default=200, help='Simulation steps')
    parser.add_argument('--runs', type=int, default=10, help='Number of simulation runs for averaging')

    args = parser.parse_args()

    # Check if trained models exist
    if not os.path.exists('models/herbivore_final.pth'):
        print("⚠️  Warning: No trained models found!")
        print("   Learning agents will use untrained neural networks.")
        print("   Run 'python train_agents.py' first for best results.\n")

    compare_agents(num_steps=args.steps, num_runs=args.runs)
