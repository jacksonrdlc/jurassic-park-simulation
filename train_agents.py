#!/usr/bin/env python3
"""
Training Script for Learning Agents
Trains dinosaur agents using reinforcement learning over multiple episodes
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from my_first_island import IslandModel
from learning_agents import LearningHerbivoreAgent, LearningCarnivoreAgent
import numpy as np
import torch


def train_agents(num_episodes=100, steps_per_episode=200, save_every=10):
    """
    Train learning agents over multiple episodes

    Args:
        num_episodes: Number of training episodes to run
        steps_per_episode: Simulation steps per episode
        save_every: Save models every N episodes
    """

    print("=" * 60)
    print("🦕 JURASSIC PARK - LEARNING AGENT TRAINING")
    print("=" * 60)
    print(f"\nTraining Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Steps per episode: {steps_per_episode}")
    print(f"  Initial exploration rate (epsilon): {LearningHerbivoreAgent.epsilon:.2f}")
    print()

    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)

    # Training metrics
    episode_rewards_herbivore = []
    episode_rewards_carnivore = []
    herbivore_survival_rates = []
    carnivore_survival_rates = []
    avg_steps_alive_herbivore = []
    avg_steps_alive_carnivore = []

    # Enable training mode
    LearningHerbivoreAgent.training_mode = True
    LearningCarnivoreAgent.training_mode = True

    for episode in range(num_episodes):
        # Create new model with learning agents
        model = IslandModel(
            width=50,
            height=50,
            num_herbivores=20,
            num_carnivores=5,
            temperature=25,
            rainfall=100,
            use_learning_agents=True
        )

        # Track initial counts
        initial_herbivores = len([a for a in model.agents if isinstance(a, LearningHerbivoreAgent)])
        initial_carnivores = len([a for a in model.agents if isinstance(a, LearningCarnivoreAgent)])

        # Run episode
        total_reward_herb = 0
        total_reward_carn = 0
        steps_alive_herb = []
        steps_alive_carn = []

        for step in range(steps_per_episode):
            model.step()

            # Collect rewards
            for agent in model.agents:
                if isinstance(agent, LearningHerbivoreAgent):
                    total_reward_herb += getattr(agent, 'total_reward', 0)
                    steps_alive_herb.append(getattr(agent, 'steps_alive', 0))
                elif isinstance(agent, LearningCarnivoreAgent):
                    total_reward_carn += getattr(agent, 'total_reward', 0)
                    steps_alive_carn.append(getattr(agent, 'steps_alive', 0))

        # Calculate survival rates
        final_herbivores = len([a for a in model.agents if isinstance(a, LearningHerbivoreAgent)])
        final_carnivores = len([a for a in model.agents if isinstance(a, LearningCarnivoreAgent)])

        herb_survival = final_herbivores / initial_herbivores if initial_herbivores > 0 else 0
        carn_survival = final_carnivores / initial_carnivores if initial_carnivores > 0 else 0

        # Calculate average steps alive
        avg_herb_steps = np.mean(steps_alive_herb) if steps_alive_herb else 0
        avg_carn_steps = np.mean(steps_alive_carn) if steps_alive_carn else 0

        # Store metrics
        episode_rewards_herbivore.append(total_reward_herb)
        episode_rewards_carnivore.append(total_reward_carn)
        herbivore_survival_rates.append(herb_survival)
        carnivore_survival_rates.append(carn_survival)
        avg_steps_alive_herbivore.append(avg_herb_steps)
        avg_steps_alive_carnivore.append(avg_carn_steps)

        # Decay exploration rate
        if episode % 10 == 0 and episode > 0:
            LearningHerbivoreAgent.epsilon = max(0.05, LearningHerbivoreAgent.epsilon * 0.95)
            LearningCarnivoreAgent.epsilon = max(0.05, LearningCarnivoreAgent.epsilon * 0.95)

        # Print progress every episode (brief) or detailed every 10
        if episode % 10 == 0 or episode == num_episodes - 1:
            # Detailed output every 10 episodes
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Herbivore - Reward: {total_reward_herb:.1f}, Survival: {herb_survival:.1%}, Avg Steps: {avg_herb_steps:.1f}")
            print(f"  Carnivore - Reward: {total_reward_carn:.1f}, Survival: {carn_survival:.1%}, Avg Steps: {avg_carn_steps:.1f}")
            print(f"  Epsilon: {LearningHerbivoreAgent.epsilon:.3f}")
            print(f"  Replay buffer size - Herbivore: {LearningHerbivoreAgent.replay_buffer.size()}, Carnivore: {LearningCarnivoreAgent.replay_buffer.size()}")
            print()
        else:
            # Brief progress indicator for other episodes
            print(f"Episode {episode + 1}/{num_episodes} - H:{final_herbivores} C:{final_carnivores} - Buffer:{LearningHerbivoreAgent.replay_buffer.size()}")

        # Save models periodically
        if (episode + 1) % save_every == 0:
            torch.save(LearningHerbivoreAgent.neural_net.state_dict(), f'models/herbivore_episode_{episode + 1}.pth')
            torch.save(LearningCarnivoreAgent.neural_net.state_dict(), f'models/carnivore_episode_{episode + 1}.pth')
            print(f"✅ Models saved at episode {episode + 1}\n")

    # Save final models
    torch.save(LearningHerbivoreAgent.neural_net.state_dict(), 'models/herbivore_final.pth')
    torch.save(LearningCarnivoreAgent.neural_net.state_dict(), 'models/carnivore_final.pth')

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal Models Saved:")
    print(f"  models/herbivore_final.pth")
    print(f"  models/carnivore_final.pth")

    # Plot training metrics
    print("\n📊 Generating training graphs...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: Total Rewards
    axes[0, 0].plot(episode_rewards_herbivore, label='Herbivore', color='green', alpha=0.7)
    axes[0, 0].plot(episode_rewards_carnivore, label='Carnivore', color='red', alpha=0.7)
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Survival Rates
    axes[0, 1].plot(herbivore_survival_rates, label='Herbivore', color='green', alpha=0.7)
    axes[0, 1].plot(carnivore_survival_rates, label='Carnivore', color='red', alpha=0.7)
    axes[0, 1].set_title('Survival Rate per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Survival Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Average Steps Alive
    axes[1, 0].plot(avg_steps_alive_herbivore, label='Herbivore', color='green', alpha=0.7)
    axes[1, 0].plot(avg_steps_alive_carnivore, label='Carnivore', color='red', alpha=0.7)
    axes[1, 0].set_title('Average Steps Alive per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Steps')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Learning Progress (moving average of rewards)
    window = 10
    herb_ma = np.convolve(episode_rewards_herbivore, np.ones(window)/window, mode='valid')
    carn_ma = np.convolve(episode_rewards_carnivore, np.ones(window)/window, mode='valid')
    axes[1, 1].plot(herb_ma, label='Herbivore (MA)', color='green', linewidth=2)
    axes[1, 1].plot(carn_ma, label='Carnivore (MA)', color='red', linewidth=2)
    axes[1, 1].set_title(f'Learning Progress (Moving Average, window={window})')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Reward (MA)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("✅ Training graphs saved as 'training_results.png'\n")

    return {
        'herbivore_rewards': episode_rewards_herbivore,
        'carnivore_rewards': episode_rewards_carnivore,
        'herbivore_survival': herbivore_survival_rates,
        'carnivore_survival': carnivore_survival_rates
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train learning agents for Jurassic Park simulation')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
    parser.add_argument('--save-every', type=int, default=10, help='Save models every N episodes')

    args = parser.parse_args()

    results = train_agents(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        save_every=args.save_every
    )

    print("🎯 Training session complete!")
    print(f"Final herbivore reward trend: {results['herbivore_rewards'][-10:]}")
    print(f"Final carnivore reward trend: {results['carnivore_rewards'][-10:]}")
