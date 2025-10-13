#!/usr/bin/env python3
"""
Jurassic Park Ecosystem Simulation
Main entry point for the refactored game

Usage:
    python run_game.py              # Traditional rule-based agents
    python run_game.py --ai         # Q-learning neural network agents
    python run_game.py --ppo        # PPO reinforcement learning agents

Controls:
    SPACE - Pause/Play
    R - Reset simulation
    Arrow Keys - Pan camera
    Left Click + Drag - Pan camera
    Right Click - Center camera
    M - Toggle minimap
    ESC - Quit
"""

import sys
from src.managers.game_manager import GameManager


def main():
    """Main entry point"""
    # Parse command line arguments
    use_ppo = '--ppo' in sys.argv
    use_learning = ('--learning-agents' in sys.argv or '--ai' in sys.argv) and not use_ppo

    # Print startup info
    print("=" * 60)
    print("🦖 JURASSIC PARK ECOSYSTEM SIMULATION")
    print("=" * 60)
    print()

    if use_ppo:
        print("Mode: PPO Agents (Proximal Policy Optimization) 🚀")
        print("  State-of-the-art reinforcement learning!")
    elif use_learning:
        print("Mode: Q-Learning Agents 🧠")
        print("  Neural network Q-learning")
    else:
        print("Mode: Traditional Rule-Based Agents")
        print("  (use --ppo for PPO agents, --ai for Q-learning agents)")

    print()
    print("Controls:")
    print("  SPACE      - Pause/Play")
    print("  R          - Reset simulation")
    print("  Arrow Keys - Pan camera")
    print("  Mouse Drag - Pan camera")
    print("  Right Click- Center camera")
    print("  M          - Toggle minimap")
    print("  ESC        - Quit")
    print()
    print("=" * 60)
    print()

    # Create and run game
    game = GameManager(
        use_learning_agents=use_learning,
        use_ppo_agents=use_ppo
    )
    game.run()


if __name__ == "__main__":
    main()
