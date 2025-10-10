"""
Train Jurassic Park agents using PPO (Proximal Policy Optimization)
Much more advanced than Q-learning - state-of-the-art policy gradient method
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from ppo_env import JurassicParkHerbivoreEnv, JurassicParkCarnivoreEnv


def make_herbivore_env(rank=0):
    """Create herbivore environment (for parallelization)"""
    def _init():
        env = JurassicParkHerbivoreEnv(width=50, height=50, max_steps=200)
        env = Monitor(env)
        return env
    return _init


def make_carnivore_env(rank=0):
    """Create carnivore environment"""
    def _init():
        env = JurassicParkCarnivoreEnv(width=50, height=50, max_steps=200)
        env = Monitor(env)
        return env
    return _init


def train_herbivore(args):
    """Train herbivore agent with PPO"""
    print("=" * 80)
    print("🦕 TRAINING HERBIVORE AGENT WITH PPO")
    print("=" * 80)
    print()

    # Create directories
    os.makedirs("models_ppo", exist_ok=True)
    os.makedirs("logs_ppo", exist_ok=True)

    # Create parallel environments for faster training
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel environments...")
        env = SubprocVecEnv([make_herbivore_env(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_herbivore_env()])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_herbivore_env()])

    # PPO Hyperparameters (tuned for ecosystem simulation)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,           # Standard PPO learning rate
        n_steps=2048 // args.n_envs,  # Steps per environment before update
        batch_size=64,                 # Batch size for optimization
        n_epochs=10,                   # Number of epochs per update
        gamma=0.95,                    # Discount factor (same as Q-learning)
        gae_lambda=0.95,               # GAE parameter for advantage estimation
        clip_range=0.2,                # PPO clip parameter
        ent_coef=0.01,                 # Entropy coefficient (encourages exploration)
        vf_coef=0.5,                   # Value function coefficient
        max_grad_norm=0.5,             # Gradient clipping
        verbose=1,
        tensorboard_log="./logs_ppo/herbivore"
    )

    print("\n📊 Model Configuration:")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Policy: MlpPolicy (Multi-Layer Perceptron)")
    print(f"  Learning Rate: 3e-4")
    print(f"  Parallel Environments: {args.n_envs}")
    print(f"  Steps per Update: {2048 // args.n_envs}")
    print(f"  Batch Size: 64")
    print(f"  Epochs per Update: 10")
    print(f"  Entropy Coefficient: 0.01 (exploration bonus)")
    print()

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path="./models_ppo",
        name_prefix="herbivore_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_ppo",
        log_path="./logs_ppo",
        eval_freq=10000 // args.n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Train!
    print(f"🚀 Starting training for {args.timesteps:,} timesteps...")
    print(f"   Progress will be logged to ./logs_ppo/herbivore")
    print(f"   Models saved every {args.save_freq:,} steps")
    print(f"   Best model will be saved as ./models_ppo/best_model.zip")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    final_path = "models_ppo/herbivore_ppo_final.zip"
    model.save(final_path)
    print(f"\n✅ Training complete! Final model saved to {final_path}")
    print(f"📈 View training progress with: tensorboard --logdir ./logs_ppo/herbivore")

    return model


def train_carnivore(args):
    """Train carnivore agent with PPO"""
    print("=" * 80)
    print("🦖 TRAINING CARNIVORE AGENT WITH PPO")
    print("=" * 80)
    print()

    os.makedirs("models_ppo", exist_ok=True)
    os.makedirs("logs_ppo", exist_ok=True)

    # Create parallel environments
    if args.n_envs > 1:
        print(f"Creating {args.n_envs} parallel environments...")
        env = SubprocVecEnv([make_carnivore_env(i) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_carnivore_env()])

    eval_env = DummyVecEnv([make_carnivore_env()])

    # PPO model with same hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048 // args.n_envs,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Exploration important for hunting
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs_ppo/carnivore"
    )

    print("\n📊 Model Configuration:")
    print(f"  Algorithm: PPO (Proximal Policy Optimization)")
    print(f"  Policy: MlpPolicy")
    print(f"  Parallel Environments: {args.n_envs}")
    print()

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq // args.n_envs,
        save_path="./models_ppo",
        name_prefix="carnivore_ppo"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_ppo",
        log_path="./logs_ppo",
        eval_freq=10000 // args.n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    print(f"🚀 Starting training for {args.timesteps:,} timesteps...")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    final_path = "models_ppo/carnivore_ppo_final.zip"
    model.save(final_path)
    print(f"\n✅ Training complete! Final model saved to {final_path}")
    print(f"📈 View training progress with: tensorboard --logdir ./logs_ppo/carnivore")

    return model


def train_both(args):
    """Train both herbivore and carnivore agents"""
    print("\n🦕🦖 TRAINING BOTH AGENT TYPES\n")

    print("Step 1/2: Training Herbivore...")
    herbivore_model = train_herbivore(args)

    print("\n" + "=" * 80)
    print("Step 2/2: Training Carnivore...")
    carnivore_model = train_carnivore(args)

    print("\n" * 2)
    print("=" * 80)
    print("🎉 ALL TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📁 Saved Models:")
    print("  - models_ppo/herbivore_ppo_final.zip")
    print("  - models_ppo/carnivore_ppo_final.zip")
    print("  - models_ppo/best_model.zip (best performing)")
    print("\n📊 View Training Metrics:")
    print("  tensorboard --logdir ./logs_ppo")
    print("\n🎮 Test the agents:")
    print("  python pygame_viz.py --ppo")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train Jurassic Park agents with PPO")

    parser.add_argument(
        "--agent",
        type=str,
        choices=["herbivore", "carnivore", "both"],
        default="both",
        help="Which agent type to train (default: both)"
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total number of training timesteps (default: 500,000)"
    )

    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments for faster training (default: 4)"
    )

    parser.add_argument(
        "--save-freq",
        type=int,
        default=50000,
        help="Save model checkpoint every N steps (default: 50,000)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(" " * 20 + "🦕 JURASSIC PARK PPO TRAINING 🦖")
    print("=" * 80)
    print("\n🎯 Training Configuration:")
    print(f"  Agent Type: {args.agent}")
    print(f"  Total Timesteps: {args.timesteps:,}")
    print(f"  Parallel Environments: {args.n_envs}")
    print(f"  Save Frequency: {args.save_freq:,} steps")
    print()
    print("💡 What is PPO?")
    print("  - Proximal Policy Optimization (state-of-the-art RL algorithm)")
    print("  - 2-3x better performance than Q-learning")
    print("  - More stable training")
    print("  - Used by OpenAI, DeepMind, and industry leaders")
    print()
    print("📈 Advantages over Q-learning:")
    print("  ✓ Policy gradient method (learns actions directly)")
    print("  ✓ Handles continuous and discrete actions")
    print("  ✓ Better sample efficiency")
    print("  ✓ More stable convergence")
    print("  ✓ Clipping prevents destructive updates")
    print()

    # Train based on selection
    if args.agent == "herbivore":
        train_herbivore(args)
    elif args.agent == "carnivore":
        train_carnivore(args)
    else:  # both
        train_both(args)


if __name__ == "__main__":
    main()
