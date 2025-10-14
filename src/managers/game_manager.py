"""
Game Manager
Main game loop and state management
"""

import random
import pygame
from my_first_island import IslandModel
from terrain_generator import generate_island
from camera import Camera
from src.config.settings import (
    WORLD_WIDTH, WORLD_HEIGHT, CELL_SIZE,
    VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
    WINDOW_WIDTH, WINDOW_HEIGHT,
    DEFAULT_NUM_HERBIVORES, DEFAULT_NUM_CARNIVORES,
    DEFAULT_TEMPERATURE, DEFAULT_RAINFALL,
    FPS, DEFAULT_SPEED, TIME_SPEED, WATER_ANIMATION_SPEED
)
from src.config.colors import COLORS
from src.managers.event_manager import EventManager
from src.rendering.terrain_renderer import TerrainRenderer
from src.rendering.ui_renderer import UIRenderer
from src.rendering.effects_renderer import EffectsRenderer


class GameManager:
    """Main game manager - coordinates all systems"""

    def __init__(self, use_learning_agents=False, use_ppo_agents=False):
        """
        Initialize game manager

        Args:
            use_learning_agents: Whether to use Q-learning agents
            use_ppo_agents: Whether to use PPO agents
        """
        pygame.init()

        # Game settings
        self.use_learning_agents = use_learning_agents
        self.use_ppo_agents = use_ppo_agents

        # Display
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Jurassic Park - Costa Rican Island")

        # Camera
        self.camera = Camera(
            VIEWPORT_WIDTH,
            VIEWPORT_HEIGHT,
            WORLD_WIDTH,
            WORLD_HEIGHT,
            CELL_SIZE
        )

        # Managers
        self.event_manager = EventManager(self.camera)

        # Renderers
        self.terrain_renderer = TerrainRenderer(self.screen, self.camera)
        self.ui_renderer = UIRenderer(self.screen, self.camera)
        self.effects_renderer = EffectsRenderer(self.screen, self.camera)

        # Game state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.speed = DEFAULT_SPEED
        self.clock = pygame.time.Clock()

        # World data
        self.model = None
        self.terrain_map = None
        self.terrain_generator = None
        self.terrain_noise = None

        # Initialize world
        self.reset_world()

    def reset_world(self):
        """Create a new world with terrain generation"""
        print("\n🏝️  Generating vast Costa Rican island...")

        # Generate terrain
        self.terrain_map, self.terrain_generator = generate_island(
            WORLD_WIDTH,
            WORLD_HEIGHT,
            seed=None  # Random seed each time
        )

        # Generate procedural noise for terrain color variation
        self.terrain_noise = [[random.randint(-10, 10) for _ in range(WORLD_WIDTH)]
                             for _ in range(WORLD_HEIGHT)]

        # Create simulation model
        self.model = IslandModel(
            width=WORLD_WIDTH,
            height=WORLD_HEIGHT,
            num_herbivores=DEFAULT_NUM_HERBIVORES,
            num_carnivores=DEFAULT_NUM_CARNIVORES,
            temperature=DEFAULT_TEMPERATURE,
            rainfall=DEFAULT_RAINFALL,
            use_learning_agents=self.use_learning_agents,
            use_ppo_agents=self.use_ppo_agents,
            terrain_map=self.terrain_map
        )

        self.step_count = 0

        # Reset renderers
        self.effects_renderer.reset()
        self.terrain_renderer.reset_tree_clusters()  # Reset tree clusters for new terrain

        # Center camera
        self.camera.center_on(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)

        # Load AI models if needed
        self._load_ai_models()

    def _load_ai_models(self):
        """Load trained AI models"""
        import os

        if self.use_ppo_agents:
            if os.path.exists('models_ppo/herbivore_ppo_final.zip') and \
               os.path.exists('models_ppo/carnivore_ppo_final.zip'):
                from ppo_agents import PPOHerbivoreAgent, PPOCarnivoreAgent
                PPOHerbivoreAgent.load_model('models_ppo/herbivore_ppo_final.zip')
                PPOCarnivoreAgent.load_model('models_ppo/carnivore_ppo_final.zip')
            else:
                print("⚠️  PPO models not found! Please train them first with:")
                print("   python train_ppo.py")

        elif self.use_learning_agents:
            if os.path.exists('models/herbivore_final.pth'):
                import torch
                from learning_agents import LearningHerbivoreAgent, LearningCarnivoreAgent, NeuralNetwork

                if LearningHerbivoreAgent.neural_net is None:
                    LearningHerbivoreAgent.neural_net = NeuralNetwork(19, 9)
                if LearningCarnivoreAgent.neural_net is None:
                    LearningCarnivoreAgent.neural_net = NeuralNetwork(19, 9)

                LearningHerbivoreAgent.neural_net.load_state_dict(torch.load('models/herbivore_final.pth'))
                LearningCarnivoreAgent.neural_net.load_state_dict(torch.load('models/carnivore_final.pth'))
                LearningHerbivoreAgent.training_mode = False
                LearningCarnivoreAgent.training_mode = False
                LearningHerbivoreAgent.epsilon = 0.0
                LearningCarnivoreAgent.epsilon = 0.0
                print("✅ Loaded trained PyTorch Q-learning models for visualization")

    def update(self):
        """Update game state"""
        if not self.paused:
            self.model.step()
            self.step_count += 1

        # Always update visual effects
        self.effects_renderer.update_time(TIME_SPEED)
        self.effects_renderer.update_particles(self.terrain_map, WORLD_WIDTH, WORLD_HEIGHT)

        # Update water animation
        water_offset = (pygame.time.get_ticks() * WATER_ANIMATION_SPEED / 1000) % 20
        self.terrain_renderer.update_water_animation(water_offset)

    def render(self):
        """Render everything"""
        # Clear screen
        self.screen.fill(COLORS['background'])

        # Draw terrain (ground only)
        self.terrain_renderer.draw_terrain(self.terrain_map, self.terrain_noise)

        # Draw trees (on top of ground, below agents)
        self.terrain_renderer.draw_trees()

        # Draw water shimmer
        self.terrain_renderer.draw_water_shimmer(self.terrain_map)

        # Draw agents (on top of trees)
        self.terrain_renderer.draw_agents(self.model)
        self.terrain_renderer.draw_grid_overlay()

        # Draw effects
        self.effects_renderer.draw_all_effects(
            self.model,
            self.terrain_map,
            WORLD_WIDTH,
            WORLD_HEIGHT
        )

        # Draw UI
        self.ui_renderer.draw_panel(
            self.model,
            self.step_count,
            self.speed,
            self.paused,
            self.use_learning_agents or self.use_ppo_agents
        )
        self.ui_renderer.draw_minimap(self.terrain_map, WORLD_WIDTH, WORLD_HEIGHT)

        # Update display
        pygame.display.flip()

    def run(self):
        """Main game loop"""
        frame_count = 0

        while self.running:
            # Handle events
            should_quit, should_pause, should_reset, should_toggle_minimap = \
                self.event_manager.handle_events()

            if should_quit:
                self.running = False
            if should_pause:
                self.paused = should_pause
            if should_reset:
                self.reset_world()
            if should_toggle_minimap:
                self.ui_renderer.toggle_minimap()

            # Sync pause state
            self.paused = self.event_manager.get_pause_state()

            # Update simulation at specified speed
            if frame_count % max(1, 60 // self.speed) == 0:
                self.update()

            # Render
            self.render()

            # Maintain FPS
            self.clock.tick(FPS)
            frame_count += 1

        pygame.quit()
