#!/usr/bin/env python3
"""
Pygame Visualization for Jurassic Park Simulation
Runs at 60 FPS with ZERO flickering - smooth as butter!

Controls:
  SPACE - Pause/Play
  R - Reset simulation
  UP/DOWN - Adjust speed
  ESC - Quit
"""

import pygame
import sys
from my_first_island import (IslandModel, GrassAgent, HerbivoreAgent, CarnivoreAgent,
                              Triceratops, Gallimimus, TRex, Velociraptor)

# Colors
COLORS = {
    'background': (20, 20, 30),
    'grass_full': (34, 139, 34),
    'grass_eaten': (139, 69, 19),
    'herbivore': (65, 105, 225),
    'carnivore': (220, 20, 60),
    # Species-specific colors
    'triceratops': (100, 180, 100),    # Green herbivore
    'gallimimus': (180, 180, 100),     # Yellow herbivore
    'trex': (200, 50, 50),             # Dark red carnivore
    'velociraptor': (128, 0, 128),     # Purple carnivore
    'text': (255, 255, 255),
    'panel': (30, 30, 40),
    'event_birth': (40, 200, 80),
    'event_death': (220, 50, 50),
    'event_hunt': (255, 165, 0),
    'event_eat': (100, 200, 100),
    'event_info': (100, 149, 237),
}

class JurassicParkViz:
    def __init__(self, width=50, height=50, cell_size=12):
        pygame.init()

        # Model parameters
        self.model_width = width
        self.model_height = height
        self.cell_size = cell_size

        # Window dimensions
        self.grid_width = width * cell_size
        self.grid_height = height * cell_size
        self.panel_width = 450
        self.window_width = self.grid_width + self.panel_width
        self.window_height = max(self.grid_height, 800)  # Ensure minimum height

        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Jurassic Park Ecosystem")

        # Fonts
        self.font_large = pygame.font.Font(None, 32)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 16)

        # Simulation state
        self.model = None
        self.running = True
        self.paused = False
        self.step_count = 0
        self.speed = 1  # Steps per second
        self.clock = pygame.time.Clock()

        # Reset model
        self.reset_model()

    def reset_model(self):
        """Create a new model"""
        self.model = IslandModel(
            width=self.model_width,
            height=self.model_height,
            num_herbivores=20,
            num_carnivores=5,
            temperature=25,
            rainfall=100
        )
        self.step_count = 0

    def handle_events(self):
        """Handle keyboard/mouse events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset_model()
                elif event.key == pygame.K_UP:
                    self.speed = min(60, self.speed + 1)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(1, self.speed - 1)

    def update(self):
        """Update simulation"""
        if not self.paused:
            self.model.step()
            self.step_count += 1

    def draw_grid(self):
        """Draw the ecosystem grid"""
        # Draw cells
        for x in range(self.model.grid.width):
            for y in range(self.model.grid.height):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # Get cell contents
                cell_contents = self.model.grid.get_cell_list_contents([(x, y)])

                # Determine what to draw (priority: carnivore > herbivore > grass)
                color = COLORS['background']
                for agent in cell_contents:
                    if isinstance(agent, TRex):
                        color = COLORS['trex']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 2 - 1
                        )
                        break
                    elif isinstance(agent, Velociraptor):
                        color = COLORS['velociraptor']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 2 - 1
                        )
                        break
                    elif isinstance(agent, Triceratops):
                        color = COLORS['triceratops']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 3
                        )
                        break
                    elif isinstance(agent, Gallimimus):
                        color = COLORS['gallimimus']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 3
                        )
                        break
                    elif isinstance(agent, CarnivoreAgent):
                        color = COLORS['carnivore']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 2 - 1
                        )
                        break
                    elif isinstance(agent, HerbivoreAgent):
                        color = COLORS['herbivore']
                        pygame.draw.circle(
                            self.screen,
                            color,
                            rect.center,
                            self.cell_size // 3
                        )
                        break
                    elif isinstance(agent, GrassAgent):
                        color = COLORS['grass_full'] if agent.energy > 0 else COLORS['grass_eaten']
                        pygame.draw.rect(self.screen, color, rect)

    def draw_panel(self):
        """Draw info panel on the right"""
        panel_x = self.grid_width

        # Background
        pygame.draw.rect(
            self.screen,
            COLORS['panel'],
            (panel_x, 0, self.panel_width, self.window_height)
        )

        y_offset = 20

        # Title
        title = self.font_large.render("JURASSIC PARK", True, COLORS['text'])
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50

        # Separator
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 1)
        y_offset += 20

        # Stats
        tri_count = len([a for a in self.model.agents if isinstance(a, Triceratops)])
        gal_count = len([a for a in self.model.agents if isinstance(a, Gallimimus)])
        trex_count = len([a for a in self.model.agents if isinstance(a, TRex)])
        vel_count = len([a for a in self.model.agents if isinstance(a, Velociraptor)])
        g_count = len([a for a in self.model.agents if isinstance(a, GrassAgent) and a.energy > 0])

        stats = [
            f"Step: {self.step_count}",
            f"Speed: {self.speed} steps/sec",
            f"Status: {'PAUSED' if self.paused else 'RUNNING'}",
            "",
            "HERBIVORES:",
            f"  Triceratops: {tri_count}",
            f"  Gallimimus: {gal_count}",
            "",
            "CARNIVORES:",
            f"  T-Rex: {trex_count}",
            f"  Velociraptor: {vel_count}",
            "",
            f"Grass: {g_count}",
            "",
            f"Temperature: {self.model.temperature}C",
            f"Rainfall: {self.model.rainfall}mm",
        ]

        for stat in stats:
            text = self.font_medium.render(stat, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 26

        # Separator
        y_offset += 10
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 1)
        y_offset += 20

        # Event log title
        log_title = self.font_medium.render("EVENT LOG", True, COLORS['text'])
        self.screen.blit(log_title, (panel_x + 20, y_offset))
        y_offset += 30

        # Calculate max space for events (leave room for controls)
        max_log_y = self.window_height - 180

        # Show events
        if hasattr(self.model, 'event_log'):
            events = list(reversed(self.model.event_log[-25:]))

            for event in events:
                if y_offset > max_log_y:
                    break

                # Determine color and prefix
                if "BIRTH" in event:
                    color = COLORS['event_birth']
                    prefix = "[+]"
                elif "DEATH" in event:
                    color = COLORS['event_death']
                    prefix = "[-]"
                elif "HUNT" in event:
                    color = COLORS['event_hunt']
                    prefix = "[!]"
                elif "EAT" in event:
                    color = COLORS['event_eat']
                    prefix = "[*]"
                elif "INFO" in event:
                    color = COLORS['event_info']
                    prefix = "[i]"
                else:
                    color = COLORS['text']
                    prefix = "   "

                # Remove emoji characters
                event = event.replace('🦕', '').replace('🦖', '').replace('🌿', '')
                event = event.replace('💀', '').replace('📊', '').replace('🟢', '')

                # Truncate long events
                if len(event) > 38:
                    event = event[:35] + "..."

                text = self.font_small.render(f"{prefix} {event}", True, color)
                self.screen.blit(text, (panel_x + 15, y_offset))
                y_offset += 17

        # Separator before controls
        y_offset = self.window_height - 160
        pygame.draw.line(self.screen, COLORS['text'],
                        (panel_x + 20, y_offset),
                        (panel_x + self.panel_width - 20, y_offset), 1)
        y_offset += 20

        # Controls section (fixed at bottom)
        controls_title = self.font_medium.render("CONTROLS", True, COLORS['text'])
        self.screen.blit(controls_title, (panel_x + 20, y_offset))
        y_offset += 28

        controls = [
            "SPACE - Pause/Play",
            "R     - Reset",
            "UP/DN - Speed +/-",
            "ESC   - Quit",
        ]

        for control in controls:
            text = self.font_small.render(control, True, COLORS['text'])
            self.screen.blit(text, (panel_x + 20, y_offset))
            y_offset += 20

    def draw_legend(self):
        """Draw legend in top-left corner"""
        legend_x = 10
        legend_y = 10

        # Semi-transparent background (larger for more species)
        legend_surface = pygame.Surface((200, 170))
        legend_surface.set_alpha(220)
        legend_surface.fill(COLORS['panel'])
        self.screen.blit(legend_surface, (legend_x, legend_y))

        # Title
        title_text = self.font_small.render("LEGEND", True, COLORS['text'])
        self.screen.blit(title_text, (legend_x + 10, legend_y + 8))

        legend_items = [
            ("Grass (full)", COLORS['grass_full'], 'rect'),
            ("Grass (eaten)", COLORS['grass_eaten'], 'rect'),
            ("Triceratops", COLORS['triceratops'], 'circle'),
            ("Gallimimus", COLORS['gallimimus'], 'circle'),
            ("T-Rex", COLORS['trex'], 'circle'),
            ("Velociraptor", COLORS['velociraptor'], 'circle'),
        ]

        y = legend_y + 30
        for label, color, shape in legend_items:
            if shape == 'circle':
                pygame.draw.circle(self.screen, color, (legend_x + 15, y + 8), 6)
            else:
                pygame.draw.rect(self.screen, color, (legend_x + 10, y + 3, 12, 12))

            text = self.font_small.render(label, True, COLORS['text'])
            self.screen.blit(text, (legend_x + 35, y))
            y += 20

    def run(self):
        """Main game loop - runs at 60 FPS!"""
        frame_count = 0

        while self.running:
            self.handle_events()

            # Update simulation at specified speed
            if frame_count % max(1, 60 // self.speed) == 0:
                self.update()

            # Draw everything
            self.screen.fill(COLORS['background'])
            self.draw_grid()
            self.draw_panel()
            self.draw_legend()

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS - smooth as butter!
            frame_count += 1

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    print("Starting Jurassic Park Simulation...")
    print("Controls: SPACE=pause, R=reset, UP/DOWN=speed, ESC=quit")

    viz = JurassicParkViz(width=50, height=50, cell_size=12)
    viz.run()
