"""
Event Manager
Handles all user input (keyboard, mouse)
"""

import pygame
from src.config.settings import VIEWPORT_WIDTH


class EventManager:
    """Manages user input and events"""

    def __init__(self, camera):
        """
        Initialize event manager

        Args:
            camera: Camera object
        """
        self.camera = camera
        self.should_quit = False
        self.should_pause = False
        self.should_reset = False
        self.should_toggle_minimap = False

    def handle_events(self):
        """
        Handle all pygame events

        Returns:
            Tuple of (should_quit, should_pause, should_reset, should_toggle_minimap)
        """
        # Reset single-frame flags
        self.should_quit = False
        self.should_reset = False
        self.should_toggle_minimap = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.should_quit = True

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(event)

            elif event.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_up(event)

            elif event.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(event)

        # Handle continuous keyboard input
        keys = pygame.key.get_pressed()
        self.camera.update_keyboard(keys)

        return self.should_quit, self.should_pause, self.should_reset, self.should_toggle_minimap

    def _handle_keydown(self, event):
        """Handle keydown events"""
        if event.key == pygame.K_ESCAPE:
            self.should_quit = True
        elif event.key == pygame.K_SPACE:
            self.should_pause = not self.should_pause
        elif event.key == pygame.K_r:
            self.should_reset = True
        elif event.key == pygame.K_m:
            self.should_toggle_minimap = True

    def _handle_mouse_down(self, event):
        """Handle mouse button down events"""
        if event.button == 1:  # Left click
            # Only drag if clicking in viewport area (not panel)
            if event.pos[0] < VIEWPORT_WIDTH:
                self.camera.start_drag(event.pos)

        elif event.button == 3:  # Right click
            # Center camera on clicked location
            world_x, world_y = self.camera.screen_to_world(event.pos[0], event.pos[1])
            self.camera.center_on(int(world_x), int(world_y))

    def _handle_mouse_up(self, event):
        """Handle mouse button up events"""
        if event.button == 1:  # Left click
            self.camera.end_drag()

    def _handle_mouse_motion(self, event):
        """Handle mouse motion events"""
        if self.camera.dragging:
            self.camera.update_drag(event.pos)

    def get_pause_state(self):
        """Get current pause state"""
        return self.should_pause
