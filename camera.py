"""
Camera System for Vast Island World
Handles viewport, panning, zooming, and coordinate transformations
"""

import pygame


class Camera:
    """
    Camera for scrolling through a large world
    Tracks camera position and converts between world/screen coordinates
    """

    def __init__(self, viewport_width, viewport_height, world_width, world_height, cell_size):
        """
        Initialize camera

        Args:
            viewport_width: Width of visible area in pixels
            viewport_height: Height of visible area in pixels
            world_width: Total world width in cells
            world_height: Total world height in cells
            cell_size: Size of each cell in pixels
        """
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.world_width = world_width
        self.world_height = world_height
        self.cell_size = cell_size

        # Camera position (in world cells) - top-left corner of viewport
        self.x = 0
        self.y = 0

        # Drag state
        self.dragging = False
        self.drag_start_mouse = None
        self.drag_start_camera = None

        # Movement momentum (for smooth scrolling with arrow keys)
        self.velocity_x = 0
        self.velocity_y = 0

        # Zoom level (for future use)
        self.zoom = 1.0

        # Center camera on middle of world
        self.center_on(world_width / 2, world_height / 2)

    def center_on(self, world_x, world_y):
        """Center camera on a specific world position"""
        # Calculate camera position to center on this point
        viewport_cells_w = self.viewport_width / self.cell_size
        viewport_cells_h = self.viewport_height / self.cell_size

        self.x = world_x - viewport_cells_w / 2
        self.y = world_y - viewport_cells_h / 2

        self._clamp_camera()

    def world_to_screen(self, world_x, world_y):
        """Convert world coordinates (in cells) to screen pixels"""
        screen_x = (world_x - self.x) * self.cell_size
        screen_y = (world_y - self.y) * self.cell_size
        return screen_x, screen_y

    def screen_to_world(self, screen_x, screen_y):
        """Convert screen pixels to world coordinates (in cells)"""
        world_x = (screen_x / self.cell_size) + self.x
        world_y = (screen_y / self.cell_size) + self.y
        return world_x, world_y

    def is_visible(self, world_x, world_y):
        """Check if a world cell is visible in viewport"""
        viewport_cells_w = self.viewport_width / self.cell_size
        viewport_cells_h = self.viewport_height / self.cell_size

        return (self.x - 1 <= world_x < self.x + viewport_cells_w + 1 and
                self.y - 1 <= world_y < self.y + viewport_cells_h + 1)

    def get_visible_bounds(self):
        """Get the range of visible cells (with buffer)"""
        viewport_cells_w = self.viewport_width / self.cell_size
        viewport_cells_h = self.viewport_height / self.cell_size

        # Add buffer around edges
        buffer = 2

        min_x = max(0, int(self.x) - buffer)
        min_y = max(0, int(self.y) - buffer)
        max_x = min(self.world_width, int(self.x + viewport_cells_w) + buffer)
        max_y = min(self.world_height, int(self.y + viewport_cells_h) + buffer)

        return min_x, min_y, max_x, max_y

    def pan(self, dx, dy):
        """Pan camera by delta amount (in cells)"""
        self.x += dx
        self.y += dy
        self._clamp_camera()

    def start_drag(self, mouse_pos):
        """Start dragging with mouse"""
        self.dragging = True
        self.drag_start_mouse = mouse_pos
        self.drag_start_camera = (self.x, self.y)

    def update_drag(self, mouse_pos):
        """Update camera position based on mouse drag"""
        if not self.dragging or self.drag_start_mouse is None:
            return

        # Calculate how far mouse has moved
        dx_pixels = self.drag_start_mouse[0] - mouse_pos[0]
        dy_pixels = self.drag_start_mouse[1] - mouse_pos[1]

        # Convert to cells
        dx_cells = dx_pixels / self.cell_size
        dy_cells = dy_pixels / self.cell_size

        # Update camera position
        self.x = self.drag_start_camera[0] + dx_cells
        self.y = self.drag_start_camera[1] + dy_cells

        self._clamp_camera()

    def end_drag(self):
        """Stop dragging"""
        self.dragging = False
        self.drag_start_mouse = None
        self.drag_start_camera = None

    def update_keyboard(self, keys_pressed, dt=1.0):
        """Update camera position based on keyboard input"""
        # Arrow key panning
        pan_speed = 0.3  # Cells per frame

        if keys_pressed[pygame.K_LEFT]:
            self.velocity_x = -pan_speed
        elif keys_pressed[pygame.K_RIGHT]:
            self.velocity_x = pan_speed
        else:
            self.velocity_x *= 0.9  # Friction

        if keys_pressed[pygame.K_UP]:
            self.velocity_y = -pan_speed
        elif keys_pressed[pygame.K_DOWN]:
            self.velocity_y = pan_speed
        else:
            self.velocity_y *= 0.9  # Friction

        # Apply velocity
        if abs(self.velocity_x) > 0.01 or abs(self.velocity_y) > 0.01:
            self.pan(self.velocity_x * dt, self.velocity_y * dt)

    def _clamp_camera(self):
        """Clamp camera position to world bounds"""
        viewport_cells_w = self.viewport_width / self.cell_size
        viewport_cells_h = self.viewport_height / self.cell_size

        # Don't let camera go past world edges
        self.x = max(0, min(self.x, self.world_width - viewport_cells_w))
        self.y = max(0, min(self.y, self.world_height - viewport_cells_h))

    def zoom_in(self):
        """Zoom in (for future use)"""
        self.zoom = min(2.0, self.zoom * 1.1)

    def zoom_out(self):
        """Zoom out (for future use)"""
        self.zoom = max(0.5, self.zoom / 1.1)

    def get_info(self):
        """Get camera info for display"""
        return {
            'position': (int(self.x), int(self.y)),
            'center': (int(self.x + self.viewport_width / self.cell_size / 2),
                      int(self.y + self.viewport_height / self.cell_size / 2)),
            'zoom': self.zoom,
            'dragging': self.dragging
        }

    def minimap_coords(self, minimap_width, minimap_height):
        """Get rectangle coordinates for minimap viewport indicator"""
        # Scale world to minimap size
        scale_x = minimap_width / self.world_width
        scale_y = minimap_height / self.world_height

        viewport_cells_w = self.viewport_width / self.cell_size
        viewport_cells_h = self.viewport_height / self.cell_size

        # Viewport rectangle in minimap coordinates
        x = int(self.x * scale_x)
        y = int(self.y * scale_y)
        w = int(viewport_cells_w * scale_x)
        h = int(viewport_cells_h * scale_y)

        return pygame.Rect(x, y, w, h)


if __name__ == "__main__":
    # Test camera
    camera = Camera(
        viewport_width=960,
        viewport_height=640,
        world_width=250,
        world_height=150,
        cell_size=16
    )

    print(f"Camera viewport: {camera.viewport_width}x{camera.viewport_height} pixels")
    print(f"World size: {camera.world_width}x{camera.world_height} cells")
    print(f"Viewport shows: {camera.viewport_width/camera.cell_size:.1f}x{camera.viewport_height/camera.cell_size:.1f} cells")
    print(f"Initial position: {camera.get_info()['position']}")
    print(f"Initial center: {camera.get_info()['center']}")

    # Test visible bounds
    min_x, min_y, max_x, max_y = camera.get_visible_bounds()
    print(f"Visible cells: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    print(f"Visible area: {(max_x-min_x) * (max_y-min_y)} cells out of {camera.world_width * camera.world_height} total")

    # Test coordinate conversion
    world_pos = (125, 75)  # Center of world
    screen_pos = camera.world_to_screen(*world_pos)
    print(f"\nWorld {world_pos} -> Screen {screen_pos}")

    back_to_world = camera.screen_to_world(*screen_pos)
    print(f"Screen {screen_pos} -> World {back_to_world}")
