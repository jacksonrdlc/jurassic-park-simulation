"""
Dinosaur Sprite Rendering
Helper functions for drawing dinosaur-shaped sprites
"""

import math
import pygame
from src.config.settings import SPRITE_SCALE_MULTIPLIER


def calculate_sprite_points(cell_size, size_ratio, direction):
    """
    Calculate dinosaur body and head points

    Args:
        cell_size: Size of cell in pixels
        size_ratio: Relative size of dinosaur (0.3-0.45)
        direction: Movement direction tuple (dx, dy)

    Returns:
        Tuple of (body_points, head_points, angle, radius)
    """
    # Calculate angle from direction
    if direction == (0, 0):
        angle = 0
    else:
        angle = math.atan2(direction[1], direction[0])

    # GBA-style scaling: dinosaurs span multiple tiles
    radius = int(cell_size * size_ratio * SPRITE_SCALE_MULTIPLIER)

    # Dinosaur dimensions
    body_length = radius
    body_width = radius * 0.6
    head_size = radius * 0.4

    # Body (ellipse-ish shape with multiple points)
    body_points = [
        (body_length * 0.3, 0),  # Neck connection
        (body_length * 0.5, -body_width * 0.3),  # Top of body
        (body_length * 0.3, -body_width * 0.5),  # Upper back
        (-body_length * 0.2, -body_width * 0.4),  # Back top
        (-body_length * 0.5, -body_width * 0.2),  # Tail start top
        (-body_length * 0.8, 0),  # Tail end
        (-body_length * 0.5, body_width * 0.2),  # Tail start bottom
        (-body_length * 0.2, body_width * 0.4),  # Back bottom
        (body_length * 0.3, body_width * 0.5),  # Lower back
        (body_length * 0.5, body_width * 0.3),  # Bottom of body
    ]

    # Head (pointing forward)
    head_points = [
        (body_length * 0.3, 0),  # Neck
        (body_length * 0.7, -head_size * 0.3),  # Top of head
        (body_length, 0),  # Snout/mouth
        (body_length * 0.7, head_size * 0.3),  # Bottom of head
    ]

    return body_points, head_points, angle, radius, body_length, head_size


def rotate_points(points, angle, center):
    """
    Rotate points around center

    Args:
        points: List of (x, y) tuples
        angle: Rotation angle in radians
        center: Center point (cx, cy)

    Returns:
        List of rotated points
    """
    cx, cy = center
    rotated = []

    for px, py in points:
        # Rotate
        rx = px * math.cos(angle) - py * math.sin(angle)
        ry = px * math.sin(angle) + py * math.cos(angle)
        # Translate to center
        rotated.append((cx + rx, cy + ry))

    return rotated


def draw_dinosaur(screen, rect, color, direction, size_ratio):
    """
    Draw a dinosaur-shaped sprite

    Args:
        screen: Pygame screen surface
        rect: Pygame rect for position
        color: Base color tuple (r, g, b)
        direction: Movement direction tuple (dx, dy)
        size_ratio: Relative size (0.3-0.45)
    """
    center = rect.center
    cell_size = rect.width

    # Calculate sprite geometry
    body_points, head_points, angle, radius, body_length, head_size = \
        calculate_sprite_points(cell_size, size_ratio, direction)

    # Rotate points around center
    body_rotated = rotate_points(body_points, angle, center)
    head_rotated = rotate_points(head_points, angle, center)

    # Draw body (main dinosaur shape)
    pygame.draw.polygon(screen, color, body_rotated)

    # Draw head (slightly lighter)
    head_color = tuple(min(255, c + 20) for c in color)
    pygame.draw.polygon(screen, head_color, head_rotated)

    # Draw outlines for definition
    outline_color = tuple(max(0, c - 50) for c in color)
    pygame.draw.polygon(screen, outline_color, body_rotated, 2)
    pygame.draw.polygon(screen, outline_color, head_rotated, 2)

    # Add eye dot
    eye_x = body_length * 0.6
    eye_y = -head_size * 0.15
    eye_rx = eye_x * math.cos(angle) - eye_y * math.sin(angle)
    eye_ry = eye_x * math.sin(angle) + eye_y * math.cos(angle)
    eye_pos = (int(center[0] + eye_rx), int(center[1] + eye_ry))

    pygame.draw.circle(screen, (0, 0, 0), eye_pos, max(2, int(radius * 0.15)))


def draw_movement_trail(screen, cell_size, agent, camera, color):
    """
    Draw fading trail showing recent movement

    Args:
        screen: Pygame screen surface
        cell_size: Size of cells in pixels
        agent: Agent with movement_history attribute
        camera: Camera object for coordinate transformation
        color: Trail color (r, g, b)
    """
    if not hasattr(agent, 'movement_history') or not agent.movement_history:
        return

    # Draw trail with fading alpha
    for i, old_pos in enumerate(agent.movement_history):
        # Check if trail position is visible
        if not camera.is_visible(old_pos[0], old_pos[1]):
            continue

        alpha = int(100 * (i + 1) / len(agent.movement_history))  # Fade from 33 to 100
        trail_surface = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)

        trail_color = (*color, alpha)
        pygame.draw.circle(trail_surface, trail_color,
                         (cell_size // 2, cell_size // 2),
                         cell_size // 4)

        # Convert world coordinates to screen coordinates
        screen_x, screen_y = camera.world_to_screen(old_pos[0], old_pos[1])
        screen.blit(trail_surface, (int(screen_x), int(screen_y)))
