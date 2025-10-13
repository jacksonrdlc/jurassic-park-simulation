"""
Particle System for Visual Effects
Volcano smoke, rain, etc.
"""

import random


class VolcanoParticle:
    """Smoke/steam particle from volcano"""

    def __init__(self, x, y):
        """
        Initialize a volcano particle

        Args:
            x: World x coordinate
            y: World y coordinate
        """
        self.x = x
        self.y = y
        self.offset_x = random.uniform(-0.5, 0.5)
        self.offset_y = 0
        self.vy = -0.05 - random.uniform(0, 0.05)  # Upward velocity
        self.life = 1.0  # 1.0 = full life, 0.0 = dead
        self.size = random.randint(2, 5)

    def update(self):
        """Update particle position and life"""
        self.offset_y += self.vy
        self.life -= 0.01

    def is_dead(self):
        """Check if particle should be removed"""
        return self.life <= 0

    def get_alpha(self):
        """Get alpha value based on remaining life"""
        return int(self.life * 180)
