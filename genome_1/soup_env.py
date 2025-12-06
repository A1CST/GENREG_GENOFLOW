# soup_env.py
"""
Lightweight Pygame environment representing a tiled "digital ocean" (primordial soup).
Grid map with water/rock tiles, resource tracking, diffusion, and rendering.
"""
import numpy as np
import pygame
from typing import List, Tuple, Optional


class SoupEnv:
    """
    A grid-based environment with:
    - Tile types: 0=water (blue), 1=rock (dark gray)
    - Resource R[i,j] tracked per water tile
    - Resource diffusion and decay
    - Pygame rendering with HUD overlay
    """
    def __init__(self, width: int, height: int, tile_size: int,
                 r_max: float = 1.0, diffusion_rate: float = 0.05,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.grid_w = width // tile_size
        self.grid_h = height // tile_size

        # Resource parameters
        self.r_max = r_max
        self.diffusion_rate = diffusion_rate

        # Tile grid: 0=water, 1=rock
        self.tile_kind = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

        # Resource grid: [0, r_max]
        self.resources = np.full((self.grid_h, self.grid_w), r_max * 0.9, dtype=np.float32)

        # RNG
        self.rng = np.random.default_rng(seed)

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Primordial Soup - Phase 0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 20)

        # Colors
        self.WATER = (30, 60, 120)
        self.ROCK = (60, 60, 60)
        self.AGENT_COLORS = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255)
        ]

        # Rendering flags
        self.show_resource_heatmap = False

        # Mass conservation tracking
        self.total_resource_spent = 0.0
        self.total_resource_returned = 0.0
        self.total_spawn_cost = 0.0

        # Phase 0 throttles: per-tile birth cooldown
        self.birth_cooldown = np.zeros((self.grid_h, self.grid_w), dtype=np.int16)

    def init_terrain(self, rock_density: float = 0.15):
        """Initialize terrain with random rock islands"""
        # Simple noise-like rock placement
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                if self.rng.random() < rock_density:
                    self.tile_kind[i, j] = 1
                    self.resources[i, j] = 0.0
                else:
                    self.tile_kind[i, j] = 0

        # Ensure there's at least one water tile at center
        center_y, center_x = self.grid_h // 2, self.grid_w // 2
        self.tile_kind[center_y, center_x] = 0
        self.resources[center_y, center_x] = self.r_max * 0.9

    def is_water(self, grid_x: int, grid_y: int) -> bool:
        """Check if tile is water (traversable)"""
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            return self.tile_kind[grid_y, grid_x] == 0
        return False

    def get_neighbors(self, grid_x: int, grid_y: int) -> List[Tuple[int, int]]:
        """Get 4-connected neighbors (up, down, left, right)"""
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = grid_x + dx, grid_y + dy
            if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                neighbors.append((nx, ny))
        return neighbors

    def diffuse_resources(self):
        """
        Mass-conserving diffusion over WATER ONLY with no-flux at rocks/borders.
        Rocks act as mirrors (no-flux), not black holes.
        """
        R = self.resources
        water = (self.tile_kind == 0).astype(np.float32)

        # Pad both R and water mask
        Rp = np.pad(R, 1, mode="edge")
        Mp = np.pad(water, 1, mode="edge")

        # Center and 4 neighbors with masks
        C = Rp[1:-1, 1:-1]
        N = Rp[0:-2, 1:-1]; MN = Mp[0:-2, 1:-1]
        S = Rp[2:,   1:-1]; MS = Mp[2:,   1:-1]
        E = Rp[1:-1, 2:  ]; ME = Mp[1:-1, 2:  ]
        W = Rp[1:-1, 0:-2]; MW = Mp[1:-1, 0:-2]

        # No-flux: if neighbor is rock, use center value (mirror condition)
        N_eff = np.where(MN > 0, N, C)
        S_eff = np.where(MS > 0, S, C)
        E_eff = np.where(ME > 0, E, C)
        W_eff = np.where(MW > 0, W, C)

        # Laplacian
        lap = (N_eff + S_eff + E_eff + W_eff) - 4.0 * C

        # Update only water tiles
        R_new = C + self.diffusion_rate * lap

        # Rocks hold zero resource, water keeps updated values
        R_new = np.where(self.tile_kind == 0, R_new, 0.0)
        self.resources = R_new.astype(np.float32)

    def add_resource(self, grid_x: int, grid_y: int, amount: float):
        """
        Add resource to a tile (e.g., from decomposition or refunds).
        NO clamping in Phase 0 - that would leak mass on refunds.
        """
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            if self.tile_kind[grid_y, grid_x] == 0:
                self.resources[grid_y, grid_x] += amount  # No clamp in Phase 0

    def consume_resource(self, grid_x: int, grid_y: int, amount: float) -> bool:
        """Try to consume resource from a tile. Returns True if successful."""
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            if self.tile_kind[grid_y, grid_x] == 0 and self.resources[grid_y, grid_x] >= amount:
                self.resources[grid_y, grid_x] -= amount
                return True
        return False

    def get_resource(self, grid_x: int, grid_y: int) -> float:
        """Get resource amount at tile"""
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            return self.resources[grid_y, grid_x]
        return 0.0

    def get_total_resource(self) -> float:
        """Get total resource in entire grid (mass conservation check)"""
        return float(np.sum(self.resources))

    def tick_cooldowns(self):
        """Decrement birth cooldown timers"""
        self.birth_cooldown[self.birth_cooldown > 0] -= 1

    def can_birth_at(self, grid_x: int, grid_y: int) -> bool:
        """Check if tile is off cooldown for births"""
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            return self.birth_cooldown[grid_y, grid_x] == 0
        return False

    def set_birth_cooldown(self, grid_x: int, grid_y: int, ticks: int = 10):
        """Set birth cooldown on a tile"""
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            self.birth_cooldown[grid_y, grid_x] = ticks

    def contract_resources(self, factor: float = 0.98):
        """
        Contract global resource max (deliberate famine mechanism).
        Returns the amount of mass REMOVED from the system.
        In Phase 0, this should NOT be called to maintain flat Avg R.
        """
        old_sum = self.get_total_resource()
        self.r_max *= factor
        # Clamp resources to new max (this REMOVES mass deliberately)
        self.resources = np.clip(self.resources, 0.0, self.r_max)
        new_sum = self.get_total_resource()
        mass_removed = old_sum - new_sum
        return mass_removed

    def get_random_water_tile(self) -> Optional[Tuple[int, int]]:
        """Get random water tile position (grid coords)"""
        water_tiles = [(j, i) for i in range(self.grid_h) for j in range(self.grid_w)
                       if self.tile_kind[i, j] == 0]
        if len(water_tiles) == 0:
            return None
        idx = self.rng.integers(0, len(water_tiles))
        return water_tiles[idx]

    def sample_poisson_disk(self, num_samples: int, min_dist: float = 8.0) -> List[Tuple[int, int]]:
        """
        Sample water tile positions using Poisson disk sampling for spatial diversity.
        Ensures spawns are at least min_dist apart.
        Phase 1 feature for founder spawn positions.
        """
        water_tiles = [(j, i) for i in range(self.grid_h) for j in range(self.grid_w)
                       if self.tile_kind[i, j] == 0]
        if len(water_tiles) == 0:
            return []

        samples = []
        max_attempts = num_samples * 100  # Give up after too many attempts
        attempts = 0

        # Start with a random water tile
        if len(water_tiles) > 0:
            idx = self.rng.integers(0, len(water_tiles))
            samples.append(water_tiles[idx])

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Pick a random water tile
            idx = self.rng.integers(0, len(water_tiles))
            candidate = water_tiles[idx]

            # Check distance to all existing samples
            valid = True
            for sample in samples:
                dx = candidate[0] - sample[0]
                dy = candidate[1] - sample[1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                samples.append(candidate)

        return samples

    def render(self, agents, tick: int, repl_rate: float, avg_r: float,
               ola_mut_rate: float, ola_loss: float, total_r: float = 0.0, mass_error: float = 0.0,
               simpson_diversity: float = 0.0, num_lineages: int = 0, RoR_ema: float = 0.0):
        """Render the environment with agents and HUD (Phase 1: adds diversity metrics)"""
        self.screen.fill((0, 0, 0))

        # Draw tiles
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                x = j * self.tile_size
                y = i * self.tile_size

                if self.tile_kind[i, j] == 1:
                    # Rock
                    pygame.draw.rect(self.screen, self.ROCK, (x, y, self.tile_size, self.tile_size))
                else:
                    # Water
                    if self.show_resource_heatmap:
                        # Color by resource level
                        r_norm = self.resources[i, j] / max(self.r_max, 0.01)
                        intensity = int(255 * r_norm)
                        color = (0, 0, intensity)
                    else:
                        color = self.WATER
                    pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))

        # Draw agents
        for agent in agents:
            px = agent.pos[0] * self.tile_size + self.tile_size // 2
            py = agent.pos[1] * self.tile_size + self.tile_size // 2
            color = self.AGENT_COLORS[agent.lineage_depth % len(self.AGENT_COLORS)]
            pygame.draw.circle(self.screen, color, (px, py), max(2, self.tile_size // 3))

            # Draw parent link (faint line for ~2 seconds)
            if agent.parent_pos is not None and agent.birth_tick + 120 > tick:
                parent_px = agent.parent_pos[0] * self.tile_size + self.tile_size // 2
                parent_py = agent.parent_pos[1] * self.tile_size + self.tile_size // 2
                faint_color = tuple(c // 2 for c in color)
                pygame.draw.line(self.screen, faint_color, (px, py), (parent_px, parent_py), 1)

        # HUD
        hud_lines = [
            f"Tick: {tick}",
            f"Agents: {len(agents)}",
            f"Repl/min: {repl_rate:.2f}",
            f"Avg R: {avg_r:.3f}",
            f"Total R: {total_r:.1f}",
            f"Mass Err: {mass_error:.2e}",
            f"OLA mut: {ola_mut_rate:.4f}"
        ]

        # Phase 1: Add diversity metrics to HUD
        if num_lineages > 0:
            hud_lines.extend([
                f"Lineages: {num_lineages}",
                f"Diversity: {simpson_diversity:.3f}",
                f"RoR EMA: {RoR_ema:.3f}"
            ])

        y_offset = 10
        for line in hud_lines:
            text_surf = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surf, (10, y_offset))
            y_offset += 22

        # Controls hint
        controls = "Space:pause R:heatmap C:contract G:spawn10 S:save"
        text_surf = self.font.render(controls, True, (200, 200, 200))
        self.screen.blit(text_surf, (10, self.height - 25))

        pygame.display.flip()

    def tick_frame(self, fps: int = 60):
        """Tick the clock and return delta time"""
        return self.clock.tick(fps)

    def get_observation_stencil(self, grid_x: int, grid_y: int, agents_list) -> np.ndarray:
        """
        Build a 3x3 stencil observation around (grid_x, grid_y).
        Returns a fixed-size observation vector (16 dims).

        Observation:
        - 9 floats: normalized resource at center and 8 neighbors (3x3)
        - 1 float: local population density (agents in 3x3 / 9)
        - 1 float: tile type at center (0=water, 1=rock, normalized to [0,1])
        - 1 float: bias term (1.0)
        - 4 floats: padding (reserved for future features)
        """
        obs = np.zeros(16, dtype=np.float32)

        # Gather 3x3 resource values
        idx = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx, ny = grid_x + dx, grid_y + dy
                if 0 <= ny < self.grid_h and 0 <= nx < self.grid_w:
                    obs[idx] = self.resources[ny, nx] / max(self.r_max, 0.01)
                else:
                    obs[idx] = 0.0
                idx += 1

        # Population density in 3x3
        count = 0
        for agent in agents_list:
            ax, ay = agent.pos
            if abs(ax - grid_x) <= 1 and abs(ay - grid_y) <= 1:
                count += 1
        obs[9] = count / 9.0

        # Tile type at center
        if 0 <= grid_y < self.grid_h and 0 <= grid_x < self.grid_w:
            obs[10] = float(self.tile_kind[grid_y, grid_x])
        else:
            obs[10] = 1.0  # treat out-of-bounds as rock

        # Bias
        obs[11] = 1.0

        # Padding (reserved for future features)
        obs[12:16] = 0.0

        return obs
