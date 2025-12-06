"""
Competitive OLA Pygame GUI
Real-time visualization of dual OLA agents competing for territory control.
"""
import sys
import time
import numpy as np
import pygame
from typing import Dict, Tuple
from dataclasses import dataclass

from competitive_ola import CompetitiveOLA, CompetitiveOLAConfig


@dataclass
class VisualizationConfig:
    """GUI configuration"""
    window_width: int = 1600
    window_height: int = 900
    grid_display_size: int = 640  # Size of grid visualization area
    target_fps: int = 30
    cell_size: int = 10  # Pixels per grid cell

    # Colors
    bg_color: Tuple[int, int, int] = (20, 20, 25)
    grid_color: Tuple[int, int, int] = (40, 40, 50)
    agent_0_color: Tuple[int, int, int] = (50, 150, 255)  # Blue
    agent_1_color: Tuple[int, int, int] = (255, 100, 50)  # Orange
    empty_color: Tuple[int, int, int] = (30, 30, 35)
    text_color: Tuple[int, int, int] = (200, 200, 200)
    highlight_color: Tuple[int, int, int] = (0, 255, 100)


class CompetitiveOLAGUI:
    """
    Real-time pygame visualization of competitive OLA evolution.

    Shows:
    - Live territory grid with agent positions
    - Real-time metrics (territory, fitness, mutation rates)
    - Dominance balance chart
    - Strategy diversity metrics
    - Co-evolution indicators
    """

    def __init__(self, config: VisualizationConfig):
        self.cfg = config

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((config.window_width, config.window_height))
        pygame.display.set_caption("Competitive OLA - Painting Battle (Goal: 100% Coverage)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        self.font_large = pygame.font.Font(None, 32)

        # Initialize competitive OLA system
        ola_config = CompetitiveOLAConfig(
            grid_width=64,
            grid_height=64,
            state_dim=128,
            action_dim=6,
            observation_dim=128,
            pop_size=32,
            elite_frac=0.15,
            mutation_rate=0.1,
            structure_add_prob=0.01,
            structure_max_dim=512,
            grow_prob=0.005,
            reward_scale=10.0,
            action_penalty=0.001,
            competitive_bonus=0.5,
            device="cuda"
        )
        self.ola_system = CompetitiveOLA(ola_config)

        # UI state
        self.running = True
        self.paused = False
        self.step_count = 0
        self.fps_actual = 0

        # Metrics tracking for graphs
        self.territory_history = [[], []]  # Track territory over time
        self.fitness_history = [[], []]  # Track fitness over time
        self.max_history_length = 200

        # Checkpoint settings
        self.checkpoint_interval = 500  # Save every N steps
        self.last_checkpoint = 0
        self.checkpoint_dir = "E:\\Genome\\competitive_checkpoints"
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def run(self):
        """Main game loop"""
        last_time = time.time()

        while self.running:
            dt = time.time() - last_time
            last_time = time.time()

            # Handle events
            self._handle_events()

            if not self.paused:
                # Step OLA system
                metrics = self.ola_system.step()
                self.step_count += 1

                # Update history
                self._update_history(metrics)

                # Checkpoint
                if self.step_count - self.last_checkpoint >= self.checkpoint_interval:
                    self._save_checkpoint(metrics)
                    self.last_checkpoint = self.step_count

                # Render
                self._render(metrics)
            else:
                # Just render current state when paused
                self._render_paused()

            # Update display
            pygame.display.flip()

            # Maintain target FPS
            self.clock.tick(self.cfg.target_fps)
            self.fps_actual = self.clock.get_fps()

        pygame.quit()

    def _handle_events(self):
        """Handle keyboard and window events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"[GUI] {'Paused' if self.paused else 'Resumed'}")

                elif event.key == pygame.K_r:
                    print("[GUI] Resetting environment...")
                    self.ola_system.reset()
                    self.step_count = 0
                    self.territory_history = [[], []]
                    self.fitness_history = [[], []]

                elif event.key == pygame.K_s:
                    # Manual save
                    print("[GUI] Saving champions...")
                    self._save_checkpoint({})

    def _update_history(self, metrics: Dict):
        """Update metric history for graphs"""
        # Territory history
        territory_counts = metrics['env_info']['territory_counts']
        for agent_id in range(2):
            self.territory_history[agent_id].append(territory_counts[agent_id])
            if len(self.territory_history[agent_id]) > self.max_history_length:
                self.territory_history[agent_id].pop(0)

        # Fitness history
        for agent_id in range(2):
            fitness = metrics[f'agent_{agent_id}']['competitive_fitness']
            self.fitness_history[agent_id].append(fitness)
            if len(self.fitness_history[agent_id]) > self.max_history_length:
                self.fitness_history[agent_id].pop(0)

    def _render(self, metrics: Dict):
        """Render the complete UI"""
        self.screen.fill(self.cfg.bg_color)

        # Layout: Left = Grid, Right = Metrics
        grid_x = 50
        grid_y = 50
        metrics_x = grid_x + self.cfg.grid_display_size + 50
        metrics_y = 50

        # Render grid
        self._render_grid(grid_x, grid_y, metrics)

        # Render metrics
        self._render_metrics(metrics_x, metrics_y, metrics)

        # Render graphs
        self._render_graphs(metrics_x, metrics_y + 400)

        # Render controls
        self._render_controls(50, self.cfg.window_height - 80)

    def _render_grid(self, x: int, y: int, metrics: Dict):
        """Render the territory grid with agent positions"""
        viz_data = self.ola_system.get_grid_visualization_data()
        grid = viz_data['grid']
        positions = viz_data['positions']
        radii = viz_data['radii']
        velocities = viz_data['velocities']

        # Calculate cell size to fit display
        grid_h, grid_w = grid.shape
        cell_size = min(
            self.cfg.grid_display_size // grid_w,
            self.cfg.grid_display_size // grid_h
        )

        # Draw grid cells
        for i in range(grid_h):
            for j in range(grid_w):
                cell_value = grid[i, j]

                if cell_value == 1:
                    color = self.cfg.agent_0_color
                elif cell_value == 2:
                    color = self.cfg.agent_1_color
                else:
                    color = self.cfg.empty_color

                rect = pygame.Rect(
                    x + j * cell_size,
                    y + i * cell_size,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(self.screen, color, rect)

                # Grid lines
                pygame.draw.rect(self.screen, self.cfg.grid_color, rect, 1)

        # Draw agent positions and radii
        for agent_id in range(2):
            pos = positions[agent_id]
            radius = radii[agent_id]
            vel = velocities[agent_id]

            # Agent center
            center_x = x + int(pos[0] * cell_size)
            center_y = y + int(pos[1] * cell_size)

            # Agent color
            color = self.cfg.agent_0_color if agent_id == 0 else self.cfg.agent_1_color

            # Draw radius circle
            pygame.draw.circle(
                self.screen,
                color,
                (center_x, center_y),
                int(radius * cell_size),
                2
            )

            # Draw agent center dot
            pygame.draw.circle(
                self.screen,
                self.cfg.highlight_color,
                (center_x, center_y),
                5
            )

            # Draw velocity vector
            if np.linalg.norm(vel) > 0.1:
                vel_end_x = center_x + int(vel[0] * cell_size * 3)
                vel_end_y = center_y + int(vel[1] * cell_size * 3)
                pygame.draw.line(
                    self.screen,
                    self.cfg.highlight_color,
                    (center_x, center_y),
                    (vel_end_x, vel_end_y),
                    2
                )

        # Draw border
        border_rect = pygame.Rect(x, y, grid_w * cell_size, grid_h * cell_size)
        pygame.draw.rect(self.screen, self.cfg.highlight_color, border_rect, 3)

        # Title
        title = self.font_large.render("Painting Battle Grid", True, self.cfg.text_color)
        self.screen.blit(title, (x, y - 40))

        # Coverage progress bar below grid
        bar_y = y + grid_h * cell_size + 20
        bar_width = grid_w * cell_size
        bar_height = 30

        # Background (unpainted)
        pygame.draw.rect(self.screen, self.cfg.empty_color,
                        (x, bar_y, bar_width, bar_height))

        # Calculate coverage
        total_tiles = grid_h * grid_w
        coverage_0 = metrics['env_info']['territory_counts'][0] / total_tiles
        coverage_1 = metrics['env_info']['territory_counts'][1] / total_tiles

        # Agent 0 bar (blue)
        bar_0_width = int(bar_width * coverage_0)
        if bar_0_width > 0:
            pygame.draw.rect(self.screen, self.cfg.agent_0_color,
                           (x, bar_y, bar_0_width, bar_height))

        # Agent 1 bar (orange) - starts after agent 0's bar
        bar_1_width = int(bar_width * coverage_1)
        if bar_1_width > 0:
            pygame.draw.rect(self.screen, self.cfg.agent_1_color,
                           (x + bar_0_width, bar_y, bar_1_width, bar_height))

        # Border
        pygame.draw.rect(self.screen, self.cfg.text_color,
                        (x, bar_y, bar_width, bar_height), 2)

        # Percentage labels
        total_painted = (coverage_0 + coverage_1) * 100
        label = self.font_small.render(
            f"Total Painted: {total_painted:.1f}% | Blue: {coverage_0*100:.1f}% | Orange: {coverage_1*100:.1f}%",
            True, self.cfg.text_color
        )
        self.screen.blit(label, (x, bar_y + bar_height + 5))

    def _render_metrics(self, x: int, y: int, metrics: Dict):
        """Render performance metrics"""
        # Calculate coverage percentages
        total_tiles = 64 * 64  # Grid size
        coverage_0 = (metrics['env_info']['territory_counts'][0] / total_tiles) * 100
        coverage_1 = (metrics['env_info']['territory_counts'][1] / total_tiles) * 100
        unpainted = 100 - coverage_0 - coverage_1

        lines = [
            "=== PAINTING COMPETITION ===",
            f"Step: {self.step_count}",
            f"FPS: {self.fps_actual:.1f}",
            f"Unpainted: {unpainted:.1f}%",
            "",
            "--- Agent 0 (Blue) ---",
            f"Territory: {metrics['env_info']['territory_counts'][0]} tiles",
            f"Coverage: {coverage_0:.1f}%",
            f"Expansion Rate: {metrics['env_info']['expansion_rates'][0]:.2f}",
            f"Fitness: {metrics['agent_0']['competitive_fitness']:.2f}",
            f"State Dim: {int(metrics['agent_0']['state_dim'])}",
            f"Mutation Rate: {metrics['agent_0']['mutation_rate']:.6f}",
            f"Win Count: {metrics['win_counts'][0]}",
            "",
            "--- Agent 1 (Orange) ---",
            f"Territory: {metrics['env_info']['territory_counts'][1]} tiles",
            f"Coverage: {coverage_1:.1f}%",
            f"Expansion Rate: {metrics['env_info']['expansion_rates'][1]:.2f}",
            f"Fitness: {metrics['agent_1']['competitive_fitness']:.2f}",
            f"State Dim: {int(metrics['agent_1']['state_dim'])}",
            f"Mutation Rate: {metrics['agent_1']['mutation_rate']:.6f}",
            f"Win Count: {metrics['win_counts'][1]}",
            "",
            "--- Co-Evolution Stats ---",
            f"Dominance Balance: {metrics['dominance_balance']:.3f}",
            f"Strategy Diversity: {metrics['avg_diversity']:.4f}",
            f"Total Painted: {coverage_0 + coverage_1:.1f}%",
        ]

        for i, line in enumerate(lines):
            # Color-code agent sections
            if "Agent 0" in line or (i >= 5 and i < 12):
                color = self.cfg.agent_0_color
            elif "Agent 1" in line or (i >= 13 and i < 20):
                color = self.cfg.agent_1_color
            else:
                color = self.cfg.text_color

            text_surface = self.font_small.render(line, True, color)
            self.screen.blit(text_surface, (x, y + i * 22))

    def _render_graphs(self, x: int, y: int):
        """Render historical graphs"""
        graph_width = 500
        graph_height = 150

        # Territory over time
        self._render_line_graph(
            x, y,
            graph_width, graph_height,
            self.territory_history,
            ["Agent 0 Territory", "Agent 1 Territory"],
            [self.cfg.agent_0_color, self.cfg.agent_1_color],
            "Territory Over Time"
        )

        # Fitness over time
        self._render_line_graph(
            x, y + graph_height + 50,
            graph_width, graph_height,
            self.fitness_history,
            ["Agent 0 Fitness", "Agent 1 Fitness"],
            [self.cfg.agent_0_color, self.cfg.agent_1_color],
            "Fitness Over Time"
        )

    def _render_line_graph(self, x: int, y: int, width: int, height: int,
                          data_series: list, labels: list, colors: list, title: str):
        """Render a multi-line graph"""
        # Background
        pygame.draw.rect(self.screen, (30, 30, 40), (x, y, width, height))
        pygame.draw.rect(self.screen, self.cfg.grid_color, (x, y, width, height), 2)

        # Title
        title_surface = self.font.render(title, True, self.cfg.text_color)
        self.screen.blit(title_surface, (x + 10, y - 25))

        # Find global min/max for scaling
        all_values = [val for series in data_series for val in series if series]
        if not all_values:
            return

        min_val = min(all_values)
        max_val = max(all_values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Draw each series
        for series, label, color in zip(data_series, labels, colors):
            if len(series) < 2:
                continue

            points = []
            for i, value in enumerate(series):
                px = x + (i / (len(series) - 1)) * width
                normalized = (value - min_val) / val_range
                py = y + height - (normalized * height)
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(self.screen, color, False, points, 2)

        # Legend
        for i, (label, color) in enumerate(zip(labels, colors)):
            legend_x = x + width - 150
            legend_y = y + 10 + i * 25
            pygame.draw.line(self.screen, color, (legend_x, legend_y + 8),
                           (legend_x + 30, legend_y + 8), 3)
            label_surface = self.font_small.render(label, True, self.cfg.text_color)
            self.screen.blit(label_surface, (legend_x + 40, legend_y))

        # Axis labels
        min_label = self.font_small.render(f"{min_val:.1f}", True, self.cfg.text_color)
        max_label = self.font_small.render(f"{max_val:.1f}", True, self.cfg.text_color)
        self.screen.blit(max_label, (x + 5, y + 5))
        self.screen.blit(min_label, (x + 5, y + height - 20))

    def _render_controls(self, x: int, y: int):
        """Render control instructions"""
        controls = [
            "[SPACE] Pause/Resume | [R] Reset | [S] Save Champions | [ESC] Quit"
        ]

        for i, text in enumerate(controls):
            surface = self.font.render(text, True, self.cfg.highlight_color)
            self.screen.blit(surface, (x, y + i * 30))

    def _render_paused(self):
        """Render paused overlay"""
        overlay = pygame.Surface((self.cfg.window_width, self.cfg.window_height))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))

        pause_text = self.font_large.render("PAUSED", True, self.cfg.highlight_color)
        text_rect = pause_text.get_rect(center=(self.cfg.window_width // 2,
                                                 self.cfg.window_height // 2))
        self.screen.blit(pause_text, text_rect)

    def _save_checkpoint(self, metrics: Dict):
        """Save champion genomes"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        metadata = {
            'step': self.step_count,
            'timestamp': timestamp,
        }

        if metrics:
            metadata.update({
                'territory_0': int(metrics['env_info']['territory_counts'][0]),
                'territory_1': int(metrics['env_info']['territory_counts'][1]),
                'wins_0': int(metrics['win_counts'][0]),
                'wins_1': int(metrics['win_counts'][1]),
                'dominance_balance': float(metrics['dominance_balance']),
            })

        import os
        path_prefix = os.path.join(self.checkpoint_dir, f"champion_step{self.step_count:06d}_{timestamp}")
        self.ola_system.save_champions(path_prefix, metadata)

        print(f"[CHECKPOINT] Saved at step {self.step_count}")


def main():
    """Entry point"""
    config = VisualizationConfig()
    gui = CompetitiveOLAGUI(config)
    gui.run()


if __name__ == "__main__":
    main()
