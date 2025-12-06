"""
Real-time genome network visualizer using pygame
Shows genomes as nodes with trust-based coloring and similarity-based edges
"""
import pygame
import numpy as np
import math
from typing import List, Dict, Tuple
from ola_genome import OLAGenome
from lsh_genome_database import LSHGenomeDatabase


class GenomeVisualizer:
    """
    Visualizes genome population as a network graph
    - Node color: trust_score (blue = low, red = high)
    - Node size: mutation_count
    - Edges: LSH similarity between genomes
    """

    def __init__(self, width: int = 1200, height: int = 800, fps: int = 30):
        """
        Initialize visualizer

        Args:
            width: Window width
            height: Window height
            fps: Target frames per second
        """
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps

        # Create window
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("OLA Genome Evolution Visualizer")

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Colors
        self.bg_color = (20, 20, 30)
        self.text_color = (200, 200, 200)
        self.edge_color = (60, 60, 80)

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Genome positions (circular layout)
        self.genome_positions: Dict[int, Tuple[float, float]] = {}

        # Running state
        self.running = True

    def compute_positions(self, genomes: List[OLAGenome]):
        """
        Compute circular layout positions for genomes

        Args:
            genomes: List of genomes to position
        """
        n = len(genomes)
        if n == 0:
            return

        # Circular layout
        center_x = self.width // 2
        center_y = self.height // 2
        radius = min(self.width, self.height) * 0.35

        for i, genome in enumerate(genomes):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.genome_positions[genome.genome_id] = (x, y)

    def trust_to_color(self, trust: float) -> Tuple[int, int, int]:
        """
        Convert trust score to color
        Low trust (0.0) = Blue
        High trust (1.0) = Red

        Args:
            trust: Trust score in [0, 1]

        Returns:
            (r, g, b) color tuple
        """
        # Blue to Red gradient
        trust = max(0.0, min(1.0, trust))

        if trust < 0.5:
            # Blue to Green
            t = trust * 2
            r = int(0)
            g = int(100 + 155 * t)
            b = int(255 - 155 * t)
        else:
            # Green to Red
            t = (trust - 0.5) * 2
            r = int(155 * t)
            g = int(255 - 155 * t)
            b = int(0)

        return (r, g, b)

    def mutation_to_size(self, mutation_count: int, min_size: int = 8, max_size: int = 30) -> int:
        """
        Convert mutation count to node size

        Args:
            mutation_count: Number of mutations
            min_size: Minimum node radius
            max_size: Maximum node radius

        Returns:
            radius: Node radius in pixels
        """
        # Logarithmic scale
        size = min_size + min(mutation_count * 2, max_size - min_size)
        return int(size)

    def draw_edges(self, genomes: List[OLAGenome], lsh_db: LSHGenomeDatabase,
                   similarity_threshold: float = 0.6):
        """
        Draw edges between similar genomes

        Args:
            genomes: List of genomes
            lsh_db: LSH database with similarity information
            similarity_threshold: Minimum similarity to draw edge
        """
        genome_ids = [g.genome_id for g in genomes]
        similarities = lsh_db.get_all_similarities(genome_ids)

        for (id1, id2), sim in similarities.items():
            if sim < similarity_threshold:
                continue

            if id1 not in self.genome_positions or id2 not in self.genome_positions:
                continue

            pos1 = self.genome_positions[id1]
            pos2 = self.genome_positions[id2]

            # Edge thickness based on similarity
            thickness = int(1 + sim * 3)

            # Edge color with alpha based on similarity
            alpha = int(50 + sim * 150)
            color = (80, 80, 120, alpha)

            # Draw line
            pygame.draw.line(self.screen, color[:3], pos1, pos2, thickness)

    def draw_nodes(self, genomes: List[OLAGenome]):
        """
        Draw genome nodes

        Args:
            genomes: List of genomes to draw
        """
        for genome in genomes:
            if genome.genome_id not in self.genome_positions:
                continue

            x, y = self.genome_positions[genome.genome_id]

            # Node color based on trust
            color = self.trust_to_color(genome.stats.trust_score)

            # Node size based on mutations
            radius = self.mutation_to_size(genome.stats.mutation_count)

            # Draw node
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

            # Draw border
            border_color = (255, 255, 255) if genome.stats.trust_score > 0.7 else (100, 100, 100)
            pygame.draw.circle(self.screen, border_color, (int(x), int(y)), radius, 2)

            # Draw genome ID
            id_text = self.small_font.render(str(genome.genome_id), True, (255, 255, 255))
            text_rect = id_text.get_rect(center=(int(x), int(y)))
            self.screen.blit(id_text, text_rect)

    def draw_stats(self, genomes: List[OLAGenome], library_stats: Dict, tick: int):
        """
        Draw statistics panel

        Args:
            genomes: List of genomes
            library_stats: Library statistics
            tick: Current tick number
        """
        y_offset = 10

        # Tick counter
        tick_text = self.font.render(f"Tick: {tick}", True, self.text_color)
        self.screen.blit(tick_text, (10, y_offset))
        y_offset += 30

        # Library stats
        stats_lines = [
            f"Genomes: {library_stats['total_genomes']}",
            f"Avg Trust: {library_stats['avg_trust']:.3f}",
            f"Min Trust: {library_stats['min_trust']:.3f}",
            f"Max Trust: {library_stats['max_trust']:.3f}",
            f"Total Mutations: {library_stats['total_mutations']}",
            f"Pruned: {library_stats.get('total_pruned', 0)}",
            f"Avg Mutations: {library_stats['avg_mutation_count']:.1f}",
            f"Avg Consistency: {library_stats['avg_consistency']:.3f}",
        ]

        for line in stats_lines:
            text = self.small_font.render(line, True, self.text_color)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

        # Legend
        y_offset += 20
        legend_text = self.font.render("Legend:", True, self.text_color)
        self.screen.blit(legend_text, (10, y_offset))
        y_offset += 25

        legend_items = [
            ("Color: Blue (low trust) -> Red (high trust)", None),
            ("Size: Mutation count", None),
            ("Edges: Genome similarity", None),
        ]

        for text, color in legend_items:
            rendered = self.small_font.render(text, True, self.text_color)
            self.screen.blit(rendered, (10, y_offset))
            y_offset += 20

    def draw_minimal_metrics(self, minimal_metrics: Dict):
        """Draw minimal metrics and health flag label (color-coded)."""
        y = 10
        x = self.width - 10
        # Static lines first
        base_lines = [
            "Minimal metrics:",
            f"avg_trust={minimal_metrics.get('avg_trust', 0):.3f}",
            f"trust_std={minimal_metrics.get('trust_std', 0):.3f}",
            f"avg_consistency_ema={minimal_metrics.get('avg_consistency_ema', 0):.3f}",
            f"mutations_in_window={minimal_metrics.get('mutations_in_window', 0)}",
            f"nn_similarity_mean={minimal_metrics.get('nn_similarity_mean', 0):.3f}",
            f"cpu_var={minimal_metrics.get('cpu_var', 0):.4f}",
            f"ram_var={minimal_metrics.get('ram_var', 0):.4f}",
        ]

        # Optional predicted MSE EMA if available
        if 'pred_mse_ema' in minimal_metrics:
            try:
                base_lines.append(f"pred_mse_ema={float(minimal_metrics.get('pred_mse_ema', 0)):.4f}")
            except Exception:
                base_lines.append("pred_mse_ema=?")
        if 'pred_rel_ema' in minimal_metrics:
            try:
                base_lines.append(f"rel_fit_ema={float(minimal_metrics.get('pred_rel_ema', 0)):.4f}")
            except Exception:
                base_lines.append("rel_fit_ema=?")
        if 'fitness_ema' in minimal_metrics:
            try:
                base_lines.append(f"fitness_ema={float(minimal_metrics.get('fitness_ema', 0)):.4f}")
            except Exception:
                base_lines.append("fitness_ema=?")

        for line in base_lines:
            surf = self.small_font.render(line, True, self.text_color)
            rect = surf.get_rect()
            rect.topright = (x, y)
            self.screen.blit(surf, rect)
            y += 18

        # Healthy line with color
        healthy = bool(minimal_metrics.get('healthy', False))
        healthy_text = f"Healthy={'YES' if healthy else 'NO'}"
        color = (0, 200, 0) if healthy else (220, 60, 60)
        surf = self.small_font.render(healthy_text, True, color)
        rect = surf.get_rect()
        rect.topright = (x, y)
        self.screen.blit(surf, rect)
        y += 18

        # Optional reason when unhealthy
        if not healthy:
            reason = minimal_metrics.get('health_reason')
            if reason:
                surf = self.small_font.render(f"Reason: {reason}", True, (220, 60, 60))
                rect = surf.get_rect()
                rect.topright = (x, y)
                self.screen.blit(surf, rect)
                y += 18

    def update(self, genomes: List[OLAGenome], lsh_db: LSHGenomeDatabase,
               library_stats: Dict, tick: int, minimal_metrics: Dict | None = None,
               roi_rect: tuple[int, int, int, int] | None = None) -> bool:
        """
        Update visualization

        Args:
            genomes: List of genomes
            lsh_db: LSH database
            library_stats: Library statistics
            tick: Current tick number

        Returns:
            running: True if window is still open
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

        if not self.running:
            return False

        # Clear screen
        self.screen.fill(self.bg_color)

        # Compute positions
        self.compute_positions(genomes)

        # Draw edges first (behind nodes)
        self.draw_edges(genomes, lsh_db)

        # Draw nodes
        self.draw_nodes(genomes)

        # Draw stats
        self.draw_stats(genomes, library_stats, tick)

        # Draw minimal metrics panel if provided
        if minimal_metrics is not None:
            self.draw_minimal_metrics(minimal_metrics)

        # Draw ROI rectangle overlay last (debug)
        if roi_rect is not None:
            try:
                x, y, w, h = roi_rect
                pygame.draw.rect(self.screen, (0, 255, 0), (int(x), int(y), int(w), int(h)), 1)
            except Exception:
                pass

        # Update display
        pygame.display.flip()

        # Cap framerate
        self.clock.tick(self.fps)

        return self.running

    def close(self):
        """Close the visualizer"""
        pygame.quit()

    def __del__(self):
        """Cleanup on deletion"""
        try:
            pygame.quit()
        except:
            pass
