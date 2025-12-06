"""
Genome Visualization for OLA Evolution
Displays genome structure at three levels of abstraction:
1. Topological: nodes = neurons, edges = connections
2. Genetic: instruction sequence view
3. Functional: color-coded by operation type
"""

import pygame
import numpy as np
import torch
import networkx as nx
from typing import Dict, Tuple, List, Optional
from collections import deque
from ola import EvoCell


class GenomeVisualizer:
    """Visualize OLA genome structure with multiple abstraction levels"""

    # Visualization modes
    MODE_TOPOLOGY = 0
    MODE_GENETIC = 1
    MODE_FUNCTIONAL = 2
    MODE_NAMES = ["Topology", "Genetic Sequence", "Functional Clusters"]

    def __init__(self, width: int = 600, height: int = 600):
        """
        Initialize genome visualizer

        Args:
            width: Visualization window width
            height: Visualization window height
        """
        self.width = width
        self.height = height
        self.mode = self.MODE_TOPOLOGY

        # Colors
        self.BG_COLOR = (20, 20, 30)
        self.TEXT_COLOR = (220, 220, 220)
        self.NODE_COLORS = {
            'input': (100, 200, 255),      # Light blue
            'hidden': (150, 255, 150),     # Light green
            'output': (255, 150, 100),     # Light orange
            'state': (255, 200, 100),      # Yellow
            'gate': (255, 100, 200),       # Pink
        }
        self.EDGE_COLOR = (80, 80, 100, 128)
        self.ACTIVE_COLOR = (255, 255, 100)
        self.ACTIVE_PATH_COLOR = (100, 255, 100)  # Bright green for active paths
        self.DELTA_POSITIVE_COLOR = (100, 255, 100)  # Green for parameter increase
        self.DELTA_NEGATIVE_COLOR = (255, 100, 100)  # Red for parameter decrease

        # Activation tracking (for dynamic highlighting)
        self.activation_history = {}  # node_id -> deque of recent activations
        self.history_length = 10

        # Parameter delta tracking
        self.prev_params = {}  # param_name -> previous tensor clone
        self.param_deltas = {}  # param_name -> delta magnitude

        # Active path tracking
        self.active_nodes = set()  # Set of node_ids that were active in last forward pass
        self.active_edges = set()  # Set of (u, v) edge tuples that were active

        # Graph cache
        self.graph = None
        self.layout = None
        self.last_genome_id = None

        # Font
        self.font = None
        self.small_font = None

    def init_fonts(self):
        """Initialize pygame fonts (call after pygame.init())"""
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def update_parameter_deltas(self, cell: EvoCell):
        """
        Track parameter changes since last tick

        Args:
            cell: EvoCell to track
        """
        import torch
        for name, param in cell.named_parameters():
            current_param = param.detach().clone()

            if name in self.prev_params:
                # Calculate delta (L2 norm of change)
                delta = torch.norm(current_param - self.prev_params[name]).item()
                self.param_deltas[name] = delta
            else:
                # First time seeing this parameter
                self.param_deltas[name] = 0.0

            # Store for next comparison
            self.prev_params[name] = current_param

    def trace_active_path(self, cell: EvoCell, last_action_idx: int):
        """
        Trace which nodes/edges were active in producing the last action

        Args:
            cell: EvoCell that produced the action
            last_action_idx: Index of the action taken (0-3)
        """
        # Reset active tracking
        self.active_nodes.clear()
        self.active_edges.clear()

        # In our simplified topology:
        # All nodes contribute to all outputs, so we highlight the full path
        # that led to the selected action

        # Mark the active output node
        active_output = f"out_{last_action_idx}"
        self.active_nodes.add(active_output)

        # Mark all nodes in the forward path
        self.active_nodes.add("input_layer")
        self.active_nodes.add("state_input")
        self.active_nodes.add("hidden_proc")
        self.active_nodes.add("gating")
        self.active_nodes.add("state_output")

        # Mark active edges
        self.active_edges.add(("input_layer", "hidden_proc"))
        self.active_edges.add(("state_input", "hidden_proc"))
        self.active_edges.add(("hidden_proc", "gating"))
        self.active_edges.add(("gating", active_output))
        self.active_edges.add(("gating", "state_output"))

    def extract_topology(self, cell: EvoCell) -> Tuple[nx.DiGraph, Dict]:
        """
        Extract SIMPLIFIED topological graph from EvoCell
        (Reduced complexity to prevent freezing with large state_dim)

        Returns:
            (graph, node_attributes)
        """
        G = nx.DiGraph()
        node_attrs = {}

        # Get dimensions
        in_dim = min(cell.in_dim, 16)  # Limit for visualization
        state_dim = min(cell.state_dim, 16)  # Limit for visualization
        out_dim = cell.out_dim

        # Input layer nodes (aggregate if too many)
        node_id = "input_layer"
        G.add_node(node_id)
        node_attrs[node_id] = {
            'type': 'input',
            'layer': 0,
            'index': 0,
            'size': 25
        }

        # State input node (aggregate)
        node_id = "state_input"
        G.add_node(node_id)
        node_attrs[node_id] = {
            'type': 'state',
            'layer': 0,
            'index': 1,
            'size': 20
        }

        # Hidden processing node
        node_id = "hidden_proc"
        G.add_node(node_id)
        node_attrs[node_id] = {
            'type': 'hidden',
            'layer': 1,
            'index': 0,
            'size': 25
        }

        # Gating node
        node_id = "gating"
        G.add_node(node_id)
        node_attrs[node_id] = {
            'type': 'gate',
            'layer': 2,
            'index': 0,
            'size': 25
        }

        # State output node
        node_id = "state_output"
        G.add_node(node_id)
        node_attrs[node_id] = {
            'type': 'state',
            'layer': 3,
            'index': 0,
            'size': 20
        }

        # Output layer nodes (one per action)
        for i in range(out_dim):
            node_id = f"out_{i}"
            G.add_node(node_id)
            node_attrs[node_id] = {
                'type': 'output',
                'layer': 3,
                'index': i + 1,
                'size': 22
            }

        # Add edges (simplified flow)
        G.add_edge("input_layer", "hidden_proc", weight=1.0)
        G.add_edge("state_input", "hidden_proc", weight=1.0)
        G.add_edge("hidden_proc", "gating", weight=1.0)
        G.add_edge("gating", "state_output", weight=1.0)

        # Gating to outputs
        for i in range(out_dim):
            G.add_edge("gating", f"out_{i}", weight=1.0)

        return G, node_attrs

    def extract_genetic_sequence(self, cell: EvoCell) -> List[Dict]:
        """
        Extract genetic sequence view (parameter listing)

        Returns:
            List of operation dictionaries
        """
        operations = []

        # Get all parameters and their shapes
        for name, param in cell.named_parameters():
            ops = {
                'name': name,
                'shape': tuple(param.shape),
                'numel': param.numel(),
                'norm': float(param.norm().item()),
                'mean': float(param.mean().item()),
                'std': float(param.std().item())
            }
            operations.append(ops)

        return operations

    def extract_functional_clusters(self, cell: EvoCell) -> Dict[str, List[str]]:
        """
        Extract functional clusters (grouped by operation type)

        Returns:
            Dictionary mapping function type to node lists
        """
        clusters = {
            'Input Processing': [],
            'State Management': [],
            'Gating & Control': [],
            'Output Generation': [],
        }

        # Analyze parameter names to cluster by function
        for name, param in cell.named_parameters():
            if 'in_proj' in name:
                clusters['Input Processing'].append(f"{name} {tuple(param.shape)}")
            elif 'next_state' in name or 'state' in name:
                clusters['State Management'].append(f"{name} {tuple(param.shape)}")
            elif 'g1' in name or 'h1' in name:
                clusters['Gating & Control'].append(f"{name} {tuple(param.shape)}")
            elif 'out' in name:
                clusters['Output Generation'].append(f"{name} {tuple(param.shape)}")
            else:
                clusters['State Management'].append(f"{name} {tuple(param.shape)}")

        return clusters

    def compute_layout(self, G: nx.DiGraph, node_attrs: Dict) -> Dict[str, Tuple[float, float]]:
        """
        Compute node positions using layered layout

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        # Group nodes by layer
        layers = {}
        for node_id, attrs in node_attrs.items():
            layer = attrs['layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        # Compute positions
        pos = {}
        num_layers = len(layers)
        margin_x = 80
        margin_y = 80
        width = self.width - 2 * margin_x
        height = self.height - 2 * margin_y

        for layer_idx, nodes in layers.items():
            x = margin_x + (width * layer_idx / (num_layers - 1)) if num_layers > 1 else self.width / 2
            num_nodes = len(nodes)

            for i, node_id in enumerate(nodes):
                if num_nodes > 1:
                    y = margin_y + (height * i / (num_nodes - 1))
                else:
                    y = self.height / 2
                pos[node_id] = (x, y)

        return pos

    def update_activation_history(self, node_id: str, activation: float):
        """Track activation for dynamic highlighting"""
        if node_id not in self.activation_history:
            self.activation_history[node_id] = deque(maxlen=self.history_length)
        self.activation_history[node_id].append(activation)

    def get_activation_intensity(self, node_id: str) -> float:
        """Get recent activation intensity (0-1)"""
        if node_id not in self.activation_history:
            return 0.0
        history = self.activation_history[node_id]
        if len(history) == 0:
            return 0.0
        return min(1.0, np.mean(list(history)))

    def render_topology(self, surface: pygame.Surface, cell: EvoCell):
        """Render topological view with active path highlighting"""
        try:
            # Build or retrieve graph
            genome_id = id(cell)
            if self.graph is None or self.last_genome_id != genome_id:
                self.graph, self.node_attrs = self.extract_topology(cell)
                self.layout = self.compute_layout(self.graph, self.node_attrs)
                self.last_genome_id = genome_id

            # Draw edges with active path highlighting
            for u, v, data in self.graph.edges(data=True):
                if u in self.layout and v in self.layout:
                    x1, y1 = self.layout[u]
                    x2, y2 = self.layout[v]

                    # Check if this edge was active in last forward pass
                    is_active = (u, v) in self.active_edges
                    edge_color = self.ACTIVE_PATH_COLOR if is_active else self.EDGE_COLOR[:3]
                    edge_width = 4 if is_active else 2

                    # Draw edge
                    pygame.draw.line(surface, edge_color,
                                   (int(x1), int(y1)), (int(x2), int(y2)), edge_width)

                    # Draw arrow head for active edges
                    if is_active:
                        self._draw_arrow_head(surface, (int(x1), int(y1)), (int(x2), int(y2)),
                                            edge_color, size=8)

            # Draw nodes with active highlighting
            for node_id, (x, y) in self.layout.items():
                attrs = self.node_attrs.get(node_id, {})
                node_type = attrs.get('type', 'hidden')
                size = attrs.get('size', 15)

                # Get base color
                color = self.NODE_COLORS.get(node_type, (150, 150, 150))

                # Highlight if node was active in last forward pass
                is_active = node_id in self.active_nodes
                if is_active:
                    # Draw glow effect for active nodes
                    glow_color = (*self.ACTIVE_PATH_COLOR, 100)
                    for glow_radius in range(size + 10, size, -2):
                        alpha = int(100 * (1 - (glow_radius - size) / 10))
                        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surf, (*self.ACTIVE_PATH_COLOR, alpha),
                                         (glow_radius, glow_radius), glow_radius)
                        surface.blit(glow_surf, (int(x) - glow_radius, int(y) - glow_radius))

                # Draw node
                pygame.draw.circle(surface, color, (int(x), int(y)), size)
                border_color = self.ACTIVE_PATH_COLOR if is_active else self.TEXT_COLOR
                border_width = 2 if is_active else 1
                pygame.draw.circle(surface, border_color, (int(x), int(y)), size, border_width)

                # Draw label
                label_color = self.ACTIVE_PATH_COLOR if is_active else self.TEXT_COLOR
                if node_id in ["input_layer", "state_input", "hidden_proc", "gating", "state_output"]:
                    label = self.small_font.render(node_id.replace("_", " "), True, label_color)
                    surface.blit(label, (int(x) + size + 5, int(y) - 8))
                elif node_id.startswith("out_"):
                    action_names = ["Up", "Down", "Left", "Right"]
                    idx = int(node_id.split("_")[1])
                    label_text = f"→ {action_names[idx]}" if is_active else action_names[idx]
                    label = self.small_font.render(label_text, True, label_color)
                    surface.blit(label, (int(x) + size + 5, int(y) - 8))

            # Draw legend
            self._draw_legend(surface)

            # Draw active path info
            if self.active_nodes:
                info_text = self.small_font.render("Green = Active path for last action",
                                                   True, self.ACTIVE_PATH_COLOR)
                surface.blit(info_text, (10, self.height - 25))

        except Exception as e:
            # Fallback: show error message
            error_text = self.font.render(f"Topology Error: {str(e)[:40]}", True, (255, 100, 100))
            surface.blit(error_text, (20, 20))

    def render_genetic(self, surface: pygame.Surface, cell: EvoCell):
        """Render genetic sequence view with parameter delta overlay"""
        try:
            operations = self.extract_genetic_sequence(cell)

            # Title
            title = self.font.render("Genetic Sequence (Parameters)", True, self.TEXT_COLOR)
            surface.blit(title, (20, 20))

            # Delta info
            delta_hint = self.small_font.render("Green=increasing, Red=decreasing, Gray=stable",
                                                True, (150, 150, 150))
            surface.blit(delta_hint, (20, 42))

            # Parameter list
            y = 70
            for i, op in enumerate(operations):
                param_name = op['name']

                # Get delta for this parameter
                delta = self.param_deltas.get(param_name, 0.0)
                delta_normalized = min(1.0, delta * 100)  # Scale for visibility

                # Color based on delta (green=increase, red=decrease, gray=stable)
                if delta > 1e-6:
                    # Increasing
                    name_color = tuple(int(c * (1 - delta_normalized) + d * delta_normalized)
                                      for c, d in zip((200, 200, 255), self.DELTA_POSITIVE_COLOR))
                elif delta < -1e-6:
                    # Decreasing (should be rare since we track abs delta)
                    name_color = tuple(int(c * (1 - delta_normalized) + d * delta_normalized)
                                      for c, d in zip((200, 200, 255), self.DELTA_NEGATIVE_COLOR))
                else:
                    # Stable
                    name_color = (200, 200, 255)

                # Operation name with delta indicator
                delta_indicator = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "—")
                name_text = self.small_font.render(f"{delta_indicator} {param_name}", True, name_color)
                surface.blit(name_text, (20, y))

                # Shape and stats
                info_text = self.small_font.render(
                    f"  shape={op['shape']} norm={op['norm']:.4f} Δ={delta:.6f}",
                    True, (180, 180, 180)
                )
                surface.blit(info_text, (40, y + 18))

                # Visual bar for norm (base layer)
                bar_width = min(280, int(op['norm'] * 50))
                bar_color = (100 + min(155, int(op['norm'] * 10)), 100, 100)
                pygame.draw.rect(surface, bar_color, (40, y + 38, bar_width, 8))

                # Delta overlay (shows change magnitude)
                if delta > 1e-6:
                    delta_bar_width = min(280, int(delta * 5000))  # Scale for visibility
                    delta_alpha = int(min(200, delta * 10000))
                    delta_color = self.DELTA_POSITIVE_COLOR if delta > 0 else self.DELTA_NEGATIVE_COLOR
                    # Draw semi-transparent overlay
                    overlay = pygame.Surface((delta_bar_width, 8), pygame.SRCALPHA)
                    overlay.fill((*delta_color, delta_alpha))
                    surface.blit(overlay, (40, y + 38))

                y += 58

                # Scroll if too many
                if y > self.height - 80:
                    more_text = self.small_font.render(f"... and {len(operations) - i - 1} more",
                                                       True, (150, 150, 150))
                    surface.blit(more_text, (20, y))
                    break

        except Exception as e:
            error_text = self.font.render(f"Genetic Error: {str(e)[:40]}", True, (255, 100, 100))
            surface.blit(error_text, (20, 20))

    def render_functional(self, surface: pygame.Surface, cell: EvoCell):
        """Render functional clusters view"""
        try:
            clusters = self.extract_functional_clusters(cell)

            # Title
            title = self.font.render("Functional Clusters", True, self.TEXT_COLOR)
            surface.blit(title, (20, 20))

            # Cluster colors
            cluster_colors = {
                'Input Processing': (100, 200, 255),
                'State Management': (255, 200, 100),
                'Gating & Control': (255, 100, 200),
                'Output Generation': (255, 150, 100),
            }

            # Render clusters
            y = 60
            for cluster_name, items in clusters.items():
                if len(items) == 0:
                    continue

                color = cluster_colors.get(cluster_name, (150, 150, 150))

                # Cluster header
                header_text = self.font.render(cluster_name, True, color)
                surface.blit(header_text, (20, y))
                y += 30

                # Cluster items
                for item in items[:5]:  # Show first 5
                    item_text = self.small_font.render(f"  • {item}", True, (180, 180, 180))
                    surface.blit(item_text, (30, y))
                    y += 20

                if len(items) > 5:
                    more_text = self.small_font.render(f"  ... and {len(items) - 5} more",
                                                       True, (150, 150, 150))
                    surface.blit(more_text, (30, y))
                    y += 20

                y += 20  # Space between clusters

        except Exception as e:
            error_text = self.font.render(f"Functional Error: {str(e)[:40]}", True, (255, 100, 100))
            surface.blit(error_text, (20, 20))

    def render(self, surface: pygame.Surface, cell: Optional[EvoCell],
               step_idx: int, champion_score: float):
        """
        Main render function

        Args:
            surface: Pygame surface to draw on
            cell: EvoCell to visualize
            step_idx: Current training step
            champion_score: Current champion score
        """
        try:
            # Clear background
            surface.fill(self.BG_COLOR)

            if cell is None:
                # No genome to display
                text = self.font.render("No genome available", True, self.TEXT_COLOR)
                surface.blit(text, (self.width // 2 - 100, self.height // 2))
                return

            # Render based on mode
            if self.mode == self.MODE_TOPOLOGY:
                self.render_topology(surface, cell)
            elif self.mode == self.MODE_GENETIC:
                self.render_genetic(surface, cell)
            elif self.mode == self.MODE_FUNCTIONAL:
                self.render_functional(surface, cell)

            # Draw header info
            self._draw_header(surface, step_idx, champion_score)

        except Exception as e:
            # Ultimate fallback
            surface.fill(self.BG_COLOR)
            error_text = self.font.render(f"Render Error: {str(e)[:50]}", True, (255, 100, 100))
            surface.blit(error_text, (20, self.height // 2))
            print(f"[Viz Error] {e}")

    def _draw_header(self, surface: pygame.Surface, step_idx: int, champion_score: float):
        """Draw header with mode and stats"""
        # Mode name
        mode_text = self.font.render(f"View: {self.MODE_NAMES[self.mode]}", True, (255, 255, 100))
        surface.blit(mode_text, (self.width - 300, 10))

        # Stats
        stats_text = self.small_font.render(f"Step: {step_idx} | Score: {champion_score:.4f}",
                                            True, self.TEXT_COLOR)
        surface.blit(stats_text, (self.width - 300, 35))

        # Controls hint
        hint_text = self.small_font.render("Press V to cycle views", True, (150, 150, 150))
        surface.blit(hint_text, (self.width - 300, 55))

    def _draw_legend(self, surface: pygame.Surface):
        """Draw node type legend for topology view"""
        x = 10
        y = self.height - 100

        legend_items = [
            ('Input', 'input'),
            ('Hidden', 'hidden'),
            ('Gate', 'gate'),
            ('Output', 'output'),
            ('State', 'state'),
        ]

        for name, node_type in legend_items:
            color = self.NODE_COLORS.get(node_type, (150, 150, 150))
            pygame.draw.circle(surface, color, (x + 10, y), 8)
            text = self.small_font.render(name, True, self.TEXT_COLOR)
            surface.blit(text, (x + 25, y - 8))
            y += 20

    def _draw_line_alpha(self, surface: pygame.Surface, color: Tuple[int, int, int, int],
                        start: Tuple[int, int], end: Tuple[int, int], width: int):
        """Draw line with alpha transparency"""
        # Create temporary surface for alpha blending
        line_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.line(line_surf, color, start, end, width)
        surface.blit(line_surf, (0, 0))

    def _draw_arrow_head(self, surface: pygame.Surface, start: Tuple[int, int],
                        end: Tuple[int, int], color: Tuple[int, int, int], size: int = 8):
        """Draw arrow head at end of line"""
        import math

        # Calculate angle
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.atan2(dy, dx)

        # Arrow head points
        arrow_angle = math.pi / 6  # 30 degrees
        p1 = end
        p2 = (int(end[0] - size * math.cos(angle - arrow_angle)),
              int(end[1] - size * math.sin(angle - arrow_angle)))
        p3 = (int(end[0] - size * math.cos(angle + arrow_angle)),
              int(end[1] - size * math.sin(angle + arrow_angle)))

        pygame.draw.polygon(surface, color, [p1, p2, p3])

    def cycle_mode(self):
        """Cycle to next visualization mode"""
        self.mode = (self.mode + 1) % 3
        # Clear graph cache when switching modes
        if self.mode != self.MODE_TOPOLOGY:
            self.graph = None
            self.layout = None

    def set_mode(self, mode: int):
        """Set visualization mode directly"""
        if 0 <= mode < 3:
            self.mode = mode
            # Clear graph cache when switching modes
            if self.mode != self.MODE_TOPOLOGY:
                self.graph = None
                self.layout = None
