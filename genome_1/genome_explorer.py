"""
Genome Explorer - Interactive Visualization of Champion Circuit Topology

Visualizes the frozen digital DNA of the champion OLA genome:
- Logical primitives (gates, operators, comparators)
- Connection topology (how signals flow between gates)
- Parameter values (weights, thresholds, routing)
- Motifs and patterns (loops, feedback structures, memory gates)

Controls:
- Mouse drag: Pan view
- Mouse wheel: Zoom
- Click node: Select and highlight connections
- 1-5: Switch visualization layers
- R: Reset view
- Space: Toggle animation
- S: Save visualization
"""

import torch
import numpy as np
import pygame
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import argparse


class CircuitNode:
    """Represents a logical primitive in the circuit"""
    def __init__(self, node_id: str, node_type: str, layer: int):
        self.id = node_id
        self.type = node_type  # "input", "gate", "output", "memory"
        self.layer = layer
        self.pos = (0.0, 0.0)
        self.connections_in: List['CircuitEdge'] = []
        self.connections_out: List['CircuitEdge'] = []
        self.gate_op: Optional[str] = None  # AND, OR, XOR, NOT, MUX, etc.
        self.parameters: Dict[str, float] = {}
        self.selected = False
        self.highlighted = False

    def degree_in(self) -> int:
        return len(self.connections_in)

    def degree_out(self) -> int:
        return len(self.connections_out)


class CircuitEdge:
    """Represents a connection between primitives"""
    def __init__(self, source: CircuitNode, target: CircuitNode, weight: float = 1.0):
        self.source = source
        self.target = target
        self.weight = weight
        self.strength = abs(weight)
        self.is_feedback = False  # loops back to earlier layer
        self.is_memory = False    # connects to memory gate
        self.highlighted = False


class GenomeTopology:
    """Analyzes and extracts circuit topology from champion genome"""

    def __init__(self, champion_cell):
        self.cell = champion_cell
        self.nodes: Dict[str, CircuitNode] = {}
        self.edges: List[CircuitEdge] = []
        self.layers: Dict[int, List[CircuitNode]] = defaultdict(list)
        self.motifs: Dict[str, List] = {
            'feedback_loops': [],
            'memory_cells': [],
            'fan_out': [],
            'fan_in': [],
            'skip_connections': []
        }

        self._extract_topology()
        self._detect_motifs()
        self._layout_nodes()

    def _extract_topology(self):
        """Extract circuit structure from champion cell parameters"""

        # Get all named parameters
        params = dict(self.cell.named_parameters())

        # Analyze each parameter to understand the circuit
        layer_idx = 0

        for name, param in params.items():
            if 'weight' in name.lower():
                self._analyze_weight_matrix(name, param, layer_idx)
                layer_idx += 1
            elif 'bias' in name.lower():
                self._analyze_bias(name, param, layer_idx - 1)

        # Add input nodes
        input_dim = self._get_input_dim()
        for i in range(min(input_dim, 32)):  # Limit to 32 input nodes for visualization
            node = CircuitNode(f"in_{i}", "input", 0)
            node.gate_op = "INPUT"
            self.nodes[node.id] = node
            self.layers[0].append(node)

        # Add output nodes
        output_dim = self._get_output_dim()
        max_layer = max(self.layers.keys()) if self.layers else 0
        for i in range(output_dim):
            node = CircuitNode(f"out_{i}", "output", max_layer + 1)
            node.gate_op = "OUTPUT"
            self.nodes[node.id] = node
            self.layers[max_layer + 1].append(node)

    def _analyze_weight_matrix(self, name: str, weight: torch.Tensor, layer: int):
        """Analyze weight matrix to extract logical connections"""
        W = weight.detach().cpu().numpy()

        if len(W.shape) != 2:
            return

        out_dim, in_dim = W.shape

        # Create nodes for this layer if they don't exist
        layer_nodes = []
        for i in range(min(out_dim, 64)):  # Limit nodes per layer
            node_id = f"L{layer}_N{i}"
            if node_id not in self.nodes:
                node = CircuitNode(node_id, "gate", layer + 1)
                node.gate_op = self._infer_gate_type(W[i])
                self.nodes[node_id] = node
                self.layers[layer + 1].append(node)
                layer_nodes.append(node)
            else:
                layer_nodes.append(self.nodes[node_id])

        # Create edges based on significant weights
        threshold = np.abs(W).mean() + np.abs(W).std() * 0.5

        for i in range(min(out_dim, 64)):
            target = layer_nodes[i]

            for j in range(min(in_dim, 64)):
                if abs(W[i, j]) > threshold:
                    # Find source node
                    if layer == 0:
                        source_id = f"in_{j}"
                    else:
                        source_id = f"L{layer - 1}_N{j}"

                    if source_id in self.nodes:
                        source = self.nodes[source_id]
                        edge = CircuitEdge(source, target, W[i, j])

                        # Check for feedback (connects to earlier or same layer)
                        if target.layer <= source.layer:
                            edge.is_feedback = True

                        source.connections_out.append(edge)
                        target.connections_in.append(edge)
                        self.edges.append(edge)

                        # Store weight in target parameters
                        target.parameters[f"w_{source.id}"] = float(W[i, j])

    def _analyze_bias(self, name: str, bias: torch.Tensor, layer: int):
        """Add bias parameters to nodes"""
        b = bias.detach().cpu().numpy()

        for i in range(min(len(b), 64)):
            node_id = f"L{layer}_N{i}"
            if node_id in self.nodes:
                self.nodes[node_id].parameters['bias'] = float(b[i])

    def _infer_gate_type(self, weights: np.ndarray) -> str:
        """Infer logical gate type from weight pattern"""
        w_mean = weights.mean()
        w_std = weights.std()
        w_max = weights.max()
        w_min = weights.min()

        # Simple heuristics to classify gate behavior
        if w_std < 0.1:
            return "CONST"
        elif w_max > 0 and w_min < 0 and abs(w_max + w_min) < 0.1:
            return "XOR"
        elif w_mean > 0.5:
            return "OR"
        elif w_mean < -0.5:
            return "NOR"
        elif w_max > abs(w_min) * 2:
            return "AND"
        elif abs(w_min) > w_max * 2:
            return "NAND"
        elif len(weights) > 1 and np.argmax(np.abs(weights)) == 0:
            return "MUX"
        else:
            return "COMB"

    def _get_input_dim(self) -> int:
        """Get input dimension from first weight layer"""
        for name, param in self.cell.named_parameters():
            if 'weight' in name.lower():
                return param.shape[1]
        return 32

    def _get_output_dim(self) -> int:
        """Get output dimension from last weight layer"""
        last_weight = None
        for name, param in self.cell.named_parameters():
            if 'weight' in name.lower():
                last_weight = param
        if last_weight is not None:
            return last_weight.shape[0]
        return 4

    def _detect_motifs(self):
        """Detect common circuit patterns and motifs"""

        # Feedback loops (directly from edges flagged during topology extraction)
        for edge in self.edges:
            if edge.is_feedback:
                self.motifs['feedback_loops'].append(edge)

        # High fan-out nodes (signal broadcasters)
        for node in self.nodes.values():
            if node.degree_out() > 5:
                self.motifs['fan_out'].append(node)

        # High fan-in nodes (signal integrators)
        for node in self.nodes.values():
            if node.degree_in() > 5:
                self.motifs['fan_in'].append(node)

        # Memory cells: any node participating in a feedback edge
        memory_nodes: Set[CircuitNode] = set()
        for edge in self.edges:
            if edge.is_feedback:
                memory_nodes.add(edge.source)
                memory_nodes.add(edge.target)
        self.motifs['memory_cells'] = list(memory_nodes)

        # Skip connections (connects across multiple layers)
        for edge in self.edges:
            layer_diff = edge.target.layer - edge.source.layer
            if layer_diff > 1:
                self.motifs['skip_connections'].append(edge)

    def _has_feedback_path(self, node: CircuitNode, visited: Optional[Set] = None) -> bool:
        """Check if node has feedback path to itself"""
        if visited is None:
            visited = set()

        if node.id in visited:
            return True

        visited.add(node.id)

        for edge in node.connections_out:
            if edge.target.layer <= node.layer:
                return True
            if self._has_feedback_path(edge.target, visited.copy()):
                return True

        return False

    def _layout_nodes(self):
        """Compute spatial layout for visualization"""

        max_layer = max(self.layers.keys()) if self.layers else 0

        for layer_idx, nodes in self.layers.items():
            n = len(nodes)
            x = layer_idx / max(max_layer, 1)

            for i, node in enumerate(nodes):
                y = (i + 0.5) / max(n, 1)
                node.pos = (x, y)


class GenomeExplorer:
    """Interactive visualization of champion genome topology"""

    def __init__(self, champion_path: Path, width: int = 1600, height: int = 900):
        self.champion_path = champion_path
        self.width = width
        self.height = height

        # Load champion
        self.champion_cell = self._load_champion()

        # Extract topology
        print("Analyzing genome topology...")
        self.topology = GenomeTopology(self.champion_cell)
        print(f"  Nodes: {len(self.topology.nodes)}")
        print(f"  Edges: {len(self.topology.edges)}")
        print(f"  Layers: {len(self.topology.layers)}")
        print(f"  Feedback loops: {len(self.topology.motifs['feedback_loops'])}")
        print(f"  Memory cells: {len(self.topology.motifs['memory_cells'])}")

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Genome Explorer - Champion Circuit Topology")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # View controls
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0
        self.dragging = False
        self.drag_start = (0, 0)

        # Visualization state
        self.selected_node: Optional[CircuitNode] = None
        self.view_layer = 0  # 0=all, 1=topology, 2=weights, 3=motifs, 4=flow
        self.animate = False
        self.animation_frame = 0

        # Colors
        self.colors = {
            'bg': (15, 15, 20),
            'node_input': (100, 200, 255),
            'node_gate': (150, 255, 150),
            'node_output': (255, 150, 100),
            'node_memory': (255, 200, 100),
            'edge_normal': (80, 80, 100),
            'edge_feedback': (255, 100, 100),
            'edge_skip': (100, 255, 200),
            'edge_highlight': (255, 255, 100),
            'text': (220, 220, 220),
            'grid': (40, 40, 50)
        }

    def _load_champion(self):
        """Load champion cell from checkpoint"""
        checkpoint = torch.load(self.champion_path, map_location='cpu', weights_only=False)

        # Try to reconstruct the cell from checkpoint
        if 'champion_state_dict' in checkpoint:
            # Need to reconstruct the EvoCell
            from ola import EvoCell

            # Get dimensions from config or state dict
            config = checkpoint.get('config', {})
            in_dim = config.get('in_dim')
            out_dim = config.get('out_dim', 4)
            state_dim = config.get('state_dim', 128)

            # If no config, infer from state dict
            if in_dim is None:
                state = checkpoint['champion_state_dict']
                first_weight = None
                for key, val in state.items():
                    if 'weight' in key and 'in_proj' in key:
                        first_weight = val
                        break

                if first_weight is not None:
                    # in_proj combines input and state, so: in_dim + state_dim = first_weight.shape[1]
                    in_dim = first_weight.shape[1] - state_dim

            if in_dim is not None:
                cell = EvoCell(in_dim, out_dim, state_dim)
                cell.load_state_dict(checkpoint['champion_state_dict'])
                return cell

        raise ValueError("Could not load champion cell from checkpoint")

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        margin = 100
        view_width = self.width - 2 * margin
        view_height = self.height - 2 * margin

        sx = margin + (x + self.offset_x) * view_width * self.zoom
        sy = margin + (y + self.offset_y) * view_height * self.zoom

        return (int(sx), int(sy))

    def screen_to_world(self, sx: int, sy: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates"""
        margin = 100
        view_width = self.width - 2 * margin
        view_height = self.height - 2 * margin

        x = (sx - margin) / (view_width * self.zoom) - self.offset_x
        y = (sy - margin) / (view_height * self.zoom) - self.offset_y

        return (x, y)

    def draw_grid(self):
        """Draw background grid"""
        grid_spacing = 50

        for i in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (i, 0), (i, self.height), 1)

        for i in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, self.colors['grid'], (0, i), (self.width, i), 1)

    def draw_edges(self):
        """Draw all circuit connections"""
        for edge in self.topology.edges:
            if self.view_layer == 3 and not edge.is_feedback and edge not in self.topology.motifs['skip_connections']:
                continue  # In motif view, only show interesting edges

            sx, sy = self.world_to_screen(*edge.source.pos)
            tx, ty = self.world_to_screen(*edge.target.pos)

            # Choose color
            if edge.highlighted:
                color = self.colors['edge_highlight']
                width = 3
            elif edge.is_feedback:
                color = self.colors['edge_feedback']
                width = 2
            elif edge in self.topology.motifs['skip_connections']:
                color = self.colors['edge_skip']
                width = 2
            else:
                # Color by weight strength
                alpha = min(255, int(edge.strength * 255))
                color = (*self.colors['edge_normal'][:3], alpha)
                width = 1

            # Draw curved line for feedback
            if edge.is_feedback:
                self._draw_curve(sx, sy, tx, ty, color, width)
            else:
                pygame.draw.line(self.screen, color, (sx, sy), (tx, ty), width)

            # Draw arrowhead
            if width > 1:
                self._draw_arrow(sx, sy, tx, ty, color)

    def _draw_curve(self, x1: int, y1: int, x2: int, y2: int, color, width: int):
        """Draw curved line (for feedback connections)"""
        # Simple quadratic bezier curve
        cx = (x1 + x2) // 2 - 50
        cy = (y1 + y2) // 2

        points = []
        for t in range(11):
            t = t / 10.0
            x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
            y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
            points.append((int(x), int(y)))

        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, width)

    def _draw_arrow(self, x1: int, y1: int, x2: int, y2: int, color):
        """Draw arrowhead at end of line"""
        angle = math.atan2(y2 - y1, x2 - x1)
        arrow_len = 10
        arrow_angle = math.pi / 6

        p1 = (
            x2 - arrow_len * math.cos(angle - arrow_angle),
            y2 - arrow_len * math.sin(angle - arrow_angle)
        )
        p2 = (
            x2 - arrow_len * math.cos(angle + arrow_angle),
            y2 - arrow_len * math.sin(angle + arrow_angle)
        )

        pygame.draw.polygon(self.screen, color, [(x2, y2), p1, p2])

    def draw_nodes(self):
        """Draw all circuit nodes"""
        for node in self.topology.nodes.values():
            sx, sy = self.world_to_screen(*node.pos)

            # Skip if off-screen
            if sx < -50 or sx > self.width + 50 or sy < -50 or sy > self.height + 50:
                continue

            # Choose color
            if node.type == "input":
                color = self.colors['node_input']
            elif node.type == "output":
                color = self.colors['node_output']
            elif node in self.topology.motifs['memory_cells']:
                color = self.colors['node_memory']
            else:
                color = self.colors['node_gate']

            # Size by degree
            degree = node.degree_in() + node.degree_out()
            radius = max(8, min(20, 8 + degree // 2))

            # Draw node
            if node.selected:
                pygame.draw.circle(self.screen, (255, 255, 100), (sx, sy), radius + 4, 3)

            pygame.draw.circle(self.screen, color, (sx, sy), radius)

            # Draw gate type label
            if self.zoom > 0.5 and node.gate_op:
                label = self.font_small.render(node.gate_op, True, (0, 0, 0))
                label_rect = label.get_rect(center=(sx, sy))
                self.screen.blit(label, label_rect)

    def draw_info_panel(self):
        """Draw information panel"""
        panel_x = 10
        panel_y = 10
        panel_w = 350
        panel_h = 250

        # Background
        s = pygame.Surface((panel_w, panel_h))
        s.set_alpha(220)
        s.fill((20, 20, 30))
        self.screen.blit(s, (panel_x, panel_y))

        # Border
        pygame.draw.rect(self.screen, (100, 100, 120), (panel_x, panel_y, panel_w, panel_h), 2)

        # Title
        title = self.font.render("Champion Genome Topology", True, self.colors['text'])
        self.screen.blit(title, (panel_x + 10, panel_y + 10))

        # Stats
        y = panel_y + 45
        stats = [
            f"Nodes: {len(self.topology.nodes)}",
            f"Connections: {len(self.topology.edges)}",
            f"Layers: {len(self.topology.layers)}",
            f"",
            f"Motifs:",
            f"  Feedback loops: {len(self.topology.motifs['feedback_loops'])}",
            f"  Memory cells: {len(self.topology.motifs['memory_cells'])}",
            f"  Fan-out hubs: {len(self.topology.motifs['fan_out'])}",
            f"  Fan-in hubs: {len(self.topology.motifs['fan_in'])}",
            f"  Skip connections: {len(self.topology.motifs['skip_connections'])}"
        ]

        for stat in stats:
            text = self.font_small.render(stat, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 15, y))
            y += 20

    def draw_selected_info(self):
        """Draw info about selected node"""
        if not self.selected_node:
            return

        panel_x = self.width - 360
        panel_y = 10
        panel_w = 350
        panel_h = 300

        # Background
        s = pygame.Surface((panel_w, panel_h))
        s.set_alpha(220)
        s.fill((20, 20, 30))
        self.screen.blit(s, (panel_x, panel_y))

        # Border
        pygame.draw.rect(self.screen, (255, 200, 100), (panel_x, panel_y, panel_w, panel_h), 2)

        # Node info
        node = self.selected_node
        y = panel_y + 10

        lines = [
            f"Node: {node.id}",
            f"Type: {node.type}",
            f"Gate: {node.gate_op or 'N/A'}",
            f"Layer: {node.layer}",
            f"",
            f"Connections:",
            f"  Inputs: {node.degree_in()}",
            f"  Outputs: {node.degree_out()}",
            f"",
            f"Parameters:"
        ]

        for line in lines:
            text = self.font_small.render(line, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 10, y))
            y += 20

        # Show some parameters
        param_count = 0
        for key, val in node.parameters.items():
            if param_count >= 5:
                text = self.font_small.render("  ...", True, self.colors['text'])
                self.screen.blit(text, (panel_x + 10, y))
                break

            text = self.font_small.render(f"  {key}: {val:.4f}", True, self.colors['text'])
            self.screen.blit(text, (panel_x + 10, y))
            y += 20
            param_count += 1

    def draw_controls(self):
        """Draw control help"""
        panel_x = 10
        panel_y = self.height - 180
        panel_w = 300
        panel_h = 170

        # Background
        s = pygame.Surface((panel_w, panel_h))
        s.set_alpha(220)
        s.fill((20, 20, 30))
        self.screen.blit(s, (panel_x, panel_y))

        # Border
        pygame.draw.rect(self.screen, (100, 100, 120), (panel_x, panel_y, panel_w, panel_h), 2)

        # Controls
        y = panel_y + 10
        controls = [
            "Controls:",
            "  Mouse drag: Pan",
            "  Mouse wheel: Zoom",
            "  Click node: Select",
            "  1-5: View layers",
            "  R: Reset view",
            "  S: Save image",
            "  ESC: Exit"
        ]

        for ctrl in controls:
            text = self.font_small.render(ctrl, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 10, y))
            y += 20

    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse click to select node"""
        wx, wy = self.screen_to_world(*pos)

        # Find nearest node
        min_dist = float('inf')
        nearest = None

        for node in self.topology.nodes.values():
            dx = node.pos[0] - wx
            dy = node.pos[1] - wy
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < min_dist and dist < 0.05:  # Within 5% of view
                min_dist = dist
                nearest = node

        # Clear previous selection
        if self.selected_node:
            self.selected_node.selected = False
            for edge in self.selected_node.connections_in + self.selected_node.connections_out:
                edge.highlighted = False

        # Select new node
        if nearest:
            self.selected_node = nearest
            nearest.selected = True

            # Highlight connected edges
            for edge in nearest.connections_in + nearest.connections_out:
                edge.highlighted = True

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.offset_x = 0.0
                    self.offset_y = 0.0
                    self.zoom = 1.0
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    self.view_layer = event.key - pygame.K_1

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.drag_start = event.pos
                elif event.button == 4:  # Scroll up
                    self.zoom *= 1.1
                elif event.button == 5:  # Scroll down
                    self.zoom *= 0.9

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if self.dragging:
                        # Check if it was a click (not a drag)
                        dx = event.pos[0] - self.drag_start[0]
                        dy = event.pos[1] - self.drag_start[1]
                        if abs(dx) < 5 and abs(dy) < 5:
                            self.handle_click(event.pos)
                    self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    dx = event.pos[0] - self.drag_start[0]
                    dy = event.pos[1] - self.drag_start[1]

                    margin = 100
                    view_width = self.width - 2 * margin
                    view_height = self.height - 2 * margin

                    self.offset_x += dx / (view_width * self.zoom)
                    self.offset_y += dy / (view_height * self.zoom)

                    self.drag_start = event.pos

        return True

    def save_screenshot(self):
        """Save current visualization as image"""
        filename = f"genome_topology_{self.view_layer}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved screenshot: {filename}")

    def run(self):
        """Main visualization loop"""
        running = True

        while running:
            running = self.handle_events()

            # Draw
            self.screen.fill(self.colors['bg'])
            self.draw_grid()
            self.draw_edges()
            self.draw_nodes()
            self.draw_info_panel()
            self.draw_selected_info()
            self.draw_controls()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description='Genome Explorer - Visualize Champion Circuit Topology')
    parser.add_argument('--champion', type=str, default='snake_champion.pt',
                        help='Path to champion checkpoint (default: snake_champion.pt)')
    parser.add_argument('--width', type=int, default=1600,
                        help='Window width (default: 1600)')
    parser.add_argument('--height', type=int, default=900,
                        help='Window height (default: 900)')
    args = parser.parse_args()

    champion_path = Path(args.champion)

    if not champion_path.exists():
        print(f"Error: Champion file not found at {champion_path}")
        print("Make sure you have trained a champion first:")
        print("  python snake_ola.py --headless --games 100")
        return

    print(f"Loading champion from {champion_path}...")
    explorer = GenomeExplorer(champion_path, args.width, args.height)

    print("\nStarting genome explorer...")
    print("Use mouse to explore the circuit topology!")

    explorer.run()


if __name__ == "__main__":
    main()
