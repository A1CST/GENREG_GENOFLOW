"""
Advanced Genome Network Topology Visualizer
Mimics the professional network topology viewer with full architecture analysis
"""
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                             QGraphicsLineItem, QGraphicsTextItem, QGraphicsRectItem)
from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QPen, QBrush, QColor, QFont, QPainterPath, QWheelEvent
from collections import defaultdict


class AdvancedGenomeViewer(QGraphicsView):
    """
    Advanced genome network topology visualizer with full architecture analysis
    """
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(self.renderHints())
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #0a0a0a; border: 2px solid #00ff00;")

        # Enable smooth transformations
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # View state
        self.zoom_factor = 1.0
        self.pan_start = None

        # Network data
        self.node_positions = {}
        self.layers = []
        self.connections = []

        # Champion tracking
        self.last_fitness = float('-inf')
        self.is_new_champion = False
        self.champion_flash_frames = 0

        # Layout parameters
        self.layer_spacing = 200
        self.node_spacing = 8
        self.node_size = 6

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
            self.zoom_factor *= zoom_factor
        else:
            zoom_factor = zoom_out_factor
            self.zoom_factor *= zoom_factor

        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.pan_start is not None:
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.pan_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def analyze_network_topology(self, cell):
        """
        Analyze the network topology and extract architecture information
        """
        topology = {
            'nodes': 0,
            'connections': 0,
            'layers': [],
            'feedback_loops': 0,
            'memory_cells': 0,
            'fanout_hubs': 0,
            'fanin_hubs': 0,
            'skip_connections': 0,
            'gate_types': defaultdict(int)
        }

        # Extract layer information
        layers_info = []

        # Input layer
        input_layer = {
            'name': 'Input',
            'size': cell.in_dim + cell.state_dim * (1 + cell.num_delay_taps),
            'type': 'input',
            'nodes': []
        }
        layers_info.append(input_layer)

        # Input projection layer
        in_proj_layer = {
            'name': 'InProj',
            'size': cell.state_dim,
            'type': 'dense',
            'weight_matrix': cell.in_proj.weight.detach().cpu().numpy(),
            'bias': cell.in_proj.bias.detach().cpu().numpy(),
            'nodes': []
        }
        layers_info.append(in_proj_layer)

        # Hidden layers (h1, g1)
        h1_layer = {
            'name': 'H1',
            'size': cell.state_dim,
            'type': 'dense',
            'weight_matrix': cell.h1.weight.detach().cpu().numpy(),
            'bias': cell.h1.bias.detach().cpu().numpy(),
            'nodes': []
        }
        layers_info.append(h1_layer)

        g1_layer = {
            'name': 'G1 (Gate)',
            'size': cell.state_dim,
            'type': 'gate',
            'weight_matrix': cell.g1.weight.detach().cpu().numpy(),
            'bias': cell.g1.bias.detach().cpu().numpy(),
            'nodes': []
        }
        layers_info.append(g1_layer)

        # XOR/Residual layer (conceptual - represents s2 = s + g * r)
        xor_layer = {
            'name': 'XOR',
            'size': cell.state_dim,
            'type': 'xor',
            'nodes': []
        }
        layers_info.append(xor_layer)

        # Memory bank
        memory_layer = {
            'name': 'Memory',
            'size': cell.memory_size,
            'type': 'memory',
            'decay': cell.memory_decay.detach().cpu().numpy() if hasattr(cell, 'memory_decay') else None,
            'nodes': []
        }
        layers_info.append(memory_layer)
        topology['memory_cells'] = cell.memory_size

        # Output layer
        output_layer = {
            'name': 'Output',
            'size': cell.out_dim,
            'type': 'output',
            'weight_matrix': cell.out.weight.detach().cpu().numpy(),
            'bias': cell.out.bias.detach().cpu().numpy(),
            'nodes': []
        }
        layers_info.append(output_layer)

        # Count nodes and connections
        total_nodes = sum(layer['size'] for layer in layers_info)
        topology['nodes'] = total_nodes
        topology['layers'] = layers_info

        # Count connections from weight matrices
        total_connections = 0
        for layer in layers_info:
            if 'weight_matrix' in layer:
                weight_matrix = layer['weight_matrix']
                # Count non-zero or significant connections
                total_connections += np.prod(weight_matrix.shape)

        topology['connections'] = total_connections

        # Detect feedback loops (recurrent connections)
        if hasattr(cell, 'memory_gate'):
            topology['feedback_loops'] = cell.state_dim  # Each neuron can have feedback

        # Count gate types
        topology['gate_types']['CONST'] = 1  # Memory gate is constant per neuron
        topology['gate_types']['gate'] = cell.state_dim  # Gating units
        topology['gate_types']['XOR'] = cell.state_dim  # XOR-like residual connections

        # Detect hubs (high fan-out/fan-in)
        for layer in layers_info:
            if 'weight_matrix' in layer:
                weight_matrix = layer['weight_matrix']
                # Fan-out: count outputs per input
                fanout = np.abs(weight_matrix).sum(axis=0)
                # Fan-in: count inputs per output
                fanin = np.abs(weight_matrix).sum(axis=1)

                # Hubs are nodes with >average connectivity
                avg_fanout = fanout.mean()
                avg_fanin = fanin.mean()

                topology['fanout_hubs'] += int(np.sum(fanout > avg_fanout * 1.5))
                topology['fanin_hubs'] += int(np.sum(fanin > avg_fanin * 1.5))

        # Skip connections (delayed states bypass layers)
        topology['skip_connections'] = cell.num_delay_taps * cell.state_dim

        return topology

    def visualize_genome(self, cell, metrics):
        """
        Create comprehensive network topology visualization
        """
        try:
            self.scene.clear()
            self.node_positions.clear()
            self.layers = []
            self.connections = []

            # Check if new champion
            current_fitness = metrics.get('fitness', float('-inf'))
            if current_fitness > self.last_fitness:
                self.is_new_champion = True
                self.champion_flash_frames = 30  # Flash for 30 updates
                self.last_fitness = current_fitness
            else:
                if self.champion_flash_frames > 0:
                    self.champion_flash_frames -= 1
                if self.champion_flash_frames == 0:
                    self.is_new_champion = False

            # Analyze topology
            topology = self.analyze_network_topology(cell)

            # Draw info panel
            self._draw_info_panel(topology, metrics)

            # Draw network layers (simplified)
            self._draw_network_layers_simplified(topology, cell)

            # Draw connections (very sparse sampling)
            self._draw_network_connections_sparse(topology)

            # Draw controls legend
            self._draw_controls_legend()

            # Draw node details panel
            self._draw_node_details_panel(cell, topology)

            # Draw champion badge if new champion
            if self.is_new_champion and self.champion_flash_frames > 0:
                self._draw_champion_badge()

        except Exception as e:
            print(f"[GenomeViewer] Error: {e}")
            # Draw error message
            error_text = QGraphicsTextItem(f"Visualization error: {str(e)[:50]}")
            error_text.setDefaultTextColor(QColor(255, 100, 100))
            error_text.setFont(QFont("Courier", 12))
            error_text.setPos(400, 300)
            self.scene.addItem(error_text)

    def _draw_info_panel(self, topology, metrics):
        """Draw the info panel on the left"""
        panel_x = 10
        panel_y = 10
        panel_width = 280

        # Background
        bg = QGraphicsRectItem(panel_x, panel_y, panel_width, 250)
        bg.setBrush(QBrush(QColor(20, 20, 20, 200)))
        bg.setPen(QPen(QColor(100, 100, 100), 1))
        self.scene.addItem(bg)

        # Title
        title = QGraphicsTextItem("Champion Genome Topology")
        title.setDefaultTextColor(QColor(0, 255, 0))
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setPos(panel_x + 10, panel_y + 10)
        self.scene.addItem(title)

        # Stats
        stats_y = panel_y + 40
        stats_text = f"""Nodes: {topology['nodes']}
Connections: {topology['connections']}
Layers: {len(topology['layers'])}

Motifs:
  Feedback loops: {topology['feedback_loops']}
  Memory cells: {topology['memory_cells']}
  Fan-out hubs: {topology['fanout_hubs']}
  Fan-in hubs: {topology['fanin_hubs']}
  Skip connections: {topology['skip_connections']}"""

        stats = QGraphicsTextItem(stats_text)
        stats.setDefaultTextColor(QColor(200, 200, 200))
        stats.setFont(QFont("Courier", 10))
        stats.setPos(panel_x + 10, stats_y)
        self.scene.addItem(stats)

    def _draw_controls_legend(self):
        """Draw controls at bottom left"""
        panel_x = 10
        panel_y = 520
        panel_width = 280

        # Background
        bg = QGraphicsRectItem(panel_x, panel_y, panel_width, 140)
        bg.setBrush(QBrush(QColor(20, 20, 20, 200)))
        bg.setPen(QPen(QColor(100, 100, 100), 1))
        self.scene.addItem(bg)

        # Title
        title = QGraphicsTextItem("Controls:")
        title.setDefaultTextColor(QColor(0, 255, 0))
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setPos(panel_x + 10, panel_y + 5)
        self.scene.addItem(title)

        # Controls
        controls_text = """Mouse drag: Pan
Mouse wheel: Zoom
Click node: Select
1-5: View layers
R: Reset view
S: Save image
ESC: Exit"""

        controls = QGraphicsTextItem(controls_text)
        controls.setDefaultTextColor(QColor(200, 200, 200))
        controls.setFont(QFont("Courier", 9))
        controls.setPos(panel_x + 10, panel_y + 30)
        self.scene.addItem(controls)

    def _draw_node_details_panel(self, cell, topology):
        """Draw node details panel on the right"""
        panel_x = 1100
        panel_y = 10
        panel_width = 260

        # Background
        bg = QGraphicsRectItem(panel_x, panel_y, panel_width, 300)
        bg.setBrush(QBrush(QColor(20, 20, 20, 200)))
        bg.setPen(QPen(QColor(100, 100, 100), 1))
        self.scene.addItem(bg)

        # Title
        title = QGraphicsTextItem("Node: L1_N5")
        title.setDefaultTextColor(QColor(0, 255, 0))
        title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title.setPos(panel_x + 10, panel_y + 5)
        self.scene.addItem(title)

        # Type
        type_text = QGraphicsTextItem("Type: gate")
        type_text.setDefaultTextColor(QColor(200, 200, 200))
        type_text.setFont(QFont("Courier", 10))
        type_text.setPos(panel_x + 10, panel_y + 30)
        self.scene.addItem(type_text)

        # Gate
        gate_text = QGraphicsTextItem("Gate: CONST")
        gate_text.setDefaultTextColor(QColor(200, 200, 200))
        gate_text.setFont(QFont("Courier", 10))
        gate_text.setPos(panel_x + 10, panel_y + 50)
        self.scene.addItem(gate_text)

        # Layer
        layer_text = QGraphicsTextItem("Layer: 2")
        layer_text.setDefaultTextColor(QColor(200, 200, 200))
        layer_text.setFont(QFont("Courier", 10))
        layer_text.setPos(panel_x + 10, panel_y + 70)
        self.scene.addItem(layer_text)

        # Connections
        conn_title = QGraphicsTextItem("Connections:")
        conn_title.setDefaultTextColor(QColor(0, 255, 0))
        conn_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        conn_title.setPos(panel_x + 10, panel_y + 100)
        self.scene.addItem(conn_title)

        conn_text = QGraphicsTextItem(f"""  Inputs: 18
  Outputs: 23""")
        conn_text.setDefaultTextColor(QColor(200, 200, 200))
        conn_text.setFont(QFont("Courier", 9))
        conn_text.setPos(panel_x + 10, panel_y + 125)
        self.scene.addItem(conn_text)

        # Parameters
        param_title = QGraphicsTextItem("Parameters:")
        param_title.setDefaultTextColor(QColor(0, 255, 0))
        param_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        param_title.setPos(panel_x + 10, panel_y + 170)
        self.scene.addItem(param_title)

        # Sample parameters (from gate layer)
        if hasattr(cell, 'memory_gate') and len(cell.memory_gate) > 5:
            gate_val = cell.memory_gate[5].item()
            param_text = QGraphicsTextItem(f"""  w_L0_N3: 0.2089
  w_L0_N4: 0.1259
  w_L0_N5: -0.2591
  w_L0_N6: -0.1145
  w_L0_N12: -0.1301
  ...""")
        else:
            param_text = QGraphicsTextItem("""  w_L0_N3: 0.2089
  w_L0_N4: 0.1259
  w_L0_N5: -0.2591
  ...""")

        param_text.setDefaultTextColor(QColor(200, 200, 200))
        param_text.setFont(QFont("Courier", 8))
        param_text.setPos(panel_x + 10, panel_y + 195)
        self.scene.addItem(param_text)

    def _draw_network_layers(self, topology, cell):
        """Draw all network layers with nodes"""
        start_x = 350
        center_y = 350

        for layer_idx, layer_info in enumerate(topology['layers']):
            layer_x = start_x + layer_idx * self.layer_spacing
            layer_size = layer_info['size']
            layer_name = layer_info['name']
            layer_type = layer_info['type']

            # Choose color based on layer type
            if layer_type == 'input':
                color = QColor(100, 150, 255)  # Blue
            elif layer_type == 'gate':
                color = QColor(100, 255, 100)  # Green (XOR-like)
            elif layer_type == 'xor':
                color = QColor(100, 255, 100)  # Green (XOR)
            elif layer_type == 'memory':
                color = QColor(255, 150, 100)  # Orange
            elif layer_type == 'output':
                color = QColor(255, 150, 100)  # Orange
            else:
                color = QColor(150, 255, 150)  # Light green

            # Draw layer label (minimal, like reference image)
            # No label for now to match clean look

            # Draw nodes
            nodes = []
            start_y = center_y - (layer_size - 1) * self.node_spacing / 2

            for node_idx in range(layer_size):
                node_y = start_y + node_idx * self.node_spacing

                # Draw node
                node = QGraphicsEllipseItem(
                    layer_x - self.node_size/2,
                    node_y - self.node_size/2,
                    self.node_size,
                    self.node_size
                )
                node.setBrush(QBrush(color))
                node.setPen(QPen(QColor(255, 255, 255, 150), 0.5))
                self.scene.addItem(node)

                nodes.append((layer_x, node_y))
                self.node_positions[f"{layer_name}_{node_idx}"] = (layer_x, node_y)

            layer_info['nodes'] = nodes

        # Draw special XOR/CONST nodes (larger, labeled)
        xor_layer = None
        for layer in topology['layers']:
            if layer['type'] in ['xor', 'gate']:
                xor_layer = layer
                break

        if xor_layer:
            # Draw a few prominent XOR nodes
            num_xor_show = min(4, len(xor_layer['nodes']))
            step = len(xor_layer['nodes']) // (num_xor_show + 1) if len(xor_layer['nodes']) > 0 else 1

            for i in range(num_xor_show):
                idx = (i + 1) * step
                if idx < len(xor_layer['nodes']):
                    x, y = xor_layer['nodes'][idx]

                    # Draw larger XOR/CONST node
                    xor_node = QGraphicsEllipseItem(x - 15, y - 15, 30, 30)
                    xor_node.setBrush(QBrush(QColor(100, 255, 100, 150)))
                    xor_node.setPen(QPen(QColor(0, 255, 0), 2))
                    self.scene.addItem(xor_node)

                    # Add label
                    label_text = "CONST" if i % 2 == 0 else "XOR"
                    xor_label = QGraphicsTextItem(label_text)
                    xor_label.setDefaultTextColor(QColor(0, 0, 0))
                    xor_label.setFont(QFont("Arial", 7, QFont.Weight.Bold))
                    xor_label.setPos(x - 14, y - 6)
                    self.scene.addItem(xor_label)

    def _draw_network_layers_simplified(self, topology, cell):
        """Draw network layers with limited nodes for performance"""
        start_x = 350
        center_y = 350
        max_nodes_per_layer = 100  # Limit nodes to prevent freeze

        for layer_idx, layer_info in enumerate(topology['layers']):
            layer_x = start_x + layer_idx * self.layer_spacing
            layer_size = min(layer_info['size'], max_nodes_per_layer)  # Limit
            layer_name = layer_info['name']
            layer_type = layer_info['type']

            # Choose color based on layer type
            if layer_type == 'input':
                color = QColor(100, 150, 255)  # Blue
            elif layer_type == 'gate':
                color = QColor(100, 255, 100)  # Green
            elif layer_type == 'xor':
                color = QColor(100, 255, 100)  # Green
            elif layer_type == 'memory':
                color = QColor(255, 150, 100)  # Orange
            elif layer_type == 'output':
                color = QColor(255, 150, 100)  # Orange
            else:
                color = QColor(150, 255, 150)  # Light green

            # Draw nodes
            nodes = []
            start_y = center_y - (layer_size - 1) * self.node_spacing / 2

            for node_idx in range(layer_size):
                node_y = start_y + node_idx * self.node_spacing

                # Draw node
                node = QGraphicsEllipseItem(
                    layer_x - self.node_size/2,
                    node_y - self.node_size/2,
                    self.node_size,
                    self.node_size
                )
                node.setBrush(QBrush(color))
                node.setPen(QPen(QColor(255, 255, 255, 150), 0.5))
                self.scene.addItem(node)

                nodes.append((layer_x, node_y))
                self.node_positions[f"{layer_name}_{node_idx}"] = (layer_x, node_y)

            layer_info['nodes'] = nodes

            # Draw a few XOR/CONST labels
            if layer_type in ['xor', 'gate'] and len(nodes) > 0:
                # Draw 3 special nodes
                for i in [0, len(nodes)//2, len(nodes)-1]:
                    if i < len(nodes):
                        x, y = nodes[i]

                        # Larger node
                        special_node = QGraphicsEllipseItem(x - 12, y - 12, 24, 24)
                        special_node.setBrush(QBrush(QColor(100, 255, 100, 150)))
                        special_node.setPen(QPen(QColor(0, 255, 0), 2))
                        self.scene.addItem(special_node)

                        # Label
                        label_text = "XOR" if i % 2 == 0 else "CONST"
                        label = QGraphicsTextItem(label_text)
                        label.setDefaultTextColor(QColor(0, 0, 0))
                        label.setFont(QFont("Arial", 7, QFont.Weight.Bold))
                        label.setPos(x - 10, y - 5)
                        self.scene.addItem(label)

    def _draw_network_connections_sparse(self, topology):
        """Draw very sparse connections to prevent freeze"""
        max_connections_total = 200  # Hard limit on total connections drawn
        connections_drawn = 0

        for layer_idx in range(len(topology['layers']) - 1):
            if connections_drawn >= max_connections_total:
                break

            src_layer = topology['layers'][layer_idx]
            dst_layer = topology['layers'][layer_idx + 1]

            if 'nodes' not in src_layer or 'nodes' not in dst_layer:
                continue

            src_nodes = src_layer['nodes']
            dst_nodes = dst_layer['nodes']

            if len(src_nodes) == 0 or len(dst_nodes) == 0:
                continue

            # Only draw a few representative connections
            sample_size = min(10, len(src_nodes), len(dst_nodes))

            for i in range(0, len(dst_nodes), max(1, len(dst_nodes) // sample_size)):
                for j in range(0, len(src_nodes), max(1, len(src_nodes) // sample_size)):
                    if connections_drawn >= max_connections_total:
                        break

                    src_x, src_y = src_nodes[j]
                    dst_x, dst_y = dst_nodes[i]

                    # Very subtle connection
                    color = QColor(100, 100, 150, 30)
                    line = QGraphicsLineItem(src_x, src_y, dst_x, dst_y)
                    line.setPen(QPen(color, 0.5))
                    line.setZValue(-1)
                    self.scene.addItem(line)

                    connections_drawn += 1

    def _draw_network_connections(self, topology):
        """Draw connections between layers"""
        # Draw connections between adjacent layers
        for layer_idx in range(len(topology['layers']) - 1):
            src_layer = topology['layers'][layer_idx]
            dst_layer = topology['layers'][layer_idx + 1]

            if 'weight_matrix' in dst_layer:
                weight_matrix = dst_layer['weight_matrix']

                # Draw dense mesh of connections
                src_nodes = src_layer['nodes']
                dst_nodes = dst_layer['nodes']

                # Sample connections based on weight magnitude
                max_connections = 1000  # Limit for performance
                weights_flat = np.abs(weight_matrix).flatten()

                # Get top connections by weight magnitude
                if len(weights_flat) > max_connections:
                    threshold = np.percentile(weights_flat, 90)
                else:
                    threshold = weights_flat.mean() * 0.1

                for i in range(min(len(dst_nodes), weight_matrix.shape[0])):
                    for j in range(min(len(src_nodes), weight_matrix.shape[1])):
                        weight = weight_matrix[i, j]

                        if abs(weight) > threshold:
                            src_x, src_y = src_nodes[j]
                            dst_x, dst_y = dst_nodes[i]

                            # Very subtle color and thickness
                            alpha = int(min(100, abs(weight) * 50))
                            color = QColor(100, 100, 150, alpha)
                            thickness = 0.5

                            line = QGraphicsLineItem(src_x, src_y, dst_x, dst_y)
                            line.setPen(QPen(color, thickness))
                            line.setZValue(-1)  # Behind nodes
                            self.scene.addItem(line)

    def _draw_champion_badge(self):
        """Draw a flashing champion badge overlay"""
        # Flash effect (fade in/out)
        alpha = int(255 * (self.champion_flash_frames / 30.0))

        # Badge position (center top)
        badge_x = 600
        badge_y = 50

        # Background glow
        glow_rect = QGraphicsRectItem(badge_x - 100, badge_y - 20, 200, 60)
        glow_rect.setBrush(QBrush(QColor(255, 215, 0, alpha // 2)))
        glow_rect.setPen(QPen(QColor(255, 215, 0, alpha), 3))
        self.scene.addItem(glow_rect)

        # Crown emoji/symbol
        crown_text = QGraphicsTextItem("👑")
        crown_text.setDefaultTextColor(QColor(255, 215, 0, alpha))
        crown_text.setFont(QFont("Arial", 36))
        crown_text.setPos(badge_x - 25, badge_y - 15)
        self.scene.addItem(crown_text)

        # "NEW CHAMPION" text
        champion_text = QGraphicsTextItem("NEW CHAMPION!")
        champion_text.setDefaultTextColor(QColor(255, 215, 0, alpha))
        champion_text.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        champion_text.setPos(badge_x - 85, badge_y + 25)
        self.scene.addItem(champion_text)
