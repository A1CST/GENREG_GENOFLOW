"""
OLA Desktop Prediction GUI
Tests OLA's ability to predict next frame from raw pixel data
"""
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSlider, QSpinBox,
                             QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
                             QGraphicsLineItem, QGraphicsTextItem)
from PyQt6.QtCore import QTimer, Qt, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPen, QBrush, QColor, QFont
import mss

from stabilized_ola import StabilizedOLA, StabilizedOLAConfig
from advanced_genome_viewer import AdvancedGenomeViewer


class GenomeViewer(QGraphicsView):
    """Real-time genome network visualizer"""
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(self.renderHints() | self.renderHints())
        self.setMinimumSize(600, 400)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #00ff00;")

        # Node positions
        self.node_positions = {}
        self.layer_spacing = 150
        self.node_spacing = 40

    def visualize_genome(self, cell, metrics):
        """
        Visualize the EvoCell architecture as a node network.
        Shows layers, connections, and special components (memory, gates, etc.)
        """
        self.scene.clear()
        self.node_positions.clear()

        # Extract architecture info
        state_dim = cell.state_dim
        in_dim = cell.in_dim
        out_dim = cell.out_dim
        memory_size = cell.memory_size
        num_delays = cell.num_delay_taps

        # Color scheme
        input_color = QColor(100, 150, 255)  # Blue
        hidden_color = QColor(150, 255, 150)  # Green
        output_color = QColor(255, 150, 100)  # Orange
        memory_color = QColor(255, 100, 255)  # Magenta
        delay_color = QColor(255, 255, 100)  # Yellow

        # Calculate simplified representation (show subset of neurons)
        max_nodes_per_layer = 12
        in_sample = min(max_nodes_per_layer, in_dim)
        hidden_sample = min(max_nodes_per_layer, state_dim)
        out_sample = min(max_nodes_per_layer, out_dim)
        mem_sample = min(8, memory_size)

        x_offset = 50
        y_center = 200

        # Layer 1: Input nodes (simplified)
        layer_x = x_offset
        self._draw_layer("Input", in_sample, layer_x, y_center, input_color, f"{in_dim}D")

        # Layer 2: Delayed states
        layer_x += self.layer_spacing
        for i in range(num_delays):
            y_pos = y_center - 60 + i * 40
            node = self._draw_node(layer_x, y_pos, delay_color, f"t-{i+1}")
            self.node_positions[f"delay_{i}"] = (layer_x, y_pos)

        # Layer 3: Hidden state (with memory gates)
        layer_x += self.layer_spacing
        hidden_nodes = self._draw_layer("Hidden", hidden_sample, layer_x, y_center,
                                       hidden_color, f"{state_dim}D")

        # Show memory gate strengths as node intensity
        if hasattr(cell, 'memory_gate'):
            gate_values = cell.memory_gate.detach().cpu().numpy()
            # Color nodes by gate value (persistence)
            for i, (node_x, node_y) in enumerate(hidden_nodes):
                if i < len(gate_values):
                    gate_val = float(gate_values[i])
                    # High gate = persistent (darker), low gate = fast (brighter)
                    intensity = int(255 * (1 - gate_val * 0.5))
                    node_color = QColor(intensity, 255, intensity)
                    self._draw_node(node_x, node_y, node_color, f"g:{gate_val:.2f}", size=20)

        # Layer 4: Memory bank
        layer_x += self.layer_spacing
        mem_nodes = self._draw_layer("Memory", mem_sample, layer_x, y_center,
                                     memory_color, f"{memory_size}D")

        # Show memory decay rates
        if hasattr(cell, 'memory_decay'):
            decay_values = cell.memory_decay.detach().cpu().numpy()
            for i, (node_x, node_y) in enumerate(mem_nodes):
                if i < len(decay_values):
                    decay_val = float(decay_values[i])
                    # Draw decay rate as text
                    text = QGraphicsTextItem(f"λ:{decay_val:.2f}")
                    text.setDefaultTextColor(QColor(200, 200, 200))
                    text.setFont(QFont("Courier", 8))
                    text.setPos(node_x + 15, node_y - 10)
                    self.scene.addItem(text)

        # Layer 5: Output
        layer_x += self.layer_spacing
        self._draw_layer("Output", out_sample, layer_x, y_center, output_color, f"{out_dim}D")

        # Draw simplified connections (sparse sampling)
        self._draw_connections()

        # Add stats overlay
        self._draw_stats_overlay(metrics, state_dim, memory_size)

    def _draw_layer(self, name, num_nodes, x, y_center, color, label):
        """Draw a layer of nodes"""
        nodes = []
        start_y = y_center - (num_nodes - 1) * self.node_spacing / 2

        # Layer label
        text = QGraphicsTextItem(name)
        text.setDefaultTextColor(QColor(255, 255, 255))
        text.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        text.setPos(x - 20, start_y - 40)
        self.scene.addItem(text)

        # Dimension label
        dim_text = QGraphicsTextItem(label)
        dim_text.setDefaultTextColor(QColor(180, 180, 180))
        dim_text.setFont(QFont("Courier", 8))
        dim_text.setPos(x - 15, start_y - 25)
        self.scene.addItem(dim_text)

        for i in range(num_nodes):
            y = start_y + i * self.node_spacing
            self._draw_node(x, y, color, str(i))
            nodes.append((x, y))
            self.node_positions[f"{name}_{i}"] = (x, y)

        return nodes

    def _draw_node(self, x, y, color, label, size=15):
        """Draw a single node"""
        node = QGraphicsEllipseItem(x - size/2, y - size/2, size, size)
        node.setBrush(QBrush(color))
        node.setPen(QPen(QColor(255, 255, 255), 1))
        self.scene.addItem(node)

        # Label (optional, only for important nodes)
        if label and len(label) < 6:
            text = QGraphicsTextItem(label)
            text.setDefaultTextColor(QColor(200, 200, 200))
            text.setFont(QFont("Courier", 7))
            text.setPos(x + size, y - 8)
            self.scene.addItem(text)

        return node

    def _draw_connections(self):
        """Draw sparse sample of connections between layers"""
        # Draw a few sample connections to show flow
        pen = QPen(QColor(100, 100, 100, 80), 1)

        # Sample a few connections
        for key1, (x1, y1) in list(self.node_positions.items())[:20]:
            for key2, (x2, y2) in list(self.node_positions.items())[20:30]:
                if x2 > x1:  # Only forward connections
                    line = QGraphicsLineItem(x1, y1, x2, y2)
                    line.setPen(pen)
                    self.scene.addItem(line)

    def _draw_stats_overlay(self, metrics, state_dim, memory_size):
        """Draw statistics overlay"""
        stats_x = 50
        stats_y = 350

        stats_text = f"""State Dim: {state_dim}
Memory Size: {memory_size}
Mutation Rate: {metrics.get('mutation_rate', 0):.4f}
Fitness: {metrics.get('fitness', 0):.4f}
Rollbacks: {metrics.get('rollback_count', 0)}"""

        text = QGraphicsTextItem(stats_text)
        text.setDefaultTextColor(QColor(0, 255, 0))
        text.setFont(QFont("Courier", 9))
        text.setPos(stats_x, stats_y)
        self.scene.addItem(text)


class OLADesktopGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLA Desktop Frame Predictor + Genome Viewer")
        self.setGeometry(100, 100, 1600, 900)

        # Capture settings
        self.target_fps = 5
        self.capture_width = 160
        self.capture_height = 120
        self.is_running = False
        self.prediction_horizon = 1  # How many frames ahead to predict

        # Screen capture
        self.sct = mss.mss()

        # Frame buffers (circular buffer for multi-frame prediction)
        self.frame_history = []  # Store last N frames
        self.max_history = 20  # Keep enough history for prediction
        self.current_frame_raw = None
        self.prev_frame_raw = None
        self.predicted_frame = None

        # Metrics
        self.frame_count = 0
        self.total_mse = 0.0
        self.total_accuracy = 0.0

        # Genome viewer update control
        self.genome_update_interval = 10  # Update genome view every N frames
        self.last_genome_update = 0

        # Champion genome saving
        self.checkpoint_interval = 50  # Save champion every N frames
        self.last_checkpoint = 0
        self.best_fitness_ever = float('-inf')

        # Create checkpoint directory
        import os
        self.checkpoint_dir = "E:\\Genome\\checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize OLA - Force CUDA
        self.device = torch.device("cuda")
        print(f"[GPU] Using device: {self.device}")
        print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[GPU] GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"[GPU] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        self.init_ola()

        # Setup UI
        self.init_ui()

        # Initial genome visualization
        self.genome_viewer.visualize_genome(self.ola.champion, {
            'mutation_rate': self.ola.mutation_rate,
            'fitness': 0.0,
            'rollback_count': 0
        })

        # Timer for capture
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_and_predict)

    def init_ola(self):
        """Initialize the OLA model"""
        # Flatten image dimensions
        latent_dim = self.capture_width * self.capture_height * 3  # RGB

        cfg = StabilizedOLAConfig(
            in_dim=latent_dim,  # raw pixel input
            out_dim=latent_dim,  # predict next frame
            state_dim=256,
            mutation_rate=0.1,
            mutation_decay=0.999,
            mutation_floor=1e-5,
            stability_factor=0.98,
            rollback_threshold=5.0,
            grow_prob=0.01,
            max_state_dim=1024,
            cosine_weight=0.2,
            reg_weight=1e-4,
            device=str(self.device),
            num_champions=3,
            variance_window=10,
            variance_threshold=0.01,
            mutation_decrease=0.9,
            mutation_increase=1.05,
            temporal_beta=0.02,
            max_no_improve_age=20
        )

        self.ola = StabilizedOLA(cfg)
        print(f"[OLA] Initialized with input/output dim: {latent_dim}")
        print(f"[OLA] Device: {self.device}")

    def init_ui(self):
        """Setup the GUI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_layout = QHBoxLayout()

        # Start/Stop button
        self.start_btn = QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.toggle_capture)
        control_layout.addWidget(self.start_btn)

        # FPS control
        control_layout.addWidget(QLabel("Target FPS:"))
        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setMinimum(1)
        self.fps_slider.setMaximum(30)
        self.fps_slider.setValue(self.target_fps)
        self.fps_slider.valueChanged.connect(self.update_fps)
        control_layout.addWidget(self.fps_slider)

        self.fps_label = QLabel(f"{self.target_fps} FPS")
        control_layout.addWidget(self.fps_label)

        # Prediction Horizon control
        control_layout.addWidget(QLabel("Predict:"))
        self.horizon_slider = QSlider(Qt.Orientation.Horizontal)
        self.horizon_slider.setMinimum(1)
        self.horizon_slider.setMaximum(10)
        self.horizon_slider.setValue(self.prediction_horizon)
        self.horizon_slider.valueChanged.connect(self.update_horizon)
        control_layout.addWidget(self.horizon_slider)

        self.horizon_label = QLabel(f"{self.prediction_horizon} frame(s)")
        control_layout.addWidget(self.horizon_label)

        # Resolution control
        control_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setMinimum(80)
        self.width_spin.setMaximum(640)
        self.width_spin.setSingleStep(20)
        self.width_spin.setValue(self.capture_width)
        self.width_spin.valueChanged.connect(self.update_resolution)
        control_layout.addWidget(self.width_spin)

        control_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setMinimum(60)
        self.height_spin.setMaximum(480)
        self.height_spin.setSingleStep(20)
        self.height_spin.setValue(self.capture_height)
        self.height_spin.valueChanged.connect(self.update_resolution)
        control_layout.addWidget(self.height_spin)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Main content layout (frames + genome viewer)
        content_layout = QHBoxLayout()

        # Left side: Frame display
        frames_layout = QVBoxLayout()

        # Image display area
        image_layout = QHBoxLayout()

        # Actual frame column
        actual_col = QVBoxLayout()
        actual_col.addWidget(QLabel("Actual Current Frame"))
        self.actual_label = QLabel()
        self.actual_label.setMinimumSize(350, 250)
        self.actual_label.setStyleSheet("border: 2px solid green; background-color: black;")
        self.actual_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        actual_col.addWidget(self.actual_label)
        image_layout.addLayout(actual_col)

        # Predicted frame column
        pred_col = QVBoxLayout()
        pred_col.addWidget(QLabel("OLA Predicted Next Frame"))
        self.predicted_label = QLabel()
        self.predicted_label.setMinimumSize(350, 250)
        self.predicted_label.setStyleSheet("border: 2px solid blue; background-color: black;")
        self.predicted_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred_col.addWidget(self.predicted_label)
        image_layout.addLayout(pred_col)

        frames_layout.addLayout(image_layout)
        content_layout.addLayout(frames_layout)

        # Right side: Genome Viewer
        genome_layout = QVBoxLayout()
        genome_label = QLabel("Live Genome Network Viewer")
        genome_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #00ff00;")
        genome_layout.addWidget(genome_label)

        self.genome_viewer = AdvancedGenomeViewer()
        genome_layout.addWidget(self.genome_viewer)

        content_layout.addLayout(genome_layout)

        main_layout.addLayout(content_layout)

        # Metrics display
        metrics_layout = QVBoxLayout()

        self.frame_label = QLabel("Frames: 0")
        self.mse_label = QLabel("MSE: 0.0000")
        self.accuracy_label = QLabel("Accuracy: 0.00%")
        self.ola_loss_label = QLabel("OLA Loss: N/A")
        self.multistep_loss_label = QLabel("Multi-step Loss: N/A")
        self.mutation_label = QLabel("Mutation Rate: N/A")
        self.state_dim_label = QLabel("State Dim: 256")
        self.fps_actual_label = QLabel("Actual FPS: 0.0")

        for label in [self.frame_label, self.mse_label, self.accuracy_label,
                     self.ola_loss_label, self.multistep_loss_label, self.mutation_label,
                     self.state_dim_label, self.fps_actual_label]:
            label.setStyleSheet("font-size: 14px; font-weight: bold;")
            metrics_layout.addWidget(label)

        main_layout.addLayout(metrics_layout)

    def toggle_capture(self):
        """Start or stop the capture"""
        if not self.is_running:
            self.is_running = True
            self.start_btn.setText("Stop Capture")
            interval_ms = int(1000 / self.target_fps)
            self.timer.start(interval_ms)
            self.frame_count = 0
            self.total_mse = 0.0
            self.total_accuracy = 0.0
            self.last_capture_time = time.time()
        else:
            self.is_running = False
            self.start_btn.setText("Start Capture")
            self.timer.stop()

    def update_fps(self, value):
        """Update target FPS"""
        self.target_fps = value
        self.fps_label.setText(f"{value} FPS")
        if self.is_running:
            interval_ms = int(1000 / self.target_fps)
            self.timer.setInterval(interval_ms)

    def update_horizon(self, value):
        """Update prediction horizon"""
        self.prediction_horizon = value
        self.horizon_label.setText(f"{value} frame(s)")

    def update_resolution(self):
        """Update capture resolution - requires restart"""
        if self.is_running:
            self.toggle_capture()
        self.capture_width = self.width_spin.value()
        self.capture_height = self.height_spin.value()
        self.init_ola()  # Reinitialize OLA with new dimensions

    def capture_screen(self):
        """Capture the primary monitor"""
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)

        # Convert to numpy array - mss returns BGRA, convert to RGB
        img = np.array(screenshot)
        # Convert BGRA to RGB
        img_rgb = img[:, :, [2, 1, 0]]  # Swap BGR to RGB, ignore alpha

        # Resize to target resolution with high quality
        from PIL import Image
        img_pil = Image.fromarray(img_rgb.astype(np.uint8))
        img_resized = img_pil.resize((self.capture_width, self.capture_height), Image.Resampling.LANCZOS)
        img_np = np.array(img_resized, dtype=np.uint8)

        return img_np

    def numpy_to_qimage(self, img_np):
        """Convert numpy array to QImage"""
        height, width, channel = img_np.shape
        bytes_per_line = 3 * width
        img_np = np.ascontiguousarray(img_np)
        qimg = QImage(img_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg

    def capture_and_predict(self):
        """Main loop: capture, predict, score"""
        start_time = time.time()

        # Capture current frame
        frame_np = self.capture_screen()  # [H, W, 3] uint8

        # Convert to tensor and normalize to [-1, 1]
        frame_tensor = torch.from_numpy(frame_np).float().to(self.device)
        frame_tensor = (frame_tensor / 127.5) - 1.0  # [H, W, 3] in [-1, 1]

        # Flatten to vector
        z_current = frame_tensor.flatten().unsqueeze(0)  # [1, H*W*3]

        # Add to frame history
        self.frame_history.append(z_current.detach().clone())
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        # Store previous frame
        z_prev = None
        if self.prev_frame_raw is not None:
            z_prev = self.prev_frame_raw

        # OLA step (train on 1-step prediction)
        metrics = self.ola.step(z_current, z_current, z_prev)

        # Multi-step prediction
        if len(self.frame_history) > self.prediction_horizon:
            # Get the frame from N steps ago
            z_start = self.frame_history[-(self.prediction_horizon + 1)]

            # Predict N steps forward iteratively
            z_pred = z_start.clone()
            for step in range(self.prediction_horizon):
                z_pred = self.ola.predict(z_pred, z_pred)

            # The target is the current frame
            z_target = z_current

            # Compute accuracy
            mse = F.mse_loss(z_pred, z_target).item()

            # Convert MSE to accuracy percentage (0-100%)
            # Assuming max possible MSE is ~4 (range [-1,1]^2)
            accuracy = max(0, 100 * (1 - mse / 4.0))

            self.total_mse += mse
            self.total_accuracy += accuracy

            # Convert prediction to image
            pred_np = z_pred.cpu().squeeze().reshape(self.capture_height, self.capture_width, 3).numpy()
            pred_np = ((pred_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Display predicted frame with smooth scaling
            pred_qimg = self.numpy_to_qimage(pred_np)
            pred_pixmap = QPixmap.fromImage(pred_qimg)
            scaled_pred = pred_pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.FastTransformation)
            self.predicted_label.setPixmap(scaled_pred)

            # Update metrics
            avg_mse = self.total_mse / self.frame_count if self.frame_count > 0 else 0
            avg_accuracy = self.total_accuracy / self.frame_count if self.frame_count > 0 else 0

            self.mse_label.setText(f"MSE: {mse:.4f} (avg: {avg_mse:.4f})")
            self.accuracy_label.setText(f"Accuracy: {accuracy:.2f}% (avg: {avg_accuracy:.2f}%)")
            self.ola_loss_label.setText(f"OLA Loss: {metrics.get('ola_loss', 0):.4f}")
            self.multistep_loss_label.setText(f"Multi-step Loss: {metrics.get('multistep_loss', 0):.4f}")
            self.mutation_label.setText(f"Mutation Rate: {metrics.get('mutation_rate', 0):.6f}")
            self.state_dim_label.setText(f"State Dim: {int(metrics.get('state_dim', 256))}")

            # Check if new champion (better fitness)
            current_fitness = metrics.get('fitness', float('-inf'))
            is_new_champion = current_fitness > self.best_fitness_ever

            if is_new_champion:
                self.best_fitness_ever = current_fitness
                print(f"[CHAMPION] New champion at frame {self.frame_count}! Fitness: {current_fitness:.6f}")

                # Update genome viewer immediately when new champion emerges
                try:
                    self.genome_viewer.visualize_genome(self.ola.champion, metrics)
                    self.last_genome_update = self.frame_count
                except Exception as e:
                    print(f"[GenomeViewer] Update failed: {e}")
            else:
                # Update genome viewer periodically (not every frame to prevent lag)
                if self.frame_count - self.last_genome_update >= self.genome_update_interval:
                    try:
                        self.genome_viewer.visualize_genome(self.ola.champion, metrics)
                        self.last_genome_update = self.frame_count
                    except Exception as e:
                        print(f"[GenomeViewer] Update failed: {e}")

            # Save champion checkpoint every N frames
            if self.frame_count - self.last_checkpoint >= self.checkpoint_interval:
                self.save_champion_checkpoint(metrics)
                self.last_checkpoint = self.frame_count

        # Display actual frame with smooth scaling
        actual_qimg = self.numpy_to_qimage(frame_np)
        actual_pixmap = QPixmap.fromImage(actual_qimg)
        scaled_actual = actual_pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.FastTransformation)
        self.actual_label.setPixmap(scaled_actual)

        # Update frame count
        self.frame_count += 1
        self.frame_label.setText(f"Frames: {self.frame_count}")

        # Calculate actual FPS
        current_time = time.time()
        if hasattr(self, 'last_capture_time'):
            actual_fps = 1.0 / (current_time - self.last_capture_time)
            self.fps_actual_label.setText(f"Actual FPS: {actual_fps:.1f}")
        self.last_capture_time = current_time

        # Store for next iteration
        self.prev_frame_raw = z_current.detach().clone()
        self.current_frame_raw = frame_np

    def save_champion_checkpoint(self, metrics):
        """Save the current champion genome to a checkpoint file"""
        try:
            import os
            import datetime

            # Create filename with timestamp and frame number
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fitness = metrics.get('fitness', 0)
            filename = f"champion_frame{self.frame_count:06d}_{timestamp}_fitness{fitness:.4f}.pt"
            filepath = os.path.join(self.checkpoint_dir, filename)

            # Prepare metadata
            metadata = {
                'frame_count': self.frame_count,
                'timestamp': timestamp,
                'fitness': fitness,
                'ola_loss': metrics.get('ola_loss', 0),
                'multistep_loss': metrics.get('multistep_loss', 0),
                'mutation_rate': metrics.get('mutation_rate', 0),
                'state_dim': metrics.get('state_dim', 256),
                'ema_loss': metrics.get('ema_loss', 0),
                'ema_score': metrics.get('ema_score', 0),
                'rollback_count': metrics.get('rollback_count', 0),
                'no_improve_age': metrics.get('no_improve_age', 0),
                'avg_mse': self.total_mse / self.frame_count if self.frame_count > 0 else 0,
                'avg_accuracy': self.total_accuracy / self.frame_count if self.frame_count > 0 else 0,
            }

            # Save using OLA's built-in save method
            self.ola.save_best_genome(filepath, metadata)

            print(f"[CHECKPOINT] Saved champion to: {filename}")

            # Keep only last 20 checkpoints to save disk space
            self.cleanup_old_checkpoints(keep_last=20)

        except Exception as e:
            print(f"[CHECKPOINT] Failed to save: {e}")

    def cleanup_old_checkpoints(self, keep_last=20):
        """Remove old checkpoint files, keeping only the most recent ones"""
        try:
            import os
            import glob

            # Get all checkpoint files
            pattern = os.path.join(self.checkpoint_dir, "champion_*.pt")
            files = glob.glob(pattern)

            # Sort by modification time
            files.sort(key=os.path.getmtime)

            # Remove older files
            if len(files) > keep_last:
                files_to_remove = files[:-keep_last]
                for f in files_to_remove:
                    os.remove(f)
                    print(f"[CHECKPOINT] Cleaned up: {os.path.basename(f)}")

        except Exception as e:
            print(f"[CHECKPOINT] Cleanup failed: {e}")


def main():
    app = QApplication(sys.argv)
    window = OLADesktopGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
