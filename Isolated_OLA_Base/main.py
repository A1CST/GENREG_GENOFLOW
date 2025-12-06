"""
Main real-time evolution loop for isolated OLA genome visualization

Process:
1. Generate random input
2. VAE encodes to latent vector
3. PatternLSTM extracts temporal features
4. All genomes process the pattern vector
5. Genomes mutate based on trust scores
6. Visualize population in real-time
"""
import torch
from collections import deque
import numpy as np
import time
import argparse
import psutil
import csv
import os
import threading
import math
import json
def worker_job(stop_event: threading.Event, intensity: int):
    """Synthetic CPU workload controlled by feedback intensity"""
    x = 0.0
    while not stop_event.is_set():
        for _ in range(int(intensity)):
            x += math.sin(x)
        # Always yield briefly to avoid starving the main thread
        time.sleep(0.001)


from vae import SimpleVAE
from pattern_lstm import PatternLSTM
from genome_library import GenomeLibrary
from lsh_genome_database import LSHGenomeDatabase
from visualizer import GenomeVisualizer
from environment.roi_mouse import ROIMouse
from genome_controller import genome_xy_control


class IsolatedOLA:
    """
    Self-contained evolution visualizer
    """

    def __init__(self, device: str = "cpu", visualize: bool = True):
        """
        Initialize the isolated OLA system

        Args:
            device: Device to run on ("cpu" or "cuda")
            visualize: Whether to show visualization
        """
        self.device = device
        self.visualize = visualize

        # Dimensions
        self.input_dim = 4096  # 64x64 grayscale flattened
        self.vae_latent_dim = 32
        self.pattern_dim = 32
        self.genome_out_dim = 16

        # Create untrained VAE
        print("[IsolatedOLA] Creating VAE...")
        self.vae = SimpleVAE(
            input_dim=self.input_dim,
            latent_dim=self.vae_latent_dim,
            hidden_dim=64
        ).to(device)
        # Attempt to load pretrained VAE weights from config/env or default path, then freeze
        try:
            vae_path = None
            # 1) Environment variable override
            vae_path = os.environ.get("VAE_WEIGHTS", None)
            # 2) ai_config.json (project root)
            if vae_path is None:
                try:
                    cfg_path = os.path.join(os.path.dirname(__file__), "ai_config.json")
                    if os.path.exists(cfg_path):
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            cfg = json.load(f)
                        # allow multiple common keys
                        vae_path = cfg.get("vae_weights") or cfg.get("vae_path") or cfg.get("vae")
                except Exception:
                    vae_path = None
            # 3) Default to local folder path
            if vae_path is None:
                candidate = os.path.join(os.path.dirname(__file__), "vae", "model.pth")
                vae_path = candidate if os.path.exists(candidate) else None

            if vae_path and os.path.exists(vae_path):
                print(f"[IsolatedOLA] Loading pretrained VAE weights from: {vae_path}")
                state = torch.load(vae_path, map_location=self.device)
                try:
                    self.vae.load_state_dict(state, strict=True)
                except Exception:
                    # Fallback to non-strict if keys don't match exactly
                    self.vae.load_state_dict(state, strict=False)
            else:
                print("[IsolatedOLA] Pretrained VAE weights not found; using current initialization")
        except Exception as e:
            print(f"[IsolatedOLA] Failed to load pretrained VAE weights: {e}")

        # Freeze VAE parameters and set eval mode (no training)
        try:
            for p in self.vae.parameters():
                p.requires_grad = False
            self.vae.eval()
        except Exception:
            pass

        # ROI embedding (project 1024 -> 32) and PatternLSTM over [latent(32)+roi(32)]
        self.roi_embed_dim = 32
        self.pattern_input_dim = int(self.vae_latent_dim + self.roi_embed_dim)

        print("[IsolatedOLA] Creating PatternLSTM...")
        self.pattern_lstm = PatternLSTM(
            input_dim=self.pattern_input_dim,
            hidden_dim=64,
            output_dim=self.pattern_dim
        ).to(device)
        self.pattern_lstm.reset_state(batch_size=1, device=device)
        # PatternLSTM mutation control
        self.plstm_mut_cooldown_ticks = 200
        self.plstm_next_mut_tick = 0

        # Create GenomeLibrary
        print("[IsolatedOLA] Creating GenomeLibrary...")
        self.genome_library = GenomeLibrary(
            in_dim=self.pattern_dim,
            out_dim=self.genome_out_dim,
            state_dim=128,
            initial_genomes=16,
            max_genomes=32,
            trust_decay=0.990,
            blacklist_threshold=0.68,
            mutation_rate=0.20,
            device=device
        )

        # Create LSH database
        print("[IsolatedOLA] Creating LSH Database...")
        self.lsh_db = LSHGenomeDatabase(
            vector_dim=self.genome_out_dim,
            n_bits=64,
            seed=42
        )

        # Create visualizer
        self.visualizer = None
        if self.visualize:
            print("[IsolatedOLA] Creating Visualizer...")
            self.visualizer = GenomeVisualizer(
                width=1200,
                height=800,
                fps=30
            )

        # State
        self.tick = 0
        self.running = True

        # Performance: cache sys metrics, buffer log writes
        self.update_sys_metrics_every = 50
        self.last_metrics_update_tick = -1
        self.cached_sysvec = torch.zeros(1, 4, device=device)
        self.flush_every = 5000
        self.metrics_buffer = []
        self.genome_metrics_buffer = []
        self.detailed_buffer = []
        self.minimal_buffer = []
        self.health_buffer = []
        # Prediction error logging (GUI-fed)
        self.pred_err_alpha = 0.10
        self.pred_err_low = 0.05
        self.pred_err_high = 0.20
        self.pred_err_mut_gain = 2.0
        self.pred_err_boost_max = 0.004
        self.pred_err_ema = None
        self.pred_rel_ema = None
        self.pred_err_buffer = []
        # Rolling-window fitness components
        self.fit_window = 128
        self._w_mse = deque(maxlen=self.fit_window)
        self._w_bl = deque(maxlen=self.fit_window)
        self._w_pm = deque(maxlen=self.fit_window)
        self._w_ps = deque(maxlen=self.fit_window)
        self.lambda_m = 0.2  # manifold penalty weight [0.1,0.5]
        self.lambda_s = 0.05 # step penalty weight [0.01,0.1]
        self.bl_small_threshold = 1e-4
        self.fitness_ema = None
        # ROI mouse and last frame cache for simulated vision
        self.fb_w = 1200
        self.fb_h = 800
        self.roi_mouse = ROIMouse(x=self.fb_w // 2 - 100, y=self.fb_h // 2 - 100)
        self.last_frame_bgr = None
        self._roi_proj = torch.randn(1024, self.roi_embed_dim, device=device, dtype=torch.float32) * 0.1
        self._roi_proj.requires_grad_(False)
        self._last_pattern_for_control = None


        # Minimal metrics + EMA and sampling
        self.consistency_ema = None
        self.ema_alpha = 0.1
        self.cpu_samples = []
        self.ram_samples = []
        self.last_total_mutations_snapshot = 0
        self.last_trust_std = None
        self.last_minimal_metrics = None
        self.health_interval = 200
        self.mut_ema = None
        # Text-driven input buffer and latest action vector
        self.text_buffer = None
        self.last_action_vector = None
        self.prev_action_vector = None
        self.text_queue = []  # list of {'text': str, 'source': str}
        self._since_last_chat_save = 0  # autosave cadence for chat/bridge mode
        self.suppress_periodic_stats = False
        # Quick rolling checkpoint path (for continuity across runs)
        self.checkpoint_path = os.path.join("sessions", "latest_checkpoint.pt")

        # Session directory for metrics logging (sessions/session_0001, 0002, ...)
        base_dir = "sessions"
        os.makedirs(base_dir, exist_ok=True)
        # Attempt to restore simple rolling checkpoint before creating a new session dir
        try:
            self.load_checkpoint()
        except Exception:
            pass
        existing = [d for d in os.listdir(base_dir)
                    if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("session_") and d[8:].isdigit()]
        next_num = (max([int(d[8:]) for d in existing]) + 1) if existing else 1
        self.session_dir = os.path.join(base_dir, f"session_{next_num:04d}")
        os.makedirs(self.session_dir, exist_ok=True)

        # Metrics logging setup
        self.metrics_log_path = os.path.join(self.session_dir, "metrics_log.csv")
        with open(self.metrics_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "cpu_usage", "ram_usage", "avg_trust", "avg_consistency"])

        # Prediction error CSV
        self.pred_err_log_path = os.path.join(self.session_dir, "prediction_error.csv")
        with open(self.pred_err_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "mse", "mse_ema", "bl_mse", "rel_fit", "rel_fit_ema"])

        # Per-genome metrics logging setup
        self.genome_metrics_path = os.path.join(self.session_dir, "genome_metrics.csv")
        with open(self.genome_metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick",
                "genome_id",
                "trust",
                "consistency",
                "mutation_count",
                "total_ticks"
            ])

        # Feedback loop setup and detailed metrics
        self.feedback_enabled = True
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.current_intensity = 1000

        self.detailed_log_path = os.path.join(self.session_dir, "detailed_metrics.csv")
        with open(self.detailed_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick","avg_trust","avg_consistency","cpu_usage","ram_usage",
                "disk_GB_read","temp_C","work_intensity"
            ])

        # Minimal metrics log (every 1k ticks)
        self.minimal_log_path = os.path.join(self.session_dir, "minimal_metrics.csv")
        with open(self.minimal_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tick","avg_trust","trust_std","avg_consistency_ema",
                "mutations_in_window","nn_similarity_mean","cpu_var","ram_var","healthy"
            ])

        # Health state log (tick-level cadence per health_interval)
        self.health_log_path = os.path.join(self.session_dir, "health_log.csv")
        with open(self.health_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "healthy", "reason"])

        # Chat vector log for moderated/suppressed outputs
        self.chat_vector_log_path = os.path.join(self.session_dir, "chat_vectors.csv")
        with open(self.chat_vector_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tick", "tag", "vector"])

        # Checkpoint directory
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Attempt to restore persisted state
        try:
            self.load_persistence()
        except Exception:
            pass
        # Attempt to restore session-local checkpoint if present
        try:
            ckpt_path = os.path.join(self.session_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                data = torch.load(ckpt_path, map_location=self.device)
                glp = data.get('genome_library')
                if glp:
                    library = GenomeLibrary(
                        in_dim=glp['in_dim'],
                        out_dim=glp['out_dim'],
                        state_dim=glp['state_dim'],
                        initial_genomes=0,
                        max_genomes=glp['max_genomes'],
                        trust_decay=glp['trust_decay'],
                        blacklist_threshold=glp['blacklist_threshold'],
                        mutation_rate=glp['mutation_rate'],
                        device=self.device
                    )
                    library.next_genome_id = glp['next_genome_id']
                    library.total_mutations = glp['total_mutations']
                    library.total_genomes_created = glp['total_genomes_created']
                    library.genomes = []
                    for genome_dict in glp['genomes']:
                        genome = OLAGenome.from_state_dict(
                            genome_dict,
                            in_dim=glp['in_dim'],
                            out_dim=glp['out_dim'],
                            device=self.device
                        )
                        library.genomes.append(genome)
                    self.genome_library = library
                ldb = data.get('lsh_db')
                if ldb:
                    if int(ldb.get('vector_dim', self.genome_out_dim)) == int(self.lsh_db.vector_dim) and int(ldb.get('n_bits', self.lsh_db.n_bits)) == int(self.lsh_db.n_bits):
                        self.lsh_db.R = ldb['R'].to('cpu')
                        gv = ldb.get('genome_vectors', {})
                        self.lsh_db.genome_vectors = {int(gid): [v.detach().cpu().float() for v in vecs] for gid, vecs in gv.items()}
                        ht = ldb.get('hash_to_genomes', {})
                        self.lsh_db.hash_to_genomes = {int(k): set(map(int, v)) for k, v in ht.items()}
                print("[OLA] Restored checkpoint.")
        except Exception:
            pass

        # No external audio input; inputs are generated internally
        self.earn_until = 3000

        print(f"[IsolatedOLA] Initialization complete! Session dir: {self.session_dir}")

    def collect_system_metrics(self):
        # Refresh only every N ticks to reduce overhead; keep last value otherwise
        if self.last_metrics_update_tick == self.tick or (
            self.last_metrics_update_tick >= 0 and (self.tick % self.update_sys_metrics_every) != 0
        ):
            return self.cached_sysvec

        cpu = psutil.cpu_percent(interval=None) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        # For input vector keep disk/temp at 0.0 to avoid heavy calls per tick
        self.cached_sysvec = torch.tensor([cpu, ram, 0.0, 0.0], device=self.device, dtype=torch.float32).unsqueeze(0)
        # Collect samples for variance calc
        self.cpu_samples.append(float(cpu))
        self.ram_samples.append(float(ram))
        self.last_metrics_update_tick = self.tick
        return self.cached_sysvec

    def generate_random_input(self) -> torch.Tensor:
        """Use live psutil hardware metrics blended with any text-derived input."""
        base = self.collect_system_metrics()
        reps = max(1, self.input_dim // 4)
        tiled = base.repeat(1, reps)[:, :self.input_dim]
        if self.text_buffer is not None:
            # Simple 50/50 blend of system metrics and text-encoded buffer
            try:
                blended = 0.5 * tiled + 0.5 * self.text_buffer
            except Exception:
                blended = tiled
            return blended
        return tiled

    def _compute_roi_embed(self) -> torch.Tensor:
        """Compute ROI-based embedding (1, roi_embed_dim) using last framebuffer and ROI position.
        Returns zeros if unavailable.
        """
        try:
            import numpy as _np
            if self.last_frame_bgr is None:
                return torch.zeros(1, self.roi_embed_dim, device=self.device)
            from vision.roi_preprocess import roi_to_vision_vec
            rect = self.roi_mouse.rect()
            vec = roi_to_vision_vec(self.last_frame_bgr, rect)
            if vec.shape != (1024,):
                return torch.zeros(1, self.roi_embed_dim, device=self.device)
            v = torch.from_numpy(_np.asarray(vec, dtype=_np.float32)).to(self.device)
            emb = (v @ self._roi_proj).unsqueeze(0)
            # Normalize to zero-mean, unit-RMS then apply slope=0.3, offset=0.0
            emb = emb - emb.mean(dim=1, keepdim=True)
            rms = torch.sqrt(torch.mean(emb * emb, dim=1, keepdim=True) + 1e-8)
            emb = emb / rms
            emb = 0.3 * emb
            return emb
        except Exception:
            return torch.zeros(1, self.roi_embed_dim, device=self.device)

    def _update_roi(self):
        """Update ROI position using active genome XY control and trust-based speed limit."""
        try:
            active = self.genome_library.get_highest_trust_genome()
            if active is None:
                return
            state_vec = self._last_pattern_for_control if self._last_pattern_for_control is not None else torch.zeros(1, self.pattern_dim, device=self.device)
            dx, dy = genome_xy_control(active, state_vec)
            if not (isinstance(dx, float) and isinstance(dy, float)):
                dx, dy = 0.0, 0.0
            # Always full speed (no clamp)
            self.roi_mouse.move(dx, dy, self.fb_w, self.fb_h, speed_scale=1.0)
        except Exception:
            pass

    def _capture_frame_from_visualizer(self):
        """Capture current pygame framebuffer into BGR numpy array for next tick."""
        if not (self.visualize and self.visualizer):
            return
        try:
            import pygame
            import numpy as _np
            surf = self.visualizer.screen
            arr = pygame.surfarray.array3d(surf)  # (W, H, 3) RGB
            arr = _np.transpose(arr, (1, 0, 2))   # (H, W, 3)
            # Convert RGB to BGR
            self.last_frame_bgr = arr[:, :, ::-1].copy()
            self.fb_h, self.fb_w = self.last_frame_bgr.shape[:2]
            # Draw ROI rectangle overlay for debugging
            try:
                pygame.draw.rect(surf, (0, 255, 0), (self.roi_mouse.x, self.roi_mouse.y, self.roi_mouse.w, self.roi_mouse.h), 1)
            except Exception:
                pass
        except Exception:
            pass

    def _encode_text_to_input(self, text: str) -> torch.Tensor:
        """Encode UTF-8 bytes of text into a 1xinput_dim float tensor in [0,1]."""
        data = text.encode("utf-8")[: self.input_dim]
        arr = torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        if len(data) > 0:
            arr[: len(data)] = torch.tensor(list(data), dtype=torch.float32, device=self.device) / 255.0
        return arr.unsqueeze(0)

    def feed_text(self, text: str, source: str = "unknown") -> None:
        """Public API: feed text into OLA sensory loop and advance one step.

        Args:
            text: Input text
            source: Origin tag (e.g., 'user', 'assistant', 'tinyllama')
        """
        self.text_buffer = self._encode_text_to_input(text)
        try:
            self.text_queue.append({"text": str(text), "source": str(source)})
        except Exception:
            pass
        # Run a single step so downstream vectors reflect this input
        self.step()
        # Chat/bridge autosave: persist every 25 messages
        try:
            self._since_last_chat_save += 1
            if self._since_last_chat_save >= 25:
                self.save_persistence()
                self._since_last_chat_save = 0
        except Exception:
            pass

    def get_action_vector(self) -> torch.Tensor:
        """Public API: latest 1xpattern_dim vector representing system action/state."""
        if self.last_action_vector is None:
            return torch.zeros(1, self.pattern_dim, device=self.device)
        return self.last_action_vector

    def get_avg_trust(self) -> float:
        """Public API: current mean trust across genomes."""
        return float(self.genome_library.get_library_stats()["avg_trust"])

    def get_latest_text(self):
        """Return the oldest queued text (string) or None if empty."""
        msg = self.get_latest_message()
        return msg.get('text') if msg else None

    def get_latest_message(self):
        """Return the oldest queued message dict with 'text' and 'source', or None."""
        if not self.text_queue:
            return None
        try:
            return self.text_queue.pop(0)
        except Exception:
            return None

    def decrease_trust(self, amount: float) -> None:
        """Public API: decrease trust for all genomes by a fixed amount [0,1]."""
        try:
            dec = float(max(0.0, amount))
        except Exception:
            dec = 0.0
        if dec <= 0.0:
            return
        for g in self.genome_library.genomes:
            try:
                g.stats.trust_score = max(0.0, float(g.stats.trust_score) - dec)
            except Exception:
                pass

    @property
    def health(self) -> str:
        """Return 'YES' if system healthy else 'NO'."""
        try:
            if self.last_minimal_metrics and bool(self.last_minimal_metrics.get('healthy', False)):
                return "YES"
        except Exception:
            pass
        return "NO"

    @property
    def novelty(self) -> float:
        """Compute novelty as relative L2 delta of last action vectors."""
        if self.last_action_vector is None or self.prev_action_vector is None:
            return 0.0
        try:
            delta = torch.norm(self.last_action_vector - self.prev_action_vector).item()
            denom = max(1e-6, torch.norm(self.prev_action_vector).item())
            return float(delta / denom)
        except Exception:
            return 0.0

    def get_minimal_metrics(self) -> dict:
        """Expose last minimal metrics dict (may be None before first health window)."""
        return dict(self.last_minimal_metrics) if self.last_minimal_metrics else {}

    def compose_prompt(self) -> str:
        """Compose a brief sensory summary and state context for language generation."""
        stats = self.genome_library.get_library_stats()
        mm = self.get_minimal_metrics()
        vec = self.last_action_vector.detach().cpu().numpy().tolist() if self.last_action_vector is not None else []
        parts = [
            f"avg_trust={stats.get('avg_trust', 0.0):.3f}",
            f"avg_consistency={stats.get('avg_consistency', 0.0):.3f}",
            f"trust_std={mm.get('trust_std', 0.0):.3f}",
            f"cpu_var={mm.get('cpu_var', 0.0):.6f}",
            f"ram_var={mm.get('ram_var', 0.0):.6f}",
            f"healthy={self.health}",
        ]
        return "State{" + ", ".join(parts) + "}\nVector:" + str(vec)

    def log_vector(self, tag: str, vector) -> None:
        """Append a tagged vector row to chat vector log."""
        try:
            if isinstance(vector, torch.Tensor):
                vec_list = vector.detach().cpu().numpy().tolist()
            elif hasattr(vector, 'tolist'):
                vec_list = vector.tolist()
            else:
                vec_list = list(vector)
        except Exception:
            vec_list = []
        try:
            with open(self.chat_vector_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.tick, str(tag), json.dumps(vec_list)])
        except Exception:
            pass

    def set_suppress_stats(self, flag: bool) -> None:
        """Enable/disable periodic console stats prints inside step()."""
        try:
            self.suppress_periodic_stats = bool(flag)
        except Exception:
            self.suppress_periodic_stats = False

    def get_stats_text(self) -> str:
        """Return a concise multi-line stats string similar to print_stats() output."""
        stats = self.genome_library.get_library_stats()
        lsh_stats = self.lsh_db.get_stats()
        lines = []
        lines.append(f"[Tick {self.tick}] Statistics:")
        lines.append(f"  Genomes: {stats['total_genomes']}")
        lines.append(f"  Avg Trust: {stats['avg_trust']:.3f}")
        lines.append(f"  Trust Range: [{stats['min_trust']:.3f}, {stats['max_trust']:.3f}]")
        lines.append(f"  Total Mutations: {stats['total_mutations']}")
        lines.append(f"  Avg Mutation Count: {stats['avg_mutation_count']:.0f}")
        lines.append(f"  Avg Consistency: {stats['avg_consistency']:.3f}")
        lines.append(f"  LSH Vectors: {lsh_stats['total_vectors']}")
        # Top 3 genomes
        top_genomes = self.genome_library.get_top_genomes(k=3)
        lines.append("  Top 3 Genomes:")
        for i, genome in enumerate(top_genomes):
            lines.append(f"    {i+1}. {genome}")
        return "\n".join(lines)

    # --- GUI feedback: prediction error ---
    def report_prediction_error(self, mse: float, bl_mse: float | None = None,
                                zhat_mu: float | None = None, zhat_std: float | None = None,
                                zenc_mu: float | None = None, zenc_std: float | None = None,
                                delta_mag: float | None = None) -> None:
        """Called from GUI thread: update EMAs and buffer a log row."""
        try:
            m = float(max(0.0, mse))
        except Exception:
            m = 0.0
        if self.pred_err_ema is None:
            self.pred_err_ema = m
        else:
            a = float(self.pred_err_alpha)
            self.pred_err_ema = a * m + (1.0 - a) * float(self.pred_err_ema)
        # Relative fitness (positive only)
        rel = 0.0
        if bl_mse is not None:
            try:
                rel = max(0.0, float(bl_mse) - float(m))
            except Exception:
                rel = 0.0
        if self.pred_rel_ema is None:
            self.pred_rel_ema = rel
        else:
            a = float(self.pred_err_alpha)
            self.pred_rel_ema = a * rel + (1.0 - a) * float(self.pred_rel_ema)
        # Rolling-window penalties
        try:
            pm = (abs(float(zhat_std) - float(zenc_std)) if (zhat_std is not None and zenc_std is not None) else 0.0) \
                 + (abs(float(zhat_mu) - float(zenc_mu)) if (zhat_mu is not None and zenc_mu is not None) else 0.0)
        except Exception:
            pm = 0.0
        try:
            ps = float(delta_mag) if (bl_mse is not None and float(bl_mse) <= self.bl_small_threshold) else 0.0
        except Exception:
            ps = 0.0
        self._w_mse.append(float(m))
        if bl_mse is not None:
            self._w_bl.append(float(bl_mse))
        self._w_pm.append(float(pm))
        self._w_ps.append(float(ps))

        # Compute windowed fitness F = clamp(mean(BL) - mean(MSE), >=0) - λm*mean(Pm) - λs*mean(Ps)
        try:
            mean_mse = (sum(self._w_mse) / len(self._w_mse)) if self._w_mse else 0.0
            mean_bl = (sum(self._w_bl) / len(self._w_bl)) if self._w_bl else 0.0
            mean_pm = (sum(self._w_pm) / len(self._w_pm)) if self._w_pm else 0.0
            mean_ps = (sum(self._w_ps) / len(self._w_ps)) if self._w_ps else 0.0
            base = max(0.0, mean_bl - mean_mse)
            fit_win = base - float(self.lambda_m) * mean_pm - float(self.lambda_s) * mean_ps
        except Exception:
            fit_win = 0.0
        if self.fitness_ema is None:
            self.fitness_ema = fit_win
        else:
            a = float(self.pred_err_alpha)
            self.fitness_ema = a * fit_win + (1.0 - a) * float(self.fitness_ema)

        # Buffer log row (tick at time of receipt)
        self.pred_err_buffer.append([
            int(self.tick),
            round(m, 6),
            round(float(self.pred_err_ema), 6),
            (round(float(bl_mse), 6) if bl_mse is not None else None),
            round(float(rel), 6),
            round(float(self.pred_rel_ema), 6),
        ])

    def step(self):
        """
        Perform one evolution step
        """
        # 1. Generate random input
        random_input = self.generate_random_input()

        # 2. Encode with VAE
        with torch.no_grad():
            latent = self.vae.sample_latent(random_input)

        # ROI control + embedding
        self._update_roi()
        roi_emb = self._compute_roi_embed()

        # 3. Extract temporal pattern with LSTM over concatenated [latent, roi_emb]
        combined_in = torch.cat([latent, roi_emb], dim=1)
        pattern_vector = self.pattern_lstm.get_pattern(combined_in)
        # Store latest action/state vector
        self.prev_action_vector = self.last_action_vector.detach() if self.last_action_vector is not None else None
        self.last_action_vector = pattern_vector.detach()
        self._last_pattern_for_control = self.last_action_vector

        # 4. Update all genomes with pattern vector with entropy-gated pruning
        #    and modulate trust boost using prediction error EMA
        allow_prune = False
        if self.tick % 50 == 0:
            stats_now = self.genome_library.get_library_stats()
            avg_trust_now = float(stats_now['avg_trust'])
            trusts_now = [float(g.stats.trust_score) for g in self.genome_library.genomes]
            trust_std_now = float(np.std(trusts_now)) if trusts_now else 0.0
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            try:
                entropy = float(self.lsh_db.get_genome_diversity(genome_ids))
            except Exception:
                entropy = 1.0
            # Link pruning to population entropy + trust conditions
            allow_prune = (trust_std_now < 0.03 and avg_trust_now > 0.8 and entropy < 0.4)
        # Apply boost override based on windowed fitness (positive only)
        fitness_signal = self.fitness_ema if self.fitness_ema is not None else self.pred_rel_ema
        if fitness_signal is not None:
            try:
                rel = float(fitness_signal)
                # Map rel in [0, +inf) to [0, pred_err_boost_max] with soft clipping
                scale = min(1.0, rel * 10.0)
                boost = min(self.pred_err_boost_max, self.pred_err_boost_max * scale)
            except Exception:
                boost = 0.0
            for g in self.genome_library.genomes:
                g.boost_base_override = float(boost) if boost > 0.0 else None
        else:
            for g in self.genome_library.genomes:
                g.boost_base_override = None

        self.genome_library.update_all_genomes(pattern_vector, current_tick=self.tick, allow_prune=allow_prune)

        # Dynamic Decay Modulation: Adjusting the Trust Floor (per-tick)
        stats_now = self.genome_library.get_library_stats()
        avg_trust_now = float(stats_now['avg_trust'])
        current_decay = float(self.genome_library.trust_decay)
        baseline_max = 0.995
        if avg_trust_now >= 0.8:
            # Lower decay (more erosion) when trust is high: 0.992 -> 0.987 as avg_trust goes 0.8 -> 1.0
            t = min(max((avg_trust_now - 0.8) / 0.2, 0.0), 1.0)
            new_decay = 0.992 - 0.005 * t
        else:
            if avg_trust_now < 0.55:
                new_decay = 0.995
            elif avg_trust_now < 0.60:
                new_decay = 0.994
            else:  # 0.60 <= avg_trust_now < 0.80
                new_decay = 0.993
        # Clamp: never raise above baseline_max
        new_decay = min(new_decay, baseline_max)
        if new_decay != current_decay:
            self.genome_library.trust_decay = new_decay
            for g in self.genome_library.genomes:
                g.trust_decay = new_decay
                # Set small positive reinforcement when population trust is very low
                if avg_trust_now < 0.4:
                    g.boost_base_override = 0.002
                else:
                    g.boost_base_override = None

        # 5. Store genome outputs in LSH database (throttled)
        if self.tick % 10 == 0:
            for genome in self.genome_library.genomes:
                with torch.no_grad():
                    output, _ = genome.cell(pattern_vector, genome.h)
                self.lsh_db.store_genome_vector(genome.genome_id, output)

        # Adaptive feedback loop: adjust synthetic CPU workload based on outputs
        if self.feedback_enabled:
            outputs = []
            for g in self.genome_library.genomes:
                if getattr(g, 'output_history', None) and len(g.output_history) > 0:
                    last = g.output_history[-1]
                    try:
                        mag = float(np.mean(np.abs(last)))
                    except Exception:
                        mag = 0.0
                else:
                    mag = 0.0
                outputs.append(mag)
            avg_out = sum(outputs) / len(outputs) if outputs else 0.0
            new_intensity = max(100, min(2000, int(800 + avg_out * 1200)))
            if abs(new_intensity - self.current_intensity) > 250:
                self.current_intensity = new_intensity
                if self.worker_thread and self.worker_thread.is_alive():
                    self.stop_event.set()
                    self.worker_thread.join()
                    self.stop_event.clear()
                self.worker_thread = threading.Thread(
                    target=worker_job, args=(self.stop_event, self.current_intensity), daemon=True
                )
                self.worker_thread.start()

        # 6. Check and mutate low-trust genomes (exploitation cadence outside probation)
        if self.tick >= self.earn_until and (self.tick % 50 == 0):
            # Diversity pressure: small trust penalty for near-duplicates (nearest-neighbor only)
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            for gid in genome_ids:
                # Find nearest neighbor similarity
                max_sim = 0.0
                for other_id in genome_ids:
                    if other_id == gid:
                        continue
                    sim = self.lsh_db.compute_genome_similarity(gid, other_id)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim > 0.7:
                    g = self.genome_library.get_genome_by_id(gid)
                    if g is not None:
                        g.stats.trust_score = max(0.0, float(g.stats.trust_score) - 0.035)

            # Anti-coast leniency: penalize only if very flat and highly consistent
            flat_eps = 1e-3
            flat_window = 50
            for g in self.genome_library.genomes:
                if getattr(g, 'output_history', None) and len(g.output_history) >= flat_window:
                    # Recent outputs variance
                    try:
                        recent = np.array(g.output_history[-flat_window:])
                        var = float(np.var(recent))
                    except Exception:
                        var = 0.0
                    if g.stats.consistency_score > 0.95 and var < flat_eps:
                        g.stats.trust_score = max(0.0, float(g.stats.trust_score) - 0.01)

            # Temporarily scale mutation rate when fitness is poor
            orig_mut_rate = float(self.genome_library.mutation_rate)
            try:
                if fitness_signal is not None:
                    rel = float(fitness_signal)
                    if rel <= 0.0:
                        # Increase mutation when not beating baseline
                        scale = 1.0 + float(self.pred_err_mut_gain) * 0.5
                        self.genome_library.mutation_rate = min(0.8, orig_mut_rate * scale)
            except Exception:
                pass

            mutated_ids = self.genome_library.check_and_mutate_blacklisted()

            # Restore original mutation rate
            self.genome_library.mutation_rate = orig_mut_rate
            if mutated_ids:
                print(f"[Tick {self.tick}] Mutated genomes: {mutated_ids}")
            # Down-weight trust slightly when not beating baseline
            if fitness_signal is not None and float(fitness_signal) <= 0.0:
                for g in self.genome_library.genomes:
                    try:
                        g.stats.trust_score = max(0.0, float(g.stats.trust_score) - 0.01)
                    except Exception:
                        pass

            # PatternLSTM evolution: mutate only when prediction error is high and cooldown passed
            try:
                if (self.tick >= self.plstm_next_mut_tick) and (self.pred_err_ema is not None):
                    ema = float(self.pred_err_ema)
                    if ema > self.pred_err_low:
                        frac = max(0.0, min(1.0, (ema - self.pred_err_low) / max(1e-6, (self.pred_err_high - self.pred_err_low))))
                        noise_scale = 0.002 + 0.008 * frac  # up to ~1% weight std
                        for p in self.pattern_lstm.parameters():
                            if p is not None and p.requires_grad and p.data is not None:
                                try:
                                    n = torch.randn_like(p.data) * float(noise_scale)
                                    p.data.add_(n)
                                except Exception:
                                    pass
                        self.plstm_next_mut_tick = int(self.tick + self.plstm_mut_cooldown_ticks)
                        print(f"[Tick {self.tick}] Mutated PatternLSTM (scale={noise_scale:.4f})")
            except Exception:
                pass

        # 7. Periodically add new genomes (pause growth while avg_trust < 0.60)
        if self.tick % 200 == 0 and self.tick > 0:
            stats_now = self.genome_library.get_library_stats()
            if stats_now['avg_trust'] >= 0.60:
                new_genome = self.genome_library.add_genome_if_needed()
            else:
                new_genome = None
            if new_genome:
                print(f"[Tick {self.tick}] Added new genome: {new_genome.genome_id}")

        # 8. Update visualization
        if self.visualize and self.visualizer:
            library_stats = self.genome_library.get_library_stats()
            # Update EMA of avg consistency
            if self.consistency_ema is None:
                self.consistency_ema = float(library_stats['avg_consistency'])
            else:
                self.consistency_ema = (
                    self.ema_alpha * float(library_stats['avg_consistency'])
                    + (1.0 - self.ema_alpha) * self.consistency_ema
                )
            self.running = self.visualizer.update(
                self.genome_library.genomes,
                self.lsh_db,
                library_stats,
                self.tick,
                self.last_minimal_metrics,
                roi_rect=self.roi_mouse.rect()
            )
            # Capture framebuffer for next tick ROI processing
            self._capture_frame_from_visualizer()

        # 9. Print stats periodically (optionally suppressed for chat UX)
        if (self.tick % 100 == 0) and (not self.suppress_periodic_stats):
            self.print_stats()

        # Log hardware/internal/per-genome/detailed metrics every 1000 ticks (buffered)
        if self.tick % 1000 == 0 and self.tick > 0:
            self.log_metrics()
            self.log_genome_metrics()
            stats = self.genome_library.get_library_stats()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            try:
                disk_counters = psutil.disk_io_counters()
                disk = (disk_counters.read_bytes / 1e9) if disk_counters else 0.0
            except Exception:
                disk = 0.0
            temp_val = 0.0
            try:
                if hasattr(psutil, "sensors_temperatures"):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        first = list(temps.values())[0]
                        if first and len(first) > 0 and getattr(first[0], 'current', None) is not None:
                            temp_val = first[0].current
            except Exception:
                temp_val = 0.0
            self.detailed_buffer.append([
                self.tick,
                round(stats["avg_trust"], 4),
                round(stats["avg_consistency"], 4),
                round(cpu, 2),
                round(ram, 2),
                round(disk, 3),
                round(temp_val, 2),
                self.current_intensity
            ])
            print(f"[Detailed] tick={self.tick} trust={stats['avg_trust']:.3f} cpu={cpu:.1f}% load={self.current_intensity}")

        # Health check and minimal metrics at higher frequency
        if self.tick % self.health_interval == 0 and self.tick > 0:
            stats_h = self.genome_library.get_library_stats()
            trusts = [float(g.stats.trust_score) for g in self.genome_library.genomes]
            trust_std = float(np.std(trusts)) if trusts else 0.0
            avg_trust = float(stats_h['avg_trust'])
            avg_consistency_ema = float(self.consistency_ema if self.consistency_ema is not None else stats_h['avg_consistency'])

            # Mutations in this health window
            total_mut = int(self.genome_library.total_mutations)
            mutations_in_window = total_mut - int(self.last_total_mutations_snapshot)
            self.last_total_mutations_snapshot = total_mut

            # EMA smoothing for mutation cadence
            if self.mut_ema is None:
                self.mut_ema = float(mutations_in_window)
            else:
                self.mut_ema = 0.9 * float(self.mut_ema) + 0.1 * float(mutations_in_window)

            # Mean nearest-neighbor similarity
            genome_ids = [g.genome_id for g in self.genome_library.genomes]
            nn_sims = []
            for gid in genome_ids:
                max_sim = 0.0
                for other_id in genome_ids:
                    if other_id == gid:
                        continue
                    sim = self.lsh_db.compute_genome_similarity(gid, other_id)
                    if sim > max_sim:
                        max_sim = sim
                if max_sim > 0.0:
                    nn_sims.append(max_sim)
            nn_similarity_mean = float(sum(nn_sims)/len(nn_sims)) if nn_sims else 0.0

            # Variance of CPU/RAM in last window
            cpu_var = float(np.var(self.cpu_samples)) if self.cpu_samples else 0.0
            ram_var = float(np.var(self.ram_samples)) if self.ram_samples else 0.0
            self.cpu_samples.clear()
            self.ram_samples.clear()

            # Health flag per rule
            N = int(stats_h['total_genomes'])
            K = max(1, N)
            effective_mut = float(self.mut_ema if self.mut_ema is not None else mutations_in_window)
            mut_ok = (effective_mut >= K and effective_mut <= 5*K)
            trust_std_ok = (trust_std >= 0.03 and trust_std <= 0.20)
            healthy = mut_ok and trust_std_ok
            fail_reasons = []
            if not mut_ok:
                fail_reasons.append("mutations out of [N,5N]")
            if not trust_std_ok:
                fail_reasons.append("trust_std outside [0.03,0.20]")
            self.last_trust_std = trust_std

            # Entropy re-injection when converged
            if avg_trust > 0.9 and trust_std < 0.03:
                for g in self.genome_library.genomes:
                    scale = float(np.random.uniform(0.97, 0.99))
                    g.stats.trust_score = max(0.0, min(1.0, float(g.stats.trust_score) * scale))

            # Consistency penalty ceiling: cap per-genome boost via external cap
            boost_cap = max(0.0, 1.0 - avg_consistency_ema)
            for g in self.genome_library.genomes:
                g.external_boost_cap = boost_cap

            self.last_minimal_metrics = {
                'avg_trust': avg_trust,
                'trust_std': trust_std,
                'avg_consistency_ema': avg_consistency_ema,
                'mutations_in_window': mutations_in_window,
                'nn_similarity_mean': nn_similarity_mean,
                'cpu_var': cpu_var,
                'ram_var': ram_var,
                'healthy': healthy,
                'health_reason': None if healthy else "; ".join(fail_reasons) if fail_reasons else "",
            }
            # Surface prediction error EMA when available
            if self.pred_err_ema is not None:
                try:
                    self.last_minimal_metrics['pred_mse_ema'] = float(self.pred_err_ema)
                except Exception:
                    pass
            if self.pred_rel_ema is not None:
                try:
                    self.last_minimal_metrics['pred_rel_ema'] = float(self.pred_rel_ema)
                except Exception:
                    pass
            if self.fitness_ema is not None:
                try:
                    self.last_minimal_metrics['fitness_ema'] = float(self.fitness_ema)
                except Exception:
                    pass

            self.minimal_buffer.append([
                self.tick,
                round(avg_trust, 4),
                round(trust_std, 4),
                round(avg_consistency_ema, 4),
                int(mutations_in_window),
                round(nn_similarity_mean, 4),
                round(cpu_var, 6),
                round(ram_var, 6),
                int(1 if healthy else 0)
            ])

            # Append to health log buffer
            self.health_buffer.append([
                self.tick,
                int(1 if healthy else 0),
                "" if healthy else "; ".join(fail_reasons)
            ])

        # (Removed) static post-probation decay set; dynamic modulation handles it per-tick

        # Periodically flush buffered logs
        if self.tick % self.flush_every == 0 and self.tick > 0:
            self.flush_logs()
        # Periodic persistence save (single file overwrite)
        if self.tick % 5000 == 0 and self.tick > 0:
            try:
                self.save_persistence()
            except Exception:
                pass
        # Quick rolling checkpoint every 100 ticks
        if self.tick % 100 == 0 and self.tick > 0:
            try:
                self.save_checkpoint()
            except Exception:
                pass
            try:
                # Per-session checkpoint in current session directory
                gl = self.genome_library
                genomes_payload = {
                    'in_dim': gl.in_dim,
                    'out_dim': gl.out_dim,
                    'state_dim': gl.state_dim,
                    'max_genomes': gl.max_genomes,
                    'trust_decay': gl.trust_decay,
                    'blacklist_threshold': gl.blacklist_threshold,
                    'mutation_rate': gl.mutation_rate,
                    'next_genome_id': gl.next_genome_id,
                    'total_mutations': gl.total_mutations,
                    'total_genomes_created': gl.total_genomes_created,
                    'genomes': [g.get_state_dict() for g in gl.genomes],
                }
                lsh_payload = {
                    'R': self.lsh_db.R.detach().cpu(),
                    'genome_vectors': {gid: [v.detach().cpu() for v in vecs] for gid, vecs in self.lsh_db.genome_vectors.items()},
                    'hash_to_genomes': {int(k): list(v) for k, v in self.lsh_db.hash_to_genomes.items()},
                    'vector_dim': int(self.lsh_db.vector_dim),
                    'n_bits': int(self.lsh_db.n_bits),
                }
                ckpt_path = os.path.join(self.session_dir, "checkpoint.pt")
                torch.save({'genome_library': genomes_payload, 'lsh_db': lsh_payload}, ckpt_path)
            except Exception:
                pass
        # Quick rolling checkpoint every 100 ticks
        if self.tick % 100 == 0 and self.tick > 0:
            try:
                self.save_checkpoint()
            except Exception:
                pass

        self.tick += 1

    def print_stats(self):
        """Print current statistics"""
        stats = self.genome_library.get_library_stats()
        lsh_stats = self.lsh_db.get_stats()

        print(f"\n[Tick {self.tick}] Statistics:")
        print(f"  Genomes: {stats['total_genomes']}")
        print(f"  Avg Trust: {stats['avg_trust']:.3f}")
        print(f"  Trust Range: [{stats['min_trust']:.3f}, {stats['max_trust']:.3f}]")
        print(f"  Total Mutations: {stats['total_mutations']}")
        print(f"  Avg Mutation Count: {stats['avg_mutation_count']:.1f}")
        print(f"  Avg Consistency: {stats['avg_consistency']:.3f}")
        print(f"  LSH Vectors: {lsh_stats['total_vectors']}")

        # Top genomes
        top_genomes = self.genome_library.get_top_genomes(k=3)
        print(f"  Top 3 Genomes:")
        for i, genome in enumerate(top_genomes):
            print(f"    {i+1}. {genome}")

    def log_metrics(self):
        """Record hardware metrics and internal stats (buffered)"""
        cpu = psutil.cpu_percent(interval=None) / 100.0
        ram = psutil.virtual_memory().percent / 100.0
        stats = self.genome_library.get_library_stats()
        self.metrics_buffer.append([
            self.tick,
            round(cpu, 4),
            round(ram, 4),
            round(stats["avg_trust"], 4),
            round(stats["avg_consistency"], 4),
        ])
        print(f"[Metrics] tick={self.tick} cpu={cpu:.2f} ram={ram:.2f} trust={stats['avg_trust']:.3f}")

    def log_genome_metrics(self):
        """Save per-genome stats (buffered)."""
        for g in self.genome_library.genomes:
            self.genome_metrics_buffer.append([
                self.tick,
                g.genome_id,
                round(g.stats.trust_score, 4),
                round(g.stats.consistency_score, 4),
                g.stats.mutation_count,
                g.stats.total_ticks
            ])
        print(f"[GenomeMetrics] Logged {len(self.genome_library.genomes)} genomes at tick {self.tick}")

    def flush_logs(self):
        """Flush buffered CSV rows to disk"""
        if self.metrics_buffer:
            with open(self.metrics_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.metrics_buffer)
            self.metrics_buffer.clear()
        if self.genome_metrics_buffer:
            with open(self.genome_metrics_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.genome_metrics_buffer)
            self.genome_metrics_buffer.clear()
        if self.detailed_buffer:
            with open(self.detailed_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.detailed_buffer)
            self.detailed_buffer.clear()
        if self.minimal_buffer:
            with open(self.minimal_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.minimal_buffer)
            self.minimal_buffer.clear()
        if self.health_buffer:
            with open(self.health_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.health_buffer)
            self.health_buffer.clear()
        if self.pred_err_buffer:
            with open(self.pred_err_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(self.pred_err_buffer)
            self.pred_err_buffer.clear()

    def run(self, max_ticks: int = 10000):
        """
        Run the evolution loop

        Args:
            max_ticks: Maximum number of ticks to run (0 = infinite)
        """
        print(f"\n[IsolatedOLA] Starting evolution loop...")
        print(f"[IsolatedOLA] Press ESC or close window to stop\n")

        start_time = time.time()

        try:
            while self.running:
                self.step()

                # Check max ticks
                if max_ticks > 0 and self.tick >= max_ticks:
                    print(f"\n[IsolatedOLA] Reached max ticks ({max_ticks})")
                    break

                # FPS limiting (if no visualizer)
                if not self.visualize:
                    if self.tick % 100 == 0:
                        time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n[IsolatedOLA] Interrupted by user")

        finally:
            # Cleanup
            elapsed = time.time() - start_time
            print(f"\n[IsolatedOLA] Evolution complete!")
            print(f"  Total ticks: {self.tick}")
            print(f"  Elapsed time: {elapsed:.1f}s")
            print(f"  Ticks/sec: {self.tick / max(elapsed, 0.001):.1f}")

            if self.visualizer:
                self.visualizer.close()

            if self.worker_thread and self.worker_thread.is_alive():
                self.stop_event.set()
                self.worker_thread.join()

            # Final stats
            self.print_stats()
            # Ensure logs are flushed
            self.flush_logs()
            # Final persistence save
            try:
                self.save_persistence()
            except Exception:
                pass

    def save_checkpoint(self, path: str | None = None):
        """
        Save system checkpoint

        Args:
            path: Path to save checkpoint
        """
        if path is not None:
            self.genome_library.save_checkpoint(path)
            print(f"[IsolatedOLA] Saved checkpoint to {path}")
            return
        try:
            gl = self.genome_library
            genomes_payload = {
                'in_dim': gl.in_dim,
                'out_dim': gl.out_dim,
                'state_dim': gl.state_dim,
                'max_genomes': gl.max_genomes,
                'trust_decay': gl.trust_decay,
                'blacklist_threshold': gl.blacklist_threshold,
                'mutation_rate': gl.mutation_rate,
                'next_genome_id': gl.next_genome_id,
                'total_mutations': gl.total_mutations,
                'total_genomes_created': gl.total_genomes_created,
                'genomes': [g.get_state_dict() for g in gl.genomes],
            }
            lsh_payload = {
                'R': self.lsh_db.R.detach().cpu(),
                'genome_vectors': {gid: [v.detach().cpu() for v in vecs] for gid, vecs in self.lsh_db.genome_vectors.items()},
                'hash_to_genomes': {int(k): list(v) for k, v in self.lsh_db.hash_to_genomes.items()},
                'vector_dim': int(self.lsh_db.vector_dim),
                'n_bits': int(self.lsh_db.n_bits),
            }
            torch.save({'genomes': genomes_payload, 'lsh': lsh_payload}, self.checkpoint_path)
        except Exception:
            pass

    def load_checkpoint(self):
        try:
            if os.path.exists(self.checkpoint_path):
                data = torch.load(self.checkpoint_path, map_location=self.device)
                glp = data.get('genomes')
                if glp:
                    library = GenomeLibrary(
                        in_dim=glp['in_dim'],
                        out_dim=glp['out_dim'],
                        state_dim=glp['state_dim'],
                        initial_genomes=0,
                        max_genomes=glp['max_genomes'],
                        trust_decay=glp['trust_decay'],
                        blacklist_threshold=glp['blacklist_threshold'],
                        mutation_rate=glp['mutation_rate'],
                        device=self.device
                    )
                    library.next_genome_id = glp['next_genome_id']
                    library.total_mutations = glp['total_mutations']
                    library.total_genomes_created = glp['total_genomes_created']
                    library.genomes = []
                    for genome_dict in glp['genomes']:
                        genome = OLAGenome.from_state_dict(
                            genome_dict,
                            in_dim=glp['in_dim'],
                            out_dim=glp['out_dim'],
                            device=self.device
                        )
                        library.genomes.append(genome)
                    self.genome_library = library
                ldb = data.get('lsh')
                if ldb:
                    if int(ldb.get('vector_dim', self.genome_out_dim)) == int(self.lsh_db.vector_dim) and int(ldb.get('n_bits', self.lsh_db.n_bits)) == int(self.lsh_db.n_bits):
                        self.lsh_db.R = ldb['R'].to('cpu')
                        gv = ldb.get('genome_vectors', {})
                        self.lsh_db.genome_vectors = {int(gid): [v.detach().cpu().float() for v in vecs] for gid, vecs in gv.items()}
                        ht = ldb.get('hash_to_genomes', {})
                        self.lsh_db.hash_to_genomes = {int(k): set(map(int, v)) for k, v in ht.items()}
                print("[OLA] Loaded checkpoint")
            else:
                print(f"[OLA] No rolling checkpoint found at {self.checkpoint_path}")
        except Exception as e:
            print(f"[OLA] Failed to load rolling checkpoint: {e}")

    def save_persistence(self):
        """Save everything in a single checkpoint file (overwritten)."""
        try:
            # Pattern LSTM component
            pattern_payload = {
                'pattern_lstm_state': self.pattern_lstm.state_dict(),
                'pattern_hidden': self.pattern_lstm.h,
                'pattern_cell': self.pattern_lstm.c,
            }

            # Genome library (mirror of GenomeLibrary.save_checkpoint structure)
            gl = self.genome_library
            genomes_payload = {
                'in_dim': gl.in_dim,
                'out_dim': gl.out_dim,
                'state_dim': gl.state_dim,
                'max_genomes': gl.max_genomes,
                'trust_decay': gl.trust_decay,
                'blacklist_threshold': gl.blacklist_threshold,
                'mutation_rate': gl.mutation_rate,
                'next_genome_id': gl.next_genome_id,
                'total_mutations': gl.total_mutations,
                'total_genomes_created': gl.total_genomes_created,
                'genomes': [g.get_state_dict() for g in gl.genomes],
            }

            # LSH DB
            lsh_payload = {
                'R': self.lsh_db.R.detach().cpu(),
                'genome_vectors': {gid: [v.detach().cpu() for v in vecs] for gid, vecs in self.lsh_db.genome_vectors.items()},
                'hash_to_genomes': {int(k): list(v) for k, v in self.lsh_db.hash_to_genomes.items()},
                'vector_dim': int(self.lsh_db.vector_dim),
                'n_bits': int(self.lsh_db.n_bits),
            }

            payload = {
                'pattern': pattern_payload,
                'genome_library': genomes_payload,
                'lsh_db': lsh_payload,
            }

            # Timestamped rotating checkpoints (keep latest 3); also refresh base file
            ts = time.strftime('%Y%m%dT%H%M%S')
            ts_path = os.path.join(self.checkpoint_dir, f'ola_state_{ts}.pt')
            base_path = os.path.join(self.checkpoint_dir, 'ola_state.pt')
            torch.save(payload, ts_path)
            # Also update the base file for compatibility
            try:
                torch.save(payload, base_path)
            except Exception:
                pass

            # Prune old timestamped checkpoints beyond 3
            try:
                entries = [
                    os.path.join(self.checkpoint_dir, f)
                    for f in os.listdir(self.checkpoint_dir)
                    if f.startswith('ola_state_') and f.endswith('.pt')
                ]
                entries.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                for old in entries[3:]:
                    try:
                        os.remove(old)
                    except Exception:
                        pass
            except Exception:
                pass

            print("[IsolatedOLA] Saved checkpoint(s):", os.path.basename(ts_path))
        except Exception:
            pass

    def load_persistence(self):
        """Restore LSTM weights/state, genome library, and LSH DB when available."""
        # Prefer newest checkpoint among timestamped files; fallback to base file
        newest_path = None
        try:
            candidates = [
                os.path.join(self.checkpoint_dir, f)
                for f in os.listdir(self.checkpoint_dir)
                if f.startswith('ola_state_') and f.endswith('.pt')
            ]
            if candidates:
                candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                newest_path = candidates[0]
        except Exception:
            newest_path = None

        if newest_path is None:
            single_path = os.path.join(self.checkpoint_dir, 'ola_state.pt')
            newest_path = single_path if os.path.exists(single_path) else None

        if newest_path is not None and os.path.exists(newest_path):
            try:
                payload = torch.load(newest_path, map_location=self.device)
                # Pattern
                patt = payload.get('pattern', {})
                if patt:
                    self.pattern_lstm.load_state_dict(patt.get('pattern_lstm_state', {}))
                    self.pattern_lstm.h = patt.get('pattern_hidden', None)
                    self.pattern_lstm.c = patt.get('pattern_cell', None)
                    if self.pattern_lstm.h is not None:
                        self.pattern_lstm.h = self.pattern_lstm.h.to(self.device)
                    if self.pattern_lstm.c is not None:
                        self.pattern_lstm.c = self.pattern_lstm.c.to(self.device)

                # Genome library
                glp = payload.get('genome_library', {})
                if glp:
                    library = GenomeLibrary(
                        in_dim=glp['in_dim'],
                        out_dim=glp['out_dim'],
                        state_dim=glp['state_dim'],
                        initial_genomes=0,
                        max_genomes=glp['max_genomes'],
                        trust_decay=glp['trust_decay'],
                        blacklist_threshold=glp['blacklist_threshold'],
                        mutation_rate=glp['mutation_rate'],
                        device=self.device
                    )
                    library.next_genome_id = glp['next_genome_id']
                    library.total_mutations = glp['total_mutations']
                    library.total_genomes_created = glp['total_genomes_created']
                    library.genomes = []
                    for genome_dict in glp['genomes']:
                        genome = OLAGenome.from_state_dict(
                            genome_dict,
                            in_dim=glp['in_dim'],
                            out_dim=glp['out_dim'],
                            device=self.device
                        )
                        library.genomes.append(genome)
                    self.genome_library = library

                # LSH DB
                ldb = payload.get('lsh_db', {})
                if ldb:
                    if int(ldb.get('vector_dim', self.genome_out_dim)) == int(self.lsh_db.vector_dim) and int(ldb.get('n_bits', self.lsh_db.n_bits)) == int(self.lsh_db.n_bits):
                        self.lsh_db.R = ldb['R'].to('cpu')
                        gv = ldb.get('genome_vectors', {})
                        self.lsh_db.genome_vectors = {int(gid): [v.detach().cpu().float() for v in vecs] for gid, vecs in gv.items()}
                        ht = ldb.get('hash_to_genomes', {})
                        self.lsh_db.hash_to_genomes = {int(k): set(map(int, v)) for k, v in ht.items()}
                print(f"[IsolatedOLA] Restored from checkpoint: {os.path.basename(newest_path)}")
                return
            except Exception as e:
                print(f"[IsolatedOLA] Failed to restore checkpoint: {e}")
        else:
            print("[IsolatedOLA] No persisted checkpoint found in 'checkpoints/'")

        # Legacy multi-file fallback
        # Pattern LSTM
        lstm_path = os.path.join(self.checkpoint_dir, 'pattern_lstm.pt')
        if os.path.exists(lstm_path):
            try:
                data = torch.load(lstm_path, map_location=self.device)
                self.pattern_lstm.load_state_dict(data.get('pattern_lstm_state', {}))
                self.pattern_lstm.h = data.get('pattern_hidden', None)
                self.pattern_lstm.c = data.get('pattern_cell', None)
                if self.pattern_lstm.h is not None:
                    self.pattern_lstm.h = self.pattern_lstm.h.to(self.device)
                if self.pattern_lstm.c is not None:
                    self.pattern_lstm.c = self.pattern_lstm.c.to(self.device)
            except Exception:
                pass

        genomes_path = os.path.join(self.checkpoint_dir, 'genomes.pt')
        if os.path.exists(genomes_path):
            try:
                self.genome_library = GenomeLibrary.load_checkpoint(genomes_path, device=self.device)
            except Exception:
                pass

        lsh_path = os.path.join(self.checkpoint_dir, 'lsh_db.pt')
        if os.path.exists(lsh_path):
            try:
                payload = torch.load(lsh_path, map_location='cpu')
                if int(payload.get('vector_dim', self.genome_out_dim)) == int(self.lsh_db.vector_dim) and int(payload.get('n_bits', self.lsh_db.n_bits)) == int(self.lsh_db.n_bits):
                    self.lsh_db.R = payload['R'].to('cpu')
                    gv = payload.get('genome_vectors', {})
                    self.lsh_db.genome_vectors = {int(gid): [v.detach().cpu().float() for v in vecs] for gid, vecs in gv.items()}
                    ht = payload.get('hash_to_genomes', {})
                    self.lsh_db.hash_to_genomes = {int(k): set(map(int, v)) for k, v in ht.items()}
            except Exception:
                pass


# Visualization toggle API for chatbot/bridge
    def enable_visualizer(self, width: int = 1200, height: int = 800, fps: int = 30):
        """Enable visualization window if not already active."""
        if self.visualizer is None:
            try:
                self.visualizer = GenomeVisualizer(width=width, height=height, fps=fps)
                self.visualize = True
                print("[IsolatedOLA] Visualizer enabled")
            except Exception as e:
                print(f"[IsolatedOLA] Failed to enable visualizer: {e}")

# Module-level singleton export for chat integration (no visualization)
ola_engine = IsolatedOLA(device=("cuda" if torch.cuda.is_available() else "cpu"), visualize=False)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Isolated OLA Genome Evolution Visualizer")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to run on")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable visualization (run in terminal only)")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Maximum ticks to run (0 = infinite)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to save checkpoint at end")

    args = parser.parse_args()

    # Create system
    system = IsolatedOLA(
        device=args.device,
        visualize=not args.no_viz
    )

    # Run
    system.run(max_ticks=args.max_ticks)

    # Save checkpoint if requested
    if args.checkpoint:
        system.save_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
