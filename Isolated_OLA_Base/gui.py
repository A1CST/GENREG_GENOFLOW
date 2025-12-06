import os, threading, time, cv2, numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk

import torch

# Reuse your engine and modules
from main import ola_engine  # exposes metrics and the running OLA engine
from vae import SimpleVAE                                        # VAE
from pattern_lstm import PatternLSTM                             # Pattern LSTM
from latent_dynamics import LatentDynamics

# Optional access to the built-in visualizer & genome state
from visualizer import GenomeVisualizer                          # pygame visualizer


# ----------------------------
# Minimal next-frame predictor
# ----------------------------
class FrameDeltaPredictor:
    """
    Map frame_t -> z_t via VAE.encode; use PatternLSTM to predict z_{t+1};
    decode predicted z to x_{t+1} (vector); render to small image.
    No gradients. Pure stateful temporal update.
    """

    def __init__(self, device: str = "cpu", input_vec_dim: int = 4096, latent_dim: int = 32, step_alpha: float = 0.3):
        self.device = device
        self.input_dim = input_vec_dim
        self.latent_dim = latent_dim
        self.step_alpha = max(0.1, min(0.5, float(step_alpha)))
        self.alpha_max = 0.5
        self.alpha_gain = 0.5  # k for adaptive alpha

        self.vae = SimpleVAE(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_dim=64).to(self.device)
        self.plstm = PatternLSTM(input_dim=self.latent_dim, hidden_dim=64, output_dim=self.latent_dim).to(self.device)
        self.plstm.reset_state(batch_size=1, device=self.device)
        # Running encoder stats for manifold snap
        self.enc_mu_ema: Optional[torch.Tensor] = None
        self.enc_std_ema: Optional[torch.Tensor] = None
        self.snap_ema = 0.05
        self.dynamics = LatentDynamics(dim=self.latent_dim, device=self.device,
                                       alpha_max=0.5, k_alpha=1.0,
                                       ema_decay=0.995, log_dir="logs", window=128)
        self.prev_z: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _frame_to_vec(self, frame_bgr: np.ndarray, out_dim: int = 4096) -> torch.Tensor:
        # Downscale, grayscale, flatten -> [-1,1]
        if frame_bgr is None:
            return torch.zeros(1, out_dim, device=self.device, dtype=torch.float32)
        h, w = (64, 64) if out_dim == 4096 else (16, 16)
        small = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        v = gray.astype(np.float32).reshape(-1)
        if v.size > out_dim:
            v = v[:out_dim]
        elif v.size < out_dim:
            v = np.pad(v, (0, out_dim - v.size), mode="constant", constant_values=0)
        v = (v / 127.5) - 1.0
        return torch.from_numpy(v).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode_latent(self, frame_bgr: np.ndarray) -> torch.Tensor:
        x_t = self._frame_to_vec(frame_bgr, out_dim=self.input_dim)
        mu, logvar = self.vae.encode(x_t)
        return mu

    @torch.no_grad()
    def _latent_to_small_img(self, z: torch.Tensor, w: int = 64, h: int = 64) -> np.ndarray:
        arr = z.detach().cpu().numpy().reshape(1, -1)
        vec = arr[0]
        import numpy as _np, cv2 as _cv2
        if vec.size < (w * h):
            vec = _np.pad(vec, (0, w * h - vec.size), mode="constant")
        else:
            vec = vec[: w * h]
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-9)
        vec = (vec * 255).astype("uint8").reshape(h, w)
        rgb = _cv2.cvtColor(vec, _cv2.COLOR_GRAY2RGB)
        return rgb

    @torch.no_grad()
    def _vec_to_small_img(self, v: torch.Tensor, w: int = 64, h: int = 64) -> np.ndarray:
        import numpy as _np, cv2 as _cv2
        arr = v.detach().cpu().numpy().reshape(h, w)
        arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(_np.uint8)
        rgb = _cv2.cvtColor(arr, _cv2.COLOR_GRAY2RGB)
        return rgb

    @torch.no_grad()
    def _frame_to_rgb_small(self, frame_bgr: np.ndarray, w: int = 64, h: int = 64) -> np.ndarray:
        import cv2 as _cv2
        if frame_bgr is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        small = _cv2.resize(frame_bgr, (w, h), interpolation=_cv2.INTER_AREA)
        rgb = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
        return rgb

    @torch.no_grad()
    def step_with_metrics(self, frame_bgr: np.ndarray):
        z_curr = self.encode_latent(frame_bgr)
        if self.prev_z is None:
            curr_small = self._frame_to_rgb_small(frame_bgr)
            self.prev_z = z_curr.detach().clone()
            empty = {
                "bl_mse": torch.tensor(0.0), "mse": torch.tensor(0.0), "margin": torch.tensor(0.0),
                "dz_norm": torch.tensor(0.0), "alpha": torch.tensor(0.0),
                "mu_enc_mean": torch.tensor(0.0), "sig_enc_mean": torch.tensor(0.0),
                "mu_hat_mean": torch.tensor(0.0), "sig_hat_mean": torch.tensor(0.0),
                "cos": torch.tensor(1.0),
                # side-channels defaults
                "I": torch.tensor(0.0), "varY": torch.tensor(0.0), "sat": torch.tensor(0.0),
                "shadow": torch.tensor(0.0), "motion": torch.tensor(0.0), "coherence": torch.tensor(1.0),
                "gate": torch.tensor(0.0), "sigma_mut": torch.tensor(0.0), "trust_delta": torch.tensor(0.0),
            }
            return curr_small, curr_small, empty

        out = self.dynamics.step(self.prev_z, z_curr, frame_bgr=frame_bgr)
        z_hat = out["z_hat"]
        x_next = self.vae.decode(z_hat)
        pred_small = self._vec_to_small_img(x_next)
        curr_small = self._frame_to_rgb_small(frame_bgr)
        self.prev_z = z_curr.detach().clone()
        return curr_small, pred_small, out

    @torch.no_grad()
    def predict_next_frame_vec(self, frame_bgr: np.ndarray, prev_pred_z: Optional[torch.Tensor] = None,
                               prev_actual_z: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Tuple[float, float], Tuple[float, float], float, float, float]:
        """
        Predict in latent delta-space, align to manifold before decode.

        Returns:
            curr_small_img_rgb, pred_small_img_rgb,
            x_t_vec, x_next_vec,
            z_t, z_out,
            delta_norm_mag, (zhat_mean, zhat_std), (zenc_mean, zenc_std),
            gating_beta, alpha_used, rel_fitness
        """
        # Vectorize current frame
        x_t = self._frame_to_vec(frame_bgr, out_dim=self.input_dim)  # [1,4096]

        # Encode to latent (use mean for stability)
        mu, logvar = self.vae.encode(x_t)
        z_t = mu  # [1, latent_dim]
        # Update running encoder stats (per-dimension mean/std not needed; track global scalar stats)
        zenc_mean_now = z_t.mean()
        zenc_std_now = z_t.std()
        if self.enc_mu_ema is None:
            self.enc_mu_ema = zenc_mean_now.detach()
            self.enc_std_ema = zenc_std_now.detach() + 1e-6
        else:
            self.enc_mu_ema = (1.0 - self.snap_ema) * self.enc_mu_ema + self.snap_ema * zenc_mean_now.detach()
            self.enc_std_ema = (1.0 - self.snap_ema) * self.enc_std_ema + self.snap_ema * (zenc_std_now.detach() + 1e-6)

        # LSTM outputs delta in latent space (Δz_t)
        delta_z = self.plstm.get_pattern(z_t)  # [1, latent_dim]
        # Normalize direction
        delta_norm = torch.norm(delta_z, dim=1, keepdim=True) + 1e-8
        delta_dir = delta_z / delta_norm
        # Adaptive alpha based on baseline vs predictor performance if previous stats available
        alpha_used = self.step_alpha
        rel_fitness = float('nan')
        if prev_pred_z is not None and prev_actual_z is not None:
            try:
                mse_pred = torch.mean((prev_pred_z.detach().cpu().float() - z_t.detach().cpu().float()) ** 2).item()
                bl_mse = torch.mean((prev_actual_z.detach().cpu().float() - z_t.detach().cpu().float()) ** 2).item()
                rel_fitness = max(0.0, bl_mse - mse_pred)
                alpha_used = min(self.alpha_max, float(self.alpha_gain * bl_mse / (mse_pred + 1e-8)))
            except Exception:
                alpha_used = self.step_alpha
        # Step in latent space
        z_hat_next = z_t + alpha_used * delta_dir

        # Manifold alignment: simple layer-norm-like (zero-mean, unit-RMS), then tanh squash
        z_mu = z_hat_next.mean(dim=1, keepdim=True)
        z_std = torch.sqrt(torch.var(z_hat_next, dim=1, keepdim=True) + 1e-8)
        # Snap to encoder running stats
        scale = (self.enc_std_ema / (z_std + 1e-6)).to(z_hat_next)
        shift = (self.enc_mu_ema - z_mu).to(z_hat_next)
        z_hat_snapped = (z_hat_next - z_mu) * scale + z_mu + shift
        # Gentle tanh to keep bounded
        z_hat_next_aligned = torch.tanh(z_hat_snapped)

        # Optional projection: decode then re-encode (use mu) to pull back to manifold
        try:
            recon_vec = self.vae.decode(z_hat_next_aligned)
            mu_proj, _ = self.vae.encode(recon_vec)
            z_proj = mu_proj
        except Exception:
            z_proj = z_hat_next_aligned

        # Confidence mixing: reduce beta when step magnitude or projection mismatch spikes
        # Gated residual to copy-last
        gating_beta = 1.0
        try:
            if prev_pred_z is not None and prev_actual_z is not None:
                mse_pred = torch.mean((prev_pred_z.detach().cpu().float() - z_t.detach().cpu().float()) ** 2).item()
                bl_mse = torch.mean((prev_actual_z.detach().cpu().float() - z_t.detach().cpu().float()) ** 2).item()
                if bl_mse <= mse_pred:
                    gating_beta = 0.0
        except Exception:
            pass
        # Fallback if cos sim low or std out-of-band
        if gating_beta > 0.0:
            try:
                cos = torch.nn.functional.cosine_similarity(z_proj.view(1, -1), z_t.view(1, -1)).item()
            except Exception:
                cos = 1.0
            try:
                zhat_std_scalar = float(torch.std(z_proj).item())
                enc_std_scalar = float(self.enc_std_ema.item()) if self.enc_std_ema is not None else zhat_std_scalar
                out_of_band = (zhat_std_scalar > 1.8 * enc_std_scalar) or (zhat_std_scalar < 0.4 * enc_std_scalar)
            except Exception:
                out_of_band = False
            if cos < 0.1 or out_of_band:
                gating_beta = min(gating_beta, 0.3)
        # Final latent output after gating
        z_out = (1.0 - gating_beta) * z_t + gating_beta * z_proj

        # Decode predicted latent to vector x_{t+1}
        x_next = self.vae.decode(z_out)  # [1, input_dim], tanh in [-1,1]

        # Render both to small RGB for panel display
        def vec_to_img(v: torch.Tensor, w: int = 64, h: int = 64) -> np.ndarray:
            arr = v.detach().cpu().numpy().reshape(h, w)
            arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            return rgb

        curr_small = vec_to_img(x_t, w=64, h=64)
        pred_small = vec_to_img(x_next, w=64, h=64)
        # Telemetry
        try:
            zhat_mean = float(z_out.mean().item())
            zhat_std = float(z_out.std().item())
        except Exception:
            zhat_mean, zhat_std = 0.0, 1.0
        try:
            zenc_mean_scalar = float(self.enc_mu_ema.item()) if self.enc_mu_ema is not None else float(zenc_mean_now.item())
            zenc_std_scalar = float(self.enc_std_ema.item()) if self.enc_std_ema is not None else float(zenc_std_now.item())
        except Exception:
            zenc_mean_scalar, zenc_std_scalar = 0.0, 1.0
        delta_norm_mag = float(delta_norm.detach().cpu().view(-1)[0]) if 'delta_norm' in locals() else 0.0
        return (curr_small, pred_small, x_t, x_next, z_t.detach(), z_out.detach(),
                delta_norm_mag, (zhat_mean, zhat_std), (zenc_mean_scalar, zenc_std_scalar),
                float(gating_beta), float(alpha_used), float(rel_fitness) if not isinstance(rel_fitness, float) or not (rel_fitness != rel_fitness) else 0.0)


# ----------------------------
# Tkinter GUI
# ----------------------------
class OLAGui(tk.Tk):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.title("OLA – Next-Frame GUI")
        self.geometry("1100x700")

        # Follow engine device and VAE
        try:
            self.device = getattr(ola_engine, 'device', device)
        except Exception:
            self.device = device
        self.predictor = FrameDeltaPredictor(device=self.device)
        # Reuse the engine's pretrained, frozen VAE
        try:
            self.predictor.vae = ola_engine.vae
        except Exception:
            pass
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_path: Optional[str] = None
        self.loop_var = tk.BooleanVar(value=False)

        # Controls
        top = ttk.Frame(self); top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        self.btn_open_vis = ttk.Button(top, text="Open Visualizer", command=self._launch_visualizer_thread)
        self.btn_open_vis.pack(side=tk.LEFT, padx=4)

        self.btn_pick = ttk.Button(top, text="Select Video", command=self._pick_video)
        self.btn_pick.pack(side=tk.LEFT, padx=4)

        self.loop_chk = ttk.Checkbutton(top, text="Loop video", variable=self.loop_var)
        self.loop_chk.pack(side=tk.LEFT, padx=8)

        # Start/Stop OLA engine
        self.btn_toggle_ola = ttk.Button(top, text="Start OLA", command=self._toggle_ola)
        self.btn_toggle_ola.pack(side=tk.LEFT, padx=8)

        # Metrics panel
        self.metrics_txt = tk.StringVar(value="Metrics: —")
        self.lbl_metrics = ttk.Label(self, textvariable=self.metrics_txt, anchor="w", justify="left")
        self.lbl_metrics.pack(side=tk.TOP, fill=tk.X, padx=8)

        # Prediction vs Actual comparison metrics
        self.compare_txt = tk.StringVar(value="Prediction error: —")
        self.lbl_compare = ttk.Label(self, textvariable=self.compare_txt, anchor="w", justify="left")
        self.lbl_compare.pack(side=tk.TOP, fill=tk.X, padx=8)

        # Two image panels
        mid = ttk.Frame(self); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.curr_canvas = tk.Label(mid); self.curr_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4)
        self.pred_canvas = tk.Label(mid); self.pred_canvas.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=4)

        # State
        self._stop = False
        self._ui_lock = threading.Lock()
        self._vis_thread = None
        self._read_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._read_thread.start()

        # OLA loop state
        self._ola_running = False
        self._ola_thread: Optional[threading.Thread] = None

        # Periodic metrics refresh if no video
        self.after(250, self._tick_metrics)

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # Local EMA for GUI-displayed prediction error
        self._mse_ema = None

    def _on_close(self):
        self._stop = True
        # Stop OLA loop
        self._ola_running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.destroy()

    def _pick_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.webm"), ("All files", "*.*")]
        )
        if path:
            self.video_path = path
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = cv2.VideoCapture(self.video_path)

    def _tick_metrics(self):
        # Query OLA metrics even if no input is flowing
        try:
            stats_txt = ola_engine.get_stats_text()
        except Exception:
            # Fallback summary from minimal metrics
            mm = ola_engine.get_minimal_metrics() if hasattr(ola_engine, "get_minimal_metrics") else {}
            stats_txt = f"avg_trust={mm.get('avg_trust','?')} healthy={mm.get('healthy','?') }"
        self.metrics_txt.set(stats_txt)
        if not self._stop:
            self.after(500, self._tick_metrics)

    def _reader_loop(self):
        """Non-blocking reader + predictor loop."""
        prev_pred_vec: Optional[torch.Tensor] = None  # previous predicted latent (aligned)
        prev_actual_z: Optional[torch.Tensor] = None  # previous actual latent
        while not self._stop:
            frame = None
            if self.cap is not None and self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    # End of file
                    if self.loop_var.get() and self.video_path:
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.video_path)
                        continue
                    else:
                        time.sleep(0.03)
                        continue

            # If no frame, pause LSTM state and fitness to avoid drift
            if frame is None:
                time.sleep(0.03)
                continue

            curr_small, pred_small, metrics = self.predictor.step_with_metrics(frame)
            # Unpack for engine reporting and label
            bl_mse = float(metrics.get("bl_mse", 0.0))
            mse = float(metrics.get("mse", 0.0))
            margin = float(metrics.get("margin", 0.0))
            dz_norm = float(metrics.get("dz_norm", 0.0))
            alpha_used = float(metrics.get("alpha", 0.0))
            zenc_mean = float(metrics.get("mu_enc_mean", 0.0))
            zenc_std = float(metrics.get("sig_enc_mean", 1.0))
            zh_mean = float(metrics.get("mu_hat_mean", 0.0))
            zh_std = float(metrics.get("sig_hat_mean", 1.0))
            cos = float(metrics.get("cos", 0.0))
            # Update metrics line per new signals
            txt = (
                "Margin={:.4f} BL_MSE={:.4f} MSE={:.4f} α={:.3f} gate={:.2f} |Δz|={:.3f} "
                "I={:.2f} varY={:.3f} sat={:.2f} shadow={:.2f} motion={:.3f} coh={:.3f} cos={:.3f} "
                "σ_mut*={:.3f} Δtrust*={:.3f}".format(
                    float(metrics.get('margin', 0.0)), float(metrics.get('bl_mse', 0.0)), float(metrics.get('mse', 0.0)),
                    float(metrics.get('alpha', 0.0)), float(metrics.get('gate', 0.0)), float(metrics.get('dz_norm', 0.0)),
                    float(metrics.get('I', 0.0)), float(metrics.get('varY', 0.0)), float(metrics.get('sat', 0.0)), float(metrics.get('shadow', 0.0)),
                    float(metrics.get('motion', 0.0)), float(metrics.get('coherence', 0.0)), float(metrics.get('cos', 0.0)),
                    float(metrics.get('sigma_mut', 0.0)), float(metrics.get('trust_delta', 0.0))
                )
            )
            try:
                self.after(0, lambda t=txt: self.compare_txt.set(t))
            except Exception:
                pass
            try:
                if hasattr(ola_engine, 'report_prediction_error'):
                    ola_engine.report_prediction_error(mse, bl_mse, zh_mean, zh_std, zenc_mean, zenc_std, dz_norm)
                if hasattr(ola_engine, 'set_mutation_scale'):
                    ola_engine.set_mutation_scale(float(metrics['sigma_mut']))
                if hasattr(ola_engine, 'nudge_trust'):
                    ola_engine.nudge_trust(float(metrics['trust_delta']))
            except Exception:
                pass

            # Shift for next tick: prev_pred_vec for cos/mse display; prev_actual_z for baseline
            prev_pred_vec = None
            prev_actual_z = None

            # Upscale for visibility (square panels)
            curr_show = cv2.resize(curr_small, (512, 512), interpolation=cv2.INTER_NEAREST)
            pred_show = cv2.resize(pred_small, (512, 512), interpolation=cv2.INTER_NEAREST)

            # Push to UI safely (create PhotoImage in Tk thread)
            def _update_panels():
                try:
                    ci = ImageTk.PhotoImage(Image.fromarray(curr_show))
                    pi = ImageTk.PhotoImage(Image.fromarray(pred_show))
                    self.curr_canvas.configure(image=ci)
                    self.curr_canvas.image = ci
                    self.curr_canvas.configure(text="Current video")
                    self.pred_canvas.configure(image=pi)
                    self.pred_canvas.image = pi
                    self.pred_canvas.configure(text="Prediction (t+1)")
                except Exception:
                    pass

            try:
                self.after(0, _update_panels)
            except Exception:
                pass

            time.sleep(0.03)  # ~33 FPS target

    def _launch_visualizer_thread(self):
        """Spawn pygame visualizer without blocking Tk."""
        if self._vis_thread and self._vis_thread.is_alive():
            return
        self._vis_thread = threading.Thread(target=self._visualizer_loop, daemon=True)
        self._vis_thread.start()

    def _toggle_ola(self):
        """Start/stop background OLA stepper."""
        if self._ola_running:
            self._ola_running = False
            try:
                self.btn_toggle_ola.configure(text="Start OLA")
            except Exception:
                pass
            return

        # Start
        self._ola_running = True
        try:
            self.btn_toggle_ola.configure(text="Stop OLA")
        except Exception:
            pass

        def _ola_loop():
            dt = 1.0 / 30.0  # 30 FPS evolution
            while self._ola_running and not self._stop:
                try:
                    ola_engine.step()
                except Exception:
                    pass
                time.sleep(dt)

        self._ola_thread = threading.Thread(target=_ola_loop, daemon=True)
        self._ola_thread.start()

    def _visualizer_loop(self):
        """
        Drive your existing GenomeVisualizer against live ola_engine state
        without freezing the Tkinter mainloop.
        """
        try:
            vis = GenomeVisualizer(width=1200, height=800, fps=30)  # pygame window
        except Exception:
            return

        import pygame
        clock = pygame.time.Clock()

        while not self._stop and getattr(vis, "running", True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    vis.running = False

            vis.screen.fill(vis.bg_color)

            # Pull live genomes + stats from OLA engine
            try:
                genomes = ola_engine.genome_library.genomes
                vis.compute_positions(genomes)
                vis.draw_edges(genomes, ola_engine.lsh_db, similarity_threshold=0.6)
                vis.draw_nodes(genomes)
                lib_stats = ola_engine.genome_library.get_library_stats()
                vis.draw_stats(genomes, lib_stats, ola_engine.tick)
            except Exception:
                pass

            pygame.display.flip()
            clock.tick(vis.fps)

        try:
            pygame.display.quit()
        except Exception:
            pass


def main():
    # Device selection: follow OLA device if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = OLAGui(device=device)
    app.mainloop()


if __name__ == "__main__":
    main()


