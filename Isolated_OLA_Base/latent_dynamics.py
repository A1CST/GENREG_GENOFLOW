import os, csv, time
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn

from compression_lstm import CompressionLSTM


class LatentDynamics(nn.Module):
    """
    Δz residual predictor with:
      - adaptive gating α vs baseline margin
      - manifold snap to encoder running stats
      - short-window CSV logging
    No gradients; use selection externally if desired.
    """

    def __init__(self, dim: int = 32, device: str = "cpu",
                 alpha_max: float = 0.5, k_alpha: float = 1.0,
                 ema_decay: float = 0.995, log_dir: str = "logs", window: int = 128):
        super().__init__()
        self.device = device
        self.dim = dim
        self.comp = CompressionLSTM(dim=dim, hidden=64).to(device)
        self.comp.reset(batch_size=1, device=device)

        self.alpha_max = alpha_max
        self.k_alpha = k_alpha
        self.ema_decay = ema_decay

        # running encoder stats for manifold snap (per-dim)
        self.mu_enc = torch.zeros(1, dim, device=device)
        self.sig_enc = torch.ones(1, dim, device=device)

        # last predicted ẑ for metric alignment
        self.prev_pred: Optional[torch.Tensor] = None

        # logging
        os.makedirs(log_dir, exist_ok=True)
        ts = int(time.time())
        self.log_path = os.path.join(log_dir, f"latent_window_{ts}.csv")
        self.window = window
        self._rows = []
        with open(self.log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "bl_mse", "mse", "margin", "dz_norm", "alpha",
                        "mu_enc_mean", "sig_enc_mean", "mu_hat_mean", "sig_hat_mean", "cos"])

        self.t = 0

    @torch.no_grad()
    def update_encoder_stats(self, z: torch.Tensor):
        # per-dim EMA mean and std
        decay = self.ema_decay
        mu = z.mean(dim=0, keepdim=True)
        sig = z.std(dim=0, unbiased=False, keepdim=True) + 1e-6
        self.mu_enc = decay * self.mu_enc + (1 - decay) * mu
        self.sig_enc = decay * self.sig_enc + (1 - decay) * sig

    @torch.no_grad()
    def _frame_metrics(self, frame_bgr: np.ndarray) -> dict:
        if frame_bgr is None:
            return {"I":0.0,"varY":0.0,"sat":0.0,"sin_h":0.0,"cos_h":1.0,"entropy":0.0,"shadow":0.0}
        small = cv2.resize(frame_bgr, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H = hsv[...,0].astype(np.float32)
        S = (hsv[...,1].astype(np.float32)) / 255.0
        V = (hsv[...,2].astype(np.float32)) / 255.0
        I = float(V.mean())
        varY = float(V.var())
        h_rad = H.mean() * (2*np.pi/180.0)
        sin_h, cos_h = float(np.sin(h_rad)), float(np.cos(h_rad))
        sat = float(S.mean())
        hist = np.histogram((V*255).astype(np.uint8), bins=32, range=(0,255))[0].astype(np.float32)
        p = hist / max(1.0, hist.sum())
        entropy = float(-(p[p>0]*np.log2(p[p>0])).sum()/5.0)
        shadow = float((V < 0.15).mean())
        return {"I":I,"varY":varY,"sat":sat,"sin_h":sin_h,"cos_h":cos_h,"entropy":entropy,"shadow":shadow}

    @torch.no_grad()
    def manifold_snap(self, z_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Per-sample normalization (axis=features), then affine to encoder stats. No tanh.
        eps = 1e-6
        mu_hat = z_hat.mean(dim=1, keepdim=True)
        sig_hat = z_hat.std(dim=1, unbiased=False, keepdim=True) + eps
        z_norm = (z_hat - mu_hat) / sig_hat
        z_snap = self.mu_enc + z_norm * self.sig_enc
        assert float(sig_hat.mean()) > 1e-4, "sig_hat collapsed; check that tanh is not applied anywhere."
        return z_snap, mu_hat, sig_hat

    @torch.no_grad()
    def step(self, z_t_prev: torch.Tensor, z_t_curr: torch.Tensor, frame_bgr: Optional[np.ndarray]=None) -> Dict[str, torch.Tensor]:
        """
        Called at actual frame t arrival.
        Inputs:
          z_t_prev: encoder latent of frame t-1  [1,D]
          z_t_curr: encoder latent of frame t    [1,D]
        Returns dict with prediction image latent and metrics.
        """
        self.t += 1
        # encoder stats and frame metrics
        self.update_encoder_stats(z_t_curr)
        fm = self._frame_metrics(frame_bgr)

        # Baseline: predict z_curr from z_prev using copy-last
        bl_mse = ((z_t_prev - z_t_curr) ** 2).mean()

        # Δz from compression LSTM on z_prev; clamp in encoder units
        dz_hat = self.comp.step(z_t_prev)  # [1,D]
        r = 0.5 * self.sig_enc
        dz_hat = torch.clamp(dz_hat, -r, r)

        # Exploration + improvement-based alpha (no deadlock when worse than baseline)
        pred_mse_prev = bl_mse if self.prev_pred is None else ((self.prev_pred - z_t_curr) ** 2).mean()
        motion = float(torch.norm(z_t_curr - z_t_prev, p=2))
        coherence = float(torch.nn.functional.cosine_similarity(z_t_curr, z_t_prev).mean())
        alpha_explore = 0.02 if motion > 1e-3 else 0.0
        improve = max(0.0, float(bl_mse - float(pred_mse_prev)))
        alpha_gain = min(self.alpha_max, self.k_alpha * improve)
        alpha = max(alpha_explore, alpha_gain)

        # Residual update
        z_hat = z_t_prev + alpha * dz_hat

        # manifold snap and metrics
        z_snap, mu_hat, sig_hat = self.manifold_snap(z_hat)

        mse = ((z_snap - z_t_curr) ** 2).mean()
        margin = bl_mse - mse
        cos = torch.nn.functional.cosine_similarity(z_snap, z_t_curr, dim=1).mean()

        # log window
        row = [self.t,
               float(bl_mse), float(mse), float(margin),
               float(torch.norm(dz_hat,p=2)), float(alpha),
               float(self.mu_enc.mean()), float(self.sig_enc.mean()),
               float(mu_hat.mean()), float(sig_hat.mean()),
               float(cos)]
        self._rows.append(row)
        if len(self._rows) >= self.window:
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerows(self._rows)
            self._rows.clear()

        self.prev_pred = z_snap.detach().clone()
        surprise = max(0.0, float(mse - bl_mse))
        sigma_mut = 0.1 * (0.5*motion + 0.5*surprise)
        trust_delta = float(max(0.0, margin)) - 0.5*surprise
        # Recompute a gate metric for telemetry (not used in alpha here)
        sat = fm["sat"]
        sat_clip = max(0.0, sat - 0.95) + max(0.0, 0.05 - sat)
        coh_term = max(0.0, 1.0 - coherence)
        g_raw = (0.6*coh_term + 0.4*motion) - (0.3*fm["shadow"] + 0.2*sat_clip)
        g = float(max(0.0, min(1.0, g_raw)))

        return {
            "z_hat": z_snap,
            "bl_mse": bl_mse, "mse": mse, "margin": margin,
            "dz_norm": torch.norm(dz_hat,p=2),
            "alpha": torch.tensor(alpha),
            "cos": cos,
            # side-channels
            "I": torch.tensor(fm["I"]), "varY": torch.tensor(fm["varY"]),
            "sat": torch.tensor(fm["sat"]), "sin_h": torch.tensor(fm["sin_h"]), "cos_h": torch.tensor(fm["cos_h"]),
            "entropy": torch.tensor(fm["entropy"]), "shadow": torch.tensor(fm["shadow"]),
            "motion": torch.tensor(motion), "coherence": torch.tensor(coherence), "gate": torch.tensor(g),
            # evolution suggestions
            "sigma_mut": torch.tensor(sigma_mut), "trust_delta": torch.tensor(trust_delta),
        }


