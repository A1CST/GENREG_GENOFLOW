from __future__ import annotations
import numpy as np


def genome_xy_control(genome, state_vec) -> tuple[float, float]:
    """
    Returns (dx_norm, dy_norm) in [-1,1].
    `state_vec` is the existing compressed/combined sensory state.
    """
    try:
        out = genome.forward(state_vec)
        # Attempt to get a flat array
        if hasattr(out, 'detach'):
            out = out.detach().cpu().numpy()
        out = np.array(out).reshape(-1)
        dx = float(np.tanh(out[0])) if out.size >= 1 else 0.0
        dy = float(np.tanh(out[1])) if out.size >= 2 else 0.0
        if not np.isfinite(dx) or not np.isfinite(dy):
            return 0.0, 0.0
        return dx, dy
    except Exception:
        return 0.0, 0.0


