# ================================================================
# GENREG v1.0 — Stateful Biological Protein Library
# Payton Miller — 2025
#
# This module implements the regulatory genome layer (proteins)
# for GENREG-based organisms. Proteins are stateful, adaptive,
# biologically-inspired regulatory units that modulate TRUST.
#
# No environment-specific logic.
# 100% forward-pass. No gradients. No backprop.
# ================================================================

import random
import math


# ================================================================
# Base Protein Class
# ================================================================
class Protein:
    def __init__(self, name, protein_type):
        self.name = name
        self.type = protein_type

        # Internal state (biological memory)
        self.state = {}

        # Cached output during one forward pass
        self.output = 0.0

        # Inputs → can be:
        # - environment signals
        # - other protein outputs
        self.inputs = []

        # Hyperparameters
        self.params = {}

    # ------------------------------------------------------------
    def bind_inputs(self, inputs):
        """Bind list of input names OR callable protein references."""
        self.inputs = inputs

    # ------------------------------------------------------------
    def mutate_param(self, key, scale=0.1):
        """Gaussian mutation."""
        if key in self.params:
            val = self.params[key]
            
            # Only mutate numeric types
            if isinstance(val, (int, float)):
                delta = random.gauss(0, scale * (abs(val) + 1e-9))
                self.params[key] = val + delta
            else:
                # Non-numeric fields (e.g., mode = "diff") mutate via categorical flips
                if isinstance(val, str):
                    # small chance to flip mode
                    if random.random() < 0.1:
                        if "mode" in key:
                            options = ["diff", "ratio", "greater", "less"]
                            self.params[key] = random.choice(options)
                # lists, dicts, others: skip for now

    # ------------------------------------------------------------
    def forward(self, signals, protein_outputs):
        """Override in subclasses."""
        raise NotImplementedError


# ================================================================
# 1. SENSOR PROTEIN
# Reads a single environment signal.
# Maintains normalization statistics.
# ================================================================
class SensorProtein(Protein):
    def __init__(self, signal_name):
        super().__init__(signal_name, "sensor")
        self.params["decay"] = 0.999
        
        self.state["running_max"] = 1.0
        self.state["count"] = 0

    def forward(self, signals, protein_outputs):
        raw = signals.get(self.name, 0.0)

        # PATCH: Distance signals need ABSOLUTE values preserved
        # Normalization destroys the meaning: "Is food 1 step away or 10?"
        # For distance, use fixed scale based on grid size (max distance = 18 for 10x10)
        if "dist" in self.name.lower() or self.name == "dist_to_food":
            # Use fixed normalization: divide by max possible distance (18 for 10x10 grid)
            # This preserves absolute magnitude: 1.0 = close, 18.0/18 = 1.0 = far
            max_distance = 18.0  # Maximum Manhattan distance on 10x10 grid
            self.output = raw / max_distance
            # Clamp to reasonable range
            self.output = max(min(self.output, 2.0), 0.0)
            return self.output
        
        # For other signals (steps_alive, etc.), use adaptive normalization
        # Track maximum absolute value seen
        self.state["running_max"] = max(
            self.params["decay"] * self.state["running_max"],
            abs(raw),
            1.0  # minimum scale
        )
        self.state["count"] += 1

        # Normalize by running max
        self.output = raw / self.state["running_max"]
        
        # Clamp the normalized output
        self.output = max(min(self.output, 5.0), -5.0)

        return self.output


# ================================================================
# 2. COMPARATOR PROTEIN
# Compares two inputs → difference, ratio, or threshold test.
# ================================================================
class ComparatorProtein(Protein):
    def __init__(self, name="comparator"):
        super().__init__(name, "comparator")
        self.params.update({
            "mode": "diff",   # diff | ratio | greater | less
            "threshold": 0.0,
        })

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        # Resolve inputs (signal or other protein)
        def resolve(x):
            return signals.get(x, protein_outputs.get(x, 0.0))

        a = resolve(self.inputs[0])
        b = resolve(self.inputs[1])

        mode = self.params["mode"]

        if mode == "diff":
            self.output = a - b

        elif mode == "ratio":
            self.output = a / (b + 1e-6)

        elif mode == "greater":
            self.output = 1.0 if a > b else -1.0

        elif mode == "less":
            self.output = 1.0 if a < b else -1.0

        return self.output


# ================================================================
# 3. TREND PROTEIN
# Detects direction and momentum of change.
# ================================================================
class TrendProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "trend")
        self.params["momentum"] = 0.9

        self.state["last"] = None
        self.state["velocity"] = 0.0

    def forward(self, signals, protein_outputs):
        # Only one input allowed
        if len(self.inputs) < 1:
            self.output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        last = self.state["last"]
        if last is None:
            self.state["last"] = x
            self.output = 0.0
            return 0.0

        # Delta
        delta = x - last
        self.state["last"] = x

        # Momentum-smoothed delta (EMA)
        self.state["velocity"] = (
            self.params["momentum"] * self.state["velocity"]
            + (1 - self.params["momentum"]) * delta
        )

        self.output = self.state["velocity"]
        return self.output


# ================================================================
# 4. INTEGRATOR PROTEIN
# Rolling accumulation or average of a signal.
# ================================================================
class IntegratorProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "integrator")
        self.params["decay"] = 0.05

        self.state["accum"] = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 1:
            self.output = 0
            return 0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        # Integrator half-life (prevents infinite accumulation)
        self.state["accum"] = self.state["accum"] * (1 - self.params["decay"]) + x
        
        # clamp output
        self.output = max(min(self.state["accum"], 10.0), -10.0)
        return self.output


# ================================================================
# 5. GATE PROTEIN
# Activates/deactivates other proteins based on conditions.
# ================================================================
class GateProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "gate")
        self.params["threshold"] = 0.0
        self.params["hysteresis"] = 0.1

        self.state["active"] = False

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 2:
            self.output = 0.0
            return 0.0

        condition = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))
        value = signals.get(self.inputs[1], protein_outputs.get(self.inputs[1], 0.0))

        if not self.state["active"] and condition > (self.params["threshold"] + self.params["hysteresis"]):
            self.state["active"] = True

        elif self.state["active"] and condition < (self.params["threshold"] - self.params["hysteresis"]):
            self.state["active"] = False

        # Gates should NOT multiply endlessly - use safe combination
        if self.state["active"]:
            # Use average to avoid exponential growth
            self.output = 0.5 * (condition + value)
        else:
            self.output = 0.0
        return self.output


# ================================================================
# 6. TRUST MODIFIER PROTEIN
# Converts protein output → trust delta.
# ================================================================
class TrustModifierProtein(Protein):
    def __init__(self, name="trust_mod"):
        super().__init__(name, "trust_modifier")
        self.params["gain"] = 1.0
        self.params["scale"] = 1.0
        self.params["decay"] = 0.999

        self.state["running"] = 0.0

        # Final trust output from this protein
        self.trust_output = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 1:
            self.trust_output = 0.0
            return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        # Rolling influence
        self.state["running"] = (
            self.params["decay"] * self.state["running"]
            + (1 - self.params["decay"]) * x
        )

        # Compute trust output
        raw_trust = self.params["gain"] * self.params["scale"] * self.state["running"]
        
        # No clamp - allow unlimited trust contributions for survival
        self.trust_output = raw_trust
        self.output = self.trust_output
        
        return self.trust_output


# ================================================================
# GENREG FORWARD PASS HELPER
# ================================================================
def run_protein_cascade(proteins, signals):
    """Runs a forward pass through all proteins in order."""
    outputs = {}

    for p in proteins:
        signal = p.forward(signals, outputs)
        # Don't clamp here - proteins self-regulate
        outputs[p.name] = signal
        p.output = signal

    # Collect trust contributions
    trust_delta = sum(
        p.trust_output
        for p in proteins
        if isinstance(p, TrustModifierProtein)
    )
    
    # Only clamp final trust delta (not individual signals)
    trust_delta = max(min(trust_delta, 5.0), -5.0)

    return outputs, trust_delta
