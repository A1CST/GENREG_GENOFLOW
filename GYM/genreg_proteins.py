# ================================================================
# GENREG v2.0 — Stateful Biological Protein Library (Clean)
# Payton Miller — 2025
#
# Regulatory genome layer for Continuous Control.
# 100% forward-pass. No gradients.
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
        self.state = {}     # Biological memory
        self.output = 0.0   # Cached output
        self.inputs = []    # Bound input names
        self.params = {}    # Evolvable hyperparameters

    def bind_inputs(self, inputs):
        self.inputs = inputs

    def mutate_param(self, key, scale=0.1):
        """Gaussian mutation for parameters."""
        if key in self.params:
            val = self.params[key]
            if isinstance(val, (int, float)):
                # Scale mutation by value magnitude to keep it relative
                delta = random.gauss(0, scale * (abs(val) + 0.1))
                self.params[key] = val + delta
            elif isinstance(val, str) and "mode" in key:
                if random.random() < 0.1:
                    options = ["diff", "ratio", "greater", "less"]
                    self.params[key] = random.choice(options)

    def forward(self, signals, protein_outputs):
        raise NotImplementedError

# ================================================================
# 1. SENSOR PROTEIN (Standardized)
# ================================================================
class SensorProtein(Protein):
    def __init__(self, signal_name):
        super().__init__(signal_name, "sensor")
        self.params["decay"] = 0.999
        self.state["running_max"] = 1.0

    def forward(self, signals, protein_outputs):
        raw = signals.get(self.name, 0.0)

        # Adaptive Normalization: Track max seen value
        self.state["running_max"] = max(
            self.params["decay"] * self.state["running_max"],
            abs(raw),
            1.0
        )
        
        # Normalize
        self.output = raw / self.state["running_max"]
        
        # Soft clamp to prevent signal explosion
        self.output = max(min(self.output, 5.0), -5.0)
        return self.output

# ================================================================
# 2. COMPARATOR PROTEIN
# ================================================================
class ComparatorProtein(Protein):
    def __init__(self, name="comparator"):
        super().__init__(name, "comparator")
        self.params["mode"] = "diff" 
        self.params["threshold"] = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 2: return 0.0
        
        def resolve(x): return signals.get(x, protein_outputs.get(x, 0.0))
        a = resolve(self.inputs[0])
        b = resolve(self.inputs[1])

        mode = self.params["mode"]
        if mode == "diff": self.output = a - b
        elif mode == "ratio": self.output = a / (b + 1e-6)
        elif mode == "greater": self.output = 1.0 if a > b else -1.0
        elif mode == "less": self.output = 1.0 if a < b else -1.0
        
        return self.output

# ================================================================
# 3. TREND PROTEIN (Derivative)
# ================================================================
class TrendProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "trend")
        self.params["momentum"] = 0.9
        self.state["last"] = None
        self.state["velocity"] = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 1: return 0.0
        
        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))
        
        if self.state["last"] is None:
            self.state["last"] = x
            return 0.0

        delta = x - self.state["last"]
        self.state["last"] = x

        # EMA Velocity
        self.state["velocity"] = (
            self.params["momentum"] * self.state["velocity"]
            + (1 - self.params["momentum"]) * delta
        )
        self.output = self.state["velocity"]
        return self.output

# ================================================================
# 4. INTEGRATOR PROTEIN (Integral)
# ================================================================
class IntegratorProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "integrator")
        self.params["decay"] = 0.05
        self.state["accum"] = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 1: return 0.0
        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))
        
        self.state["accum"] = self.state["accum"] * (1 - self.params["decay"]) + x
        self.output = max(min(self.state["accum"], 10.0), -10.0)
        return self.output

# ================================================================
# 5. TRUST MODIFIER PROTEIN (The Reward Function)
# ================================================================
class TrustModifierProtein(Protein):
    def __init__(self, name="trust_mod"):
        super().__init__(name, "trust_modifier")
        self.params["gain"] = 1.0
        self.params["scale"] = 1.0
        self.params["decay"] = 0.9

        self.state["running"] = 0.0
        self.trust_output = 0.0

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 1: return 0.0

        x = signals.get(self.inputs[0], protein_outputs.get(self.inputs[0], 0.0))

        # Smooth the signal
        self.state["running"] = (
            self.params["decay"] * self.state["running"]
            + (1 - self.params["decay"]) * x
        )

        # Compute Trust: Gain * Scale * Signal
        # Scale can be negative (for punishment)
        self.trust_output = self.params["gain"] * self.params["scale"] * self.state["running"]
        
        self.output = self.trust_output
        return self.trust_output

# ================================================================
# CASCADE RUNNER
# ================================================================
def run_protein_cascade(proteins, signals):
    outputs = {}
    
    # 1. Forward Pass
    for p in proteins:
        signal = p.forward(signals, outputs)
        outputs[p.name] = signal

    # 2. Sum Trust
    trust_delta = sum(
        p.trust_output for p in proteins if isinstance(p, TrustModifierProtein)
    )
    
    # Clamp trust delta per step to prevent infinity
    trust_delta = max(min(trust_delta, 10.0), -10.0)
    
    return outputs, trust_delta