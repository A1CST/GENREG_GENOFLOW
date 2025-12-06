# ================================================================
# GENREG v2.0 — Generic Biological Protein Library
# ================================================================
# Purpose: The "Regulatory Layer".
#          Proteins read signals (Inputs) and modify internal states
#          to produce Trust (Fitness) and internal messages.
# Updates:
#   - Removed all hardcoded Snake grid references.
#   - Added 'norm_scale' to SensorProtein for adaptive normalization.
#   - Optimized logic for continuous error signals (Vector Distance).
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
        self.state = {}
        self.output = 0.0
        self.inputs = []
        self.params = {}

    def bind_inputs(self, inputs):
        """Connect this protein to environment signals or other proteins."""
        self.inputs = inputs

    def mutate_param(self, key, scale=0.1):
        """Gaussian mutation for hyperparameters."""
        if key in self.params:
            val = self.params[key]
            if isinstance(val, (int, float)):
                # Adaptive mutation scale based on value magnitude
                # Adds 0.01 to prevent stagnation at 0.0
                delta = random.gauss(0, scale * (abs(val) + 0.01))
                self.params[key] = val + delta
            elif isinstance(val, str) and "mode" in key:
                # Chance to flip categorical modes
                if random.random() < 0.1:
                    options = ["diff", "ratio", "greater", "less"]
                    self.params[key] = random.choice(options)

    def forward(self, signals, protein_outputs):
        """
        Calculate output based on inputs.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


# ================================================================
# 1. SENSOR PROTEIN
# Reads environment signals (e.g., error_magnitude, token_hit).
# ================================================================
class SensorProtein(Protein):
    def __init__(self, signal_name):
        super().__init__(signal_name, "sensor")
        # norm_scale: Divisor to bring large signals into neural range (-1 to 1)
        # Default 1.0, but evolution can tune this to 'zoom in' on small errors.
        self.params["norm_scale"] = 1.0 
        self.params["offset"] = 0.0

    def forward(self, signals, protein_outputs):
        raw = signals.get(self.name, 0.0)
        
        # Linear scaling: (raw / scale) + offset
        # We assume the signal comes in 'raw' (e.g., distance 0.5 or 15.0)
        scale = self.params["norm_scale"] if abs(self.params["norm_scale"]) > 1e-6 else 1.0
        val = (raw / scale) + self.params["offset"]
        
        # Soft clamp (Tanh) to prevent exploding values inside the biological network
        # Multiplied by 2.0 to give it a bit more range (-2 to 2)
        self.output = math.tanh(val) * 2.0 
        return self.output


# ================================================================
# 2. TREND PROTEIN (The Derivative)
# Crucial for "Hot/Cold" learning. Detects if Error is shrinking.
# ================================================================
class TrendProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "trend")
        self.params["momentum"] = 0.5
        self.state["last"] = None
        self.state["velocity"] = 0.0

    def forward(self, signals, protein_outputs):
        if not self.inputs: return 0.0
        
        # Get input (usually a Sensor like 'error_magnitude')
        inp_name = self.inputs[0]
        # Try to find in signals first, then protein outputs
        curr = signals.get(inp_name, protein_outputs.get(inp_name, 0.0))

        if self.state["last"] is None:
            self.state["last"] = curr
            self.output = 0.0
            return 0.0

        # Delta: 
        # Positive = Value went UP (Bad for error)
        # Negative = Value went DOWN (Good for error)
        delta = curr - self.state["last"]
        self.state["last"] = curr

        # EMA smoothing (Momentum) to filter out noise
        m = self.params["momentum"]
        self.state["velocity"] = (m * self.state["velocity"]) + ((1 - m) * delta)

        self.output = self.state["velocity"]
        return self.output


# ================================================================
# 3. INTEGRATOR PROTEIN
# Accumulates value over time (e.g., persistence, total hits).
# ================================================================
class IntegratorProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "integrator")
        self.params["decay"] = 0.1  # Leak rate (forgetting factor)
        self.state["accum"] = 0.0

    def forward(self, signals, protein_outputs):
        if not self.inputs: return 0.0
        
        inp_name = self.inputs[0]
        val = signals.get(inp_name, protein_outputs.get(inp_name, 0.0))

        # Accumulate with leak
        self.state["accum"] = (self.state["accum"] * (1.0 - self.params["decay"])) + val
        
        # Hard clamp to prevent infinite accumulation/overflow
        self.output = max(min(self.state["accum"], 10.0), -10.0)
        return self.output


# ================================================================
# 4. TRUST MODIFIER PROTEIN
# The only protein that affects Genome Fitness (Trust).
# ================================================================
class TrustModifierProtein(Protein):
    def __init__(self, name="trust_mod"):
        super().__init__(name, "trust_modifier")
        self.params["gain"] = 1.0
        self.params["scale"] = 1.0  
        
        self.trust_output = 0.0

    def forward(self, signals, protein_outputs):
        if not self.inputs: return 0.0

        inp_name = self.inputs[0]

        # Priority: 1. Environment Signal, 2. Protein Output
        val = signals.get(inp_name, protein_outputs.get(inp_name, 0.0))

        # Allow negative trust to flow through (removed hard gate to enable penalties)
        # Previously blocked val <= 0.0, but we need negative signals for regression/wrong predictions
        self.trust_output = val * self.params["scale"] * self.params["gain"]

        self.output = self.trust_output
        return self.trust_output


# ================================================================
# 5. GATE PROTEIN (Logic Switch)
# Can inhibit or excite signals based on a threshold.
# ================================================================
class GateProtein(Protein):
    def __init__(self, name):
        super().__init__(name, "gate")
        self.params["threshold"] = 0.5
        self.state["active"] = False

    def forward(self, signals, protein_outputs):
        if len(self.inputs) < 2: return 0.0
        
        def resolve(x): return signals.get(x, protein_outputs.get(x, 0.0))
        
        # Input 0: The Control Signal (The Switch)
        cond = resolve(self.inputs[0])
        # Input 1: The Value Signal (The Payload)
        val = resolve(self.inputs[1])

        # Hysteresis toggle (prevents rapid flickering)
        if not self.state["active"] and cond > self.params["threshold"]:
            self.state["active"] = True
        elif self.state["active"] and cond < (self.params["threshold"] - 0.1):
            self.state["active"] = False

        self.output = val if self.state["active"] else 0.0
        return self.output


# ================================================================
# CASCADE RUNNER
# ================================================================
def run_protein_cascade(proteins, signals):
    """
    Runs one forward pass through the protein regulatory network.
    Returns:
      outputs (dict): The values of all proteins this step.
      total_trust_delta (float): The net change in fitness.
    """
    outputs = {}
    total_trust_delta = 0.0

    # DEBUG: Track accumulation
    debug_info = []

    # Proteins must be run in order (defined by template list)
    # to ensure inputs are ready if they depend on previous proteins.
    for p in proteins:
        # Forward pass for this protein
        out = p.forward(signals, outputs)
        outputs[p.name] = out

        # Accumulate trust if applicable
        # Use duck typing instead of isinstance() to avoid import path issues
        # (isinstance fails when TrustModifierProtein is imported via different paths)
        trust_val = getattr(p, 'trust_output', None)

        if hasattr(p, 'trust_output'):
            total_trust_delta += p.trust_output
            debug_info.append(f"  ✓ {p.name}: trust_output={p.trust_output:.4f}, total={total_trust_delta:.4f}")
        else:
            debug_info.append(f"  ✗ {p.name}: has_trust_output=False, trust_val={trust_val}")

    # DEBUG: Print accumulation trace (only occasionally to avoid spam)
    import random
    if random.random() < 0.01:  # 1% sample rate
        print(f"\n[CASCADE DEBUG] Protein accumulation trace:")
        for line in debug_info:
            print(line)
        print(f"[CASCADE DEBUG] Final total_trust_delta: {total_trust_delta:.4f}\n")

    return outputs, total_trust_delta

