# ================================================================
# GENREG Controller - Neural Network for Action Selection
# Payton Miller — 2025
#
# Simple feed-forward network. Weights mutated, not trained.
# ================================================================

import random
import math


class Controller:
    """
    Feed-forward neural network controller.
    Input: environment signals → Hidden: tanh → Output: 4 actions
    """

    def __init__(self, input_size=11, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier initialization
        scale1 = math.sqrt(2.0 / (input_size + hidden_size))
        scale2 = math.sqrt(2.0 / (hidden_size + output_size))

        # Weights and biases
        self.w1 = [[random.gauss(0, scale1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[random.gauss(0, scale2) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b2 = [0.0] * output_size

    def forward(self, inputs):
        """
        Forward pass through the network.

        Args:
            inputs: list of floats (signal values)

        Returns:
            list of 4 output values (logits for each action)
        """
        # Pad or truncate inputs to match expected size
        x = list(inputs)
        while len(x) < self.input_size:
            x.append(0.0)
        x = x[:self.input_size]

        # Hidden layer
        h = []
        for i in range(self.hidden_size):
            val = self.b1[i]
            for j in range(self.input_size):
                val += self.w1[i][j] * x[j]
            h.append(math.tanh(val))

        # Output layer
        out = []
        for i in range(self.output_size):
            val = self.b2[i]
            for j in range(self.hidden_size):
                val += self.w2[i][j] * h[j]
            out.append(val)

        return out

    def select_action(self, signals, signal_order):
        """
        Select action from signal dictionary.

        Args:
            signals: dict of signal_name → value
            signal_order: list of signal names in order (from Sensor)

        Returns:
            int: action index (0-3)
        """
        inputs = [signals.get(k, 0.0) for k in signal_order]
        outputs = self.forward(inputs)

        # Argmax
        return outputs.index(max(outputs))

    def mutate(self, rate=0.1, scale=0.3):
        """
        Mutate weights with Gaussian noise.

        Args:
            rate: probability of mutating each weight
            scale: std dev of mutation
        """
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                if random.random() < rate:
                    self.w1[i][j] += random.gauss(0, scale)
            if random.random() < rate:
                self.b1[i] += random.gauss(0, scale)

        for i in range(self.output_size):
            for j in range(self.hidden_size):
                if random.random() < rate:
                    self.w2[i][j] += random.gauss(0, scale)
            if random.random() < rate:
                self.b2[i] += random.gauss(0, scale)

    def clone(self):
        """Create a deep copy of this controller."""
        c = Controller(self.input_size, self.hidden_size, self.output_size)
        c.w1 = [row[:] for row in self.w1]
        c.b1 = self.b1[:]
        c.w2 = [row[:] for row in self.w2]
        c.b2 = self.b2[:]
        return c

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        }

    @classmethod
    def from_dict(cls, d):
        """Deserialize from dictionary."""
        c = cls(d["input_size"], d["hidden_size"], d["output_size"])
        
        # If weights are empty or missing, keep the randomly initialized ones from __init__
        # Otherwise, load the saved weights
        if d.get("w1") and len(d.get("w1", [])) > 0:
            c.w1 = d["w1"]
        if d.get("b1") and len(d.get("b1", [])) > 0:
            c.b1 = d["b1"]
        if d.get("w2") and len(d.get("w2", [])) > 0:
            c.w2 = d["w2"]
        if d.get("b2") and len(d.get("b2", [])) > 0:
            c.b2 = d["b2"]
        # If any weights were empty, weights from __init__ are already randomly initialized
        
        return c
