# ================================================================
# GENREG v2.0 — Neural Controller (Continuous)
# ================================================================
# Forward-pass neural network.
# Inputs:  MuJoCo Observation Vector (17 floats)
# Outputs: Joint Torques (6 floats, -1.0 to 1.0)
# ================================================================

import random
import math
import copy

def tanh(x):
    return math.tanh(x)

class GENREGController:
    def __init__(self, input_size=17, hidden_size=32, output_size=6):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights: Input -> Hidden
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)]
                   for _ in range(hidden_size)]
        self.b1 = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

        # Weights: Hidden -> Output
        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                   for _ in range(output_size)]
        self.b2 = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

    def clone(self):
        new = GENREGController(self.input_size, self.hidden_size, self.output_size)
        new.w1 = copy.deepcopy(self.w1)
        new.w2 = copy.deepcopy(self.w2)
        new.b1 = copy.deepcopy(self.b1)
        new.b2 = copy.deepcopy(self.b2)
        return new

    def mutate(self, rate=0.05, scale=0.1):
        """Gaussian mutation."""
        def mutate_mat(mat):
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if random.random() < rate:
                        mat[i][j] += random.gauss(0, scale)
        
        def mutate_vec(vec):
            for i in range(len(vec)):
                if random.random() < rate:
                    vec[i] += random.gauss(0, scale)

        mutate_mat(self.w1)
        mutate_mat(self.w2)
        mutate_vec(self.b1)
        mutate_vec(self.b2)

    def forward(self, inputs):
        """
        Returns continuous vector of size `output_size` (range -1 to 1).
        """
        # Hidden Layer (tanh)
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(self.input_size):
                s += self.w1[i][j] * inputs[j]
            hidden.append(tanh(s))

        # Output Layer (tanh for continuous control [-1, 1])
        outputs = []
        for i in range(self.output_size):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(tanh(s))

        return outputs