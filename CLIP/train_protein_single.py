"""
Single-file CLIP training with inline protein system.
All proteins and training code in one place for easy debugging.
"""

import os
import random
import math
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import clip
import tkinter as tk

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
POP_SIZE = 100
MUTATION_RATE = 0.08
CHECKPOINT_DIR = "checkpoints_protein_single"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Protein config - BALANCED FOR LEARNING (reward 80% accuracy, punish 10%)
PROTEIN_CONFIG = {
    "distance_scale": 3.0,
    "proximity_max": 1.0,
    "proximity_halflife": 2.0,
    "proximity_steepness": 1.0,
    "proximity_penalty": 0.3,  # REDUCED from 0.8 - reward close predictions
    "proximity_scale": 1.5,    # INCREASED from 0.5 - make distance matter

    "category_match_bonus": 1.0,   # INCREASED - reward correctness strongly
    "category_mismatch_penalty": 2.0,  # HIGHER penalty for wrong answers
    "category_scale": 3.0,     # INCREASED from 1.0 - correctness is key

    "improvement_bonus": 0.2,
    "regression_penalty": 0.2,
    "improvement_scale": 0.3,
}

# ============================================================
# PROTEINS (Inline)
# ============================================================

class Protein:
    """Base protein class."""
    def __init__(self, name):
        self.name = name
        self.output = 0.0
        self.params = {}
        self.state = {}

    def forward(self, signals):
        raise NotImplementedError

class ProximityProtein(Protein):
    """Gives reward/penalty based on distance to target."""
    def __init__(self, name="proximity"):
        super().__init__(name)
        self.params["max_reward"] = 1.0
        self.params["half_life"] = 2.0
        self.params["steepness"] = 1.0
        self.params["penalty_baseline"] = 0.5

    def forward(self, signals):
        distance = signals.get("distance", 10.0)

        # Exponential decay
        decay = math.exp(-self.params["steepness"] * distance / self.params["half_life"])

        # Subtract penalty baseline (makes far predictions negative)
        reward = (self.params["max_reward"] * decay) - self.params["penalty_baseline"]

        # Floor at -1.0
        self.output = max(-1.0, reward)
        return self.output

class CategoryProtein(Protein):
    """Reward for correct category, penalty for wrong."""
    def __init__(self, name="category"):
        super().__init__(name)
        self.params["match_bonus"] = 0.8
        self.params["mismatch_penalty"] = 0.5

    def forward(self, signals):
        is_correct = signals.get("is_correct", False)

        if is_correct:
            self.output = self.params["match_bonus"]
        else:
            self.output = -self.params["mismatch_penalty"]

        return self.output

class ImprovementProtein(Protein):
    """Tracks if predictions are improving over time."""
    def __init__(self, name="improvement"):
        super().__init__(name)
        self.params["improvement_bonus"] = 0.3
        self.params["regression_penalty"] = 0.2
        self.params["momentum"] = 0.7
        self.state["last_distance"] = None
        self.state["trend"] = 0.0

    def forward(self, signals):
        distance = signals.get("distance", 10.0)

        if self.state["last_distance"] is None:
            self.state["last_distance"] = distance
            self.output = 0.0
            return 0.0

        # Calculate change (negative = getting closer = good)
        delta = distance - self.state["last_distance"]
        self.state["last_distance"] = distance

        # Smooth trend
        m = self.params["momentum"]
        self.state["trend"] = m * self.state["trend"] + (1 - m) * delta

        # Convert to reward/penalty
        if self.state["trend"] < -0.01:
            # Improving
            self.output = self.params["improvement_bonus"]
        elif self.state["trend"] > 0.01:
            # Getting worse
            self.output = -self.params["regression_penalty"]
        else:
            self.output = 0.0

        return self.output

def create_protein_network(config):
    """Create protein network with config."""
    proteins = []

    # Proximity protein
    proximity = ProximityProtein("proximity")
    proximity.params["max_reward"] = config["proximity_max"]
    proximity.params["half_life"] = config["proximity_halflife"]
    proximity.params["steepness"] = config["proximity_steepness"]
    proximity.params["penalty_baseline"] = config["proximity_penalty"]
    proteins.append(proximity)

    # Category protein
    category = CategoryProtein("category")
    category.params["match_bonus"] = config["category_match_bonus"]
    category.params["mismatch_penalty"] = config["category_mismatch_penalty"]
    proteins.append(category)

    # Improvement protein
    improvement = ImprovementProtein("improvement")
    improvement.params["improvement_bonus"] = config["improvement_bonus"]
    improvement.params["regression_penalty"] = config["regression_penalty"]
    proteins.append(improvement)

    return proteins, config

def run_protein_cascade(proteins, signals, config):
    """Run proteins and calculate total trust delta."""
    total_trust = 0.0

    for p in proteins:
        output = p.forward(signals)

        # Apply scale based on protein type
        if p.name == "proximity":
            scaled = output * config["proximity_scale"]
        elif p.name == "category":
            scaled = output * config["category_scale"]
        elif p.name == "improvement":
            scaled = output * config["improvement_scale"]
        else:
            scaled = output

        total_trust += scaled

    return total_trust

# ============================================================
# SIMPLE GEOMETRIC PREDICTOR
# ============================================================

class SimplePredictor(torch.nn.Module):
    """Simple learnable transformation from CLIP image embedding to text embedding."""

    def __init__(self, dim=512, virtual_context_size=4):
        super().__init__()
        self.dim = dim
        self.virtual_context_size = virtual_context_size

        # Learnable position-dependent weights - START RANDOM
        self.position_weights = torch.nn.Parameter(
            torch.randn(virtual_context_size, dim) * 0.5  # Increased from 0.01
        )

        # Learnable dimension weights - START RANDOM, NOT IDENTITY!
        self.dim_weights = torch.nn.Parameter(
            torch.randn(dim) * 0.3 + 0.5  # Random centered around 0.5, NOT 1.0
        )

    def forward(self, x):
        """x: [batch, 512] CLIP image embedding -> [batch, 512] predicted text embedding"""
        # Apply dimension weights
        x = x * self.dim_weights

        # Add position-dependent perturbations
        for i in range(self.virtual_context_size):
            x = x + self.position_weights[i] * (0.1 / (i + 1))

        # Normalize
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def clone(self):
        """Deep copy for reproduction."""
        new = SimplePredictor(self.dim, self.virtual_context_size)
        new.position_weights.data = self.position_weights.data.clone()
        new.dim_weights.data = self.dim_weights.data.clone()
        return new

    def mutate(self, rate=0.08):
        """Add Gaussian noise to parameters."""
        with torch.no_grad():
            self.position_weights += torch.randn_like(self.position_weights) * rate
            self.dim_weights += torch.randn_like(self.dim_weights) * rate * 0.1

# ============================================================
# GENOME
# ============================================================

class Genome:
    _id_counter = 0

    def __init__(self):
        self.predictor = SimplePredictor().to(DEVICE)
        self.trust = 1.0
        self.genome_id = Genome._id_counter
        Genome._id_counter += 1

        # Create protein network
        self.proteins, self.protein_config = create_protein_network(PROTEIN_CONFIG)

    def forward(self, x):
        return self.predictor(x)

    def run_proteins(self, signals):
        """Run protein cascade and return trust delta."""
        return run_protein_cascade(self.proteins, signals, self.protein_config)

    def clone(self):
        new_genome = Genome()
        new_genome.predictor = self.predictor.clone()
        new_genome.trust = self.trust
        # Copy protein state
        import copy
        new_genome.proteins = copy.deepcopy(self.proteins)
        return new_genome

    def mutate(self, rate=MUTATION_RATE):
        self.predictor.mutate(rate)

# ============================================================
# LOAD TEACHER CLIP
# ============================================================

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

@torch.no_grad()
def encode_text(text):
    tokens = clip.tokenize([text]).to(DEVICE)
    txt = clip_model.encode_text(tokens).float()
    return torch.nn.functional.normalize(txt, dim=-1)

# ============================================================
# DATASET
# ============================================================

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

dataset = ImageFolder("caltech-101/101_ObjectCategories/101_ObjectCategories", transform=transform)
labels = list(dataset.class_to_idx.keys())

# Precompute text latents
text_latents = {cls: encode_text(cls).detach() for cls in labels}
all_txt_tensor = torch.stack([text_latents[c].squeeze(0) for c in labels]).to(DEVICE)

def preprocess_image(img):
    pil_img = T.ToPILImage()(img.squeeze().cpu())
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)

# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_genome(genome, num_samples=200):
    """Evaluate genome accuracy on random samples."""
    correct = 0
    total = 0
    sample = random.sample(list(dataset), num_samples)

    for img, idx in sample:
        img = img.unsqueeze(0).to(DEVICE)

        # Get CLIP embedding
        t = clip_model.encode_image(preprocess_image(img)).float()
        t = t / t.norm(dim=-1, keepdim=True)

        # Get student prediction
        s = genome.forward(t).squeeze(0)  # [512]

        # Compare to all text embeddings
        sims = (s @ all_txt_tensor.t())
        pred_idx = sims.argmax().item()

        if pred_idx == idx:
            correct += 1
        total += 1

    return correct / total

# ============================================================
# TRAINING DASHBOARD
# ============================================================

class TrainingDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Protein Trainer Monitor")
        self.root.configure(bg="#1f1f1f")
        self.root.geometry("360x320")

        header = tk.Label(self.root, text="Training Status", bg="#1f1f1f", fg="white", font=("Segoe UI", 14, "bold"))
        header.pack(pady=(10, 5))

        self.step_value = tk.Label(self.root, text="Step: 0", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.step_value.pack()

        self.trust_value = tk.Label(self.root, text="Trust: 0.000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.trust_value.pack()

        self.trust_delta_value = tk.Label(self.root, text="Trust Δ: +0.000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.trust_delta_value.pack()

        self.accuracy_value = tk.Label(self.root, text="Accuracy: 0.0000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.accuracy_value.pack()

        self.highest_acc_value = tk.Label(self.root, text="Highest Acc: 0.0000", bg="#1f1f1f", fg="#52c41a", font=("Segoe UI", 11, "bold"))
        self.highest_acc_value.pack()

        self.eval_button = tk.Button(
            self.root,
            text="Run Eval Now",
            bg="#1890ff",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            command=self._request_eval,
            relief="raised",
            pady=5
        )
        self.eval_button.pack(pady=(10, 5))

        self._highest_accuracy = 0.0
        self.manual_eval_requested = False

    def _request_eval(self):
        self.manual_eval_requested = True
        self.eval_button.configure(text="Eval Queued...", bg="#52c41a")

    def update(self, trust, trust_delta, accuracy, step):
        if not self.root:
            return

        self.trust_value.configure(text=f"Trust: {trust:.3f}")

        # Color trust delta (green for positive, red for negative)
        delta_color = "#52c41a" if trust_delta >= 0 else "#ff4d4f"
        delta_sign = "+" if trust_delta >= 0 else ""
        self.trust_delta_value.configure(
            text=f"Trust Δ: {delta_sign}{trust_delta:.3f}",
            fg=delta_color
        )

        self.accuracy_value.configure(text=f"Accuracy: {accuracy:.4f}")
        self.step_value.configure(text=f"Step: {step}")

        if accuracy > self._highest_accuracy:
            self._highest_accuracy = accuracy
            self.highest_acc_value.configure(text=f"Highest Acc: {self._highest_accuracy:.4f}")

        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.root = None

    def close(self):
        if self.root:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self.root = None

# ============================================================
# TRAINING LOOP
# ============================================================

def train():
    genomes = [Genome() for _ in range(POP_SIZE)]
    dashboard = TrainingDashboard()

    step = 0
    last_acc = 0.0
    last_trust_delta = 0.0

    print("Starting protein-based training...")
    print(f"Protein scales: proximity={PROTEIN_CONFIG['proximity_scale']}, category={PROTEIN_CONFIG['category_scale']}, improvement={PROTEIN_CONFIG['improvement_scale']}")

    try:
        while True:
            step += 1

            # Sample training data
            img, idx = random.choice(dataset)
            img = img.unsqueeze(0).to(DEVICE)

            # Get CLIP embedding once
            with torch.no_grad():
                pil_img = T.ToPILImage()(img.squeeze().cpu())
                preprocessed_img = preprocess(pil_img).unsqueeze(0).to(DEVICE)
                t_img = clip_model.encode_image(preprocessed_img).float()
                t_img = t_img / t_img.norm(dim=-1, keepdim=True)
                t_img = t_img.detach()
                del preprocessed_img, pil_img

            correct_txt = text_latents[labels[idx]].squeeze(0)

            # Evaluate each genome with proteins
            for g in genomes:
                with torch.no_grad():
                    # Get prediction
                    student = g.forward(t_img).squeeze(0)

                    # Check if correct
                    sims = (student @ all_txt_tensor.t())
                    pred_idx = sims.argmax().item()
                    is_correct = (pred_idx == idx)

                    # Calculate distance to correct text
                    distance = torch.dist(student, correct_txt, p=2).item()

                    # Create signals for proteins
                    signals = {
                        "distance": distance,
                        "is_correct": is_correct,
                    }

                    # Run protein cascade
                    trust_delta = g.run_proteins(signals)

                    # Apply trust delta (with small scaling)
                    g.trust += trust_delta * 0.01

                    del student

            # Sort by trust
            genomes.sort(key=lambda g: g.trust, reverse=True)
            best = genomes[0]

            # Track trust delta from best genome for display
            if step > 1:
                # Re-run best genome's proteins to get delta for display
                with torch.no_grad():
                    student = best.forward(t_img).squeeze(0)
                    sims = (student @ all_txt_tensor.t())
                    pred_idx = sims.argmax().item()
                    is_correct = (pred_idx == idx)
                    distance = torch.dist(student, correct_txt, p=2).item()
                    signals = {"distance": distance, "is_correct": is_correct}
                    last_trust_delta = best.run_proteins(signals)
                    del student

            # Every 100 steps: print status AND protein debug
            if step % 100 == 0:
                median_trust = genomes[len(genomes)//2].trust

                # DEBUG: Sample 10 predictions and show average protein outputs
                debug_proximity = []
                debug_category = []
                debug_total = []
                debug_correct_count = 0

                for _ in range(10):
                    test_img, test_idx = random.choice(dataset)
                    test_img = test_img.unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        pil = T.ToPILImage()(test_img.squeeze().cpu())
                        prep = preprocess(pil).unsqueeze(0).to(DEVICE)
                        t_test = clip_model.encode_image(prep).float()
                        t_test = t_test / t_test.norm(dim=-1, keepdim=True)

                        student = best.forward(t_test).squeeze(0)
                        sims = (student @ all_txt_tensor.t())
                        pred_idx = sims.argmax().item()
                        is_correct = (pred_idx == test_idx)

                        if is_correct:
                            debug_correct_count += 1

                        distance = torch.dist(student, text_latents[labels[test_idx]].squeeze(0), p=2).item()

                        # Run proteins
                        signals = {"distance": distance, "is_correct": is_correct}
                        for p in best.proteins:
                            p.forward(signals)

                        prox_out = best.proteins[0].output * PROTEIN_CONFIG["proximity_scale"]
                        cat_out = best.proteins[1].output * PROTEIN_CONFIG["category_scale"]
                        total = prox_out + cat_out

                        debug_proximity.append(prox_out)
                        debug_category.append(cat_out)
                        debug_total.append(total)

                avg_prox = sum(debug_proximity) / len(debug_proximity)
                avg_cat = sum(debug_category) / len(debug_category)
                avg_total = sum(debug_total) / len(debug_total)
                avg_correct = debug_correct_count / 10.0

                print(f"\n[Step {step}]")
                print(f"  Trust: Best={best.trust:.3f} | Median={median_trust:.3f}")
                print(f"  Last trust Δ: {last_trust_delta:.3f}")
                print(f"  [DEBUG - Quick check, NOT official eval]:")
                print(f"    Protein avg (10 samples): Proximity={avg_prox:+.3f} Category={avg_cat:+.3f} Total={avg_total:+.3f}")
                print(f"    Quick check accuracy: {avg_correct:.1%} ({debug_correct_count}/10)")
                print(f"  OFFICIAL EVAL (last 200 samples): {last_acc:.4f}")

            # Update dashboard
            dashboard.update(best.trust, last_trust_delta, last_acc, step)

            # Manual evaluation
            if dashboard.manual_eval_requested:
                acc = evaluate_genome(best)
                last_acc = acc
                print(f"[{step}] MANUAL EVAL: Accuracy={acc:.4f}")
                dashboard.manual_eval_requested = False
                dashboard.eval_button.configure(text="Run Eval Now", bg="#1890ff")

                if acc >= 0.99:
                    print(f"\n>>> Training complete! Accuracy: {acc:.4f}")
                    break

            # Auto evaluation every 500 steps (200 samples from dataset)
            if step % 500 == 0:
                print(f"\n[{step}] === RUNNING OFFICIAL EVAL (200 random samples) ===")
                acc = evaluate_genome(best, num_samples=200)
                last_acc = acc
                print(f"[{step}] === OFFICIAL ACCURACY: {acc:.4f} ===")

                if acc >= 0.99:
                    print(f"\n>>> Training complete! Accuracy: {acc:.4f}")
                    break

            # Check if stuck (no improvement in accuracy)
            if step % 1000 == 0 and last_acc < 0.05:
                # STUCK! Inject diversity - clone top 10 genomes with HEAVY mutation
                num_to_replace = POP_SIZE // 5  # Replace bottom 20%
                num_top = 10  # Use top 10 as parents
                print(f"\n[{step}] STUCK! Accuracy={last_acc:.4f} - Replacing bottom {num_to_replace} with heavily mutated top-{num_top}")

                genomes.sort(key=lambda g: g.trust, reverse=True)
                top_genomes = genomes[:num_top]  # Top 10

                for i in range(num_to_replace):
                    idx = len(genomes) - 1 - i
                    # Pick random parent from top 10
                    parent = random.choice(top_genomes)
                    genomes[idx] = parent.clone()
                    # HEAVY mutation (5x normal rate)
                    genomes[idx].mutate(MUTATION_RATE * 5.0)
                    genomes[idx].trust = parent.trust * 0.8
                    print(f"  Replaced rank {idx+1} with heavily mutated clone of rank {genomes.index(parent)+1}")

            # Normal reproduction: replace worst with mutated best
            else:
                worst_idx = len(genomes) - 1
                old_id = genomes[worst_idx].genome_id
                genomes[worst_idx] = best.clone()
                genomes[worst_idx].genome_id = old_id

                # Increase mutation rate when stuck
                mutation_rate = MUTATION_RATE * 3.0 if last_acc < 0.05 else MUTATION_RATE
                genomes[worst_idx].mutate(mutation_rate)
                genomes[worst_idx].trust = best.trust * 0.95

    finally:
        dashboard.close()

if __name__ == "__main__":
    train()
