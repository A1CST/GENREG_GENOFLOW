import os
import random
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
CHECKPOINT_DIR = "checkpoints_simple"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================
# SIMPLE GEOMETRIC PREDICTOR
# ============================================================

class SimplePredictor(torch.nn.Module):
    """Simple learnable transformation from CLIP image embedding to text embedding."""

    def __init__(self, dim=512, virtual_context_size=4):
        super().__init__()
        self.dim = dim
        self.virtual_context_size = virtual_context_size

        # Learnable position-dependent weights
        self.position_weights = torch.nn.Parameter(
            torch.randn(virtual_context_size, dim) * 0.01
        )

        # Learnable dimension weights
        self.dim_weights = torch.nn.Parameter(
            torch.ones(dim) * 1.0
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
        self.accuracy = 0.0  # Track actual accuracy
        self.genome_id = Genome._id_counter
        Genome._id_counter += 1

    def forward(self, x):
        return self.predictor(x)

    def clone(self):
        new_genome = Genome()
        new_genome.predictor = self.predictor.clone()
        new_genome.trust = self.trust
        new_genome.accuracy = self.accuracy
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
# SIMPLE TRUST CALCULATION
# ============================================================

def calculate_trust_delta(is_correct, distance):
    """
    Simple trust calculation:
    - Correct prediction: +1.0
    - Wrong prediction: -0.5
    - Bonus for close predictions: up to +0.5
    - Penalty for far predictions: up to -0.5
    """
    # Base trust from correctness
    if is_correct:
        base_trust = 1.0
    else:
        base_trust = -0.5

    # Distance bonus/penalty (distance typically 0.5 to 3.0)
    # Close predictions (dist < 1.0): bonus
    # Far predictions (dist > 2.0): penalty
    if distance < 1.0:
        distance_bonus = (1.0 - distance) * 0.5  # Up to +0.5
    elif distance > 2.0:
        distance_bonus = -(distance - 2.0) * 0.5  # Up to -0.5
    else:
        distance_bonus = 0.0

    return base_trust + distance_bonus

# ============================================================
# TRAINING DASHBOARD
# ============================================================

class TrainingDashboard:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Simple Trainer Monitor")
        self.root.configure(bg="#1f1f1f")
        self.root.geometry("360x300")

        header = tk.Label(self.root, text="Training Status", bg="#1f1f1f", fg="white", font=("Segoe UI", 14, "bold"))
        header.pack(pady=(10, 5))

        self.step_value = tk.Label(self.root, text="Step: 0", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.step_value.pack()

        self.trust_value = tk.Label(self.root, text="Trust: 0.000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.trust_value.pack()

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

    def update(self, trust, accuracy, step):
        if not self.root:
            return

        self.trust_value.configure(text=f"Trust: {trust:.3f}")
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

    print("Starting simple training...")

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

            # Evaluate each genome and update trust
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

                    # Update trust
                    trust_delta = calculate_trust_delta(is_correct, distance)
                    g.trust += trust_delta * 0.01  # Scale down

                    del student

            # Sort by trust
            genomes.sort(key=lambda g: g.trust, reverse=True)
            best = genomes[0]

            # Every 100 steps: print status
            if step % 100 == 0:
                median_trust = genomes[len(genomes)//2].trust
                print(f"\n[Step {step}]")
                print(f"  Trust: Best={best.trust:.3f} | Median={median_trust:.3f}")
                print(f"  Last accuracy: {last_acc:.4f}")

            # Update dashboard
            dashboard.update(best.trust, last_acc, step)

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

            # Auto evaluation every 5000 steps
            if step % 5000 == 0:
                acc = evaluate_genome(best)
                last_acc = acc
                print(f"[{step}] AUTO EVAL: Accuracy={acc:.4f}")

                if acc >= 0.99:
                    print(f"\n>>> Training complete! Accuracy: {acc:.4f}")
                    break

            # Reproduce: replace worst with mutated best
            worst_idx = len(genomes) - 1
            old_id = genomes[worst_idx].genome_id
            genomes[worst_idx] = best.clone()
            genomes[worst_idx].genome_id = old_id
            genomes[worst_idx].mutate()
            genomes[worst_idx].trust = best.trust * 0.95

    finally:
        dashboard.close()

if __name__ == "__main__":
    train()
