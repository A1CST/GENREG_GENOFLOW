import os
import json
import random
import csv
from datetime import datetime
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import clip
import tkinter as tk
from tkinter import filedialog
import pygame
import threading

# Protein system imports
from GENREG.proteins import run_protein_cascade
from clip_proteins import (
    CLIPGeometricPredictor,
    create_clip_protein_network,
    CLIP_PROTEIN_CONFIG,
    compute_protein_signals,
    get_predicted_class,
)

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
LATENT_DIM = 512
POP_SIZE = 100
MUTATION_RATE = 0.08
TRUST_STEP = 0.01
CHECKPOINT_DIR = "checkpoints_full"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_SHORT_DIR = "checkpoints_short"
os.makedirs(CHECKPOINT_SHORT_DIR, exist_ok=True)

# Protein trust scale factor (converts protein trust_delta to match old +/-0.01 magnitude)
PROTEIN_TRUST_SCALE = 0.01  # Increased from 0.001 - allows trust to accumulate 10x faster

# ============================================================
# GENOME
# ============================================================

class Genome:
    _id_counter = 0

    def __init__(self):
        # NEW: Use CLIPGeometricPredictor instead of Linear
        self.predictor = CLIPGeometricPredictor(
            embedding_dim=512,
            virtual_context_size=4,
            config=CLIP_PROTEIN_CONFIG,
            device=DEVICE
        )

        self.trust = 1.0
        self.low_trust_steps = 0  # Track consecutive steps with trust < 0.02
        self.age = 0  # Track genome age in steps
        self.genome_id = Genome._id_counter
        self.last_mutation_rate = MUTATION_RATE
        Genome._id_counter += 1

        # NEW: Add protein network
        self.proteins = create_clip_protein_network(CLIP_PROTEIN_CONFIG)

        # DEBUG: Verify proteins were created (only log first genome)
        if self.genome_id == 0:
            print(f"[DEBUG] Genome #{self.genome_id} initialized with {len(self.proteins)} proteins:")
            for p in self.proteins:
                print(f"  - {p.name} ({p.__class__.__name__})")

    def forward(self, x):
        """Forward pass through predictor."""
        return self.predictor(x)

    def run_proteins(self, signals):
        """Run protein cascade and return trust delta."""
        outputs, trust_delta = run_protein_cascade(self.proteins, signals)

        # DEBUG: Check TrustModifierProtein values
        if self.genome_id == 0 and not hasattr(self, '_debug_logged'):
            print(f"\n[DEBUG] Checking TrustModifierProteins:")
            from GENREG.proteins import TrustModifierProtein
            for p in self.proteins:
                if isinstance(p, TrustModifierProtein):
                    print(f"  {p.name}: trust_output={p.trust_output}, output={p.output}")
            print(f"  Final trust_delta from cascade: {trust_delta}")
            self._debug_logged = True

        return trust_delta

    def clone(self):
        """Deep copy genome for reproduction."""
        new_genome = Genome()
        new_genome.predictor = self.predictor.clone()
        new_genome.trust = self.trust
        new_genome.low_trust_steps = 0
        new_genome.age = 0
        new_genome.proteins = create_clip_protein_network(CLIP_PROTEIN_CONFIG)
        return new_genome

    def mutate(self, current_phase=None, avg_div=None, mutation_rate_multiplier=1.0):
        # Dynamic mutation rate based on trust
        # Only activate after 200 consecutive steps of trust < 0.02
        if self.low_trust_steps >= 200 and self.trust < 0.02:
            # Scale mutation rate inversely with trust
            # Lower trust = higher mutation rate
            trust_normalized = (0.02 - self.trust) / (0.02 - (-1.0))  # 0 to 1
            trust_multiplier = 1.0 + (trust_normalized * 1.5)  # 1.0 to 2.5x
            dynamic_rate = MUTATION_RATE * trust_multiplier * mutation_rate_multiplier
        else:
            dynamic_rate = MUTATION_RATE * mutation_rate_multiplier

        # Reduce mutation rate in Phase 2 if diversity is too low
        if current_phase == 2 and avg_div is not None and avg_div < 0.006:
            dynamic_rate *= 0.5

        self.last_mutation_rate = float(dynamic_rate)

        # NEW: Mutate predictor instead of Linear weights
        self.predictor.mutate(rate=dynamic_rate, current_phase=current_phase)

# ============================================================
# LOAD TEACHER CLIP
# ============================================================

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# ============================================================
# TEXT ENCODER
# ============================================================

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

# precompute text latents
text_latents = {cls: encode_text(cls).detach() for cls in labels}

# Pre-stack all 101 text latents for 101-way ranking
all_txt_tensor = torch.stack(
    [text_latents[c].squeeze(0) for c in labels]
).to(DEVICE)   # shape: [101, 512]

# ============================================================
# PREPROCESS WRAPPER
# ============================================================

def preprocess_image(img):
    pil_img = T.ToPILImage()(img.squeeze().cpu())
    return preprocess(pil_img).unsqueeze(0).to(DEVICE)

# ============================================================
# ZERO SHOT EVAL
# ============================================================

@torch.no_grad()
def zero_shot_eval(genome):
    correct, total = 0, 0
    sample = random.sample(list(dataset), 200)

    for img, idx in sample:
        img = img.unsqueeze(0).to(DEVICE)

        t = clip_model.encode_image(preprocess_image(img)).float()
        t = t / t.norm(dim=-1, keepdim=True)

        s = genome.forward(t).squeeze(0)  # [512]

        sims = (s @ all_txt_tensor.t())  # [101]
        pred_idx = sims.argmax().item()

        if pred_idx == idx:
            correct += 1
        total += 1

    return correct / total

# ============================================================
# COSINE
# ============================================================

def cosine(a, b):
    return (a @ b.t()).item()

# ============================================================
# CSV LOGGING
# ============================================================

class CSVLogger:
    def __init__(self, log_file="training_log.csv"):
        self.log_file = log_file
        self.file = open(self.log_file, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Write header
        self.writer.writerow(['timestamp', 'step', 'genome_id', 'phase', 'trust', 'accuracy'])
        self.file.flush()
    
    def log(self, step, genome_id, phase, trust, accuracy):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.writer.writerow([timestamp, step, genome_id, phase, f'{trust:.6f}', f'{accuracy:.6f}'])
        self.file.flush()
    
    def close(self):
        self.file.close()

# ============================================================
# TRAINING DASHBOARD GUI
# ============================================================

class TrainingDashboard:
    STATE_CONFIG = {
        "Collapsed": {"color": "#ff4d4f"},
        "Stagnating": {"color": "#faad14"},
        "Healthy": {"color": "#52c41a"},
        "Aggressive": {"color": "#1890ff"},
        "Noisy": {"color": "#d4380d"},
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Minimal Trainer Monitor")
        self.root.configure(bg="#1f1f1f")
        self.root.geometry("360x380")

        header = tk.Label(self.root, text="Training Status", bg="#1f1f1f", fg="white", font=("Segoe UI", 14, "bold"))
        header.pack(pady=(10, 5))

        self.step_value = tk.Label(self.root, text="Step: 0", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.step_value.pack()

        self.phase_value = tk.Label(self.root, text="Phase: 1 (1-neg)", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.phase_value.pack()

        self.diversity_value = tk.Label(self.root, text="Diversity: 0.0000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.diversity_value.pack()

        self.trust_value = tk.Label(self.root, text="Trust: 0.000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.trust_value.pack()

        self.accuracy_value = tk.Label(self.root, text="Accuracy: 0.0000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.accuracy_value.pack()

        self.highest_acc_value = tk.Label(self.root, text="Highest Acc: 0.0000", bg="#1f1f1f", fg="#52c41a", font=("Segoe UI", 11, "bold"))
        self.highest_acc_value.pack()

        self.mrate_value = tk.Label(self.root, text="Mutation Rate: 0.0000", bg="#1f1f1f", fg="white", font=("Segoe UI", 11))
        self.mrate_value.pack()

        self.trend_label = tk.Label(self.root, text="Trend: =",
                                    bg="#1f1f1f", fg="white", font=("Segoe UI", 12, "bold"))
        self.trend_label.pack(pady=(0, 10))

        states_frame = tk.Frame(self.root, bg="#1f1f1f")
        states_frame.pack(pady=5, fill="x", padx=15)

        self.state_labels = {}
        for state, cfg in self.STATE_CONFIG.items():
            lbl = tk.Label(
                states_frame,
                text=state,
                width=18,
                pady=4,
                bg="#3a3a3a",
                fg="white",
                relief="groove",
                font=("Segoe UI", 10, "bold")
            )
            lbl.pack(pady=2, fill="x")
            self.state_labels[state] = lbl

        self._active_state = None
        self._highest_accuracy = 0.0
        self.manual_eval_requested = False
        
        # Manual evaluation button
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
    
    def _request_eval(self):
        """Set flag to request manual evaluation on next training tick"""
        self.manual_eval_requested = True
        self.eval_button.configure(text="Eval Queued...", bg="#52c41a")

    def _diversity_state(self, value):
        if value < 0.002:
            return "Collapsed"
        if value < 0.006:
            return "Stagnating"
        if value < 0.015:
            return "Healthy"
        if value < 0.03:
            return "Aggressive"
        return "Noisy"

    def update(self, diversity, trust, trend, accuracy, phase_name, phase_number, step, mutation_rate):
        if not self.root:
            return

        state = self._diversity_state(diversity)
        if state != self._active_state:
            for name, lbl in self.state_labels.items():
                if name == state:
                    lbl.configure(bg=self.STATE_CONFIG[name]["color"], fg="#000000")
                else:
                    lbl.configure(bg="#3a3a3a", fg="white")
            self._active_state = state

        self.diversity_value.configure(text=f"Diversity: {diversity:.4f}")
        self.trust_value.configure(text=f"Trust: {trust:.3f}")
        self.accuracy_value.configure(text=f"Accuracy: {accuracy:.4f}")
        
        # Update highest accuracy record
        if accuracy > self._highest_accuracy:
            self._highest_accuracy = accuracy
            self.highest_acc_value.configure(text=f"Highest Acc: {self._highest_accuracy:.4f}", fg="#52c41a")
        
        self.phase_value.configure(text=f"Phase: {phase_number} ({phase_name})")
        self.step_value.configure(text=f"Step: {step}")
        self.mrate_value.configure(text=f"Mutation Rate: {mutation_rate:.4f}")

        trend_symbol = {"up": "▲", "down": "▼", "flat": "="}.get(trend, "=")
        trend_color = {"up": "#52c41a", "down": "#ff4d4f", "flat": "#ffffff"}.get(trend, "#ffffff")
        self.trend_label.configure(text=f"Trend: {trend_symbol}", fg=trend_color)

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
# TRUST CHART (PYGAME)
# ============================================================

class TrustChart:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Trust Over Time")
        self.clock = pygame.time.Clock()
        
        # Data storage
        self.trust_history = []  # List of (step, trust) tuples
        self.max_history = 10000  # Keep last 10k points
        self.update_interval = 100  # Update chart every N steps
        self.last_update_step = 0
        
        # Chart settings
        self.padding = 60
        self.chart_width = width - 2 * self.padding
        self.chart_height = height - 2 * self.padding
        
        # Colors
        self.bg_color = (20, 20, 30)
        self.grid_color = (40, 40, 50)
        self.line_color = (100, 200, 255)
        self.text_color = (200, 200, 200)
        
        self.running = True
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
    
    def add_data_point(self, step, trust):
        """Add a trust value at a given step"""
        self.trust_history.append((step, trust))
        # Keep only recent history to prevent memory issues
        if len(self.trust_history) > self.max_history:
            self.trust_history = self.trust_history[-self.max_history:]
    
    def _update_loop(self):
        """Separate thread to handle pygame events and updates"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
            
            self._draw()
            self.clock.tick(10)  # 10 FPS is enough for a chart
    
    def _draw(self):
        """Draw the chart"""
        self.screen.fill(self.bg_color)
        
        if len(self.trust_history) < 2:
            # Not enough data yet
            text = self.font.render("Waiting for data...", True, self.text_color)
            self.screen.blit(text, (self.width // 2 - 100, self.height // 2))
            pygame.display.flip()
            return
        
        # Get data range
        steps = [s for s, _ in self.trust_history]
        trusts = [t for _, t in self.trust_history]
        
        min_step = min(steps)
        max_step = max(steps)
        min_trust = min(trusts)
        max_trust = max(trusts)
        
        # Add some padding to the trust range
        trust_range = max_trust - min_trust
        if trust_range < 1.0:
            trust_range = 1.0
        min_trust_display = min_trust - trust_range * 0.1
        max_trust_display = max_trust + trust_range * 0.1
        
        step_range = max_step - min_step
        if step_range == 0:
            step_range = 1
        
        # Draw grid
        num_grid_lines = 10
        for i in range(num_grid_lines + 1):
            # Horizontal grid lines
            y = self.padding + (self.chart_height * i / num_grid_lines)
            pygame.draw.line(self.screen, self.grid_color, 
                           (self.padding, y), 
                           (self.width - self.padding, y), 1)
            
            # Trust value labels
            trust_val = max_trust_display - (max_trust_display - min_trust_display) * (i / num_grid_lines)
            trust_text = self.small_font.render(f"{trust_val:.2f}", True, self.text_color)
            self.screen.blit(trust_text, (5, y - 10))
        
        # Vertical grid lines
        for i in range(num_grid_lines + 1):
            x = self.padding + (self.chart_width * i / num_grid_lines)
            pygame.draw.line(self.screen, self.grid_color, 
                           (x, self.padding), 
                           (x, self.height - self.padding), 1)
            
            # Step labels
            step_val = min_step + step_range * (i / num_grid_lines)
            step_text = self.small_font.render(f"{int(step_val)}", True, self.text_color)
            self.screen.blit(step_text, (x - 20, self.height - self.padding + 5))
        
        # Draw trust line
        if len(self.trust_history) > 1:
            points = []
            for step, trust in self.trust_history:
                x = self.padding + ((step - min_step) / step_range) * self.chart_width
                y = self.padding + self.chart_height - ((trust - min_trust_display) / (max_trust_display - min_trust_display)) * self.chart_height
                points.append((int(x), int(y)))
            
            # Draw line
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.line_color, False, points, 2)
            
            # Draw current point
            if points:
                pygame.draw.circle(self.screen, self.line_color, points[-1], 4)
        
        # Draw title and current stats
        title = self.font.render("Trust Over Time", True, self.text_color)
        self.screen.blit(title, (self.padding, 10))
        
        if self.trust_history:
            last_step, last_trust = self.trust_history[-1]
            stats_text = self.small_font.render(
                f"Step: {last_step} | Trust: {last_trust:.3f} | Points: {len(self.trust_history)}", 
                True, self.text_color
            )
            self.screen.blit(stats_text, (self.padding, 35))
        
        pygame.display.flip()
    
    def close(self):
        """Close the chart window"""
        self.running = False
        pygame.quit()

# ============================================================
# CHECKPOINT SAVE/LOAD
# ============================================================

def save_checkpoint(genomes, phase, step, avg_div, labels, text_latents, path):
    torch.save({
        "step": step,
        "phase": phase,
        "avg_div": avg_div,
        "labels": labels,
        "text_latents": {k: v.cpu() for k, v in text_latents.items()},
        "genomes": [
            {
                "predictor_state": g.predictor.state_dict(),  # NEW: Save predictor state
                "trust": g.trust,
                "age": g.age,
                "low_trust_steps": g.low_trust_steps,
                "last_mutation_rate": g.last_mutation_rate,
                "genome_id": g.genome_id,
            }
            for g in genomes
        ],
    }, path)

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)

    phase = ckpt["phase"]
    step = ckpt["step"]
    avg_div = ckpt["avg_div"]
    labels = ckpt["labels"]

    text_latents = {k: v.to(device) for k, v in ckpt["text_latents"].items()}

    genomes = []
    for gdata in ckpt["genomes"]:
        g = Genome()  # Will create new predictor
        g.predictor.load_state_dict(gdata["predictor_state"])  # NEW: Load predictor state
        g.predictor.to(device)
        g.trust = gdata["trust"]
        g.age = gdata["age"]
        g.low_trust_steps = gdata["low_trust_steps"]
        g.last_mutation_rate = gdata["last_mutation_rate"]
        g.genome_id = gdata["genome_id"]
        genomes.append(g)

    return genomes, phase, step, avg_div, labels, text_latents

def prompt_checkpoint_file():
    """Open a file dialog so the user can select a checkpoint to resume from."""
    try:
        root = tk.Tk()
        root.withdraw()
        initial_dir = CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else "."
        file_path = filedialog.askopenfilename(
            title="Select checkpoint to resume (Cancel for fresh start)",
            initialdir=initial_dir,
            filetypes=[("Checkpoint files", "*.pth"), ("All files", "*.*")]
        )
        root.destroy()
        return file_path or None
    except tk.TclError:
        return None

# ============================================================
# PHASED TRAINING
# ============================================================

def evaluate_phase_1(student, correct_txt, neg_txt):
    """Phase 1: 1 random negative (binary comparison)"""
    pos = cosine(student, correct_txt)
    neg = cosine(student, neg_txt)
    return pos > neg

def evaluate_phase_n(student, correct_txt, neg_txts):
    """Phases 2-9: N negatives"""
    pos = cosine(student, correct_txt)
    neg_sims = [cosine(student, neg_txt) for neg_txt in neg_txts]
    max_neg = max(neg_sims)
    return pos > max_neg

def evaluate_phase_stable(student, correct_txt, neg_txts):
    """Phase ≥5: two positives with clamped negative pressure"""
    pos1 = cosine(student, correct_txt)
    pos2 = cosine(student, correct_txt)
    pos = max(pos1, pos2)

    neg_sims = [cosine(student, n) for n in neg_txts]

    max_neg = max(neg_sims)
    max_neg = min(max_neg, pos - 0.05)

    return pos > max_neg

############################################################
#   UNIFIED TRIPLET BUILDER (Training, Eval)
############################################################

def build_training_triplet(genome, pos_clip_emb, correct_txt, neg_txts, device):
    """
    Unified triplet construction using pre-computed CLIP embedding to avoid repeated preprocessing.
    Returns: (pos_sim, neg_sims_list)
    pos_sim: similarity between student embedding and correct text
    neg_sims: similarities between student embedding and negative texts
    """
    # EXACT same path training uses, but with pre-computed CLIP embedding
    with torch.no_grad():
        # pos_clip_emb is already normalized CLIP embedding passed in
        # Get student embedding through genome
        pos_student_emb = genome.forward(pos_clip_emb)
        pos_student_emb = pos_student_emb / pos_student_emb.norm(dim=-1, keepdim=True)
        
        # Get correct text embedding (already normalized)
        correct_txt_emb = correct_txt.squeeze(0).to(device)  # [D]
        
        # Get text embeddings for negatives (already normalized)
        neg_txt_embs = torch.stack([neg_txt.squeeze(0) for neg_txt in neg_txts]).to(device)  # [K, D]
    
        # Strict cosine comparison like training
        pos_sim = (pos_student_emb @ correct_txt_emb.T).item()  # student vs correct text
        neg_sims = (pos_student_emb @ neg_txt_embs.T).squeeze(0)  # [K] student vs negative texts
        neg_sims_list = neg_sims.cpu().tolist()
        
        # Explicitly delete intermediate tensors to free GPU memory
        del pos_student_emb, correct_txt_emb, neg_txt_embs, neg_sims
    
    return float(pos_sim), neg_sims_list

############################################################
#   NEW TRAINING EVALUATION MODES
############################################################

def eval_soft(student, correct, neg_txts):
    # original stable/clamped behavior
    pos = cosine(student, correct)
    neg_sims = [cosine(student, n) for n in neg_txts]
    max_neg = max(neg_sims)
    max_neg = min(max_neg, pos - 0.05)   # keep soft clamp
    return pos > max_neg

def eval_medium(student, correct, neg_txts):
    # no clamp, but allow a tiny margin buffer
    pos = cosine(student, correct)
    neg_sims = [cosine(student, n) for n in neg_txts]
    max_neg = max(neg_sims)
    return pos > max_neg - 0.01          # small buffer

def eval_hard(student, correct, neg_txts):
    # strict CLIP-style ranking
    pos = cosine(student, correct)
    neg_sims = [cosine(student, n) for n in neg_txts]
    return pos > max(neg_sims)

############################################################
#   TRAINING MODE SELECTOR PER PHASE
############################################################

def get_training_eval_fn(phase):
    if phase <= 2:
        return eval_soft          # nursery mode
    elif phase <= 5:
        return eval_medium        # intermediate mode
    else:
        return eval_hard          # strict mode

def evaluate_phase_10(student, idx):
    """Phase 10: Full 101-way ranking"""
    sims = (student @ all_txt_tensor.t())  # [101]
    pred_idx = sims.argmax().item()
    return pred_idx == idx

# ============================================================
# TRAIN LOOP — 8 GENOMES
# ============================================================

def train():
    start_step = 0
    genomes = None

    print("Select a checkpoint to resume (cancel the dialog to start fresh).")
    selected_ckpt = prompt_checkpoint_file()
    if selected_ckpt:
        try:
            genomes, saved_phase, saved_step, _, _, _ = load_checkpoint(selected_ckpt, DEVICE)
            start_step = saved_step
            print(f"Resumed {len(genomes)} genomes from checkpoint '{selected_ckpt}' (saved step {saved_step}). Restarting curriculum from Phase 1.")
        except Exception as exc:
            print(f"Failed to load checkpoint '{selected_ckpt}': {exc}")
            genomes = None

    if genomes is None:
        genomes = [Genome() for _ in range(POP_SIZE)]  # Predictor handles device internally
        print("Starting fresh training run with new genomes.")

    logger = CSVLogger()
    dashboard = TrainingDashboard()
    trust_chart = TrustChart()

    # Phase tracking - 12 phases, each requiring 20 cumulative trust before the final 101-way
    phase = 1
    phase_names = {
        1: "1-neg",
        2: "2-neg",
        3: "3-neg",
        4: "5-neg",
        5: "8-neg",
        6: "12-neg",
        7: "18-neg",
        8: "25-neg",
        9: "40-neg",
        10: "60-neg",
        11: "80-neg",
        12: "101-way",
    }
    phase_thresholds = {
        1: 20.0,
        2: 40.0,
        3: 60.0,
        4: 120.0,
        5: 140.0,
        6: 190.0,
        7: 260.0,
        8: 300.0,
        9: 350.0,
        10: 425.0,
        11: 500.0,
    }
    phase_neg_counts = {
        1: 1,
        2: 2,
        3: 3,
        4: 5,
        5: 8,
        6: 12,
        7: 18,
        8: 25,
        9: 40,
        10: 60,
        11: 80,
    }

    # Track previous metrics
    prev_median_trust = None
    prev_best_trust = None
    last_acc = 0.0

    step = start_step
    try:
        while True:
            step += 1

            img, idx = random.choice(dataset)
            img = img.unsqueeze(0).to(DEVICE)

            # Preprocess image ONCE on CPU, then encode with CLIP ONCE on GPU
            # This eliminates repeated PIL conversions and CLIP encoding in the genome loop
            with torch.no_grad():
                # Do PIL conversion on CPU first (CPU-intensive, but only once)
                pil_img = T.ToPILImage()(img.squeeze().cpu())
                # Preprocess and move to GPU
                preprocessed_img = preprocess(pil_img).unsqueeze(0).to(DEVICE)
                # Encode with CLIP once (GPU-intensive, but only once)
                t_img = clip_model.encode_image(preprocessed_img).float()
                t_img = t_img / t_img.norm(dim=-1, keepdim=True)
                # Detach to break computation graph and free memory
                t_img = t_img.detach()
                # Clear preprocessed image from GPU memory
                del preprocessed_img, pil_img

            # Get best genome to check phase transition
            # Cache sort result to avoid multiple sorts per step
            genomes.sort(key=lambda g: g.trust, reverse=True)
            best = genomes[0]

            # Check phase transition based on trust thresholds
            # When in phase N, check if trust >= threshold[N] to advance to phase N+1
            current_threshold = phase_thresholds.get(phase, 999999)
            if phase < 12 and best.trust >= current_threshold:
                old_phase = phase
                phase += 1
                print(f"\n>>> Phase transition: Phase {old_phase} -> {phase} ({phase_names[phase]}) <<<")
                print(f">>> Trust: {best.trust:.3f} >= Threshold[{old_phase}]: {current_threshold:.1f}\n")

            # ================================================================
            # NEW: Unified Protein-Based Trust Update (All Phases)
            # ================================================================
            # Instead of phase-specific binary trust updates, use protein network
            # to compute gradient-rich trust signals based on:
            # - prediction_distance: How close to correct text embedding
            # - category_match: Semantic category similarity
            # - token_hit: Binary hit if within threshold
            # ================================================================

            correct_txt = text_latents[labels[idx]]

            # Pre-compute all text embeddings tensor for efficient prediction
            all_txt_tensor = torch.stack([text_latents[label].squeeze(0) for label in labels])  # [101, 512]

            for g in genomes:
                with torch.no_grad():
                    student = g.forward(t_img).squeeze(0)  # [512]

                # NEW: Compute protein signals
                predicted_idx = get_predicted_class(student, all_txt_tensor)

                signals = compute_protein_signals(
                    student_embedding=student,
                    correct_text_embedding=correct_txt.squeeze(0),
                    predicted_class_idx=predicted_idx,
                    true_class_idx=idx,
                    labels=labels
                )

                # NEW: Run protein cascade
                trust_delta = g.run_proteins(signals)

                # DEBUG: Log ACTUAL trust_delta during training (not logging call)
                if step % 100 == 0 and g.genome_id == 0:
                    print(f"\n[TRAINING LOOP DEBUG] Step {step}, Genome #{g.genome_id}")
                    print(f"  ACTUAL trust_delta from cascade: {trust_delta}")
                    print(f"  Signals: {signals}")
                    print(f"  Protein trust_outputs:")
                    from GENREG.proteins import TrustModifierProtein
                    for p in g.proteins:
                        if isinstance(p, TrustModifierProtein):
                            print(f"    {p.name}: trust_output={p.trust_output}, output={p.output}, inputs={p.inputs}")

                # DEBUG: Log first genome's protein details on first few steps
                if step <= 5 and g.genome_id == 0:
                    print(f"\n[DEBUG] Step {step}, Genome #{g.genome_id}")
                    print(f"  Signals: {signals}")
                    print(f"  Trust delta: {trust_delta}")
                    print(f"  Protein outputs:")
                    for p in g.proteins:
                        print(f"    {p.name}: {p.output}")

                # NEW: Apply protein-based trust (scaled to match old system magnitude)
                g.trust += trust_delta * PROTEIN_TRUST_SCALE

                # Clear student tensor to free memory
                del student

                # Update low trust step counter for dynamic mutation rate
                if g.trust < 0.02:
                    g.low_trust_steps += 1
                else:
                    g.low_trust_steps = 0  # Reset if trust rises above threshold

                # Increment genome age
                g.age += 1

            # Clean up
            del all_txt_tensor

            # ================================================================
            # LOGGING: Genome stats and protein signals
            # ================================================================
            # Sort genomes by trust for reporting
            sorted_genomes = sorted(genomes, key=lambda g: g.trust, reverse=True)
            best_genome = sorted_genomes[0]
            worst_genome = sorted_genomes[-1]
            median_genome = sorted_genomes[len(genomes)//2]

            # Get last protein signals from best genome (for visibility)
            # Re-compute signals for best genome to show what it's "seeing"
            with torch.no_grad():
                best_student = best_genome.forward(t_img).squeeze(0)
                all_txt_tensor_log = torch.stack([text_latents[label].squeeze(0) for label in labels])
                best_pred_idx = get_predicted_class(best_student, all_txt_tensor_log)
                best_signals = compute_protein_signals(
                    best_student,
                    text_latents[labels[idx]].squeeze(0),
                    best_pred_idx,
                    idx,
                    labels
                )
                # Also compute trust delta to show what proteins are producing
                best_trust_delta = best_genome.run_proteins(best_signals)
                best_trust_scaled = best_trust_delta * PROTEIN_TRUST_SCALE
                del best_student, all_txt_tensor_log

            # Only log every 100 steps to reduce spam
            if step % 100 == 0:
                print(f"\n[Step {step}] Phase {phase} ({phase_names[phase]})")
                print(f"  Trust: Best={best_genome.trust:.4f} | Median={median_genome.trust:.4f} | Worst={worst_genome.trust:.4f}")
                print(f"  Best Genome #{best_genome.genome_id} Signals: dist={best_signals['prediction_distance']:.3f} | cat_match={best_signals['category_match']:.1f} | hit={best_signals['token_hit']:.0f}")
                print(f"  Best predicted: '{labels[best_pred_idx]}' | True: '{labels[idx]}'")
                print(f"  Protein trust_delta: {best_trust_delta:.2f} (scaled: {best_trust_scaled:.6f})")

                # DEBUG: Show all protein outputs to diagnose trust=0 issue
                print(f"  [DEBUG] Protein outputs:")
                for p in best_genome.proteins:
                    print(f"    {p.name}: output={p.output:.4f}", end="")
                    if hasattr(p, 'trust_output'):
                        print(f" | trust_output={p.trust_output:.4f}", end="")
                    if hasattr(p, 'inputs') and p.inputs:
                        print(f" | inputs={p.inputs}", end="")
                    print()

            # Calculate diversity for mutation rate adjustment
            # Use predictor weights for diversity calculation
            weights = [
                torch.cat([
                    g.predictor.position_weights.flatten(),
                    g.predictor.dim_weights.flatten()
                ]).detach().cpu()
                for g in genomes
            ]
            dists = []
            for i in range(len(weights)-1):
                dist = (weights[i] - weights[i+1]).abs().mean().item()
                dists.append(dist)
            avg_div = sum(dists) / len(dists) if dists else 0.0
            # Clear weights list to free memory
            del weights, dists

            # Log diversity (only every 100 steps)
            if step % 100 == 0:
                print(f"  Diversity: {avg_div:.6f}")

            # Maintain minimum diversity floor: force-mutate 1 genome if diversity too low
            if avg_div < 0.002:
                # Force-mutate a random genome (excluding the best) to inject diversity
                # Use already-sorted genomes from phase check above
                # Pick from genomes ranked 2-8 (avoid mutating the best)
                target_idx = random.randint(1, min(len(genomes) - 1, 7))
                force_target = genomes[target_idx]
                force_target.mutate(current_phase=phase, avg_div=avg_div)
                print(f"  [DIVERSITY INJECTION] Genome #{force_target.genome_id} force-mutated (diversity={avg_div:.6f} < 0.002)")
            
            # select best + reproduce
            # genomes already sorted from phase check above, so best is already genomes[0]
            best = genomes[0]

            # Update dashboard with current metrics
            trust_trend = "flat"
            if prev_best_trust is not None:
                if best.trust > prev_best_trust + 1e-4:
                    trust_trend = "up"
                elif best.trust < prev_best_trust - 1e-4:
                    trust_trend = "down"
            dashboard.update(
                diversity=avg_div,
                trust=best.trust,
                trend=trust_trend,
                accuracy=last_acc,
                phase_name=phase_names[phase],
                phase_number=phase,
                step=step,
                mutation_rate=best.last_mutation_rate
            )
            prev_best_trust = best.trust
            
            # Update trust chart periodically
            if step - trust_chart.last_update_step >= trust_chart.update_interval:
                trust_chart.add_data_point(step, best.trust)
                trust_chart.last_update_step = step
            
            # Gentle culling every 2500 steps (only from Phase 2 onwards)
            if step % 2500 == 0 and phase >= 2:
                # Calculate median trust
                trust_values = sorted([g.trust for g in genomes])
                median_trust = trust_values[len(trust_values) // 2]
                
                # RULE 4: No culling when trust is rising
                if prev_median_trust is not None and median_trust > prev_median_trust:
                    # Skip culling, population is improving
                    prev_median_trust = median_trust
                else:
                    # RULE 1: Cull only if trust < median(trust)
                    candidates = [g for g in genomes if g.trust < median_trust]
                    
                    if candidates:
                        if phase >= 5:
                            # Phase 5+: Aggressive culling with double mutation rate
                            # Sort candidates by trust (lowest first)
                            candidates.sort(key=lambda g: g.trust)
                            
                            # Replace lowest genome with completely NEW initialized genome
                            lowest = candidates[0]
                            new_genome = Genome()  # NEW: No .to(DEVICE) needed, predictor handles device
                            # Replace the lowest genome with the new one
                            lowest_idx = genomes.index(lowest)
                            genomes[lowest_idx] = new_genome
                            print(f"[{step}] PHASE {phase} CULLING: Replaced lowest Genome {lowest.genome_id} (trust={lowest.trust:.3f} < median={median_trust:.3f}) with NEW initialized genome {new_genome.genome_id}")
                            
                            # Replace rest of candidates (other genomes below median) with clones of best + double mutation
                            best = genomes[0]  # Best genome
                            for target in candidates[1:]:  # Skip the lowest one we already replaced
                                target_idx = genomes.index(target)
                                old_id = target.genome_id
                                # Clone best genome
                                genomes[target_idx] = best.clone()
                                genomes[target_idx].genome_id = old_id  # Keep original ID
                                # Double mutation rate for culling in phase 5+
                                genomes[target_idx].mutate(current_phase=phase, avg_div=avg_div, mutation_rate_multiplier=2.0)
                                genomes[target_idx].trust = best.trust * 0.9
                                print(f"[{step}] PHASE {phase} CULLING: Replaced Genome {old_id} (trust < median={median_trust:.3f}) with clone of best Genome {best.genome_id} (2x mutation rate)")
                            
                            print(f"[{step}] PHASE {phase} CULLING: Culled {len(candidates)} genomes (1 new + {len(candidates)-1} clones with 2x mutation)")
                            # Re-sort genomes after culling to ensure best is at index 0
                            genomes.sort(key=lambda g: g.trust, reverse=True)
                        else:
                            # Phase 2-4: Original gentle culling (only replace ONE genome)
                            target = random.choice(candidates)
                            target_idx = genomes.index(target)
                            old_id = target.genome_id

                            # RULE 3: Copy top genome 80% of the time, top-2 genome 20% of the time
                            top1, top2 = genomes[0], genomes[1]
                            parent = top1 if random.random() < 0.8 else top2

                            # Clone parent
                            genomes[target_idx] = parent.clone()
                            genomes[target_idx].genome_id = old_id
                            genomes[target_idx].mutate(current_phase=phase, avg_div=avg_div)
                            genomes[target_idx].trust = parent.trust * 0.9

                            print(f"[{step}] GENTLE CULLING: Replaced Genome {old_id} (trust < median={median_trust:.3f}) with clone of Genome {parent.genome_id}")
                
                # Update median trust tracking
                prev_median_trust = median_trust
            else:
                # Normal reproduction: replace worst with best mutation
                worst_idx = len(genomes) - 1
                old_id = genomes[worst_idx].genome_id
                genomes[worst_idx] = best.clone()
                genomes[worst_idx].genome_id = old_id
                genomes[worst_idx].mutate(current_phase=phase, avg_div=avg_div)
                genomes[worst_idx].trust = best.trust * 0.9

            # Check for manual evaluation request
            if dashboard.manual_eval_requested:
                # Clear CUDA cache before evaluation to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                acc = zero_shot_eval(best)
                last_acc = acc
                phase_name = phase_names[phase]
                print(f"[{step}] [MANUAL] Genome {best.genome_id} | Phase {phase} ({phase_name}) | trust={best.trust:.3f} acc={acc:.4f} diversity={avg_div:.6f}")
                logger.log(step, best.genome_id, phase_name, best.trust, acc)
                dashboard.manual_eval_requested = False
                dashboard.eval_button.configure(text="Run Eval Now", bg="#1890ff")
                
                # Stop training if accuracy reaches 1.0 (100%)
                if acc >= 1.0:
                    print(f"\n>>> TRAINING COMPLETE: Accuracy reached 1.0 at step {step} <<<")
                    torch.save(best.predictor.state_dict(), f"final_model_acc_1.0_step_{step}.pth")
                    print(f"Model saved to: final_model_acc_1.0_step_{step}.pth")
                    break
            
            # Evaluate every 20,000 steps for all phases
            if step % 20000 == 0:
                # Clear CUDA cache before evaluation to free memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                acc = zero_shot_eval(best)
                last_acc = acc
                phase_name = phase_names[phase]
                print(f"[{step}] Genome {best.genome_id} | Phase {phase} ({phase_name}) | trust={best.trust:.3f} acc={acc:.4f} diversity={avg_div:.6f}")
                logger.log(step, best.genome_id, phase_name, best.trust, acc)
                
                # Stop training if accuracy reaches 1.0 (100%)
                if acc >= 1.0:
                    print(f"\n>>> TRAINING COMPLETE: Accuracy reached 1.0 at step {step} <<<")
                    torch.save(best.predictor.state_dict(), f"final_model_acc_1.0_step_{step}.pth")
                    print(f"Model saved to: final_model_acc_1.0_step_{step}.pth")
                    break

            if step % 2000 == 0:
                # Clear CUDA cache periodically to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                phase_dir = os.path.join(CHECKPOINT_SHORT_DIR, f"phase{phase}")
                os.makedirs(phase_dir, exist_ok=True)
                short_path = os.path.join(phase_dir, f"checkpoint_step_{step}.pth")
                torch.save(best.predictor.state_dict(), short_path)
                print(f"[{step}] Short checkpoint saved to {short_path}")
            
            # Save full checkpoint every 15,000 steps
            if step % 15000 == 0:
                # Clear CUDA cache before saving checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                save_checkpoint(
                    genomes=genomes,
                    phase=phase,
                    step=step,
                    avg_div=avg_div,
                    labels=labels,
                    text_latents=text_latents,
                    path=os.path.join(CHECKPOINT_DIR, f"checkpoint_step_{step}.pth")
                )
                print(f"[{step}] Full checkpoint saved to {CHECKPOINT_DIR}/checkpoint_step_{step}.pth")
    finally:
        logger.close()
        dashboard.close()
        trust_chart.close()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
