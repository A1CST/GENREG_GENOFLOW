from tabnanny import check
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import clip

# ----------------------------------------------------------
# 1. Load Caltech101 dataset
# ----------------------------------------------------------
DATASET_PATH = "caltech-101/101_ObjectCategories/101_ObjectCategories"

def load_caltech101():
    images = []
    labels = []
    classes = sorted(os.listdir(DATASET_PATH))

    cls_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(DATASET_PATH, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(cls_path, fname))
                labels.append(cls_to_idx[cls])
    return images, labels, classes

# ----------------------------------------------------------
# 2. Load CLIP model for image/text encoding
# ----------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# Transform for dataset loading (simple resize)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# ----------------------------------------------------------
# 3. O-CLIP Student Encoder: Linear projection only
# ----------------------------------------------------------
class OCLIP_Encoder(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        out_dim, in_dim = weight.shape
        self.proj = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            self.proj.weight.copy_(weight)
            if bias is not None:
                self.proj.bias.copy_(bias)
            else:
                self.proj.bias.zero_()

    def forward(self, x):
        # x is already a CLIP image embedding [batch, 512]
        return torch.nn.functional.normalize(self.proj(x), dim=-1)

# ----------------------------------------------------------
# 4. Zero-shot evaluation
# ----------------------------------------------------------
def zeroshot_eval():
    checkpoint_file ="checkpoint_step_1395000.pth"
    print("Loading checkpoint...")

    ckpt = torch.load(f"checkpoints_full/{checkpoint_file}", map_location=DEVICE, weights_only=False)
    print(f"{checkpoint_file} loaded")
    
    # Handle full checkpoint format (with genomes list) or short checkpoint format (direct state_dict)
    if "genomes" in ckpt:
        # Full checkpoint: extract best genome (highest trust)
        genomes = ckpt["genomes"]
        best_genome = max(genomes, key=lambda g: g.get("trust", 0.0))
        state_dict = best_genome["state_dict"]
        print(f"Loaded best genome (trust={best_genome.get('trust', 'N/A'):.3f}) from full checkpoint")
    else:
        # Short checkpoint: direct state_dict
        state_dict = ckpt
    
    W = state_dict["proj.weight"]
    b = state_dict.get("proj.bias", None)

    model = OCLIP_Encoder(W, b).to(DEVICE).eval()

    print("Loading dataset...")
    images, labels, class_names = load_caltech101()

    print(f"Loaded {len(images)} images across {len(class_names)} classes.")

    # Pre-compute CLIP text embeddings for all classes
    print("Pre-computing CLIP text embeddings...")
    with torch.no_grad():
        text_tokens = clip.tokenize(class_names).to(DEVICE)
        text_embeddings = clip_model.encode_text(text_tokens).float()
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        # Stack for efficient computation: [num_classes, 512]
        text_tensor = text_embeddings

    correct = 0
    total = len(images)

    with torch.no_grad():
        for img_path, label in tqdm(zip(images, labels), total=total):

            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = clip_preprocess(img).unsqueeze(0).to(DEVICE)
            
            # Encode image with CLIP
            clip_img_emb = clip_model.encode_image(img_tensor).float()
            clip_img_emb = clip_img_emb / clip_img_emb.norm(dim=-1, keepdim=True)
            
            # Pass through O-CLIP projection
            student_emb = model(clip_img_emb).squeeze(0)  # [512]

            # Compute similarities with all text embeddings
            sims = (student_emb @ text_tensor.t()).cpu().numpy()  # [num_classes]
            
            pred = np.argmax(sims)

            if pred == label:
                correct += 1

    top1 = correct / total
    print("========================================")
    print(f"ZERO-SHOT TOP-1 ACCURACY: {top1:.4f}")
    print("========================================")


if __name__ == "__main__":
    zeroshot_eval()
