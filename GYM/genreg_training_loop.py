# ================================================================
# GENREG v2.0 — Training Loop (MuJoCo HalfCheetah)
# ================================================================

import argparse
import gymnasium as gym
import numpy as np
import pickle
import os
import glob
from genreg_genome import GENREGPopulation
from genreg_proteins import SensorProtein, TrustModifierProtein

# ================================================================
# PATCH: V2.2 "FLOOR IS LAVA" (Anti-Scoot Penalty)
# ================================================================
def build_cheetah_template():
    proteins = []

    # --- SENSORS ---
    proteins.append(SensorProtein("root_vel_x")) 
    
    # "Pain" is injected by the loop logic when height < 0.4
    proteins.append(SensorProtein("pain")) 

    # --- REGULATION (IMMUTABLE LAWS) ---
    
    # 1. TRUST_VELOCITY (The Engine)
    # Gain 5.0 means 1m/s = 5 Trust.
    t_vel = TrustModifierProtein("trust_velocity")
    t_vel.bind_inputs(["root_vel_x"])
    t_vel.params["gain"] = 5.0    
    t_vel.params["scale"] = 1.0   
    t_vel.params["decay"] = 0.0   
    proteins.append(t_vel)

    # 2. TRUST_PAIN (The Electric Fence)
    # If the loop detects low height, it sends signal["pain"] = 1.0
    # This protein applies a massive -10.0 penalty.
    # Since we froze mutations, the agents CANNOT turn this off.
    t_pain = TrustModifierProtein("trust_pain")
    t_pain.bind_inputs(["pain"])
    t_pain.params["gain"] = 10.0
    t_pain.params["scale"] = -1.0 
    proteins.append(t_pain)

    return proteins

# ================================================================
# CHECKPOINTING
# ================================================================
CHECKPOINT_DIR = "checkpoints"

def save_checkpoint(population, generation, obs_dim, act_dim, input_size, hidden_size, output_size):
    """Save checkpoint after a generation completes. Keeps only the 5 most recent checkpoints."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_gen_{generation:04d}.pkl")
    checkpoint_data = {
        "generation": generation,
        "population": population,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"[CHECKPOINT] Saved generation {generation} to {checkpoint_path}")
    
    # Keep only the 5 most recent checkpoints
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, "checkpoint_gen_*.pkl")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if len(checkpoint_files) > 5:
        # Sort by generation number (extract from filename)
        def get_gen_num(path):
            filename = os.path.basename(path)
            gen_str = filename.replace("checkpoint_gen_", "").replace(".pkl", "")
            return int(gen_str)
        
        checkpoint_files.sort(key=get_gen_num, reverse=True)
        
        # Delete all but the 5 most recent
        for old_checkpoint in checkpoint_files[5:]:
            os.remove(old_checkpoint)
            print(f"[CHECKPOINT] Deleted old checkpoint: {os.path.basename(old_checkpoint)}")

def load_latest_checkpoint():
    """Load the latest checkpoint if it exists."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Find all checkpoint files
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, "checkpoint_gen_*.pkl")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by generation number (extract from filename)
    def get_gen_num(path):
        filename = os.path.basename(path)
        # Extract number from "checkpoint_gen_XXXX.pkl"
        gen_str = filename.replace("checkpoint_gen_", "").replace(".pkl", "")
        return int(gen_str)
    
    checkpoint_files.sort(key=get_gen_num, reverse=True)
    latest_checkpoint = checkpoint_files[0]
    generation = get_gen_num(latest_checkpoint)
    
    print(f"[CHECKPOINT] Loading latest checkpoint: {latest_checkpoint} (generation {generation})")
    
    with open(latest_checkpoint, "rb") as f:
        checkpoint_data = pickle.load(f)
    
    return checkpoint_data

# ================================================================
# MAIN LOOP
# ================================================================
def train_genreg_cheetah(headless=False):
    # Setup environment with optional rendering
    render_mode = None if headless else "human"
    env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    mode_str = "Headless" if headless else "Visual"
    print(f"[INIT] HalfCheetah ({mode_str}) | Obs: {obs_dim} | Act: {act_dim}")

    # Try to load checkpoint or create new population
    checkpoint = load_latest_checkpoint()
    
    if checkpoint:
        # Resume from checkpoint
        pop = checkpoint["population"]
        start_gen = checkpoint["generation"] + 1
        input_size = checkpoint["input_size"]
        hidden_size = checkpoint["hidden_size"]
        output_size = checkpoint["output_size"]
        print(f"[RESUME] Resuming from generation {start_gen}")
    else:
        # Create new population
        template = build_cheetah_template()
        input_size = obs_dim
        hidden_size = 64
        output_size = act_dim
        pop = GENREGPopulation(
            template_proteins=template,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            size=20
        )
        start_gen = 0
        print(f"[NEW] Starting new training run")

    # --- BENCHMARKING VARS ---
    TOTAL_ENV_STEPS = 0
    BEST_BENCHMARK_SCORE = -float('inf')

    # Run indefinitely (no generation limit)
    gen = start_gen
    while True:
        
        # Track average benchmark score for this generation
        gen_benchmark_scores = []
        
        for i in range(pop.size):
            genome = pop.get_active()
            
            obs, info = env.reset()
            done = False
            truncated = False
            
            # Reset Metrics
            total_dist = 0.0
            total_gym_reward = 0.0 # <--- THE SHADOW METRIC
            episode_steps = 0

            while not (done or truncated) and episode_steps < 1000:
                # 1. Controller Action
                action_vec = genome.controller.forward(obs)
                
                # 2. Step Environment
                obs, reward, done, truncated, info = env.step(action_vec)
                
                # 3. TRACKING
                TOTAL_ENV_STEPS += 1
                episode_steps += 1
                total_gym_reward += reward  # <--- Accumulate the "Official Score"
                
                # 4. GENREG Logic (Your Trust)
                # A. Map Signals
                signals = {
                    "root_vel_x": obs[8],
                }
                
                # --- THE PURIST PATCH ---
                # This is "Reward Shaping" (Legal)
                # If height is low, we inject a "pain" signal that hurts trust.
                if obs[0] < 0.4:
                    signals["pain"] = 1.0  # Active Pain
                else:
                    signals["pain"] = 0.0  # No Pain
                
                # B. Controller Forward is already done above
                # Now forward through GENREG
                genome.forward(signals)
                
                # Log actual distance just for us
                total_dist += signals["root_vel_x"] * 0.05 
            
            # Store metrics
            genome.metrics["distance"] = total_dist
            genome.metrics["gym_score"] = total_gym_reward # Store for logging
            
            gen_benchmark_scores.append(total_gym_reward)
            
            # Check for new high score
            if total_gym_reward > BEST_BENCHMARK_SCORE:
                BEST_BENCHMARK_SCORE = total_gym_reward
            
            pop.next_genome()

        # Stats for the Generation
        avg_score = sum(gen_benchmark_scores) / len(gen_benchmark_scores)
        max_score = max(gen_benchmark_scores)
        
        print(f"\n[GEN {gen}] Total Steps: {TOTAL_ENV_STEPS:,.0f}")
        print(f"  > Gym Score (Benchmark): Avg={avg_score:.1f} | Max={max_score:.1f} | All-Time Best={BEST_BENCHMARK_SCORE:.1f}")
        
        # Show your Trust for context
        best_trust_genome = max(pop.genomes, key=lambda g: g.trust)
        print(f"  > Internal Trust: {best_trust_genome.trust:.1f} | Dist: {best_trust_genome.metrics['distance']:.2f}")

        # Evolve
        pop.evolve()
        
        # Save checkpoint after each generation
        save_checkpoint(pop, gen, obs_dim, act_dim, input_size, hidden_size, output_size)
        
        # Increment generation for next iteration
        gen += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GENREG on HalfCheetah")
    parser.add_argument("--headless", action="store_true", 
                        help="Run without rendering (faster)")
    args = parser.parse_args()
    
    train_genreg_cheetah(headless=args.headless)