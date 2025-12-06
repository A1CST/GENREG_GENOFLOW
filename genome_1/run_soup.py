# run_soup.py
"""
Main Pygame simulation loop for "Primordial Soup" Phase 0.
Integration of SoupEnv, ProtoMolecule, and OLAFoundry.

Phase 0 goal: proto-molecules learn to replicate.
Mutations only enabled at foundry level, not on replication events.
"""
import argparse
import time
import os
import csv
from collections import deque

import torch
import numpy as np
import pygame

from soup_env import SoupEnv
from protomolecule import ProtoMolecule
from ola_foundry import OLAFoundry
from stabilized_ola import StabilizedOLAConfig


def main():
    parser = argparse.ArgumentParser(description="Primordial Soup - Phase 0")
    parser.add_argument("--width", type=int, default=960, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--tile", type=int, default=8, help="Tile size in pixels")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--primordial_burst", type=int, default=200, help="Initial agent spawn count")
    parser.add_argument("--spawn_rate", type=int, default=5, help="Emit new genome every K spawns during burst")
    parser.add_argument("--phase", type=int, default=1, help="Simulation phase (0=replication-only, 1=evolution)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Soup] Using device: {device}")

    # Create environment
    env = SoupEnv(args.width, args.height, args.tile, seed=args.seed)
    env.init_terrain(rock_density=0.15)

    # Create OLA foundry
    obs_dim = 16  # ProtoMolecule observation vector size
    foundry_cfg = StabilizedOLAConfig(
        in_dim=obs_dim,
        out_dim=obs_dim,
        state_dim=128,
        mutation_rate=0.1,
        grow_prob=0.0,
        max_state_dim=256,
        device=str(device)
    )
    foundry = OLAFoundry(foundry_cfg)

    # Phase 1: Initialize founder pool with diversity
    if args.phase == 1:
        num_founders = 8
        foundry.init_founder_pool(num_founders, phase=args.phase)
        print(f"[Soup] Phase 1: Initialized {num_founders} diverse founders")
    else:
        foundry.init_founder_pool(1, phase=args.phase)

    # Agent list
    agents = []

    # Phase 1: Primordial burst with diverse founders and Poisson disk sampling
    if args.phase == 1 and len(foundry.founder_pool) > 0:
        num_founders = len(foundry.founder_pool)
        agents_per_founder = args.primordial_burst // num_founders

        # Get spatially diverse spawn positions
        spawn_positions = env.sample_poisson_disk(num_founders, min_dist=8.0)
        if len(spawn_positions) < num_founders:
            print(f"[Soup] Warning: Could only find {len(spawn_positions)} spawn positions, expected {num_founders}")
            # Fill remaining with random positions
            while len(spawn_positions) < num_founders:
                pos = env.get_random_water_tile()
                if pos:
                    spawn_positions.append(pos)

        print(f"[Soup] Phase 1 burst: spawning {args.primordial_burst} agents from {num_founders} founders...")

        for founder_idx, founder_genome in enumerate(foundry.founder_pool):
            lineage_id = foundry.next_lineage_id
            foundry.next_lineage_id += 1

            for _ in range(agents_per_founder):
                # Clone founder genome
                genome = foundry._clone_genome(founder_genome)

                # Random position near founder spawn point
                base_pos = spawn_positions[founder_idx % len(spawn_positions)]
                pos = env.get_random_water_tile()  # Start simple, can improve later
                if pos is None:
                    break

                # Create agent with lineage tracking
                agent = ProtoMolecule(
                    genome=genome,
                    pos=pos,
                    energy=0.5,
                    device=device,
                    lineage_depth=0,
                    parent_pos=None,
                    birth_tick=0,
                    lineage_id=lineage_id
                )
                agent.phase = args.phase  # Set phase for mutation logic
                agents.append(agent)

    else:
        # Phase 0: Original random spawn
        print(f"[Soup] Phase 0 burst: spawning {args.primordial_burst} agents...")
        for i in range(args.primordial_burst):
            # Emit genome from foundry
            genome = foundry.emit_genome(every_k=args.spawn_rate)

            # Random water tile
            pos = env.get_random_water_tile()
            if pos is None:
                print("[Soup] Warning: no water tiles available for spawn")
                break

            # Create agent
            agent = ProtoMolecule(
                genome=genome,
                pos=pos,
                energy=0.5,
                device=device,
                lineage_depth=0,
                parent_pos=None,
                birth_tick=0,
                lineage_id=0
            )
            agent.phase = args.phase
            agents.append(agent)

    print(f"[Soup] Spawned {len(agents)} agents")

    # Metrics tracking
    tick = 0
    replication_events = deque(maxlen=600)  # Last 600 ticks (~10 sec at 60fps)
    paused = False
    running = True

    # CSV logging
    os.makedirs("runs", exist_ok=True)
    csv_path = "runs/soup_metrics.csv"
    csv_file = open(csv_path, "w", newline="")

    # Phase 1: Extended CSV columns for diversity and RoR
    if args.phase == 1:
        csv_fieldnames = [
            "tick", "live_agents", "repl_events_window", "avg_R", "total_R", "mass_error",
            "OLA_mut_rate", "OLA_ema_loss", "simpson_diversity", "num_lineages", "RoR_ema"
        ]
    else:
        csv_fieldnames = [
            "tick", "live_agents", "repl_events_window", "avg_R", "total_R", "mass_error",
            "OLA_mut_rate", "OLA_ema_loss"
        ]

    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()

    # Contraction schedule (DISABLED in Phase 0 for mass conservation)
    contraction_interval = 999999  # Effectively disabled
    last_contraction_tick = 0

    # Transition criterion tracking
    repl_windows = deque(maxlen=3)
    target_repl_rate = 1.0  # replications per minute threshold
    banner_printed = False

    # Phase 1: Immigrant injection config
    immigrant_interval = 600  # Inject every 600 ticks (~10 seconds at 60fps)
    immigrants_per_drop = 2
    last_immigrant_tick = 0

    # Phase 1: Lineage stats per tick
    lineage_births_this_tick = {}  # lineage_id -> birth count this tick
    lineage_R_spent_this_tick = {}  # lineage_id -> R spent this tick

    print("[Soup] Starting simulation loop. Press Space to pause, R for heatmap, C for contraction, G to spawn 10, S to save.")

    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[Soup] {'Paused' if paused else 'Resumed'}")
                elif event.key == pygame.K_r:
                    env.show_resource_heatmap = not env.show_resource_heatmap
                    print(f"[Soup] Resource heatmap: {'ON' if env.show_resource_heatmap else 'OFF'}")
                elif event.key == pygame.K_c:
                    mass_removed = env.contract_resources(0.98)
                    print(f"[Soup] Manual contraction: R_max={env.r_max:.4f}, mass_removed={mass_removed:.2f}")
                elif event.key == pygame.K_g:
                    # Spawn 10 new genomes with LOW initial energy (0.1 instead of 0.5)
                    # to minimize spawn resource cost
                    for _ in range(10):
                        genome = foundry.emit_genome(every_k=1)  # Force mutation
                        pos = env.get_random_water_tile()
                        if pos:
                            agent = ProtoMolecule(genome, pos, 0.1, device, 0, None, tick)
                            agents.append(agent)
                    print(f"[Soup] Spawned 10 new genomes (low energy). Total: {len(agents)}")
                elif event.key == pygame.K_s:
                    # Save champion genome
                    os.makedirs("genomes", exist_ok=True)
                    save_path = f"genomes/champion_{tick}.pt"
                    foundry.save_best_genome(save_path)
                    print(f"[Soup] Saved genome to {save_path}")

        if not paused:
            tick += 1

            # Mass conservation: track total resource before this tick
            R_before = env.get_total_resource()

            # Reset per-tick counters
            repl_resource_spent = 0.0
            decomp_resource_returned = 0.0
            spawn_resource_cost = 0.0

            # Phase 1: Reset lineage stats for this tick
            if args.phase == 1:
                lineage_births_this_tick.clear()
                lineage_R_spent_this_tick.clear()

            # Agent tick: observe, decide, act
            new_agents = []
            repl_count_this_tick = 0

            for agent in agents:
                # Observe and step genome
                rep_score, is_dead = agent.observe_and_step(env, agents)

                if is_dead:
                    returned = agent.decompose(env)
                    decomp_resource_returned += returned
                    continue

                # Attempt replication
                if rep_score >= agent.s_threshold:
                    child = agent.attempt_replication(env, agents, tick)
                    if child is not None:
                        new_agents.append(child)
                        repl_count_this_tick += 1
                        repl_resource_spent += agent.R_replicate_cost

                        # Phase 1: Track lineage stats
                        if args.phase == 1:
                            lineage_id = agent.lineage_id
                            if lineage_id not in lineage_births_this_tick:
                                lineage_births_this_tick[lineage_id] = 0
                                lineage_R_spent_this_tick[lineage_id] = 0.0
                            lineage_births_this_tick[lineage_id] += 1
                            lineage_R_spent_this_tick[lineage_id] += agent.R_replicate_cost
                else:
                    # Try to move (energy-only cost, does NOT touch resources)
                    agent.attempt_move(env, agents)

            # Merge new agents
            agents.extend(new_agents)

            # Remove dead agents
            agents = [a for a in agents if a.energy > 0.0]

            # Track replication events
            replication_events.append(repl_count_this_tick)

            # Resource diffusion (mass-conserving)
            env.diffuse_resources()

            # Phase 0 throttles: tick birth cooldowns
            env.tick_cooldowns()

            # Periodic resource contraction (DISABLED in Phase 0)
            contraction_mass_removed = 0.0
            if tick - last_contraction_tick >= contraction_interval:
                contraction_mass_removed = env.contract_resources(0.98)
                last_contraction_tick = tick

            # Mass conservation check
            R_after = env.get_total_resource()
            expected_delta = -repl_resource_spent + decomp_resource_returned - spawn_resource_cost - contraction_mass_removed
            actual_delta = R_after - R_before
            mass_error = abs(actual_delta - expected_delta)

            # Update cumulative counters
            env.total_resource_spent += repl_resource_spent
            env.total_resource_returned += decomp_resource_returned
            env.total_spawn_cost += spawn_resource_cost

            # Warn if mass conservation is violated
            if mass_error > 1e-3:
                print(f"[Soup] WARNING tick {tick}: Mass conservation violated! Error={mass_error:.6f} "
                      f"(expected Δ={expected_delta:.6f}, actual Δ={actual_delta:.6f})")

            # Phase 1: Immigrant injection every 600 ticks
            if args.phase == 1 and tick - last_immigrant_tick >= immigrant_interval:
                for _ in range(immigrants_per_drop):
                    immigrant_genome = foundry.emit_immigrant()
                    pos = env.get_random_water_tile()
                    if pos:
                        immigrant_lineage_id = foundry.next_lineage_id
                        foundry.next_lineage_id += 1
                        immigrant = ProtoMolecule(
                            genome=immigrant_genome,
                            pos=pos,
                            energy=0.5,
                            device=device,
                            lineage_depth=0,
                            parent_pos=None,
                            birth_tick=tick,
                            lineage_id=immigrant_lineage_id
                        )
                        immigrant.phase = args.phase
                        agents.append(immigrant)
                last_immigrant_tick = tick
                print(f"[Soup] Tick {tick}: Injected {immigrants_per_drop} immigrants")

            # Phase 1: Update foundry lineage stats
            if args.phase == 1:
                for lineage_id in lineage_births_this_tick:
                    foundry.update_lineage_stats(
                        lineage_id,
                        lineage_births_this_tick[lineage_id],
                        lineage_R_spent_this_tick[lineage_id]
                    )

            # Phase 1: Compute diversity metrics
            simpson_diversity = 0.0
            num_lineages = 0
            if args.phase == 1:
                lineage_counts = {}
                for agent in agents:
                    lid = agent.lineage_id
                    lineage_counts[lid] = lineage_counts.get(lid, 0) + 1
                simpson_diversity = foundry.compute_simpson_diversity(lineage_counts)
                num_lineages = len(lineage_counts)

                # Anneal mutation rate based on diversity
                foundry.anneal_mutation_rate(simpson_diversity, min_rate=0.01, max_rate=0.2)

            # OLA foundry tick
            avg_r = float(np.mean(env.resources))
            foundry_metrics = foundry.tick(avg_r, len(agents))

            # Compute replication rate (events per minute)
            repl_rate = sum(replication_events) / max(1, len(replication_events)) * 60.0

            # Transition criterion: check if replication is stable
            repl_windows.append(repl_rate)
            if len(repl_windows) == 3 and all(r > target_repl_rate for r in repl_windows) and not banner_printed:
                print("\n" + "=" * 60)
                print("REPLICATION ACHIEVED!")
                print("Ready to enable mutation/selection in Phase 1.")
                print("=" * 60 + "\n")
                banner_printed = True

            # CSV logging
            if args.phase == 1:
                csv_writer.writerow({
                    "tick": tick,
                    "live_agents": len(agents),
                    "repl_events_window": sum(replication_events),
                    "avg_R": avg_r,
                    "total_R": R_after,
                    "mass_error": mass_error,
                    "OLA_mut_rate": foundry_metrics["mutation_rate"],
                    "OLA_ema_loss": foundry_metrics["ema_loss"],
                    "simpson_diversity": simpson_diversity,
                    "num_lineages": num_lineages,
                    "RoR_ema": foundry.RoR_ema
                })
            else:
                csv_writer.writerow({
                    "tick": tick,
                    "live_agents": len(agents),
                    "repl_events_window": sum(replication_events),
                    "avg_R": avg_r,
                    "total_R": R_after,
                    "mass_error": mass_error,
                    "OLA_mut_rate": foundry_metrics["mutation_rate"],
                    "OLA_ema_loss": foundry_metrics["ema_loss"]
                })

            # Render
            env.render(agents, tick, repl_rate, avg_r,
                      foundry_metrics["mutation_rate"],
                      foundry_metrics["ema_loss"],
                      R_after, mass_error,
                      simpson_diversity, num_lineages, foundry.RoR_ema)

        else:
            # Paused: still render but don't update
            avg_r = float(np.mean(env.resources))
            total_r = env.get_total_resource()
            foundry_metrics = foundry.get_metrics()
            repl_rate = sum(replication_events) / max(1, len(replication_events)) * 60.0

            # Phase 1: Compute diversity metrics for paused rendering
            paused_simpson_diversity = 0.0
            paused_num_lineages = 0
            if args.phase == 1:
                lineage_counts = {}
                for agent in agents:
                    lid = agent.lineage_id
                    lineage_counts[lid] = lineage_counts.get(lid, 0) + 1
                paused_simpson_diversity = foundry.compute_simpson_diversity(lineage_counts)
                paused_num_lineages = len(lineage_counts)

            env.render(agents, tick, repl_rate, avg_r,
                      foundry_metrics["mutation_rate"],
                      foundry_metrics["ema_loss"],
                      total_r, 0.0,
                      paused_simpson_diversity, paused_num_lineages, foundry.RoR_ema)

        env.tick_frame(args.fps)

    # Cleanup
    csv_file.close()
    pygame.quit()
    print(f"[Soup] Simulation ended at tick {tick}. Metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
