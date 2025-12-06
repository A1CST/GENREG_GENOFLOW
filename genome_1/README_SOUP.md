# Primordial Soup - Phase 0 Simulation

A minimal Pygame simulation of a "digital ocean" where proto-molecules learn to replicate using evolved neural genomes.

## Overview

This simulation implements Phase 0 of an artificial life experiment:
- **Goal**: Proto-molecules learn to replicate in a resource-constrained environment
- **Genome**: Each agent has an `EvoCell` recurrent neural network that processes local observations
- **Evolution**: A single `StabilizedOLA` "foundry" generates candidate genomes via mutation
- **Phase 0 constraint**: Mutation only occurs at the foundry level, not during agent replication

## Files

- `soup_env.py` - Grid environment with water/rock tiles, resource tracking, and diffusion
- `protomolecule.py` - Agent with EvoCell genome, replication logic, and movement
- `ola_foundry.py` - Genome generator using StabilizedOLA
- `run_soup.py` - Main simulation loop

## Requirements

- Python 3.10+
- PyTorch (CPU or CUDA)
- Pygame
- NumPy

Install dependencies:
```bash
pip install torch pygame numpy
```

## Usage

Basic run:
```bash
python run_soup.py
```

With options:
```bash
python run_soup.py --width 1280 --height 720 --tile 10 --seed 123 --primordial_burst 300 --fps 60
```

### Command-line Arguments

- `--width` - Window width in pixels (default: 960)
- `--height` - Window height in pixels (default: 720)
- `--tile` - Tile size in pixels (default: 8)
- `--seed` - Random seed for reproducibility (default: 42)
- `--primordial_burst` - Initial number of agents to spawn (default: 200)
- `--spawn_rate` - Emit new genome every K spawns during burst (default: 5)
- `--phase` - Simulation phase, currently only 0 supported (default: 0)
- `--fps` - Target frames per second (default: 60)

### Keybindings

- **Space** - Pause/resume simulation
- **R** - Toggle resource heatmap overlay (shows resource levels as blue intensity)
- **C** - Trigger immediate resource contraction (lowers R_max by 2%)
- **G** - Spawn 10 new genomes from foundry at random water tiles
- **S** - Save current champion genome to `genomes/champion_<tick>.pt`

## Environment

### Tiles
- **Water (blue)** - Traversable tiles where agents can move and replicate
- **Rock (dark gray)** - Non-traversable barriers

### Resources
- Each water tile has a resource level `R[i,j]` in range `[0, R_max]`
- Initial `R_max = 1.0`, starts at `~0.9` to trigger "primordial burst"
- Resources diffuse slowly between adjacent water tiles each tick
- Replication consumes resources from the tile
- Dead agents decompose, returning energy as resource to neighbors

### Resource Dynamics (Mass-Conserving in Phase 0)
- **Diffusion**: Strictly mass-conserving 5-point Laplacian with no-flux boundaries
- **Diffusion rate**: 0.05 (smooth gradient formation)
- **Decomposition**: 100% energy return (no entropy leak in Phase 0)
- **Contraction**: DISABLED in Phase 0 to maintain flat Avg R (can be manually triggered with 'C' key)
- **Replication cost**: 0.12 resource units per replication event
- **Mass conservation**: Every tick verifies `ΔR = -replication_spent + decomposition_returned`

## Agents (Proto-Molecules)

### State
- **Position**: Grid coordinates `(x, y)`
- **Energy**: Range `[0, 1.0]`, starts at 0.5
- **Genome**: EvoCell recurrent network (weights define the genome)
- **Hidden state**: Neural state vector for temporal dynamics
- **Lineage depth**: Tracks generations from original spawn

### Behavior

Each tick, an agent:
1. **Observes** a 3×3 stencil of local resources and population density (16-dim vector)
2. **Runs genome forward** to update hidden state `h`
3. **Computes replication score** `s` using a small linear head on `h`
4. **Attempts replication** if:
   - Tile resource `R >= 0.12` (replication cost)
   - Replication score `s >= 0.6` (threshold)
   - Free adjacent water tile exists
5. **Spawns child** with identical genome (no mutation in Phase 0)
   - Parent and child each pay energy cost
   - Resource consumed from tile
6. **Moves randomly** if replication fails (tiny energy cost)
7. **Dies** when energy reaches 0, decomposing to return resources

### Visualization
- Agents are colored circles based on lineage depth (6-color palette)
- Faint lines connect children to parents for ~2 seconds after birth

## OLA Foundry

A single `StabilizedOLA` instance acts as the "genome foundry":
- Maintains a **champion genome** that evolves over time
- Emits genomes for new spawns:
  - Most spawns get a **clone** of the current champion
  - Every Kth spawn gets a **mutated candidate** (default K=5)
- Tracks fitness via pooled resource and population stats
- Adapts mutation rate based on loss variance

### Mutation Policy (Phase 0)
- **Foundry level**: Genomes are mutated when emitted every K spawns
- **Replication level**: Children receive exact copy of parent genome (no mutation)
- Future phases may enable mutation on replication events

## Output

### Console
- Device info (CPU/CUDA)
- Spawn count during burst
- Pause/resume and control feedback
- **Transition banner** when replication rate stabilizes above threshold for 3 consecutive windows

### CSV Metrics (`runs/soup_metrics.csv`)
Columns:
- `tick` - Simulation tick
- `live_agents` - Current population size
- `repl_events_window` - Replication events in last 600 ticks
- `avg_R` - Average resource across all tiles
- `total_R` - Total resource in system (mass conservation check)
- `mass_error` - Absolute error in mass conservation (should be ~0)
- `OLA_mut_rate` - Current OLA mutation rate
- `OLA_ema_loss` - OLA exponential moving average loss

### Genome Snapshots (`genomes/`)
Press `S` to save current champion genome:
- Filename: `champion_<tick>.pt`
- Contains: champion state dict, config, metrics, and metadata

## HUD Display

Top-left overlay shows:
- **Tick**: Current simulation tick
- **Agents**: Live population count
- **Food**: Active food blocks / total food blocks
- **Repl/min**: Replication events per minute (rolling average)
- **Avg R**: Average resource level across all tiles
- **Total R**: Total resource in system (should stay flat in Phase 0)
- **Mass Err**: Mass conservation error (should be ~0.0)
- **OLA mut**: Current OLA mutation rate

Bottom shows control hints.

## Transition to Phase 1

When the replication rate exceeds `1.0 events/min` for 3 consecutive windows (1800 ticks), a banner is printed:

```
============================================================
REPLICATION ACHIEVED!
Ready to enable mutation/selection in Phase 1.
============================================================
```

This signals that the population has stabilized and replication is self-sustaining. Future phases can enable:
- Mutation on replication events (child genomes differ from parent)
- Selection pressure based on resource efficiency
- Complex behaviors (foraging, cooperation, competition)

## Mass Conservation (Phase 0)

The resource system is strictly mass-conserving in Phase 0:

### Key Properties
1. **Diffusion is conservative**: Uses 5-point Laplacian with no-flux boundaries, NO clipping
2. **Decomposition returns 100%**: Dead agents return all remaining energy as resources
3. **No background decay**: Resources only change via replication (spent) and death (returned)
4. **Contraction disabled**: Automatic resource contraction is turned off to maintain flat Avg R
5. **Verification**: Every tick computes `mass_error = |ΔR - (-repl_spent + decomp_returned)|`

### Expected Behavior
- **Total R should stay constant** until a population boom occurs
- **Avg R drifts smoothly** as resources diffuse and redistribute
- **Mass error should be ~0** (< 1e-6 in theory, < 1e-3 tolerance in practice)
- Warning printed to console if mass conservation is violated

### Energy vs Resource Ledgers
- **Resource (R)**: Environmental stock in tiles, strictly conserved
- **Agent energy**: Private battery, replenished by food (separate from R)
- **Food**: External energy source, does NOT affect resource ledger
- **Movement**: Costs agent energy only, does NOT touch resources
- **Replication**: Only operation that transfers R → agent energy

## Technical Details

### Integration with Existing Code
- Uses `EvoCell` from `ola.py` as the genome substrate
- Uses `StabilizedOLA` from `stabilized_ola.py` for genome evolution
- Does **not** import `ola_gui.py` (separate PyQt application)

### Observation Vector (16 dims)
- 9 floats: normalized resources in 3×3 stencil
- 1 float: local population density (agents in 3×3 / 9)
- 1 float: center tile type (0=water, 1=rock)
- 1 float: bias term (1.0)
- 4 floats: padding (reserved for future features)

### Deterministic Seeding
- NumPy RNG: `np.random.seed(args.seed)`
- PyTorch: `torch.manual_seed(args.seed)`
- Enables reproducible runs with same `--seed`

### Performance
- Runs smoothly on CPU with ~200 agents at 60 FPS
- CUDA acceleration available for larger populations
- Tile size controls visual resolution (smaller = more agents visible)

## Examples

### Quick test run (small window, few agents)
```bash
python run_soup.py --width 640 --height 480 --tile 8 --primordial_burst 50 --fps 30
```

### Large-scale run (high resolution, many agents)
```bash
python run_soup.py --width 1920 --height 1080 --tile 6 --primordial_burst 500 --fps 60
```

### Controlled experiment (fixed seed)
```bash
python run_soup.py --seed 123 --primordial_burst 200
```

## Troubleshooting

**No water tiles available for spawn**
- Increase tile size or reduce rock density in `soup_env.py` (`init_terrain` method)

**Population dies out quickly**
- Increase initial resource levels in `soup_env.py` (start closer to `R_max`)
- Reduce contraction rate or increase interval in `run_soup.py`

**Replication never achieved**
- Lower `s_threshold` in `protomolecule.py` (default 0.6)
- Increase `primordial_burst` to seed more genetic diversity
- Adjust `spawn_rate` to introduce more mutated genomes

**Slow performance**
- Increase `--tile` size to reduce grid resolution
- Reduce `--primordial_burst` to spawn fewer agents
- Lower `--fps` target

## License

This code integrates with the existing OLA codebase and follows the same license.
