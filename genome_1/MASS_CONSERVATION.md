# Mass Conservation Implementation - Phase 0

This document describes the strict mass-conserving resource system implemented in the Primordial Soup simulation.

## Problem Statement

In a closed ecosystem simulation, total resource should remain constant unless deliberately added or removed. Any "leak" causes Avg R to drift downward, making it impossible to distinguish natural redistribution from implementation bugs.

## Solution: 9-Point Surgical Fix

### 1. Mass-Conserving Diffusion ✓

**Old approach** (broken):
```python
# Averaged neighbors then clipped - leaks mass every tick
R_new = R + d * (avg_neighbors - R)
R = clip(R_new, 0, R_max)  # ← CLIPS AWAY MASS
```

**New approach** (conservative):
```python
# 5-point Laplacian with no-flux boundaries
# No clipping - preserves total mass exactly
R_pad = copy edges for no-flux boundary
laplacian = north + south + east + west - 4*center
R_new = R + d * laplacian
# NO CLIPPING HERE
```

**Location**: `soup_env.py:103-142`

### 2. No Background Resource Decay ✓

Removed all `R *= (1 - decay)` or evaporation terms. Resources only change via:
- Replication (consumes R)
- Death/decomposition (returns R)

### 3. Single Charge Per Replication ✓

**Verified**: Only one `consume_resource()` call per replication event.
- Parent pays `R_replicate_cost = 0.12` from tile
- Child inherits energy from parent's internal battery
- No double-charging

**Location**: `protomolecule.py:66-113`

### 4. 100% Decomposition Yield (Phase 0) ✓

**Old**: `decomp_yield = 0.5 * energy` (50% leak)

**New**: `decomp_yield = 1.0 * energy` (100% return)

Dead agents return ALL remaining energy as resources to neighbors. Zero entropy leak.

**Location**: `protomolecule.py:146-162`

### 5. Resource Contraction Disabled ✓

**Old**: `contraction_interval = 1500` (automatic famine every 25 seconds)

**New**: `contraction_interval = 999999` (effectively disabled)

Contraction deliberately removes mass - turned off in Phase 0 to maintain flat Avg R.

Can still be manually triggered with 'C' key for testing, and tracks `mass_removed`.

**Location**: `run_soup.py:107-108`

### 6. Energy-Only Movement Costs ✓

**Verified**: `attempt_move()` only touches `agent.energy`, never accesses resources.

Movement is an internal energy drain, not a resource sink.

**Location**: `protomolecule.py:115-132`

### 7. No-Flux Boundary Conditions ✓

Diffusion uses proper no-flux boundaries:
- Edge cells copy from nearest interior cell
- No mass bleeds out of grid edges
- Laplacian is computed on padded grid

**Location**: `soup_env.py:113-124`

### 8. Foundry Spawn Cost Tracked ✓

Initial spawns start with energy, but this is an **external injection** in Phase 0.

**Implementation**:
- Primordial burst agents start with `energy=0.5`
- Manual spawns (G key) start with `energy=0.1` (lower to minimize impact)
- Future: could charge tiles for spawn cost, but Phase 0 treats initial burst as "external"

**Location**: `run_soup.py:70-91, 133-142`

### 9. Full Instrumentation ✓

Added comprehensive tracking and verification:

**Per-tick counters**:
```python
repl_resource_spent      # Sum of all R_replicate_cost charged
decomp_resource_returned # Sum of all decomposition yields
spawn_resource_cost      # Cost of new spawns (currently 0)
contraction_mass_removed # Mass removed by contraction (if any)
```

**Conservation law**:
```python
R_before = env.get_total_resource()
# ... tick events ...
R_after = env.get_total_resource()

expected_delta = -repl_spent + decomp_returned - spawn_cost - contraction
actual_delta = R_after - R_before
mass_error = |actual_delta - expected_delta|

if mass_error > 1e-3:
    print(f"WARNING: Mass conservation violated! Error={mass_error}")
```

**Location**: `run_soup.py:151-222`

**CSV logging**:
- `total_R` - Total resource in system
- `mass_error` - Conservation error per tick

**HUD display**:
- `Total R` - Visual verification
- `Mass Err` - Real-time error monitoring

## Expected Behavior

### Phase 0 (No Population Boom)
- **Total R**: Flat (constant within numerical precision)
- **Avg R**: Drifts smoothly as resources diffuse
- **Mass Err**: ~0.0 (< 1e-3 tolerance)

### Phase 0 (Population Boom)
- **Total R**: Still flat (replication spending balanced by death returns)
- **Avg R**: May drop temporarily if population spikes (R locked in agent energy)
- **Mass Err**: Still ~0.0

### If Mass Conservation Fails
- Console warning printed immediately
- Check CSV for `mass_error` spike
- Debug which term (replication/decomposition/spawn) is leaking

## Two-Ledger System

The simulation maintains **two separate accounts**:

### Resource Ledger (R)
- Environmental stock in tiles
- **Strictly conserved** in Phase 0
- Changes only via:
  - Replication: R → agent energy (decreases R)
  - Death: agent energy → R (increases R)
  - Contraction: R → void (decreases R, disabled in Phase 0)

### Energy Ledger (per agent)
- Private battery in `agent.energy`
- **NOT conserved** (external food source)
- Changes via:
  - Replication: parent pays cost (decreases)
  - Movement: tiny drain (decreases)
  - Food: restored to 1.0 (increases, external injection)

**Food is an external energy source** that does NOT affect the resource ledger. This allows agents to survive indefinitely while still maintaining strict resource conservation.

## Verification Checklist

- [x] Diffusion uses 5-point Laplacian with no-flux boundaries
- [x] No clipping in diffusion step
- [x] Decomposition returns 100% energy
- [x] No background resource decay
- [x] Contraction disabled (interval = 999999)
- [x] Movement costs energy only, not resources
- [x] Replication charges resources exactly once
- [x] Per-tick mass balance computed and logged
- [x] Warning printed if mass_error > 1e-3
- [x] Total R and Mass Err displayed in HUD
- [x] CSV includes total_R and mass_error columns

## Testing

Run simulation and verify:
```bash
python run_soup.py --primordial_burst 100 --fps 60
```

Monitor HUD:
- `Total R` should stay constant (~10800 for default grid)
- `Mass Err` should show ~0.00e+00

Check CSV after 1000 ticks:
```python
import pandas as pd
df = pd.read_csv('runs/soup_metrics.csv')
print(df['total_R'].std())  # Should be < 0.01
print(df['mass_error'].max())  # Should be < 1e-3
```

## Future Phases

**Phase 1 (Introduce Entropy)**:
- Lower `decomp_yield` to 0.95 (5% leak)
- Enable contraction every 1500 ticks
- Avg R will now decline → famine pressure

**Phase 2 (Complex Ecology)**:
- Multiple resource types
- Predator/prey dynamics
- Spatial competition

The strict conservation foundation makes it possible to introduce controlled sinks later while maintaining clear accounting.
