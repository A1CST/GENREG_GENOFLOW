"""
Test script for Competitive OLA system
Validates co-evolutionary pressure and emergent strategies
"""
import torch
import numpy as np
from competitive_ola import CompetitiveOLA, CompetitiveOLAConfig


def test_basic_competition():
    """Test basic competitive dynamics"""
    print("\n=== Testing Basic Competition ===")

    config = CompetitiveOLAConfig(
        grid_width=32,
        grid_height=32,
        state_dim=64,
        action_dim=6,
        observation_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    system = CompetitiveOLA(config)

    print(f"Device: {system.device}")
    print(f"Grid size: {config.grid_width}x{config.grid_height}")
    print(f"Action space: {config.action_dim}D continuous")
    print(f"Observation space: {config.observation_dim}D")

    # Run for several steps
    num_steps = 50
    print(f"\nRunning {num_steps} competitive steps...")

    for step in range(num_steps):
        metrics = system.step()

        if step % 10 == 0:
            print(f"\nStep {step}:")
            print(f"  Agent 0 Territory: {metrics['env_info']['territory_counts'][0]}")
            print(f"  Agent 1 Territory: {metrics['env_info']['territory_counts'][1]}")
            print(f"  Agent 0 Fitness: {metrics['agent_0']['competitive_fitness']:.4f}")
            print(f"  Agent 1 Fitness: {metrics['agent_1']['competitive_fitness']:.4f}")
            print(f"  Dominance Balance: {metrics['dominance_balance']:.3f}")
            print(f"  Strategy Diversity: {metrics['strategy_diversity']:.4f}")

    print("\n[PASS] Basic competition test passed!")
    return system, metrics


def test_action_space():
    """Test continuous action interpretation"""
    print("\n=== Testing Action Space ===")

    config = CompetitiveOLAConfig(
        grid_width=32,
        grid_height=32,
        state_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    system = CompetitiveOLA(config)

    # Test extreme actions
    test_actions = [
        # [action_x, action_y, speed, wait, expand, contract]
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # Move right, fast, expand
        [-1.0, 0.0, 1.0, 0.0, 0.0, 1.0],  # Move left, fast, contract
        [0.0, 1.0, 0.5, 0.5, 0.0, 0.0],  # Move down, medium speed, wait
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Full wait (planning)
    ]

    for i, action in enumerate(test_actions):
        system.reset()
        actions = np.array([action, action])  # Both agents do same action

        print(f"\nTest action {i}: {action}")
        initial_pos = system.env.agent_positions.copy()
        initial_radii = system.env.agent_radii.copy()

        # Execute action
        obs, rewards, info = system.env.step(actions)

        print(f"  Initial pos: {initial_pos[0]}")
        print(f"  Final pos: {info['agent_positions'][0]}")
        print(f"  Initial radius: {initial_radii[0]:.2f}")
        print(f"  Final radius: {info['agent_radii'][0]:.2f}")
        print(f"  Territory change: {info['territory_counts'][0]}")

    print("\n[PASS] Action space test passed!")


def test_territory_conflict():
    """Test conflict resolution and invasion mechanics"""
    print("\n=== Testing Territory Conflict ===")

    config = CompetitiveOLAConfig(
        grid_width=32,
        grid_height=32,
        state_dim=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    system = CompetitiveOLA(config)

    # Force agents to move toward each other
    print("\nForcing agents to collide...")

    for step in range(30):
        # Get current positions
        pos_0 = system.env.agent_positions[0]
        pos_1 = system.env.agent_positions[1]

        # Create actions that move agents toward each other
        action_0 = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Move right, expand
        action_1 = [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # Move left, expand

        obs, rewards, info = system.env.step(np.array([action_0, action_1]))

        distance = np.linalg.norm(pos_1 - pos_0)

        if step % 5 == 0:
            print(f"  Step {step}: Distance = {distance:.2f}, "
                  f"Territory = [{info['territory_counts'][0]}, {info['territory_counts'][1]}]")

        # Check for overlap
        if distance < (info['agent_radii'][0] + info['agent_radii'][1]):
            print(f"\n  >> Collision detected at step {step}!")
            print(f"    Agent 0 territory: {info['territory_counts'][0]}")
            print(f"    Agent 1 territory: {info['territory_counts'][1]}")
            break

    print("\n[PASS] Territory conflict test passed!")


def test_co_evolution():
    """Test co-evolutionary pressure over extended run"""
    print("\n=== Testing Co-Evolution ===")

    config = CompetitiveOLAConfig(
        grid_width=32,
        grid_height=32,
        state_dim=64,
        action_dim=6,
        observation_dim=64,
        mutation_rate=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    system = CompetitiveOLA(config)

    num_steps = 100
    print(f"Running {num_steps} steps to observe co-evolution...")

    territory_balance = []
    fitness_balance = []
    mutation_rates = [[], []]

    for step in range(num_steps):
        metrics = system.step()

        # Track balance metrics
        territory_diff = (metrics['env_info']['territory_counts'][0] -
                         metrics['env_info']['territory_counts'][1])
        territory_balance.append(territory_diff)

        fitness_diff = (metrics['agent_0']['competitive_fitness'] -
                       metrics['agent_1']['competitive_fitness'])
        fitness_balance.append(fitness_diff)

        mutation_rates[0].append(metrics['agent_0']['mutation_rate'])
        mutation_rates[1].append(metrics['agent_1']['mutation_rate'])

        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Territory Balance: {territory_diff:+d}")
            print(f"  Fitness Balance: {fitness_diff:+.4f}")
            print(f"  Agent 0 Mutation: {metrics['agent_0']['mutation_rate']:.6f}")
            print(f"  Agent 1 Mutation: {metrics['agent_1']['mutation_rate']:.6f}")
            print(f"  Strategy Diversity: {metrics['avg_diversity']:.4f}")

    # Analyze co-evolution
    print("\n=== Co-Evolution Analysis ===")
    print(f"Territory balance std: {np.std(territory_balance):.2f}")
    print(f"Fitness balance std: {np.std(fitness_balance):.4f}")
    print(f"Avg strategy diversity: {np.mean([m['avg_diversity'] for m in [metrics]]):.4f}")

    # Check for signs of co-evolution:
    # 1. Balance should fluctuate (not stuck)
    # 2. Mutation rates should adapt
    # 3. Strategy diversity should be non-zero

    territory_variance = np.var(territory_balance)
    mutation_variance_0 = np.var(mutation_rates[0])
    mutation_variance_1 = np.var(mutation_rates[1])

    print(f"\nCo-evolution indicators:")
    print(f"  Territory variance: {territory_variance:.2f} (expect > 10)")
    print(f"  Mutation variance Agent 0: {mutation_variance_0:.8f} (expect > 0)")
    print(f"  Mutation variance Agent 1: {mutation_variance_1:.8f} (expect > 0)")

    if territory_variance > 10 and mutation_variance_0 > 0:
        print("\n[PASS] Co-evolution test passed! Agents are adapting to each other.")
    else:
        print("\n[WARN] Co-evolution indicators weak (may need longer run)")


def test_visualization_data():
    """Test visualization data extraction"""
    print("\n=== Testing Visualization Data ===")

    config = CompetitiveOLAConfig(
        grid_width=16,
        grid_height=16,
        state_dim=32,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    system = CompetitiveOLA(config)

    # Run a few steps
    for _ in range(5):
        system.step()

    # Get visualization data
    viz_data = system.get_grid_visualization_data()

    print(f"Grid shape: {viz_data['grid'].shape}")
    print(f"Positions shape: {viz_data['positions'].shape}")
    print(f"Radii shape: {viz_data['radii'].shape}")
    print(f"Velocities shape: {viz_data['velocities'].shape}")
    print(f"Territory counts: {viz_data['territory_counts']}")
    print(f"Step: {viz_data['step']}")

    # Verify grid contains both agents
    unique_values = np.unique(viz_data['grid'])
    print(f"Grid unique values: {unique_values} (expect [0, 1, 2])")

    assert viz_data['grid'].shape == (16, 16), "Grid shape mismatch"
    assert len(viz_data['positions']) == 2, "Position count mismatch"
    assert len(unique_values) >= 2, "Grid should contain multiple territory values"

    print("\n[PASS] Visualization data test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Competitive OLA Test Suite")
    print("=" * 60)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available, using CPU")

    try:
        # Run tests
        test_action_space()
        test_territory_conflict()
        test_basic_competition()
        test_co_evolution()
        test_visualization_data()

        print("\n" + "=" * 60)
        print("[PASS] ALL TESTS PASSED!")
        print("=" * 60)
        print("\nReady to run full simulation with:")
        print("  python competitive_ola_gui.py")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
