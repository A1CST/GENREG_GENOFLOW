import gymnasium as gym
import time

# Load the HalfCheetah environment
env = gym.make("HalfCheetah-v5", render_mode="human") 

# Check the dimensions for setting up your GENREG Controller
print(f"Observation Space Size: {env.observation_space.shape}")  # Should be (17,)
print(f"Action Space Size: {env.action_space.shape}")          # Should be (6,)

observation, info = env.reset(seed=42)

for _ in range(200):
    # Action must be a vector of 6 continuous torques (floats)
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment (if render_mode="human")
    # You should see the HalfCheetah window pop up
    env.render() 

    if terminated or truncated:
        observation, info = env.reset()

env.close()