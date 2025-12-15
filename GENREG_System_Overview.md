# GENREG: An Evolved Genetic Regulatory Network for Agent Control

## 1. Overview

GENREG is a Python-based project that explores a novel approach to training autonomous agents, using a "snake" game as the testbed. Instead of traditional reinforcement learning with explicit reward functions, GENREG employs an evolutionary algorithm guided by a simulated **Genetic Regulatory Network (GENREG)**.

The core principle is to evolve a neural network controller for an agent (the "functional" part of its genome) by assessing its performance not with a simple score, but with a "trust" value. This trust is dynamically calculated by a network of simulated "proteins" (the "regulatory" part of its genome) that interpret raw sensory data from the environment.

The system is designed for high-throughput experimentation, with multiple execution modes ranging from real-time visualizers to a high-performance, GPU-accelerated headless trainer.

## 2. Core Components

The project is modular, with each file representing a key biological or systemic analogue.

### 2.1. The Environment (`genreg_snake_env.py`)

-   **Purpose**: Provides a minimal, challenging environment for the agent.
-   **Functionality**:
    -   A standard "snake" game on a 10x10 grid.
    -   **Crucially, it does not provide any reward signal.** Its sole responsibility is to provide a dictionary of sensory `signals` to the agent at each step.
    -   **Signals include**: `dist_to_food`, `steps_alive`, `near_wall`, `head_x`, `head_y`, etc.

### 2.2. The Regulatory Genome (`genreg_proteins.py`)

This is the conceptual heart of the GENREG system. It defines a library of "proteins" that process signals and ultimately determine the agent's fitness.

-   **`SensorProtein`**: The primary interface to the environment. It reads a specific signal (e.g., `dist_to_food`) and normalizes its value.
-   **`TrendProtein`**: Acts like a calculus derivative. It measures the rate and direction of change of an input signal over time. This allows the system to know not just *what* the distance to the food is, but whether the agent is getting *closer* or *farther away*.
-   **`TrustModifierProtein`**: The most critical protein. It converts the output of other proteins into a `trust_delta`—the change in fitness for a given time step. This is where the "reward" is implicitly defined.
    -   **Example**: A `TrendProtein` tracking `dist_to_food` will have a negative output if the snake is getting closer. A `TrustModifierProtein` can be configured with a negative `scale` parameter. When it receives the negative trend, it multiplies it by its negative scale, resulting in a **positive trust delta**. This effectively "rewards" the agent for approaching food, without the environment ever providing a reward.

### 2.3. The Functional Genome (`genreg_controller.py`)

-   **Purpose**: To act as the agent's "brain" or neural network.
-   **Architecture**: A simple, feed-forward neural network with a single hidden layer and `tanh` activation functions.
-   **Function**:
    -   Receives the raw signal list from the environment as input.
    -   Outputs a single action (up, down, left, or right).
    -   **No Backpropagation**: Its weights are not trained with gradient descent. Instead, they are evolved through mutation over generations.

### 2.4. The Genome & Population (`genreg_genome.py`)

-   **`GENREGGenome`**: Represents a single individual agent. It encapsulates both the regulatory part (the list of proteins) and the functional part (the neural controller). Its fitness is its `trust` score.
-   **`GENREGPopulation`**: Manages the entire population of genomes and orchestrates the evolutionary process.
    1.  **Evaluation**: Each genome in the population is evaluated in the snake environment for one "lifetime".
    2.  **Selection**: After evaluation, genomes are sorted by their final `trust` score. The top performers (e.g., the top 20%) are selected as "survivors".
    3.  **Reproduction & Mutation**: A new generation is created by cloning the survivors and applying small, random mutations to the weights of their neural controllers and the parameters of their proteins.
    4.  **Trust Inheritance**: To ensure stability and forward progress, child genomes inherit a percentage of their parent's final trust score. This gives the offspring of successful parents a competitive advantage.

### 2.5. Checkpoint System (`genreg_checkpoint.py`)

-   **Purpose**: To persist training progress.
-   **Functionality**: Allows for saving and loading the complete state of a `GENREGPopulation`, including all genomes, their trust scores, controller weights, protein parameters, and the current generation number.

## 3. Execution & Visualization

The system can be run in several modes to suit different needs.

-   **`genreg_visualizer.py`**: A real-time visualizer using Pygame to render the snake game and a Tkinter window to display detailed metrics, charts, and protein/signal states. Ideal for observing a specific genome's behavior.
-   **`genreg_headless.py` / `pure_headless.py`**: Runs the training loop without any game GUI for significantly faster performance. The `gpu_headless.py` version is the most advanced, using PyTorch and a CUDA-enabled GPU to process the entire population's neural network inferences in parallel batches. These scripts often output charts (using PyQt6 or Matplotlib) to track training progress over thousands of generations, as seen in the provided screenshots.
-   **`viz.py` ("The Trust Landscape")**: An abstract visualizer that provides a conceptual overview of the evolutionary process. It represents genomes as particles in a 2D space, visualizing their fitness, diversity, and the selection process in a more artistic and intuitive manner.
-   **`run_best_genome.py`**: An inference script to load the best-performing genome from a saved checkpoint and watch it play without any ongoing training.

![GENREG Training Statistics](Screenshot 2025-11-29 110249.png)
_Figure 1: Training statistics chart showing the agent's performance (food eaten, steps survived) and the population's fitness (trust) improving over tens of thousands of generations._

## 4. Summary

GENREG is a powerful and flexible framework for experimenting with evolutionary computation and emergent agent behavior. By separating the agent's "brain" (the controller) from its "value system" (the regulatory proteins), it creates a rich, dynamic fitness landscape where sophisticated strategies for survival and success can evolve from simple, domain-agnostic principles like "getting closer to resources is good" and "surviving longer is good."
