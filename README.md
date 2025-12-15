# 🧬 GenoFlow: The Visual IDE for Genetic Regulatory AI
** Note this application is still in development. Please pin or note bugs. 

**GenoFlow** (formerly *Genome Studio*) is a visual, node-based development environment for designing, training, and tuning **GENREG** (Genetic Regulatory) AI models.

Inspired by biological gene regulatory networks, GenoFlow lets users visually construct complex, self-adapting AI architectures that combine a functional neural network layer with a stateful, trust-modulating protein network.

---

## Key Features

### Visual, Node-Based Architecture

Build and visualize your entire AI model using an intuitive, ComfyUI-like graph editor powered by **LiteGraph.js**.

### Biologically Inspired Layers (GENREG)

**Protein Network (Regulatory Layer)**
Stateless, self-adapting *proteins* (e.g., `Sensor`, `Trend`, `Comparator`, `TrustModifier`) process environmental signals to generate a **Trust Delta** (fitness signal).

**Controller Network (Functional Layer)**
A simple feed-forward neural network that selects actions based on processed signals.

### Trust-Based Evolution

Genome fitness is determined by accumulated **Trust**, promoting robust and adaptive behaviors through evolutionary pressure.

### Real-Time Environment

Train and observe AI behavior in a simulated **Snake Environment**, with real-time visualization via a separate **Pygame** window.

### Client–Server Architecture

A Python **FastAPI / WebSocket** backend handles all heavy processing (evolution, environment steps), while the JavaScript frontend provides the interactive IDE.

---

## Quick Start

### 1. Prerequisites

GenoFlow requires **Python** and several libraries, including `fastapi` and `uvicorn` for the server, and optionally `pygame` for environment visualization.

```bash
# Recommended: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
```

### 2. Installation

Install the required Python packages:

```bash
pip install -r requirements.txt

# To enable environment visualization
pip install pygame
```

> **Note**
> `requirements.txt` includes core dependencies such as `fastapi`, `uvicorn`, and `websockets`.

### 3. Running the Server

Start the backend server using `uvicorn` as defined in `start_server.py`:

```bash
python start_server.py
```

The server typically starts at:

```
http://0.0.0.0:8000
```

### 4. Accessing the IDE

Open your browser and navigate to:

```
http://localhost:8000
```

Load the provided `snake_training_template.json` to begin exploring the training flow.

---

## Architecture Overview

GenoFlow is divided into distinct layers reflecting its biological inspiration and client–server design.

| Component        | Technology                  | Role                                               | Core Files                                 |
| ---------------- | --------------------------- | -------------------------------------------------- | ------------------------------------------ |
| Frontend IDE     | LiteGraph.js, Vanilla JS    | Visual graph editor, monitoring, real-time control | `static/js/*.js`, `static/index.html`      |
| Backend Server   | Python (FastAPI, WebSocket) | AI processing, evolution, simulation               | `start_server.py`                          |
| Controller Layer | Python (`Controller` class) | Feed-forward NN for action selection               | `genreg_controller.py`                     |
| Regulatory Layer | Python (Protein classes)    | Generates Trust Delta fitness signals              | `genreg_proteins.py`                       |
| Evolution Core   | Python (Genome, Population) | Evolutionary process and selection                 | `genreg_genome.py`, `genreg_population.py` |

---

## The GENREG Model Flow

The training process is orchestrated by the node graph in a continuous loop:

1. **Environment Step** (`SnakeEnvironment`) outputs signals.
2. **Protein Network** processes signals and calculates **Trust Delta**.
3. **Controller Network** selects an **Action**.
4. **Action** is fed back into the environment.
5. Episode ends and total **Trust** determines Genome fitness.
6. **Generation Manager** triggers population evolution.

---

## Node Types

The IDE provides modular nodes to construct and monitor AI behavior.

| Category    | Example Nodes                         | Purpose                                    |
| ----------- | ------------------------------------- | ------------------------------------------ |
| Regulatory  | Sensor, Trend, Trust Modifier         | Process signals and influence Genome Trust |
| Functional  | Controller Network                    | Select actions from processed signals      |
| Environment | Snake Environment, Visualize (Pygame) | Simulate and visualize behavior            |
| Evolution   | Population Controller, Episode Runner | Manage training and mutation lifecycle     |

---

## Contribution

GenoFlow is an evolving project. Contributions are welcome.

Areas of interest:

* New protein types
* Additional environments
* Visualization tools
* Extensions to evolutionary algorithms

Pull requests and design discussions are encouraged.
