from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from collections import defaultdict, deque
import json
import traceback
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio
import threading

app = FastAPI()

# Mount static files directory
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Import GENREG system components
try:
    from nodes.envs.snake import SnakeEnvironment
    from genreg_proteins import run_protein_cascade
    from genreg_controller import Controller
    from genreg_genome import Genome
    from genreg_population import Population
    GENREG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GENREG modules not fully available: {e}")
    print("Some features may not work.")
    GENREG_AVAILABLE = False

# Client state management - each WebSocket connection gets isolated state
client_states: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def read_root():
    """Serve the frontend HTML file."""
    return FileResponse(str(static_path / "index.html"))


def topological_sort(nodes, connections):
    """
    Perform topological sort on nodes based on connections.
    Returns a list of node IDs in execution order.
    """
    # Build graph and in-degree count
    graph = defaultdict(list)
    in_degree = {node["id"]: 0 for node in nodes}
    
    # Create adjacency list and count in-degrees
    for conn in connections:
        from_id = conn["from"]
        to_id = conn["to"]
        graph[from_id].append(to_id)
        if to_id in in_degree:
            in_degree[to_id] += 1
    
    # Find all node IDs that exist
    node_ids = {node["id"]: node for node in nodes}
    
    # Initialize in-degree for nodes that might not have connections
    for node_id in node_ids:
        if node_id not in in_degree:
            in_degree[node_id] = 0
    
    # Kahn's algorithm for topological sort
    queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
    result = []
    
    while queue:
        node_id = queue.popleft()
        result.append(node_id)
        
        for neighbor in graph[node_id]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles (if result length doesn't match node count)
    if len(result) != len(node_ids):
        # Find nodes that weren't included (cycle detection)
        missing = set(node_ids.keys()) - set(result)
        raise ValueError(f"Cycle detected in graph. Missing nodes: {missing}")
    
    return result


def get_client_state(websocket_id: str) -> Dict[str, Any]:
    """Get or create isolated state for a client."""
    if websocket_id not in client_states:
        client_states[websocket_id] = {
            "env": None,
            "genome": None,
            "population": None,
            "current_genome_index": 0
        }
    return client_states[websocket_id]


def cleanup_client_state(websocket_id: str):
    """Clean up client state when connection closes."""
    if websocket_id in client_states:
        del client_states[websocket_id]


async def handle_reset_env(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle reset_env command - creates new SnakeEnv and returns initial signals."""
    try:
        grid_size = data.get("grid_size", 10)
        print(f"  -> Resetting env (grid={grid_size})")

        if GENREG_AVAILABLE:
            state["env"] = SnakeEnvironment(grid_size=grid_size)
            signals = state["env"].get_signals()
        else:
            # Fallback if GENREG not available
            signals = {
                "steps_alive": 0,
                "energy": 100.0,
                "dist_to_food": 0.0,
                "head_x": 5.0,
                "head_y": 5.0,
                "food_x": 7.0,
                "food_y": 7.0,
                "food_dx": 2.0,
                "food_dy": 2.0,
                "near_wall": 0.0,
                "alive": 1.0
            }
        
        return {
            "type": "reset_env_complete",
            "signals": signals,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error resetting environment: {str(e)}",
            "success": False
        }


async def handle_step(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle step command - steps environment with action and returns new signals."""
    try:
        if state["env"] is None:
            return {
                "type": "error",
                "message": "Environment not initialized. Call reset_env first.",
                "success": False
            }

        action = data.get("action", 0)
        # Log every 50 steps
        if state["env"] and hasattr(state["env"], "steps_alive") and state["env"].steps_alive % 50 == 0:
            print(f"  -> Step {state['env'].steps_alive}: action={action}, alive={state['env'].alive}, energy={state['env'].energy}")
        if not isinstance(action, int) or action < 0 or action > 3:
            return {
                "type": "error",
                "message": f"Invalid action: {action}. Must be 0-3.",
                "success": False
            }
        
        if GENREG_AVAILABLE:
            signals, done = state["env"].step(action)
            if done:
                print(f"  -> Snake died! steps={state['env'].steps_alive}, food={state['env'].food_eaten}")
        else:
            # Fallback
            signals = state["env"].get_signals() if hasattr(state["env"], "get_signals") else {}
            done = False
        
        return {
            "type": "step_complete",
            "signals": signals,
            "done": done,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error stepping environment: {str(e)}",
            "success": False
        }


def pygame_window_thread(env, grid_size, window_id, state_ref):
    """Run Pygame window in a separate thread."""
    import pygame
    
    # Initialize Pygame
    pygame.init()
    
    # Calculate window size (40 pixels per grid cell, plus padding)
    cell_size = 40
    window_width = grid_size * cell_size + 100  # Extra space for info
    window_height = grid_size * cell_size + 100
    
    # Create window
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"GENREG Snake Environment - Window {window_id}")
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    target_fps = 30  # Default FPS
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    DARK_GREEN = (0, 150, 0)
    GRAY = (128, 128, 128)
    
    try:
        while running:
            # Check state for pause/play/FPS updates
            if state_ref:
                paused = state_ref.get("pygame_paused", False)
                target_fps = state_ref.get("pygame_fps", 30)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Toggle pause with spacebar
                        if state_ref:
                            state_ref["pygame_paused"] = not state_ref.get("pygame_paused", False)
                            paused = state_ref["pygame_paused"]
            
            # Skip update if paused
            if paused:
                clock.tick(target_fps)
                pygame.display.flip()
                continue
            
            # Clear screen
            screen.fill(BLACK)
            
            # Draw grid
            offset_x = 50
            offset_y = 50
            
            # Draw grid lines
            for i in range(grid_size + 1):
                x = offset_x + i * cell_size
                y = offset_y + i * cell_size
                pygame.draw.line(screen, GRAY, (offset_x, y), (offset_x + grid_size * cell_size, y), 1)
                pygame.draw.line(screen, GRAY, (x, offset_y), (x, offset_y + grid_size * cell_size), 1)
            
            # Draw food
            try:
                food_x = int(env.food_x)
                food_y = int(env.food_y)
                food_rect = pygame.Rect(
                    offset_x + food_x * cell_size + 2,
                    offset_y + food_y * cell_size + 2,
                    cell_size - 4,
                    cell_size - 4
                )
                pygame.draw.ellipse(screen, RED, food_rect)
            except (AttributeError, TypeError):
                pass
            
            # Draw snake head
            try:
                head_x = int(env.head_x)
                head_y = int(env.head_y)
                head_rect = pygame.Rect(
                    offset_x + head_x * cell_size + 2,
                    offset_y + head_y * cell_size + 2,
                    cell_size - 4,
                    cell_size - 4
                )
                # Use different color if alive/dead
                color = GREEN if env.alive else DARK_GREEN
                pygame.draw.rect(screen, color, head_rect)
            except (AttributeError, TypeError):
                pass
            
            # Draw tail if enabled
            if hasattr(env, 'tail') and env.tail:
                try:
                    for tail_pos in env.tail:
                        tail_x, tail_y = tail_pos
                        tail_rect = pygame.Rect(
                            offset_x + tail_x * cell_size + 4,
                            offset_y + tail_y * cell_size + 4,
                            cell_size - 8,
                            cell_size - 8
                        )
                        pygame.draw.rect(screen, DARK_GREEN, tail_rect)
                except (AttributeError, TypeError):
                    pass
            
            # Draw info text
            try:
                font = pygame.font.Font(None, 24)
                info_texts = [
                    f"Steps: {env.steps_alive}",
                    f"Energy: {env.energy}/{env.max_energy}",
                    f"Food: {env.food_eaten}",
                    f"Status: {'Alive' if env.alive else 'Dead'}"
                ]
                y_offset = 10
                for text in info_texts:
                    text_surface = font.render(text, True, WHITE)
                    screen.blit(text_surface, (10, y_offset))
                    y_offset += 25
            except (AttributeError, TypeError):
                pass
            
            # Update display
            pygame.display.flip()
            clock.tick(target_fps)  # Use target FPS from state
            
    except Exception as e:
        print(f"Error in Pygame window thread: {e}")
        traceback.print_exc()
    finally:
        pygame.quit()


async def handle_launch_env(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle launch_env command - launches Pygame window for environment visualization."""
    try:
        env_type = data.get("env_type", "snake")
        grid_size = data.get("grid_size", 10)
        
        # Try to import pygame
        try:
            import pygame
        except ImportError:
            return {
                "type": "error",
                "message": "Pygame not installed. Install with: pip install pygame",
                "success": False
            }
        
        # Initialize the environment in state
        if GENREG_AVAILABLE:
            from nodes.envs.snake import SnakeEnvironment
            state["env"] = SnakeEnvironment(grid_size=grid_size)
            
            # Reset to get initial state
            signals = state["env"].reset()
        else:
            # Fallback - create a simple mock environment
            class MockEnv:
                def __init__(self, grid_size):
                    self.grid_size = grid_size
                    self.head_x = grid_size // 2
                    self.head_y = grid_size // 2
                    self.food_x = grid_size - 2
                    self.food_y = 2
                    self.food_eaten = 0
                    self.steps_alive = 0
                    self.energy = 25
                    self.max_energy = 25
                    self.alive = True
                    self.tail = []
            
            state["env"] = MockEnv(grid_size)
            signals = {
                "steps_alive": 0.0,
                "energy": 25.0,
                "dist_to_food": 10.0,
                "head_x": float(state["env"].head_x),
                "head_y": float(state["env"].head_y),
                "food_x": float(state["env"].food_x),
                "food_y": float(state["env"].food_y),
                "food_dx": 2.0,
                "food_dy": -2.0,
                "near_wall": 0.0,
                "alive": 1.0
            }
        
        # Close existing window if any
        if "pygame_thread" in state and state["pygame_thread"] and state["pygame_thread"].is_alive():
            # Can't easily kill thread, but we can start a new one
            pass
        
        # Launch Pygame window in separate thread
        window_id = id(state)  # Use state ID as window identifier
        pygame_thread = threading.Thread(
            target=pygame_window_thread,
            args=(state["env"], grid_size, window_id, state),
            daemon=True  # Thread will exit when main process exits
        )
        pygame_thread.start()
        
        # Store thread reference and control state in state
        state["pygame_thread"] = pygame_thread
        state["pygame_paused"] = False
        state["pygame_fps"] = 30  # Default FPS
        
        response = {
            "type": "launch_env_complete",
            "status": "success",
            "message": f"Environment launched: {env_type} (Pygame window opened)",
            "grid_size": grid_size,
            "signals": signals,
            "success": True
        }
        
        return response
    except Exception as e:
        traceback.print_exc()
        return {
            "type": "error",
            "message": f"Error launching environment: {str(e)}",
            "success": False
        }


async def handle_play_env(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle play_env command - resumes Pygame window rendering."""
    try:
        state["pygame_paused"] = False
        return {
            "type": "play_env_complete",
            "status": "success",
            "message": "Environment resumed",
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error resuming environment: {str(e)}",
            "success": False
        }


async def handle_pause_env(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle pause_env command - pauses Pygame window rendering."""
    try:
        state["pygame_paused"] = True
        return {
            "type": "pause_env_complete",
            "status": "success",
            "message": "Environment paused",
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error pausing environment: {str(e)}",
            "success": False
        }


async def handle_set_fps(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle set_fps command - sets Pygame window frame rate."""
    try:
        fps = data.get("fps", 30)
        fps = max(1, min(60, int(fps)))  # Clamp between 1 and 60
        state["pygame_fps"] = fps
        return {
            "type": "set_fps_complete",
            "status": "success",
            "message": f"FPS set to {fps}",
            "fps": fps,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error setting FPS: {str(e)}",
            "success": False
        }


async def handle_run_proteins(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle run_proteins command - runs protein cascade and returns trust_delta."""
    try:
        if state["genome"] is None:
            return {
                "type": "error",
                "message": "Genome not set. Call create_genome or set_genome first.",
                "success": False
            }

        signals = data.get("signals", {})

        if GENREG_AVAILABLE and hasattr(state["genome"], "proteins"):
            # Run protein cascade
            outputs, trust_delta = run_protein_cascade(state["genome"].proteins, signals)
            state["genome"].trust += trust_delta
        else:
            trust_delta = 0.0
            outputs = {}

        return {
            "type": "run_proteins_complete",
            "trust_delta": trust_delta,
            "protein_outputs": outputs,
            "total_trust": state["genome"].trust if state["genome"] else 0.0,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error running proteins: {str(e)}",
            "success": False
        }


async def handle_run_controller(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle run_controller command - runs controller and returns action."""
    try:
        if state["genome"] is None:
            return {
                "type": "error",
                "message": "Genome not set. Call create_genome first.",
                "success": False
            }

        signals = data.get("signals", {})

        if GENREG_AVAILABLE and hasattr(state["genome"], "controller"):
            # Get signal_order from genome
            signal_order = state["genome"].signal_order
            if not signal_order:
                return {
                    "type": "error",
                    "message": "Genome signal_order not set. Must be provided from Sensor.",
                    "success": False
                }

            action = state["genome"].controller.select_action(signals, signal_order)
            outputs = state["genome"].controller.forward(
                [signals.get(k, 0.0) for k in signal_order]
            )
        else:
            action = 0
            outputs = [0, 0, 0, 0]

        return {
            "type": "run_controller_complete",
            "action": int(action),
            "outputs": outputs,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error running controller: {str(e)}",
            "success": False
        }


async def handle_evolve(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evolve command - runs one generation of evolution."""
    try:
        if state["population"] is None:
            return {
                "type": "error",
                "message": "Population not initialized. Call create_population first.",
                "success": False
            }

        print(f"  -> Evolving population...")
        if GENREG_AVAILABLE:
            stats = state["population"].evolve()
            print(f"  -> Evolution complete: gen={stats.get('generation')}, best={stats.get('best_trust', 0):.2f}, avg={stats.get('avg_trust', 0):.2f}")
        else:
            stats = {"generation": 0, "best_trust": 0, "avg_trust": 0}

        return {
            "type": "evolve_complete",
            "stats": stats,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error during evolution: {str(e)}",
            "success": False
        }


async def handle_create_genome(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle create_genome command - creates a new genome."""
    try:
        signal_order = data.get("signal_order", None)
        print(f"  -> Creating new genome (signal_order: {signal_order})")
        if GENREG_AVAILABLE:
            state["genome"] = Genome(signal_order=signal_order)
            print(f"  -> Genome created with {len(state['genome'].proteins)} proteins")
            return {
                "type": "create_genome_complete",
                "genome": state["genome"].to_dict(),
                "success": True
            }
        else:
            return {
                "type": "error",
                "message": "GENREG modules not available",
                "success": False
            }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error creating genome: {str(e)}",
            "success": False
        }


async def handle_create_population(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle create_population command - creates a new population."""
    try:
        size = data.get("size", 50)
        print(f"  -> Creating population of {size} genomes")

        if GENREG_AVAILABLE:
            state["population"] = Population(size=size)
            print(f"  -> Population created: {state['population'].get_stats()}")
            return {
                "type": "create_population_complete",
                "stats": state["population"].get_stats(),
                "success": True
            }
        else:
            return {
                "type": "error",
                "message": "GENREG modules not available",
                "success": False
            }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error creating population: {str(e)}",
            "success": False
        }


async def handle_evaluate_population(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle evaluate_population command - evaluates all genomes in environment."""
    try:
        if state["population"] is None:
            return {
                "type": "error",
                "message": "Population not initialized.",
                "success": False
            }

        steps_per_life = data.get("steps_per_life", 200)

        if GENREG_AVAILABLE:
            env = SnakeEnvironment()
            state["population"].evaluate(env, steps_per_life)
            return {
                "type": "evaluate_complete",
                "stats": state["population"].get_stats(),
                "success": True
            }
        else:
            return {
                "type": "error",
                "message": "GENREG modules not available",
                "success": False
            }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error evaluating population: {str(e)}",
            "success": False
        }


async def handle_get_state(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle get_state command - returns current env state for browser rendering."""
    try:
        if state["env"] is None:
            return {
                "type": "error",
                "message": "Environment not initialized.",
                "success": False
            }

        env = state["env"]
        return {
            "type": "state",
            "grid_size": env.grid_size,
            "head_x": env.head_x,
            "head_y": env.head_y,
            "food_x": env.food_x,
            "food_y": env.food_y,
            "tail": list(env.tail) if hasattr(env, 'tail') else [],
            "alive": env.alive,
            "food_eaten": env.food_eaten,
            "energy": env.energy,
            "steps_alive": env.steps_alive,
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error getting state: {str(e)}",
            "success": False
        }


async def handle_set_genome(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle set_genome command - sets the current genome."""
    try:
        genome_data = data.get("genome")
        if genome_data is None:
            return {
                "type": "error",
                "message": "No genome data provided",
                "success": False
            }
        
        # Store genome (assuming it's a serialized genome that can be reconstructed)
        # In a real implementation, you'd deserialize and create a Genome object
        state["genome"] = genome_data  # For now, store as-is
        
        return {
            "type": "set_genome_complete",
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error setting genome: {str(e)}",
            "success": False
        }


async def handle_set_population(state: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle set_population command - sets the current population."""
    try:
        population_data = data.get("population")
        if population_data is None:
            return {
                "type": "error",
                "message": "No population data provided",
                "success": False
            }
        
        # Store population
        state["population"] = population_data  # For now, store as-is
        
        return {
            "type": "set_population_complete",
            "success": True
        }
    except Exception as e:
        return {
            "type": "error",
            "message": f"Error setting population: {str(e)}",
            "success": False
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_id = id(websocket)
    state = get_client_state(str(websocket_id))
    
    try:
        while True:
            # Receive JSON data
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            # Check if this is a command-based message (new format)
            if "cmd" in payload:
                # Handle command-based messages
                command = payload.get("cmd")
                request_id = payload.get("request_id")
                command_data = {k: v for k, v in payload.items() if k not in ("cmd", "request_id")}

                # Log received command
                print(f"[WS] Received: {command}", end="")
                if command == "step":
                    print(f" action={command_data.get('action', '?')}", end="")
                elif command == "create_population":
                    print(f" size={command_data.get('size', '?')}", end="")
                print()

                response = None

                if command == "reset_env":
                    response = await handle_reset_env(state, command_data)
                elif command == "step":
                    response = await handle_step(state, command_data)
                elif command == "run_proteins":
                    response = await handle_run_proteins(state, command_data)
                elif command == "run_controller":
                    response = await handle_run_controller(state, command_data)
                elif command == "evolve":
                    response = await handle_evolve(state, command_data)
                elif command == "set_genome":
                    response = await handle_set_genome(state, command_data)
                elif command == "set_population":
                    response = await handle_set_population(state, command_data)
                elif command == "ping":
                    response = {
                        "type": "pong",
                        "timestamp": payload.get("timestamp"),
                        "success": True
                    }
                elif command == "launch_env":
                    response = await handle_launch_env(state, command_data)
                elif command == "create_genome":
                    response = await handle_create_genome(state, command_data)
                elif command == "create_population":
                    response = await handle_create_population(state, command_data)
                elif command == "evaluate_population":
                    response = await handle_evaluate_population(state, command_data)
                elif command == "get_state":
                    response = await handle_get_state(state, command_data)
                elif command == "pause_env":
                    response = await handle_pause_env(state, command_data)
                elif command == "play_env":
                    response = await handle_play_env(state, command_data)
                elif command == "set_fps":
                    response = await handle_set_fps(state, command_data)
                else:
                    response = {
                        "type": "error",
                        "message": f"Unknown command: {command}",
                        "success": False
                    }

                # Include request_id in response for async handling
                if request_id is not None:
                    response["request_id"] = request_id

                # Send response
                await websocket.send_json(response)
                
            else:
                # Legacy format - handle old graph execution (for backward compatibility)
                nodes = payload.get("nodes", [])
                connections = payload.get("connections", [])
                
                # Validate input
                if not nodes:
                    await websocket.send_json({
                        "error": "No nodes provided",
                        "success": False
                    })
                    continue
                
                # Perform topological sort
                try:
                    execution_order = topological_sort(nodes, connections)
                except ValueError as e:
                    await websocket.send_json({
                        "error": str(e),
                        "success": False
                    })
                    continue
                
                # Create shared namespace for execution
                namespace = {}
                node_results = {}
                
                # Execute nodes in topological order
                for node_id in execution_order:
                    # Find the node data
                    node_data = next((n for n in nodes if n["id"] == node_id), None)
                    if not node_data:
                        await websocket.send_json({
                            "error": f"Node {node_id} not found",
                            "node_id": node_id,
                            "success": False
                        })
                        break
                    
                    code = node_data.get("code", "")
                    
                    # Capture stdout/stderr
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    stdout_capture = StringIO()
                    stderr_capture = StringIO()
                    
                    try:
                        sys.stdout = stdout_capture
                        sys.stderr = stderr_capture
                        
                        # Execute the node's code
                        exec(code, namespace)
                        
                        # Get captured output
                        stdout_output = stdout_capture.getvalue()
                        stderr_output = stderr_capture.getvalue()
                        
                        # Store results
                        node_results[node_id] = {
                            "stdout": stdout_output,
                            "stderr": stderr_output,
                            "success": True
                        }
                        
                    except Exception as e:
                        # Capture error information
                        error_traceback = traceback.format_exc()
                        stderr_output = stderr_capture.getvalue()
                        
                        node_results[node_id] = {
                            "stdout": stdout_capture.getvalue(),
                            "stderr": stderr_output + "\n" + error_traceback,
                            "error": str(e),
                            "success": False
                        }
                        
                        # Send error response
                        await websocket.send_json({
                            "error": f"Execution failed at node {node_id}",
                            "node_id": node_id,
                            "error_message": str(e),
                            "traceback": error_traceback,
                            "node_results": node_results,
                            "success": False
                        })
                        break
                        
                    finally:
                        # Restore stdout/stderr
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                
                # If we completed all nodes successfully
                if len(node_results) == len(execution_order):
                    await websocket.send_json({
                        "success": True,
                        "execution_order": execution_order,
                        "node_results": node_results,
                        "namespace_keys": list(namespace.keys())
                    })
            
    except WebSocketDisconnect:
        cleanup_client_state(str(websocket_id))
    except json.JSONDecodeError:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid JSON format",
            "success": False
        })
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": f"Unexpected error: {str(e)}",
            "traceback": traceback.format_exc(),
            "success": False
        })
        cleanup_client_state(str(websocket_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



