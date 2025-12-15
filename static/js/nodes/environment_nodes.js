// Environment Node Definitions
function registerEnvironmentNodes() {
    // SnakeEnvironmentNode
    function SnakeEnvironmentNode() {
        // INPUT: Takes action from previous step (0=up, 1=down, 2=left, 3=right)
        this.addInput("action", "number");

        // OUTPUTS
        this.addOutput("signals", "signal");
        this.addOutput("done", "boolean");  // True when snake dies

        // Visual properties
        this.color = "#27AE60";
        this.bgcolor = "#1E5631";
        this.size = [200, 100];

        this.properties = {
            grid_size: 10,
            episodes: 0,
            current_score: 0,
            best_score: 0,
            alive: true,
            // Initialize with default signals (will be updated by backend)
            current_signals: {
                "steps_alive": 0,
                "energy": 25,
                "dist_to_food": 10,
                "head_x": 5,
                "head_y": 5,
                "food_x": 8,
                "food_y": 3,
                "food_dx": 3,
                "food_dy": -2,
                "near_wall": 0.0,
                "alive": 1.0
            },
            done: false
        };

        this.last_done = false;  // Track done state for edge detection
    }
    
    SnakeEnvironmentNode.title = "Snake Environment";
    SnakeEnvironmentNode.desc = "Snake game environment - outputs all signals";
    
    SnakeEnvironmentNode.prototype.onExecute = function() {
        var action = this.getInputData(0);
        var actionReceived = (action !== undefined && action !== null);
        
        // Log action reception status
        if (!actionReceived) {
            console.warn("[Snake Env] No action received from input - defaulting to action 0. Check controller connection.");
            action = 0;
        } else {
            // Validate action is a number in valid range
            if (typeof action !== 'number' || action < 0 || action > 3) {
                console.warn("[Snake Env] Invalid action received:", action, "- defaulting to 0");
                action = 0;
            } else {
                console.log("[Snake Env] Action received:", action, "(" + ["UP", "DOWN", "LEFT", "RIGHT"][action] + ")");
            }
        }

        // OUTPUT current signals (will be null until first backend response)
        this.setOutputData(0, this.properties.current_signals);
        this.setOutputData(1, this.properties.done || false);

        // Kick off next step
        if (window.BackendConnection && window.BackendConnection.isConnected()) {
            console.log("[Snake Env] Sending action to backend:", action);
            window.BackendConnection.sendCommand("step", { action: action })
                .then(function(response) {
                    if (response && response.signals) {
                        this.properties.current_signals = response.signals;
                        this.properties.alive = response.signals.alive !== 0;
                        this.properties.done = response.done || false;

                        if (response.done) {
                            this.properties.episodes++;
                        }
                    }
                }.bind(this))
                .catch(function(e) {
                    console.error("[Snake Env] Backend step error:", e);
                });
        }
    };

    SnakeEnvironmentNode.prototype.onAdded = function() {
        // Initialize environment on backend when node is added
        if (window.BackendConnection && window.BackendConnection.isConnected()) {
            window.BackendConnection.sendCommand("reset_env", { grid_size: this.properties.grid_size })
                .then(function(response) {
                    if (response && response.signals) {
                        this.properties.current_signals = response.signals;
                    }
                }.bind(this))
                .catch(function(e) {
                    console.error("[Snake Env] Reset error:", e);
                });
        }
    };
    
    SnakeEnvironmentNode.prototype.onDrawForeground = function(ctx) {
        // Draw stats on the node
        const stats = [
            `Ep: ${this.properties.episodes || 0}`,
            `Score: ${this.properties.current_score || 0}`,
            `Best: ${this.properties.best_score || 0}`
        ];
        
        // Set text style
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        
        // Draw stats text
        let yOffset = 25;
        stats.forEach((stat, index) => {
            ctx.fillText(stat, 10, yOffset + (index * 15));
        });
        
        // Draw alive/dead indicator
        const alive = this.properties.alive !== false;
        ctx.fillStyle = alive ? "#2ECC71" : "#E74C3C";
        ctx.beginPath();
        ctx.arc(this.size[0] - 15, 15, 5, 0, Math.PI * 2);
        ctx.fill();
    };
    
    LiteGraph.registerNodeType("environment/snake", SnakeEnvironmentNode);
    
    // VisualizeNode - Opens Pygame window to visualize environment
    function VisualizeNode() {
        // INPUT: Takes signals from environment node
        this.addInput("signals", "signal");

        // NO OUTPUTS - This node is just for visualization triggering

        // Visual properties
        this.color = "#3498DB";  // Blue for visualization
        this.bgcolor = "#2C3E50";
        this.size = [180, 80];

        this.properties = {
            name: "visualize_1"
        };
    }

    VisualizeNode.title = "Visualize";
    VisualizeNode.desc = "Visualizes environment in Pygame window (no outputs)";

    VisualizeNode.prototype.onExecute = function() {
        // This node just needs to exist - it doesn't process signals
        // Its existence triggers pygame window launch
        // No output needed
    };

    VisualizeNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "center";
            ctx.fillStyle = "#3498DB";
            ctx.font = "bold 12px Arial";
            ctx.fillText("Pygame Visualizer", this.size[0] / 2, this.size[1] - 20);
        }
    };

    LiteGraph.registerNodeType("environment/visualize", VisualizeNode);

    // CanvasVisualizeNode - Renders snake game in browser canvas (no Pygame needed)
    function CanvasVisualizeNode() {
        this.addInput("signals", "signal");

        this.color = "#9B59B6";
        this.bgcolor = "#2C3E50";
        this.size = [220, 240];

        this.properties = {
            grid_size: 10,
            cell_size: 18,
            head_x: 5,
            head_y: 5,
            food_x: 7,
            food_y: 3,
            alive: true,
            energy: 25,
            steps_alive: 0
        };
    }

    CanvasVisualizeNode.title = "Canvas Visualize";
    CanvasVisualizeNode.desc = "Renders snake game directly in browser";

    CanvasVisualizeNode.prototype.onExecute = function() {
        const signals = this.getInputData(0);
        if (signals) {
            this.properties.head_x = Math.floor(signals.head_x || 5);
            this.properties.head_y = Math.floor(signals.head_y || 5);
            this.properties.food_x = Math.floor(signals.food_x || 7);
            this.properties.food_y = Math.floor(signals.food_y || 3);
            this.properties.alive = signals.alive !== 0;
            this.properties.energy = Math.floor(signals.energy || 0);
            this.properties.steps_alive = Math.floor(signals.steps_alive || 0);
        }
        this.setDirtyCanvas(true);
    };

    CanvasVisualizeNode.prototype.onDrawForeground = function(ctx) {
        const p = this.properties;
        const cellSize = p.cell_size;
        const gridSize = p.grid_size;
        const offsetX = 10;
        const offsetY = 30;

        // Draw background
        ctx.fillStyle = "#1a1a2e";
        ctx.fillRect(offsetX, offsetY, gridSize * cellSize, gridSize * cellSize);

        // Draw grid lines
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= gridSize; i++) {
            ctx.beginPath();
            ctx.moveTo(offsetX + i * cellSize, offsetY);
            ctx.lineTo(offsetX + i * cellSize, offsetY + gridSize * cellSize);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(offsetX, offsetY + i * cellSize);
            ctx.lineTo(offsetX + gridSize * cellSize, offsetY + i * cellSize);
            ctx.stroke();
        }

        // Draw food
        ctx.fillStyle = "#e74c3c";
        ctx.beginPath();
        ctx.arc(
            offsetX + p.food_x * cellSize + cellSize / 2,
            offsetY + p.food_y * cellSize + cellSize / 2,
            cellSize / 2 - 2,
            0, Math.PI * 2
        );
        ctx.fill();

        // Draw snake head
        ctx.fillStyle = p.alive ? "#2ecc71" : "#666";
        ctx.fillRect(
            offsetX + p.head_x * cellSize + 1,
            offsetY + p.head_y * cellSize + 1,
            cellSize - 2,
            cellSize - 2
        );

        // Draw stats below grid
        const statsY = offsetY + gridSize * cellSize + 15;
        ctx.fillStyle = "#fff";
        ctx.font = "10px monospace";
        ctx.textAlign = "left";
        ctx.fillText(`Steps: ${p.steps_alive}  Energy: ${p.energy}`, offsetX, statsY);
        ctx.fillText(`${p.alive ? 'ALIVE' : 'DEAD'}`, offsetX, statsY + 12);
    };

    LiteGraph.registerNodeType("environment/canvas_visualize", CanvasVisualizeNode);
}
