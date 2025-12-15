// Global references for graph and canvas
let globalGraph = null;
let globalCanvasNode = null;

// Global Backend Connection Manager - Direct WebSocket connection to backend
let BackendConnection = {
    ws: null,
    url: "ws://localhost:8000/ws",
    is_connected: false,
    reconnect_timer: null,
    reconnect_attempts: 0,
    max_reconnect_attempts: 10,
    reconnect_interval: 3000,
    message_callbacks: [],
    pending_commands: [],
    pending_promises: {},  // Map of command type -> resolve function
    request_id: 0,
    
    connect: function() {
        if (this.is_connected && this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log("[BackendConnection] Already connected");
            return;
        }
        
        if (this.ws) {
            this.disconnect();
        }
        
        try {
            console.log("[BackendConnection] Connecting to", this.url);
            this.ws = new WebSocket(this.url);
            
            var self = this;
            
            this.ws.onopen = function() {
                console.log("[BackendConnection] Connected to backend");
                self.is_connected = true;
                self.reconnect_attempts = 0;
                
                // Process any pending commands
                while (self.pending_commands.length > 0) {
                    var cmd = self.pending_commands.shift();
                    self.sendCommand(cmd.command, cmd.data);
                }
            };
            
            this.ws.onmessage = function(event) {
                try {
                    var response = JSON.parse(event.data);

                    // Resolve pending promise if exists
                    if (response.request_id && self.pending_promises[response.request_id]) {
                        self.pending_promises[response.request_id](response);
                        delete self.pending_promises[response.request_id];
                    }

                    // Notify all callbacks
                    for (var i = 0; i < self.message_callbacks.length; i++) {
                        if (self.message_callbacks[i]) {
                            try {
                                self.message_callbacks[i](response);
                            } catch (error) {
                                console.error("[BackendConnection] Callback error:", error);
                            }
                        }
                    }
                } catch (error) {
                    console.error("[BackendConnection] Error parsing message:", error);
                }
            };
            
            this.ws.onerror = function(error) {
                console.error("[BackendConnection] WebSocket error:", error);
                self.is_connected = false;
            };
            
            this.ws.onclose = function() {
                console.log("[BackendConnection] WebSocket closed");
                self.is_connected = false;
                self.ws = null;
                
                // Attempt to reconnect
                if (self.reconnect_attempts < self.max_reconnect_attempts) {
                    self.reconnect_attempts++;
                    console.log("[BackendConnection] Attempting to reconnect (" + self.reconnect_attempts + "/" + self.max_reconnect_attempts + ")");
                    self.reconnect_timer = setTimeout(function() {
                        self.connect();
                    }, self.reconnect_interval);
                } else {
                    console.error("[BackendConnection] Max reconnect attempts reached");
                }
            };
            
        } catch (error) {
            console.error("[BackendConnection] Connection error:", error);
            this.is_connected = false;
        }
    },
    
    disconnect: function() {
        if (this.reconnect_timer) {
            clearTimeout(this.reconnect_timer);
            this.reconnect_timer = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        this.is_connected = false;
    },
    
    sendCommand: function(command, data) {
        var self = this;
        return new Promise(function(resolve, reject) {
            if (!self.is_connected || !self.ws || self.ws.readyState !== WebSocket.OPEN) {
                console.warn("[BackendConnection] Not connected, queueing command:", command);
                self.pending_commands.push({ command: command, data: data || {}, resolve: resolve });

                // Try to connect if not already attempting
                if (!self.ws || self.ws.readyState === WebSocket.CLOSED) {
                    self.connect();
                }
                return;
            }

            try {
                self.request_id++;
                var reqId = self.request_id;
                var message = Object.assign({ cmd: command, request_id: reqId }, data || {});

                // Store resolve function for this request
                self.pending_promises[reqId] = resolve;

                self.ws.send(JSON.stringify(message));

                // Timeout after 5 seconds
                setTimeout(function() {
                    if (self.pending_promises[reqId]) {
                        delete self.pending_promises[reqId];
                        resolve({ success: false, error: "timeout" });
                    }
                }, 5000);
            } catch (error) {
                console.error("[BackendConnection] Error sending command:", error);
                reject(error);
            }
        });
    },

    isConnected: function() {
        return this.is_connected && this.ws && this.ws.readyState === WebSocket.OPEN;
    },
    
    onMessage: function(callback) {
        if (this.message_callbacks.indexOf(callback) === -1) {
            this.message_callbacks.push(callback);
        }
    },
    
    removeMessageCallback: function(callback) {
        var index = this.message_callbacks.indexOf(callback);
        if (index !== -1) {
            this.message_callbacks.splice(index, 1);
        }
    }
};

// Chart Manager
let ChartManager = {
    chartElements: new Map(), // Use a map to store { node_id: { container, title, canvas } }
    panel: null,
    content: null,
    title: null,
    toggleBtn: null,

    init: function() {
        this.panel = document.getElementById("chart-panel");
        this.content = document.getElementById("chart-panel-content");
        this.title = document.getElementById("chart-panel-title");
        this.toggleBtn = document.getElementById("chart-panel-toggle-btn");

        if (this.toggleBtn) {
            this.toggleBtn.addEventListener("click", () => this.togglePanel());
        }

        // Start a rendering loop
        this.renderLoop();
    },

    register: function(node) {
        if (!this.chartElements.has(node.id)) {
            const container = document.createElement("div");
            container.className = "chart-container";
            container.id = "chart-container-" + node.id;

            const title = document.createElement("h3");
            title.textContent = node.properties.name || ("Chart " + node.id);
            container.appendChild(title);

            const canvas = document.createElement("canvas");
canvas.className = "chart-canvas";
            canvas.width = this.content.clientWidth > 0 ? this.content.clientWidth - 20 : 300;
            canvas.height = 150;
            container.appendChild(canvas);

            this.content.appendChild(container);
            
            this.chartElements.set(node.id, { node, container, title, canvas });
        }
        this.updatePanelVisibility();
    },

    unregister: function(node) {
        if (this.chartElements.has(node.id)) {
            const { container } = this.chartElements.get(node.id);
            if (container) {
                this.content.removeChild(container);
            }
            this.chartElements.delete(node.id);
        }
        this.updatePanelVisibility();
    },

    togglePanel: function() {
        this.panel.classList.toggle("collapsed");
        this.toggleBtn.textContent = this.panel.classList.contains("collapsed") ? "▶" : "◀";
    },

    updatePanelVisibility: function() {
        if (this.chartElements.size > 0) {
            this.panel.style.display = "flex";
        } else {
            this.panel.style.display = "none";
        }
        this.updateTitle();
    },

    updateTitle: function() {
        if (this.chartElements.size === 1) {
            const { node } = this.chartElements.values().next().value;
            this.title.textContent = node.properties.name;
        } else if (this.chartElements.size > 1) {
            this.title.textContent = "Charts (" + this.chartElements.size + ")";
        } else {
            this.title.textContent = "Chart";
        }
    },

    renderLoop: function() {
        requestAnimationFrame(() => this.renderLoop());
        if (this.chartElements.size > 0 && !this.panel.classList.contains("collapsed")) {
            for (const [nodeId, { node, canvas }] of this.chartElements.entries()) {
                this.drawChartOnCanvas(canvas, node);
            }
        }
    },

    drawChartOnCanvas: function(canvas, node) {
        const ctx = canvas.getContext("2d");
        const chart_width = canvas.width;
        const chart_height = canvas.height;

        let history_A, history_B, current_value_A, current_value_B, line_color_A, line_color_B, show_value;

        // Determine if it's a DualChartMonitorNode or ChartMonitorNode
        if (node.signal_history_A && node.signal_history_B) { // DualChartMonitorNode
            history_A = node.signal_history_A;
            history_B = node.signal_history_B;
            current_value_A = node.current_value_A;
            current_value_B = node.current_value_B;
            line_color_A = node.properties.line_color_A;
            line_color_B = node.properties.line_color_B;
            show_value = node.properties.show_value;

            // Combine histories for auto-scaling
            const combined_history = [];
            if (history_A.length > 0) combined_history.push(...history_A);
            if (history_B.length > 0) combined_history.push(...history_B);

            if (combined_history.length < 2) {
                ctx.fillStyle = "#888";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Waiting for data...", chart_width / 2, chart_height / 2);
                return;
            }

            var y_min, y_max;
            if (node.properties.auto_scale) {
                y_min = node.observed_min; // Should already be calculated in onExecute
                y_max = node.observed_max; // Should already be calculated in onExecute
                var range = y_max - y_min;
                if (range < 0.01) range = 1; // Prevent division by zero
                y_min -= range * 0.1;
                y_max += range * 0.1;
            } else {
                y_min = node.properties.min_value;
                y_max = node.properties.max_value;
            }

            var y_range = y_max - y_min;
            if (y_range < 0.01) y_range = 1;
            
            // Clear and draw background/grid
            ctx.clearRect(0, 0, chart_width, chart_height);
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, chart_width, chart_height);
            
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i <= 4; i++) {
                const y = (chart_height / 4) * i;
                ctx.moveTo(0, y);
                ctx.lineTo(chart_width, y);
            }
            for (let i = 0; i <= 4; i++) {
                const x = (chart_width / 4) * i;
                ctx.moveTo(x, 0);
                ctx.lineTo(x, chart_height);
            }
            ctx.stroke();

            if (y_min < 0 && y_max > 0) {
                const zero_y = chart_height * (1 - (0 - y_min) / y_range);
                ctx.strokeStyle = "#555";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, zero_y);
                ctx.lineTo(chart_width, zero_y);
                ctx.stroke();
            }

            // Function to draw a signal line
            const drawSignal = (history, color) => {
                if (history.length < 2) return;
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                const hist_len = history.length;
                const max_index = Math.max(1, hist_len - 1);
                
                for (let i = 0; i < hist_len; i++) {
                    const value = history[i];
                    const x = (i / max_index) * chart_width;
                    const normalized = (value - y_min) / y_range;
                    const y = chart_height * (1 - normalized);
                    
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            };

            drawSignal(history_A, line_color_A);
            drawSignal(history_B, line_color_B);

            if (show_value) {
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "left";
                
                ctx.fillStyle = line_color_A;
                ctx.fillText("A: " + current_value_A.toFixed(3), 5, chart_height - 20);
                
                ctx.fillStyle = line_color_B;
                ctx.fillText("B: " + current_value_B.toFixed(3), 5, chart_height - 5);
            }
        } else { // ChartMonitorNode
            const history = node.signal_history;
            const current_value = node.current_value;
            const line_color = node.properties.line_color;
            show_values = node.properties.show_value;

            if (history.length < 2) {
                ctx.fillStyle = "#888";
                ctx.font = "12px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Waiting for data...", chart_width / 2, chart_height / 2);
                return;
            }

            var y_min, y_max;
            if (node.properties.auto_scale) {
                y_min = node.observed_min;
                y_max = node.observed_max;
                var range = y_max - y_min;
                if (range < 0.01) range = 1;
                y_min -= range * 0.1;
                y_max += range * 0.1;
            } else {
                y_min = node.properties.min_value;
                y_max = node.properties.max_value;
            }
            
            var y_range = y_max - y_min;
            if (y_range < 0.01) y_range = 1;

            ctx.clearRect(0, 0, chart_width, chart_height);
            
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(0, 0, chart_width, chart_height);
            
            ctx.strokeStyle = "#333";
            ctx.lineWidth = 1;
            ctx.beginPath();
            for (let i = 0; i <= 4; i++) {
                const y = (chart_height / 4) * i;
                ctx.moveTo(0, y);
                ctx.lineTo(chart_width, y);
            }
            for (let i = 0; i <= 4; i++) {
                const x = (chart_width / 4) * i;
                ctx.moveTo(x, 0);
                ctx.lineTo(x, chart_height);
            }
            ctx.stroke();

            if (y_min < 0 && y_max > 0) {
                const zero_y = chart_height * (1 - (0 - y_min) / y_range);
                ctx.strokeStyle = "#555";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(0, zero_y);
                ctx.lineTo(chart_width, zero_y);
                ctx.stroke();
            }
            
            ctx.strokeStyle = line_color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            const hist_len = history.length;
            const max_index = Math.max(1, hist_len - 1);
            
            for (let i = 0; i < hist_len; i++) {
                const value = history[i];
                const x = (i / max_index) * chart_width;
                const normalized = (value - y_min) / y_range;
                const y = chart_height * (1 - normalized);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            if (show_values) {
                ctx.fillStyle = line_color;
                ctx.font = "bold 12px Arial";
                ctx.textAlign = "left";
                const value_text = "Value: " + current_value.toFixed(3);
                ctx.fillText(value_text, 5, chart_height - 5);
            }
        }
    }
};



// Console Output Panel Manager
let ConsoleLogManager = {
    logs: [],
    maxLogs: 1000,
    executionOrder: 0,
    
    addLog: function(level, label, timestamp, type, data, nodeId) {
        // Add log with execution order
        var logEntry = {
            order: this.executionOrder++,
            level: level,
            label: label,
            timestamp: timestamp,
            type: type,
            data: data,
            nodeId: nodeId,
            time: Date.now()
        };
        
        this.logs.push(logEntry);
        
        // Limit log history
        if (this.logs.length > this.maxLogs) {
            this.logs.shift();
        }
        
        // Render to panel
        this.renderToPanel();
    },
    
    renderToPanel: function() {
        var consoleContent = document.getElementById("console-content");
        if (!consoleContent) return;

        const buffer = 10; // Small buffer for scroll position
        const isScrolledToBottom = consoleContent.scrollTop + consoleContent.clientHeight >= consoleContent.scrollHeight - buffer;
        
        // Sort by execution order
        var sortedLogs = this.logs.slice().sort(function(a, b) {
            return a.order - b.order;
        });
        
        // Clear and rebuild
        consoleContent.innerHTML = "";
        
        sortedLogs.forEach(function(log) {
            var entry = document.createElement("div");
            entry.className = "console-log-entry " + log.level;
            
            var parts = [];
            
            // Timestamp
            if (log.timestamp) {
                var ts = document.createElement("span");
                ts.className = "console-log-timestamp";
                ts.textContent = log.timestamp + " ";
                entry.appendChild(ts);
            }
            
            // Label
            if (log.label) {
                var lbl = document.createElement("span");
                lbl.className = "console-log-label";
                lbl.textContent = log.label + " ";
                entry.appendChild(lbl);
            }
            
            // Type
            if (log.type) {
                var typ = document.createElement("span");
                typ.className = "console-log-type";
                typ.textContent = log.type + " ";
                entry.appendChild(typ);
            }
            
            // Data
            var dataSpan = document.createElement("span");
            if (typeof log.data === "object" && log.data !== null) {
                dataSpan.textContent = JSON.stringify(log.data, null, 2);
            } else {
                dataSpan.textContent = String(log.data);
            }
            entry.appendChild(dataSpan);
            
            consoleContent.appendChild(entry);
        });
        
        if (isScrolledToBottom) {
            consoleContent.scrollTop = consoleContent.scrollHeight;
        }
    },
    
    clear: function() {
        this.logs = [];
        this.executionOrder = 0;
        var consoleContent = document.getElementById("console-content");
        if (consoleContent) {
            consoleContent.innerHTML = "";
        }
    }
};

// Execution Controller
let ExecutionController = {
    isPlaying: false,
    updateRate: 30, // FPS
    intervalId: null,
    graph: null,
    
    init: function(graph) {
        this.graph = graph;
        this.setupUI();
        // Start paused
        this.pause();
    },
    
    setupUI: function() {
        var playPauseBtn = document.getElementById("play-pause-btn");
        var stepBtn = document.getElementById("step-btn");
        var resetBtn = document.getElementById("reset-btn");
        var rateSlider = document.getElementById("rate-slider");
        var rateInput = document.getElementById("rate-input");
        var statusSpan = document.getElementById("execution-status");
        
        if (!playPauseBtn || !stepBtn || !resetBtn || !rateSlider || !rateInput || !statusSpan) {
            console.error("Execution control UI elements not found");
            return;
        }
        
        // Play/Pause button
        playPauseBtn.addEventListener("click", function() {
            if (ExecutionController.isPlaying) {
                ExecutionController.pause();
            } else {
                ExecutionController.play();
            }
        });
        
        // Step button
        stepBtn.addEventListener("click", function() {
            ExecutionController.step();
        });
        
        // Reset button
        resetBtn.addEventListener("click", function() {
            ExecutionController.reset();
        });
        
        // Rate slider
        rateSlider.addEventListener("input", function(e) {
            var rate = parseInt(e.target.value);
            ExecutionController.setRate(rate);
            rateInput.value = rate;
        });
        
        // Rate input
        rateInput.addEventListener("change", function(e) {
            var rate = parseInt(e.target.value);
            if (rate < 1) rate = 1;
            if (rate > 60) rate = 60;
            ExecutionController.setRate(rate);
            rateSlider.value = rate;
            e.target.value = rate;
        });
        
        // Initialize UI
        this.updateUI();
    },
    
    play: function() {
        if (this.isPlaying) return;
        
        // Check if visualize node exists - only launch pygame if it does
        var visualizeNode = findVisualizeNodeInGraph();
        if (visualizeNode) {
            // Visualize node exists - find connected environment node
            var envNode = findEnvironmentNodeFromVisualize(visualizeNode);
            if (!envNode) {
                // Visualize node exists but no environment node connected
                alert("Visualize node found but no Environment node connected. Please connect an Environment node to the Visualize node.");
                return;
            }
            
            // Send launch command to backend
            sendLaunchEnvironmentCommand(envNode);
        } else {
            // No visualize node - execution can continue without pygame
            console.log("No Visualize node found - Pygame window will not open");
        }
        
        this.isPlaying = true;
        // Enable graph execution
        if (this.graph) {
            this.graph._executionEnabled = true;
            // Also try to stop any internal LiteGraph execution loop
            if (this.graph.stop) {
                this.graph.stop();
            }
        }
        this.startExecutionLoop();
        
        // Send play command to backend
        sendPlayCommandToBackend();
        
        this.updateUI();
    },
    
    pause: function() {
        if (!this.isPlaying) return;
        
        this.isPlaying = false;
        // Disable graph execution
        if (this.graph) {
            this.graph._executionEnabled = false;
            // Stop any internal LiteGraph execution loop
            if (this.graph.stop) {
                this.graph.stop();
            }
        }
        this.stopExecutionLoop();
        
        // Send pause command to backend
        sendPauseCommandToBackend();
        
        this.updateUI();
    },
    
    step: function() {
        // Send step command to backend first
        sendStepCommandToBackend();
        
        // Execute graph once (temporarily enable execution)
        if (this.graph) {
            var wasEnabled = this.graph._executionEnabled;
            this.graph._executionEnabled = true;
            this.graph.runStep();
            this.graph._executionEnabled = wasEnabled;
        }
    },
    
    reset: function() {
        // Pause first
        this.pause();
        
        // Reset all nodes by clearing their state
        if (this.graph && this.graph._nodes) {
            for (var i = 0; i < this.graph._nodes.length; i++) {
                var node = this.graph._nodes[i];
                // Clear any cached values or state
                if (node.onReset) {
                    node.onReset();
                }
                // Clear output data
                if (node.outputs) {
                    for (var j = 0; j < node.outputs.length; j++) {
                        node.outputs[j].data = null;
                    }
                }
            }
        }
        
        // Clear console logs
        ConsoleLogManager.clear();
        
        // Force graph redraw
        if (this.graph && this.graph.setDirty) {
            this.graph.setDirty(true);
        }
    },
    
    setRate: function(rate) {
        this.updateRate = Math.max(1, Math.min(60, rate));
        
        // Send FPS update to backend
        sendFPSCommandToBackend(rate);
        
        // If currently playing, restart the loop with new rate
        if (this.isPlaying) {
            this.stopExecutionLoop();
            this.startExecutionLoop();
        }
        
        this.updateUI();
    },
    
    startExecutionLoop: function() {
        this.stopExecutionLoop(); // Clear any existing loop
        
        if (!this.graph) return;
        
        var intervalMs = 1000 / this.updateRate;
        
        this.intervalId = setInterval(function() {
            if (ExecutionController.graph && ExecutionController.isPlaying) {
                ExecutionController.graph.runStep();
            }
        }, intervalMs);
    },
    
    stopExecutionLoop: function() {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
    },
    
    updateUI: function() {
        var playPauseBtn = document.getElementById("play-pause-btn");
        var statusSpan = document.getElementById("execution-status");
        var rateSlider = document.getElementById("rate-slider");
        var rateInput = document.getElementById("rate-input");
        
        if (playPauseBtn) {
            if (this.isPlaying) {
                playPauseBtn.textContent = "⏸";
                playPauseBtn.classList.add("playing");
            } else {
                playPauseBtn.textContent = "▶";
                playPauseBtn.classList.remove("playing");
            }
        }
        
        if (statusSpan) {
            if (this.isPlaying) {
                statusSpan.textContent = "Playing";
                statusSpan.className = "execution-status playing";
            } else {
                statusSpan.textContent = "Paused";
                statusSpan.className = "execution-status paused";
            }
        }
        
        if (rateSlider) {
            rateSlider.value = this.updateRate;
        }
        
        if (rateInput) {
            rateInput.value = this.updateRate;
        }
    }
};

function initializeExecutionController(graph) {
    ExecutionController.init(graph);
    
    // Initialize backend connection - connect automatically
    BackendConnection.connect();
}

// Helper functions for finding visualize and environment nodes
function findVisualizeNodeInGraph() {
    if (!globalGraph) return null;
    
    var nodes = globalGraph._nodes;
    if (!nodes) return null;
    
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i] && nodes[i].type === "environment/visualize") {
            return nodes[i];
        }
    }
    return null;
}

function findEnvironmentNodeFromVisualize(visualizeNode) {
    if (!visualizeNode || !globalGraph) return null;
    
    // Check if visualize node has a connected input
    if (visualizeNode.inputs && visualizeNode.inputs.length > 0) {
        var input = visualizeNode.inputs[0];
        if (input && input.link !== null && input.link !== undefined) {
            // Find the node that's connected to this input
            var nodes = globalGraph._nodes;
            if (!nodes) return null;
            
            for (var i = 0; i < nodes.length; i++) {
                if (nodes[i] && nodes[i].outputs) {
                    for (var j = 0; j < nodes[i].outputs.length; j++) {
                        if (nodes[i].outputs[j].links && nodes[i].outputs[j].links.indexOf(input.link) !== -1) {
                            // Found the connected node - check if it's an environment node
                            if (nodes[i].type && nodes[i].type.startsWith("environment/") && nodes[i].type !== "environment/visualize") {
                                return nodes[i];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fallback: find any environment node (not visualize)
    var nodes = globalGraph._nodes;
    if (!nodes) return null;
    
    for (var i = 0; i < nodes.length; i++) {
        if (nodes[i] && nodes[i].type && nodes[i].type.startsWith("environment/") && nodes[i].type !== "environment/visualize") {
            return nodes[i];
        }
    }
    
    return null;
}

function findEnvironmentNodeInGraph() {
    // This function is kept for backward compatibility but now checks for visualize node first
    var visualizeNode = findVisualizeNodeInGraph();
    if (visualizeNode) {
        return findEnvironmentNodeFromVisualize(visualizeNode);
    }
    return null;
}

function showNoEnvironmentModal() {
    // Simple alert for now (can be fancier modal later)
    alert("No Visualize node in graph. Please add a Visualize node connected to an Environment node before trying to run the model.");
}

function sendLaunchEnvironmentCommand(envNode) {
    // Extract environment type from node type (e.g., "snake" from "environment/snake")
    var envType = "snake"; // Default
    if (envNode.type) {
        var parts = envNode.type.split("/");
        if (parts.length > 1) {
            envType = parts[1];
        }
    }
    
    // Get grid_size from node properties
    var gridSize = 10; // Default
    if (envNode.properties && envNode.properties.grid_size !== undefined) {
        gridSize = envNode.properties.grid_size;
    }
    
    console.log("[BackendConnection] Sending launch_env command:", { env_type: envType, grid_size: gridSize });
    
    // Send via global backend connection
    return BackendConnection.sendCommand("launch_env", {
        env_type: envType,
        grid_size: gridSize
    });
}

function sendPlayCommandToBackend() {
    BackendConnection.sendCommand("play_env", {});
}

function sendPauseCommandToBackend() {
    BackendConnection.sendCommand("pause_env", {});
}

function sendStepCommandToBackend() {
    // Step with action 0 (up) for now - later this will come from controller
    BackendConnection.sendCommand("step", { action: 0 });
}

function sendFPSCommandToBackend(fps) {
    BackendConnection.sendCommand("set_fps", { fps: fps });
}

// Graph Persistence Functions
function saveGraphState(graph) {
    try {
        const data = graph.serialize();
        localStorage.setItem("genreg_graph_state", JSON.stringify(data));
        console.log("Graph state saved to localStorage");
        return true;
    } catch (error) {
        console.error("Error saving graph state:", error);
        return false;
    }
}

function saveGraphToLocalStorage(graph) {
    const success = saveGraphState(graph);
    if (success) {
        alert("Graph saved successfully!");
    } else {
        alert("Error saving graph. Check console for details.");
    }
}

function loadGraphFromLocalStorage(graph) {
    if (confirm("This will replace the current graph. Continue?")) {
        const success = loadGraphState(graph);
        if (success) {
            // Force canvas to redraw
            if (globalCanvasNode) {
                globalCanvasNode.setDirty(true, true);
            }
            alert("Graph loaded successfully!");
        } else {
            alert("No saved graph found or error loading graph.");
        }
    }
}

function exportGraphToJSON(graph) {
    try {
        const data = graph.serialize();
        const jsonString = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonString], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "genreg_graph_" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("Graph exported to JSON");
        alert("Graph exported successfully!");
    } catch (error) {
        console.error("Error exporting graph:", error);
        alert("Error exporting graph. Check console for details.");
    }
}

function importGraphFromJSON(graph, file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            if (confirm("This will replace the current graph. Continue?")) {
                graph.configure(data);
                // Sync widget values with properties after loading
                // Use setTimeout to ensure widgets are fully initialized
                setTimeout(function() {
                    syncAllNodeWidgets(graph);
                    // Also save to localStorage
                    localStorage.setItem("genreg_graph_state", JSON.stringify(data));
                    // Force canvas to redraw
                    if (globalCanvasNode) {
                        globalCanvasNode.setDirty(true, true);
                    }
                }, 100);
                console.log("Graph imported from JSON");
                alert("Graph imported successfully!");
            }
        } catch (error) {
            console.error("Error importing graph:", error);
            alert("Error importing graph. Invalid JSON file. Check console for details.");
        }
    };
    reader.onerror = function() {
        alert("Error reading file.");
    };
    reader.readAsText(file);
}

function syncNodeWidgets(node) {
    // Sync widget values with properties after loading
    if (!node.widgets || !node.properties) return;
    
    // Iterate through all properties and try to match them to widgets
    for (var propName in node.properties) {
        if (!node.properties.hasOwnProperty(propName)) continue;
        
        var propValue = node.properties[propName];
        if (propValue === undefined || propValue === null) continue;
        
        // Find matching widget by trying different name variations
        var widget = null;
        
        // Try exact match
        for (var i = 0; i < node.widgets.length; i++) {
            if (node.widgets[i] && node.widgets[i].name === propName) {
                widget = node.widgets[i];
                break;
            }
        }
        
        // Try case-insensitive match
        if (!widget) {
            var lowerPropName = propName.toLowerCase();
            for (var i = 0; i < node.widgets.length; i++) {
                if (node.widgets[i] && node.widgets[i].name && 
                    node.widgets[i].name.toLowerCase().replace(/\s+/g, '_') === lowerPropName) {
                    widget = node.widgets[i];
                    break;
                }
            }
        }
        
        // Try matching by converting widget name to property name format
        if (!widget) {
            for (var i = 0; i < node.widgets.length; i++) {
                var w = node.widgets[i];
                if (!w || !w.name) continue;
                
                // Convert widget name to possible property names
                var wLower = w.name.toLowerCase().replace(/\s+/g, '_');
                var wCamel = w.name.charAt(0).toLowerCase() + w.name.slice(1).replace(/\s+([a-z])/g, function(g) { return g[1].toUpperCase(); });
                
                if (propName === wLower || propName === wCamel || propName.toLowerCase() === wLower) {
                    widget = w;
                    break;
                }
            }
        }
        
        // Update widget if found
        if (widget) {
            widget.value = propValue;
            
            // Update widget's input element if it exists
            if (widget.input) {
                if (widget.input.type === "checkbox" || widget.type === "toggle") {
                    widget.input.checked = !!propValue;
                } else {
                    widget.input.value = propValue;
                }
            }
            
            // Some widgets store value in options
            if (widget.options) {
                widget.options.value = propValue;
            }
        }
    }
}

function syncAllNodeWidgets(graph) {
    // Sync all node widgets with their properties
    if (graph && graph._nodes) {
        for (var i = 0; i < graph._nodes.length; i++) {
            syncNodeWidgets(graph._nodes[i]);
        }
    }
}

function loadGraphState(graph) {
    try {
        const saved = localStorage.getItem("genreg_graph_state");
        if (saved) {
            const data = JSON.parse(saved);
            graph.configure(data);
            // Sync widget values with properties after loading
            // Use setTimeout to ensure widgets are fully initialized
            setTimeout(function() {
                syncAllNodeWidgets(graph);
                if (globalCanvasNode) {
                    globalCanvasNode.setDirty(true, true);
                }
            }, 100);
            console.log("Graph state loaded from localStorage");
            return true;
        }
    } catch (error) {
        console.error("Error loading graph state:", error);
        // Clear corrupted data
        localStorage.removeItem("genreg_graph_state");
    }
    return false;
}

function setupGraphPersistence(graph, canvas_node) {
    // Debounced save for frequent events (like dragging)
    let saveTimeout = null;
    const debouncedSave = function() {
        if (saveTimeout) {
            clearTimeout(saveTimeout);
        }
        saveTimeout = setTimeout(function() {
            saveGraphState(graph);
        }, 500); // Save 500ms after last change
    };
    
    // Save on node changes (add, remove)
    if (graph.add) {
        const originalAdd = graph.add.bind(graph);
        graph.add = function(node) {
            const result = originalAdd(node);
            debouncedSave();
            return result;
        };
    }
    
    if (graph.remove) {
        const originalRemove = graph.remove.bind(graph);
        graph.remove = function(node) {
            const result = originalRemove(node);
            debouncedSave();
            return result;
        };
    }
    
    // Save on connection changes
    // LiteGraph handles connections through canvas, so we'll track via canvas events
    // The graph.connect/disconnect might not exist, so we'll use canvas connection events
    
    // Save on connection changes via canvas
    if (canvas_node) {
        // Track when connections are made
        const originalOnConnectionStart = canvas_node.onConnectionStart;
        canvas_node.onConnectionStart = function() {
            if (originalOnConnectionStart) {
                originalOnConnectionStart.apply(canvas_node, arguments);
            }
        };
        
        const originalOnConnectionEnd = canvas_node.onConnectionEnd;
        canvas_node.onConnectionEnd = function() {
            if (originalOnConnectionEnd) {
                originalOnConnectionEnd.apply(canvas_node, arguments);
            }
            debouncedSave(); // Save when connection is completed
        };
        
        // Save on node position changes (dragging)
        // Listen to canvas mouse events for node dragging
        const originalOnMouseMove = canvas_node.onMouseMove;
        canvas_node.onMouseMove = function(e) {
            if (originalOnMouseMove) {
                originalOnMouseMove.call(canvas_node, e);
            }
            // If a node is being dragged, save after a delay
            if (canvas_node.node_dragged) {
                debouncedSave();
            }
        };
        
        // Save when mouse is released after dragging
        const originalOnMouseUp = canvas_node.onMouseUp;
        canvas_node.onMouseUp = function(e) {
            if (originalOnMouseUp) {
                originalOnMouseUp.call(canvas_node, e);
            }
            debouncedSave();
        };
        
        // Save on property changes (when widgets are modified)
        const originalOnNodeChanged = canvas_node.onNodeChanged;
        canvas_node.onNodeChanged = function(node) {
            if (originalOnNodeChanged) {
                originalOnNodeChanged.call(canvas_node, node);
            }
            debouncedSave();
        };
    }
    
    // Save on graph property changes
    if (graph.onNodeAdded) {
        const originalOnNodeAdded = graph.onNodeAdded;
        graph.onNodeAdded = function(node) {
            if (originalOnNodeAdded) {
                originalOnNodeAdded.call(this, node);
            }
            debouncedSave();
        };
    }
    
    if (graph.onNodeRemoved) {
        const originalOnNodeRemoved = graph.onNodeRemoved;
        graph.onNodeRemoved = function(node) {
            if (originalOnNodeRemoved) {
                originalOnNodeRemoved.call(this, node);
            }
            debouncedSave();
        };
    }
}

// File Menu Setup
function setupFileMenu(graph) {
    const fileMenuBtn = document.getElementById("file-menu-btn");
    const fileMenuDropdown = document.getElementById("file-menu-dropdown");
    const fileSave = document.getElementById("file-save");
    const fileLoad = document.getElementById("file-load");
    const fileExport = document.getElementById("file-export");
    const fileImport = document.getElementById("file-import");
    const fileImportInput = document.getElementById("file-import-input");
    
    if (!fileMenuBtn || !fileMenuDropdown) {
        console.error("File menu elements not found");
        return;
    }
    
    // Show dropdown on hover
    fileMenuBtn.addEventListener("mouseenter", function() {
        fileMenuDropdown.style.display = "block";
    });
    
    fileMenuBtn.addEventListener("mouseleave", function() {
        // Delay hiding to allow moving to dropdown
        setTimeout(function() {
            if (!fileMenuDropdown.matches(":hover")) {
                fileMenuDropdown.style.display = "none";
            }
        }, 100);
    });
    
    fileMenuDropdown.addEventListener("mouseleave", function() {
        fileMenuDropdown.style.display = "none";
    });
    
    // Save to localStorage
    if (fileSave) {
        fileSave.addEventListener("click", function() {
            saveGraphToLocalStorage(graph);
            fileMenuDropdown.style.display = "none";
        });
    }
    
    // Load from localStorage
    if (fileLoad) {
        fileLoad.addEventListener("click", function() {
            loadGraphFromLocalStorage(graph);
            fileMenuDropdown.style.display = "none";
        });
    }
    
    // Export to JSON file
    if (fileExport) {
        fileExport.addEventListener("click", function() {
            exportGraphToJSON(graph);
            fileMenuDropdown.style.display = "none";
        });
    }
    
    // Import from JSON file
    if (fileImport && fileImportInput) {
        fileImport.addEventListener("click", function() {
            fileImportInput.click();
            fileMenuDropdown.style.display = "none";
        });
        
        fileImportInput.addEventListener("change", function(e) {
            const file = e.target.files[0];
            if (file) {
                importGraphFromJSON(graph, file);
            }
            // Reset input so same file can be selected again
            e.target.value = "";
        });
    }
}

// Wait for DOM and LiteGraph to be ready
document.addEventListener("DOMContentLoaded", function() {
    // Check if LiteGraph is loaded
    if (typeof LiteGraph === "undefined") {
        console.error("LiteGraph library not loaded!");
        return;
    }

    // Enable UUIDs to avoid ID conflicts
    LiteGraph.use_uuids = true;

    // Initialize LiteGraph
    let canvas = document.getElementById("litegraph-canvas");
    if (!canvas) {
        console.error("Canvas element not found!");
        return;
    }

    globalGraph = new LGraph();
    globalCanvasNode = new LGraphCanvas("#litegraph-canvas", globalGraph);
    
    // Disable auto-execution - we'll control it manually
    if (globalGraph.config) {
        globalGraph.config.mode = LGraph.MODE_NEVER; // Don't auto-execute
    }
    
    // Override graph.runStep to only execute when we allow it
    var originalRunStep = globalGraph.runStep;
    if (originalRunStep) {
        globalGraph.runStep = function() {
            // Check execution flag - if disabled, do nothing
            if (!this._executionEnabled) {
                return;
            }
            return originalRunStep.call(this);
        };
    }
    
    // Also override executeAllNodes if it exists (some LiteGraph versions use this)
    if (globalGraph.executeAllNodes) {
        var originalExecuteAllNodes = globalGraph.executeAllNodes;
        globalGraph.executeAllNodes = function() {
            if (!this._executionEnabled) {
                return;
            }
            return originalExecuteAllNodes.call(this);
        };
    }
    
    // Override onExecute if graph has it
    if (globalGraph.onExecute) {
        var originalOnExecute = globalGraph.onExecute;
        globalGraph.onExecute = function() {
            if (!this._executionEnabled) {
                return;
            }
            return originalOnExecute.call(this);
        };
    }
    
    // Store execution control flag - start disabled
    globalGraph._executionEnabled = false;
    
    // Also prevent canvas node from auto-executing
    if (globalCanvasNode) {
        // Override the canvas update loop to prevent auto-execution
        var originalDraw = globalCanvasNode.draw;
        if (originalDraw) {
            globalCanvasNode.draw = function() {
                // Always allow drawing/rendering
                var wasEnabled = globalGraph._executionEnabled;
                // Temporarily disable execution during draw
                globalGraph._executionEnabled = false;
                try {
                    originalDraw.call(this);
                } finally {
                    // Restore execution state
                    globalGraph._executionEnabled = wasEnabled;
                }
            };
        }
        
        // Override process method if it exists
        var originalProcess = globalCanvasNode.process;
        if (originalProcess) {
            globalCanvasNode.process = function() {
                // Only process/execute if execution is enabled
                if (globalGraph._executionEnabled) {
                    return originalProcess.call(this);
                }
                // Still allow rendering, just not execution
                if (this.draw) {
                    this.draw();
                }
                return;
            };
        }
        
        // Override onExecute if canvas has it
        if (globalCanvasNode.onExecute) {
            var originalOnExecute = globalCanvasNode.onExecute;
            globalCanvasNode.onExecute = function() {
                if (globalGraph._executionEnabled) {
                    return originalOnExecute.call(this);
                }
                return;
            };
        }
    }
    let graph = globalGraph;
    let canvas_node = globalCanvasNode;

    // Disable all default LiteGraph context menus and node panels
    canvas_node.show_info = false;
    canvas_node.allow_searchbox = false;
    
    // Clear all registered node types to prevent default node creation menu
    if (LiteGraph.registered_node_types) {
        LiteGraph.registered_node_types = {};
    }
    
    // Override the showNodeMenu method to prevent default menu
    canvas_node.onShowNodeMenu = function() {
        return false; // Prevent default menu
    };
    
    // Override context menu handler on canvas
    canvas_node.onMenu = function() {
        return false; // Prevent default menu
    };
    
    // Override processContextMenu to prevent default behavior
    if (canvas_node.processContextMenu) {
        const originalProcessContextMenu = canvas_node.processContextMenu.bind(canvas_node);
        canvas_node.processContextMenu = function(e) {
            // Don't call original - completely prevent default
            return false;
        };
    }
    
    // Disable search box
    if (canvas_node.searchbox) {
        canvas_node.searchbox.style.display = "none";
    }
    
    // Periodically remove any LiteGraph menus that might appear
    setInterval(function() {
        const lgMenus = document.querySelectorAll(".litemenu, .litegraph-menu, .lgraphcontextmenu, .lgraph-searchbox");
        lgMenus.forEach(menu => {
            menu.style.display = "none";
            menu.remove();
        });
    }, 100);

    // Set canvas to fill window (accounting for console panel and control panel)
    function resizeCanvas() {
        var consolePanel = document.getElementById("console-panel");
        var controlPanel = document.getElementById("execution-control-panel");
        var panelHeight = consolePanel && !consolePanel.classList.contains("collapsed") ? 300 : 35;
        var controlHeight = controlPanel ? 45 : 0;
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight - panelHeight - controlHeight;
        if (canvas_node && canvas_node.resize) {
            canvas_node.resize();
        }
    }

    // Initialize console panel
    function initializeConsolePanel() {
        var consolePanel = document.getElementById("console-panel");
        var consoleToggle = document.getElementById("console-toggle-btn");
        var consoleClear = document.getElementById("console-clear-btn");
        
        if (!consolePanel) return;
        
        // Toggle collapse/expand
        if (consoleToggle) {
            consoleToggle.addEventListener("click", function() {
                consolePanel.classList.toggle("collapsed");
                var isCollapsed = consolePanel.classList.contains("collapsed");
                consoleToggle.textContent = isCollapsed ? "▲" : "▼";
                resizeCanvas();
            });
        }
        
        // Clear console
        if (consoleClear) {
            consoleClear.addEventListener("click", function() {
                ConsoleLogManager.clear();
            });
        }
        
        // Show panel when first console node is added
        // (Panel is visible by default, but we ensure it's shown)
        consolePanel.style.display = "flex";
    }

    // Initial resize
    resizeCanvas();

    // Handle window resize
    window.addEventListener("resize", resizeCanvas);
    
    // Initialize console panel
    initializeConsolePanel();
    
    // Make canvas focusable for keyboard events
    canvas.setAttribute("tabindex", "0");
    canvas.style.outline = "none";
    
    // Focus canvas on click to ensure keyboard events work
    canvas.addEventListener("click", function() {
        canvas.focus();
    }, true);

    // Enable grid background
    canvas_node.show_grid = true;

    // Register all protein node types (must be before context menu initialization)
    registerProteinNodes();
    
    // Register all environment node types
    registerEnvironmentNodes();
    
    // Register all routing node types
    registerRoutingNodes();
    
    // Register all population node types
    registerPopulationNodes();
    
    // Register all mutator node types
    registerMutatorNodes();
    
    // Register all genome controller node types (loaded from separate file)
    if (typeof registerGenomeControllerNodes === "function") {
        registerGenomeControllerNodes();
    }
    
    // Register all utility node types
    registerUtilityNodes();
    
    // Register all controller node types
    registerControllerNodes();
    
    // Register episode nodes (if function exists - loaded from separate file)
    if (typeof registerEpisodeNodes === "function") {
        registerEpisodeNodes();
    } else if (typeof registerEpisodeRunnerNode === "function") {
        // Fallback for older version
        registerEpisodeRunnerNode();
    }
    
    // Load saved graph state from localStorage
    loadGraphState(graph);
    
    // Set up auto-save on graph changes
    setupGraphPersistence(graph, canvas_node);
    
    // Initialize File menu
    setupFileMenu(graph);
    
    // Initialize Chart Manager
    ChartManager.init();

    // Initialize context menu and event handlers
    initializeContextMenu(canvas, canvas_node, graph);

    // Initialize execution controller (must be before graph.start())
    initializeExecutionController(graph);
    
    // Ensure execution is disabled before starting
    graph._executionEnabled = false;
    
    // Start the graph rendering loop (but execution will be disabled)
    // Note: graph.start() starts rendering, but we control execution separately
    graph.start();
    
    // Force canvas redraw after a short delay to ensure everything is loaded
    setTimeout(function() {
        if (canvas_node && canvas_node.setDirty) {
            canvas_node.setDirty(true, true);
        }
    }, 100);
    
    // Immediately disable execution again (graph.start() might have enabled it)
    graph._executionEnabled = false;
    
    // Also try to stop any internal execution if graph has a stop method
    if (graph.stop) {
        try {
            graph.stop();
        } catch (e) {
            // Ignore errors - stop might not exist
        }
    }
    
    // Ensure controller is paused
    if (ExecutionController) {
        ExecutionController.pause();
    }
    
    // Force execution to be disabled one more time after everything initializes
    setTimeout(function() {
        if (graph) {
            graph._executionEnabled = false;
        }
    }, 100);
    
    console.log("LiteGraph initialized successfully");
});

// Custom context menu initialization
function initializeContextMenu(canvas, canvas_node, graph) {

    // Custom context menu
    let customContextMenu = null;
    
    // Store the last context menu event for node creation
    let lastContextMenuEvent = null;
    
    // Helper function to create a node at mouse position
    function createNodeAtPosition(nodeType) {
        if (!lastContextMenuEvent || !globalCanvasNode) return;
        
        const canvas = globalCanvasNode.canvas;
        const rect = canvas.getBoundingClientRect();
        const canvasX = lastContextMenuEvent.clientX - rect.left;
        const canvasY = lastContextMenuEvent.clientY - rect.top;
        
        const offset = globalCanvasNode.offset || [0, 0];
        const scale = globalCanvasNode.scale || 1;
        const graphX = (canvasX / scale) - offset[0];
        const graphY = (canvasY / scale) - offset[1];
        
        const node = LiteGraph.createNode(nodeType);
        if (node) {
            node.pos = [graphX, graphY];
            globalGraph.add(node);
            saveGraphState(globalGraph);
        }
        
        if (customContextMenu) {
            customContextMenu.classList.remove("show");
        }
        
        lastContextMenuEvent = null;
    }
    
    let contextMenuItems = {
        "Proteins": [
            {
                name: "Sensor",
                action: function() {
                    createNodeAtPosition("proteins/sensor");
                }
            },
            {
                name: "Comparator",
                action: function() {
                    createNodeAtPosition("proteins/comparator");
                }
            },
            {
                name: "Trend",
                action: function() {
                    createNodeAtPosition("proteins/trend");
                }
            },
            {
                name: "Integrator",
                action: function() {
                    createNodeAtPosition("proteins/integrator");
                }
            },
            {
                name: "Gate",
                action: function() {
                    createNodeAtPosition("proteins/gate");
                }
            },
            {
                name: "Trust Modifier",
                action: function() {
                    createNodeAtPosition("proteins/trust_modifier");
                }
            },
            {
                name: "Trust Aggregator",
                action: function() {
                    createNodeAtPosition("proteins/trust_aggregator");
                }
            }
        ],
        "Environment": [
            {
                name: "Snake",
                action: function() {
                    createNodeAtPosition("environment/snake");
                }
            },
            {
                name: "Visualize",
                action: function() {
                    createNodeAtPosition("environment/visualize");
                }
            }
        ],
        "Routing": [
            {
                name: "Strip",
                action: function() {
                    createNodeAtPosition("routing/strip");
                }
            },
            {
                name: "Clone",
                action: function() {
                    createNodeAtPosition("routing/clone");
                }
            },
            {
                name: "Signal Combiner",
                action: function() {
                    createNodeAtPosition("routing/combiner");
                }
            }
        ],
        "Mutator": [
            {
                name: "Mutator Controller",
                action: function() {
                    createNodeAtPosition("mutator/controller");
                }
            }
        ],
        "Population": [
            {
                name: "Population Controller",
                action: function() {
                    createNodeAtPosition("population/controller");
                }
            }
        ],
        "Genome Controller": [
            {
                name: "Genome Builder",
                action: function() {
                    createNodeAtPosition("genome/builder");
                }
            },
            {
                name: "Genome Loader",
                action: function() {
                    createNodeAtPosition("genome/loader");
                }
            },
            {
                name: "Genome Saver",
                action: function() {
                    createNodeAtPosition("genome/saver");
                }
            },
            {
                name: "Episode Runner",
                action: function() {
                    createNodeAtPosition("episode/runner");
                }
            },
            {
                name: "Generation Manager",
                action: function() {
                    createNodeAtPosition("generation/manager");
                }
            },
            {
                name: "Statistics Display",
                action: function() {
                    createNodeAtPosition("statistics/display");
                }
            }
        ],
        "Controller": [
            {
                name: "Controller Input",
                action: function() {
                    createNodeAtPosition("controller/input");
                }
            },
            {
                name: "Controller Network",
                action: function() {
                    createNodeAtPosition("controller/network");
                }
            }
        ],
        "Utility": [
            {
                name: "Chart Monitor",
                action: function() {
                    createNodeAtPosition("utility/chart_monitor");
                }
            },
            {
                name: "Console Output",
                action: function() {
                    createNodeAtPosition("utility/console_output");
                }
            },
            {
                name: "Dual Chart Monitor",
                action: function() {
                    createNodeAtPosition("utility/dual_chart_monitor");
                }
            },
            {
                name: "Test Signal Generator",
                action: function() {
                    createNodeAtPosition("utility/test_signal_generator");
                }
            },
            {
                name: "Signal Inspector",
                action: function() {
                    createNodeAtPosition("utility/signal_inspector");
                }
            }
        ]
    };

    function createContextMenu() {
        // Remove existing menu if any
        if (customContextMenu) {
            customContextMenu.remove();
        }

        // Create menu element
        customContextMenu = document.createElement("div");
        customContextMenu.className = "custom-context-menu";
        document.body.appendChild(customContextMenu);

        // Create menu items
        Object.keys(contextMenuItems).forEach((groupName) => {
            const menuItem = document.createElement("div");
            menuItem.className = "custom-context-menu-item has-submenu";
            menuItem.textContent = groupName;

            // Create submenu
            const submenu = document.createElement("div");
            submenu.className = "custom-context-submenu";

            // Add placeholder items (empty for now)
            if (contextMenuItems[groupName].length === 0) {
                const placeholder = document.createElement("div");
                placeholder.className = "custom-context-submenu-item";
                placeholder.textContent = "(Empty)";
                placeholder.style.color = "#888";
                placeholder.style.cursor = "default";
                submenu.appendChild(placeholder);
            } else {
                contextMenuItems[groupName].forEach((item) => {
                    const submenuItem = document.createElement("div");
                    submenuItem.className = "custom-context-submenu-item";
                    submenuItem.textContent = item.name;
                    submenuItem.onclick = item.action;
                    submenu.appendChild(submenuItem);
                });
            }

            menuItem.appendChild(submenu);
            customContextMenu.appendChild(menuItem);
            
            // Add hover handlers with delay for submenu
            let hideTimeout = null;
            const HIDE_DELAY = 200; // 200ms delay before hiding
            
            menuItem.addEventListener("mouseenter", function() {
                // Clear any pending hide
                if (hideTimeout) {
                    clearTimeout(hideTimeout);
                    hideTimeout = null;
                }
                // Show submenu immediately on hover
                submenu.classList.add("show");
            });
            
            menuItem.addEventListener("mouseleave", function(e) {
                // Check if mouse is moving to submenu
                const relatedTarget = e.relatedTarget;
                if (relatedTarget && (submenu.contains(relatedTarget) || relatedTarget === submenu)) {
                    // Mouse is moving to submenu, don't hide
                    return;
                }
                
                // Delay hiding the submenu
                hideTimeout = setTimeout(function() {
                    submenu.classList.remove("show");
                    hideTimeout = null;
                }, HIDE_DELAY);
            });
            
            // Also handle submenu mouse events
            submenu.addEventListener("mouseenter", function() {
                // Clear any pending hide when mouse enters submenu
                if (hideTimeout) {
                    clearTimeout(hideTimeout);
                    hideTimeout = null;
                }
                submenu.classList.add("show");
            });
            
            submenu.addEventListener("mouseleave", function() {
                // Hide submenu when mouse leaves
                hideTimeout = setTimeout(function() {
                    submenu.classList.remove("show");
                    hideTimeout = null;
                }, HIDE_DELAY);
            });
        });
    }

    // Initialize context menu
    createContextMenu();

    // Completely disable default LiteGraph context menu
    // Override canvas context menu handler
    canvas.addEventListener("contextmenu", function(e) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();

        // Hide any existing LiteGraph menus
        const lgMenus = document.querySelectorAll(".litemenu, .litegraph-menu, .lgraphcontextmenu");
        lgMenus.forEach(menu => {
            menu.style.display = "none";
            menu.remove();
        });

        // Hide any existing custom menu and all submenus
        if (customContextMenu) {
            customContextMenu.classList.remove("show");
            // Hide all submenus
            const submenus = customContextMenu.querySelectorAll(".custom-context-submenu");
            submenus.forEach(submenu => {
                submenu.classList.remove("show");
            });
        }

        // Show custom context menu
        if (customContextMenu) {
            // Store the event for node creation
            lastContextMenuEvent = e;
            customContextMenu.style.left = e.pageX + "px";
            customContextMenu.style.top = e.pageY + "px";
            customContextMenu.classList.add("show");
        }

        return false;
    }, true); // Use capture phase to intercept before LiteGraph
    
    // Also prevent context menu on the canvas_node element if it exists
    if (canvas_node.canvas) {
        canvas_node.canvas.addEventListener("contextmenu", function(e) {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            return false;
        }, true);
    }

    // Hide context menu when clicking elsewhere
    document.addEventListener("click", function(e) {
        if (customContextMenu && !customContextMenu.contains(e.target)) {
            customContextMenu.classList.remove("show");
        }
    });

    // Handle keyboard shortcuts (on both document and canvas)
    function handleKeyDown(e) {
    // Hide context menu on escape key
        if (e.key === "Escape" && customContextMenu) {
            customContextMenu.classList.remove("show");
        }
        
        // Delete selected nodes on Delete or Backspace key
        if ((e.key === "Delete" || e.key === "Backspace") && globalCanvasNode && globalGraph) {
            // Don't delete if user is typing in an input field
            if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.isContentEditable) {
                return;
            }
            
            // Prevent default browser behavior (e.g., going back in history)
            e.preventDefault();
            e.stopPropagation();
            
            // Get selected nodes - try multiple methods
            var selected_nodes = [];
            
            // Method 1: Check canvas_node.selected_nodes array
            if (globalCanvasNode.selected_nodes && Array.isArray(globalCanvasNode.selected_nodes)) {
                selected_nodes = globalCanvasNode.selected_nodes.slice();
            }
            
            // Method 2: Check canvas_node.selected_node (singular)
            if (selected_nodes.length === 0 && globalCanvasNode.selected_node) {
                selected_nodes = [globalCanvasNode.selected_node];
            }
            
            // Method 3: Iterate through all nodes and check their selected property
            // This is the most reliable method - ALWAYS check all nodes for selection flag
            if (globalGraph._nodes) {
                for (var i = 0; i < globalGraph._nodes.length; i++) {
                    var node = globalGraph._nodes[i];
                    if (node) {
                        // Check multiple possible selection indicators
                        var isSelected = false;
                        if (node.flags && node.flags.selected) {
                            isSelected = true;
                        }
                        if (node.selected === true) {
                            isSelected = true;
                        }
                        // Check if node is in canvas selection arrays
                        if (globalCanvasNode.selected_nodes && Array.isArray(globalCanvasNode.selected_nodes) && globalCanvasNode.selected_nodes.indexOf(node) !== -1) {
                            isSelected = true;
                        }
                        if (globalCanvasNode.selected_node === node) {
                            isSelected = true;
                        }
                        
                        // Add to selected_nodes if not already there
                        if (isSelected && selected_nodes.indexOf(node) === -1) {
                            selected_nodes.push(node);
                        }
                    }
                }
            }
            
            // Method 4: Check if canvas has a getSelectedNodes method
            if (selected_nodes.length === 0 && globalCanvasNode.getSelectedNodes) {
                try {
                    selected_nodes = globalCanvasNode.getSelectedNodes() || [];
                } catch (err) {
                    console.log("getSelectedNodes not available");
                }
            }
            
            if (selected_nodes && selected_nodes.length > 0) {
                // Remove each selected node from the graph
                for (var i = 0; i < selected_nodes.length; i++) {
                    var node = selected_nodes[i];
                    if (node && globalGraph.remove) {
                        try {
                            globalGraph.remove(node);
                        } catch (err) {
                            console.error("Error removing node:", err);
                        }
                    }
                }
                
                // Clear selection
                if (globalCanvasNode.selected_nodes) {
                    globalCanvasNode.selected_nodes = [];
                }
                if (globalCanvasNode.selected_node) {
                    globalCanvasNode.selected_node = null;
                }
                
                // Clear node selection flags
                if (globalGraph._nodes) {
                    for (var j = 0; j < globalGraph._nodes.length; j++) {
                        var n = globalGraph._nodes[j];
                        if (n && n.flags) {
                            n.flags.selected = false;
                        }
                    }
                }
                
                // Trigger canvas update
                if (globalCanvasNode.setDirtyCanvas) {
                    globalCanvasNode.setDirtyCanvas(true);
                }
                
                // Save state after deletion
                if (globalGraph) {
                    saveGraphState(globalGraph);
                }
            }
        }
    }
    
    // Attach keyboard handler to both document and canvas
    document.addEventListener("keydown", handleKeyDown);
    if (canvas) {
        canvas.addEventListener("keydown", handleKeyDown);
    }
}