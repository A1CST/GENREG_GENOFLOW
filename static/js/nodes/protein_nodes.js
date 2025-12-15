// Protein Node Definitions
function registerProteinNodes() {
    // SensorProtein Node
    function SensorNode() {
        // NO physical input connection - reads from environment dynamically
        // Outputs will be created dynamically for each signal detected
        
        this.properties = {
            auto_detect_env: true,          // Automatically find environment node
            name: "sensor_1"
        };
        
        this.color = "#4A90E2";
        this.bgcolor = "#2C3E50";
        this.size = [180, 100];
        
        // Cache for found environment node
        this.cached_env_node = null;
        this.detected_signals = [];  // List of signal names detected
        this.signal_values = {};      // Current values for each signal
    }
    SensorNode.title = "Sensor";
    SensorNode.desc = "Reads all signals from environment";
    
    SensorNode.prototype.onAdded = function() {
        // Auto-detect toggle (could manually specify env node ID later)
        this.addWidget("toggle", "Auto-Detect", this.properties.auto_detect_env,
            function(v) { 
                this.properties.auto_detect_env = v;
                this.cached_env_node = null;
                this.updateOutputs();
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Refresh outputs button
        this.addWidget("button", "Refresh", null,
            function() {
                this.cached_env_node = null;
                this.updateOutputs();
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Initial output update
        this.updateOutputs();
    };
    
    SensorNode.prototype.updateOutputs = function() {
        // Ensure outputs exist in correct order:
        // Index 0: "all_signals" (object)
        // Index 1: "signal_names" (array)
        // Index 2+: individual signal outputs

        var needsRebuild = false;

        // Check if outputs are in correct order
        if (!this.outputs || this.outputs.length < 2) {
            needsRebuild = true;
        } else if (this.outputs[0].name !== "all_signals" || this.outputs[1].name !== "signal_names") {
            needsRebuild = true;
        }

        if (needsRebuild) {
            // Save individual signal outputs (skip "all_signals" and "signal_names")
            var existingSignalOutputs = [];
            if (this.outputs && this.outputs.length > 0) {
                for (var i = 0; i < this.outputs.length; i++) {
                    var name = this.outputs[i].name;
                    if (name !== "all_signals" && name !== "signal_names") {
                        existingSignalOutputs.push(name);
                    }
                }
            }

            // Remove all outputs
            while (this.outputs && this.outputs.length > 0) {
                this.removeOutput(0);
            }

            // Add in correct order
            this.addOutput("all_signals", "object");    // Index 0
            this.addOutput("signal_names", "array");    // Index 1

            // Re-add individual signal outputs
            for (var j = 0; j < existingSignalOutputs.length; j++) {
                this.addOutput(existingSignalOutputs[j], "signal");
            }
        }
        
        // Find environment node
        var env_node = this.findEnvironmentNode();
        
        if (!env_node) {
            // No environment - remove all individual signal outputs (keep "all_signals" and "signal_names")
            while (this.outputs && this.outputs.length > 2) {
                this.removeOutput(this.outputs.length - 1);
            }
            this.detected_signals = [];
            return;
        }

        // Get signals from environment
        var signals = null;
        if (env_node.properties && env_node.properties.current_signals) {
            signals = env_node.properties.current_signals;
        } else if (env_node.outputs && env_node.outputs.length > 0) {
            for (var i = 0; i < env_node.outputs.length; i++) {
                if (env_node.outputs[i].name === "signals" || env_node.outputs[i].name === "signals_dict") {
                    signals = env_node.getOutputData(i);
                    break;
                }
            }
        }

        if (!signals || typeof signals !== "object") {
            // No signals available - remove all individual signal outputs
            while (this.outputs && this.outputs.length > 2) {
                this.removeOutput(this.outputs.length - 1);
            }
            this.detected_signals = [];
            return;
        }

        // Get list of signal names
        var signal_names = Object.keys(signals);
        signal_names.sort(); // Sort for consistent ordering

        // Update outputs to match detected signals (skip index 0="all_signals" and index 1="signal_names")
        var current_outputs = this.outputs || [];
        var current_names = [];
        for (var i = 2; i < current_outputs.length; i++) {
            current_names.push(current_outputs[i].name);
        }

        // Remove outputs that no longer exist (skip index 0 and 1)
        for (var i = current_outputs.length - 1; i >= 2; i--) {
            if (signal_names.indexOf(current_outputs[i].name) === -1) {
                this.removeOutput(i);
            }
        }

        // Add outputs for new signals (after "all_signals" and "signal_names")
        for (var j = 0; j < signal_names.length; j++) {
            var sig_name = signal_names[j];
            if (current_names.indexOf(sig_name) === -1) {
                this.addOutput(sig_name, "signal");
            }
        }

        // Update detected signals list
        this.detected_signals = signal_names;
    };
    
    SensorNode.prototype.findEnvironmentNode = function() {
        // If we have a cached node, verify it's still valid
        if (this.cached_env_node) {
            if (this.graph && this.graph._nodes && this.graph._nodes.indexOf(this.cached_env_node) !== -1) {
                return this.cached_env_node;
            }
            this.cached_env_node = null;
        }
        
        // Search for environment node in the graph
        if (!this.graph) return null;
        
        var nodes = this.graph._nodes || [];
        for (var i = 0; i < nodes.length; i++) {
            var node = nodes[i];
            // Check if this is an environment node (has "environment" in type)
            // or test signal generator (can also provide signals)
            if (node.type && (
                node.type.indexOf("environment/") === 0 ||
                node.type === "utility/test_signal_generator"
            )) {
                this.cached_env_node = node;
                return node;
            }
        }
        
        return null;
    };
    
    SensorNode.prototype.onExecute = function() {
        // Update outputs if needed (in case environment signals changed)
        this.updateOutputs();
        
        // Find environment node
        var env_node = this.findEnvironmentNode();
        
        // Get signals from environment node
        var signals = null;
        
        if (!env_node) {
            // No environment found - output empty
            this.setOutputData(0, {});     // all_signals
            this.setOutputData(1, []);     // signal_names
            for (var i = 2; i < this.outputs.length; i++) {
                this.setOutputData(i, 0);
            }
            return;
        }

        // Try to get signals from environment node's properties
        if (env_node.properties && env_node.properties.current_signals) {
            signals = env_node.properties.current_signals;
        }

        // Also try to get from output if available
        if (!signals && env_node.outputs && env_node.outputs.length > 0) {
            // Find the "signals" output
            for (var i = 0; i < env_node.outputs.length; i++) {
                if (env_node.outputs[i].name === "signals" || env_node.outputs[i].name === "signals_dict") {
                    signals = env_node.getOutputData(i);
                    break;
                }
            }
        }

        if (!signals || typeof signals !== "object") {
            // No signals available - output empty
            this.setOutputData(0, {});     // all_signals
            this.setOutputData(1, []);     // signal_names
            for (var j = 2; j < this.outputs.length; j++) {
                this.setOutputData(j, 0);
            }
            return;
        }

        // Output all signals as a dictionary to index 0 ("all_signals")
        var all_signals_dict = {};
        for (var key in signals) {
            if (signals.hasOwnProperty(key)) {
                all_signals_dict[key] = signals[key];
            }
        }
        this.setOutputData(0, all_signals_dict);

        // Output signal names array to index 1 ("signal_names")
        this.setOutputData(1, this.detected_signals);

        // Output each individual signal value on its corresponding output (starting at index 2)
        for (var k = 2; k < this.outputs.length; k++) {
            var output_name = this.outputs[k].name;
            var signal_value = signals[output_name];

            if (signal_value === undefined || signal_value === null) {
                signal_value = 0;
            }

            // Store for display
            this.signal_values[output_name] = signal_value;

            // Output the signal value
            this.setOutputData(k, signal_value);
        }
    };
    
    SensorNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";
            
            // Show signal count
            ctx.fillStyle = "#3498DB";
            ctx.font = "10px Arial";
            ctx.fillText("Signals: " + this.detected_signals.length, 10, this.size[1] - 10);
            
            // Show warning if no environment found
            var env = this.findEnvironmentNode();
            if (!env) {
                ctx.fillStyle = "#E74C3C";
                ctx.font = "9px Arial";
                ctx.fillText("⚠ No Environment", 10, this.size[1] - 25);
            } else if (this.detected_signals.length === 0) {
                ctx.fillStyle = "#F39C12";
                ctx.font = "9px Arial";
                ctx.fillText("⚠ No Signals", 10, this.size[1] - 25);
            }
        }
    };
    
    LiteGraph.registerNodeType("proteins/sensor", SensorNode);

    // ComparatorProtein Node
    function ComparatorNode() {
        this.addInput("signal1", "signal");
        this.addInput("signal2", "signal");
        this.addOutput("output", "signal");
        this.color = "#9B59B6";
        this.bgcolor = "#2C3E50";
        this.properties = {
            name: "comparator_1"
        };
        this.size = [180, 80];
    }
    ComparatorNode.title = "Comparator";
    ComparatorNode.desc = "Compares two signals";
    ComparatorNode.prototype.onExecute = function() {
        // Get input signals
        var signal1 = this.getInputData(0);
        var signal2 = this.getInputData(1);
        
        // Handle undefined/null inputs
        if (signal1 === undefined || signal1 === null) {
            signal1 = 0;
        }
        if (signal2 === undefined || signal2 === null) {
            signal2 = 0;
        }
        
        // Convert to numbers if needed
        signal1 = Number(signal1) || 0;
        signal2 = Number(signal2) || 0;
        
        // Compare: output the difference (signal1 - signal2)
        // Positive = signal1 > signal2
        // Negative = signal1 < signal2
        // Zero = signal1 == signal2
        var result = signal1 - signal2;
        
        // Output the comparison result
        this.setOutputData(0, result);
    };
    LiteGraph.registerNodeType("proteins/comparator", ComparatorNode);

    // TrendProtein Node
    function TrendNode() {
        this.addInput("input", "signal");
        this.addOutput("output", "signal");
        this.color = "#E67E22";
        this.bgcolor = "#2C3E50";
        this.properties = {
            name: "trend_1",
            momentum: 0.9  // Default value from Python
        };
        this.size = [180, 100];
    }
    TrendNode.title = "Trend";
    TrendNode.desc = "Tracks signal velocity/momentum";
    
    TrendNode.prototype.onAdded = function() {
        // Add momentum widget
        this.addWidget("number", "Momentum", this.properties.momentum,
            function(v) { 
                this.properties.momentum = Math.max(0, Math.min(1, v)); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 1, step: 0.05, precision: 2 }
        );
    };
    
    TrendNode.prototype.onExecute = function() {
        // Get input signal
        var x = this.getInputData(0);
        
        if (x === undefined || x === null) {
            this.setOutputData(0, 0);
            return;
        }
        
        x = Number(x) || 0;
        
        // Initialize state if needed
        if (this.state === undefined) {
            this.state = {
                last: null,
                velocity: 0.0
            };
        }
        
        // First time - initialize last value
        if (this.state.last === null) {
            this.state.last = x;
            this.setOutputData(0, 0);
            return;
        }
        
        // Calculate delta (change since last frame)
        var delta = x - this.state.last;
        this.state.last = x;
        
        // Apply momentum smoothing (EMA - Exponential Moving Average)
        var momentum = this.properties.momentum || 0.9;
        this.state.velocity = momentum * this.state.velocity + (1 - momentum) * delta;
        
        // Output the velocity
        this.setOutputData(0, this.state.velocity);
    };
    LiteGraph.registerNodeType("proteins/trend", TrendNode);

    // IntegratorProtein Node
    function IntegratorNode() {
        this.addInput("input", "signal");
        this.addOutput("output", "signal");
        this.color = "#2ECC71";
        this.bgcolor = "#2C3E50";
        this.properties = {
            name: "integrator_1"
        };
        this.size = [180, 60];
    }
    IntegratorNode.title = "Integrator";
    IntegratorNode.desc = "Accumulates signal values";
    
    IntegratorNode.prototype.onAdded = function() {
        // Add decay parameter widget
        this.addWidget("number", "Decay", this.properties.decay || 0.05,
            function(v) { 
                this.properties.decay = Math.max(0, Math.min(1, v)); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 1, step: 0.01, precision: 3 }
        );
    };
    
    IntegratorNode.prototype.onExecute = function() {
        // Get input signal
        var x = this.getInputData(0);
        
        if (x === undefined || x === null) {
            this.setOutputData(0, 0);
            return;
        }
        
        x = Number(x) || 0;
        
        // Initialize state if needed
        if (this.state === undefined) {
            this.state = {
                accum: 0.0
            };
        }
        
        // Get decay parameter (default 0.05)
        var decay = this.properties.decay !== undefined ? this.properties.decay : 0.05;
        
        // Accumulate with decay (prevents infinite growth)
        this.state.accum = this.state.accum * (1 - decay) + x;
        
        // Clamp output to prevent overflow
        var output = Math.max(Math.min(this.state.accum, 10.0), -10.0);
        
        this.setOutputData(0, output);
    };
    LiteGraph.registerNodeType("proteins/integrator", IntegratorNode);

    // GateProtein Node
    function GateNode() {
        this.addInput("condition", "signal");
        this.addInput("value", "signal");
        this.addOutput("output", "signal");
        this.color = "#F39C12";
        this.bgcolor = "#2C3E50";
        this.properties = {
            name: "gate_1"
        };
        this.size = [180, 80];
    }
    GateNode.title = "Gate";
    GateNode.desc = "Gates a value based on condition";
    
    GateNode.prototype.onAdded = function() {
        // Add threshold parameter
        this.addWidget("number", "Threshold", this.properties.threshold || 0.0,
            function(v) { 
                this.properties.threshold = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -10, max: 10, step: 0.1, precision: 2 }
        );
        
        // Add hysteresis parameter
        this.addWidget("number", "Hysteresis", this.properties.hysteresis || 0.1,
            function(v) { 
                this.properties.hysteresis = Math.max(0, v); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 2, step: 0.05, precision: 2 }
        );
    };
    
    GateNode.prototype.onExecute = function() {
        // Get inputs
        var condition = this.getInputData(0);
        var value = this.getInputData(1);
        
        if (condition === undefined || condition === null || value === undefined || value === null) {
            this.setOutputData(0, 0);
            return;
        }
        
        condition = Number(condition) || 0;
        value = Number(value) || 0;
        
        // Initialize state if needed
        if (this.state === undefined) {
            this.state = {
                active: false
            };
        }
        
        // Get parameters
        var threshold = this.properties.threshold !== undefined ? this.properties.threshold : 0.0;
        var hysteresis = this.properties.hysteresis !== undefined ? this.properties.hysteresis : 0.1;
        
        // Check if gate should activate (with hysteresis to prevent oscillation)
        if (!this.state.active && condition > (threshold + hysteresis)) {
            this.state.active = true;
        } else if (this.state.active && condition < (threshold - hysteresis)) {
            this.state.active = false;
        }
        
        // Output value only if active, otherwise 0
        // Use average to prevent exponential growth
        var output = this.state.active ? (0.5 * (condition + value)) : 0.0;
        
        this.setOutputData(0, output);
    };
    LiteGraph.registerNodeType("proteins/gate", GateNode);

    // TrustModifierProtein Node
    function TrustModifierNode() {
        this.addInput("input", "signal");
        this.addOutput("trust", "signal");
        this.color = "#E74C3C";
        this.bgcolor = "#922B21";
        this.properties = {
            name: "trust_modifier_1",
            gain: 1.0,
            scale: 1.0
        };
        this.size = [180, 100];
        // Make Trust Modifier visually distinct
        this.shape = 1; // Rounded rectangle
        
        // Track current trust for display
        this.current_trust = 0;
    }
    TrustModifierNode.title = "Trust Modifier";
    TrustModifierNode.desc = "Converts signal to trust delta";
    
    TrustModifierNode.prototype.onAdded = function() {
        // Gain widget
        this.addWidget("number", "Gain", this.properties.gain,
            function(v) {
                this.properties.gain = Number(v) || 1.0;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Scale widget
        this.addWidget("number", "Scale", this.properties.scale,
            function(v) {
                this.properties.scale = Number(v) || 1.0;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    TrustModifierNode.prototype.onExecute = function() {
        // Get input signal
        var input_signal = this.getInputData(0);
        
        if (input_signal === undefined || input_signal === null) {
            this.setOutputData(0, 0);  // Output 0 if no input
            this.current_trust = 0;
            return;
        }
        
        // Convert to number
        input_signal = Number(input_signal) || 0;
        
        // Simple trust calculation (matching Python logic)
        // For now: trust_output = gain * scale * input_signal
        var gain = this.properties.gain || 1.0;
        var scale = this.properties.scale || 1.0;
        
        var trust_output = gain * scale * input_signal;
        
        // Store for display
        this.current_trust = trust_output;
        
        // IMPORTANT: Actually set the output data
        this.setOutputData(0, trust_output);
    };
    
    TrustModifierNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "bold 10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("Trust: " + this.current_trust.toFixed(3), 10, this.size[1] - 10);
        }
    };
    LiteGraph.registerNodeType("proteins/trust_modifier", TrustModifierNode);

    // TrustAggregatorNode
    function TrustAggregatorNode() {
        // INPUTS: Multiple trust signals from TrustModifier proteins
        // Start with 4 inputs, can be extended dynamically
        this.addInput("trust_1", "signal");
        this.addInput("trust_2", "signal");
        this.addInput("trust_3", "signal");
        this.addInput("trust_4", "signal");
        
        // OUTPUTS: 
        this.addOutput("trust_delta", "number");  // Final aggregated trust
        this.addOutput("raw_sum", "number");      // Raw sum before clamping
        
        this.properties = {
            aggregation_mode: "sum",    // sum, average, weighted_sum, max, min
            clamp_min: -5.0,            // Clamp final trust delta
            clamp_max: 5.0,
            enable_clamping: true,      // Toggle clamping on/off
            weights: [1.0, 1.0, 1.0, 1.0]  // For weighted_sum mode
        };
        
        this.color = "#E74C3C";         // Red - critical trust computation
        this.bgcolor = "#922B21";
        this.size = [220, 180];
        
        // Track values for display
        this.current_trust = 0;
        this.raw_trust = 0;
        this.input_values = [0, 0, 0, 0];
    }
    
    TrustAggregatorNode.title = "Trust Aggregator";
    TrustAggregatorNode.desc = "Aggregate trust deltas from TrustModifier proteins";
    
    TrustAggregatorNode.prototype.onAdded = function() {
        // Aggregation mode
        this.addWidget("combo", "Mode", this.properties.aggregation_mode,
            function(v) { 
                this.properties.aggregation_mode = v;
                this.updateWidgetVisibility();
                this.setDirtyCanvas(true);
            }.bind(this),
            { values: ["sum", "average", "weighted_sum", "max", "min"] }
        );
        
        // Clamping toggle
        this.addWidget("toggle", "Clamp", this.properties.enable_clamping,
            function(v) { 
                this.properties.enable_clamping = v;
                this.updateWidgetVisibility();
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Clamp min
        this.clamp_min_widget = this.addWidget("number", "Min", this.properties.clamp_min,
            function(v) { 
                this.properties.clamp_min = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        // Clamp max
        this.clamp_max_widget = this.addWidget("number", "Max", this.properties.clamp_max,
            function(v) { 
                this.properties.clamp_max = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        // Add/Remove input buttons
        this.addWidget("button", "Add Input", null,
            function() {
                var idx = this.inputs.length;
                this.addInput("trust_" + (idx + 1), "signal");
                this.properties.weights.push(1.0);
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        this.addWidget("button", "Remove Input", null,
            function() {
                if (this.inputs.length > 1) {
                    this.removeInput(this.inputs.length - 1);
                    this.properties.weights.pop();
                    this.setDirtyCanvas(true);
                }
            }.bind(this)
        );
        
        this.updateWidgetVisibility();
    };
    
    TrustAggregatorNode.prototype.updateWidgetVisibility = function() {
        // Show/hide clamp widgets based on toggle
        if (this.clamp_min_widget) {
            this.clamp_min_widget.hidden = !this.properties.enable_clamping;
        }
        if (this.clamp_max_widget) {
            this.clamp_max_widget.hidden = !this.properties.enable_clamping;
        }
    };
    
    TrustAggregatorNode.prototype.onExecute = function() {
        // Collect all input trust values
        var trust_values = [];
        this.input_values = [];
        
        for (var i = 0; i < this.inputs.length; i++) {
            var value = this.getInputData(i);
            if (value !== undefined && value !== null) {
                trust_values.push(value);
                this.input_values.push(value);
            } else {
                this.input_values.push(0);
            }
        }
        
        // Compute aggregated trust based on mode
        var trust_delta = 0;
        
        switch(this.properties.aggregation_mode) {
            case "sum":
                trust_delta = trust_values.reduce(function(a, b) { return a + b; }, 0);
                break;
                
            case "average":
                if (trust_values.length > 0) {
                    trust_delta = trust_values.reduce(function(a, b) { return a + b; }, 0) / trust_values.length;
                }
                break;
                
            case "weighted_sum":
                trust_delta = 0;
                for (var i = 0; i < trust_values.length; i++) {
                    var weight = this.properties.weights[i] || 1.0;
                    trust_delta += trust_values[i] * weight;
                }
                break;
                
            case "max":
                if (trust_values.length > 0) {
                    trust_delta = Math.max.apply(null, trust_values);
                }
                break;
                
            case "min":
                if (trust_values.length > 0) {
                    trust_delta = Math.min.apply(null, trust_values);
                }
                break;
        }
        
        // Store raw value before clamping
        this.raw_trust = trust_delta;
        
        // Apply clamping if enabled
        if (this.properties.enable_clamping) {
            trust_delta = Math.max(
                this.properties.clamp_min,
                Math.min(this.properties.clamp_max, trust_delta)
            );
        }
        
        this.current_trust = trust_delta;
        
        // Output both clamped and raw values
        this.setOutputData(0, trust_delta);
        this.setOutputData(1, this.raw_trust);
    };
    
    TrustAggregatorNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Display current trust value prominently
            var trust_text = "Trust: " + this.current_trust.toFixed(3);
            var color = this.current_trust > 0 ? "#2ECC71" : 
                        this.current_trust < 0 ? "#E74C3C" : "#95A5A6";
            
            ctx.fillStyle = color;
            ctx.font = "bold 14px Arial";
            ctx.textAlign = "left";
            ctx.fillText(trust_text, 10, this.size[1] - 10);
            
            // Show raw value if clamping occurred
            if (this.properties.enable_clamping && this.raw_trust !== this.current_trust) {
                ctx.fillStyle = "#F39C12";
                ctx.font = "10px Arial";
                ctx.fillText("(Raw: " + this.raw_trust.toFixed(3) + ")", 10, this.size[1] - 25);
            }
            
            // Show input count
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "9px Arial";
            ctx.fillText("Inputs: " + this.inputs.length, 10, this.size[1] - 40);
        }
    };
    
    LiteGraph.registerNodeType("proteins/trust_aggregator", TrustAggregatorNode);
}
