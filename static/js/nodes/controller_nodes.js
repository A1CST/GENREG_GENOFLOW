// Controller Network Node Definitions
function registerControllerNodes() {
    // ControllerInputNode
    function ControllerInputNode() {
        // INPUTS:
        this.addInput("signals", "*");           // Environment signals (dict or list)
        this.addInput("signal_names", "array");  // Signal names from Sensor (REQUIRED)

        // OUTPUTS:
        this.addOutput("input_vector", "array");   // Flattened signal vector
        this.addOutput("signal_order", "array");   // Signal names in order (for genome)

        this.properties = {
            signal_order: null  // Will be set by Sensor node
        };

        this.color = "#16A085";
        this.bgcolor = "#0E6655";
        this.size = [200, 100];

        this.current_vector = [];
        this.current_signal_order = [];
    }
    
    ControllerInputNode.title = "Controller Input";
    ControllerInputNode.desc = "Converts environment signals to neural network input vector";
    
    ControllerInputNode.prototype.onExecute = function() {
        var signals = this.getInputData(0);
        var signal_names = this.getInputData(1);

        // REQUIRE signal_names from Sensor - no fallback
        if (!signal_names || !Array.isArray(signal_names) || signal_names.length === 0) {
            // No signal names provided - cannot process
            this.setOutputData(0, null);
            this.setOutputData(1, null);
            this.current_vector = [];
            this.current_signal_order = [];
            return;
        }

        // Store signal order
        this.current_signal_order = signal_names;
        this.properties.signal_order = signal_names;

        // Output signal order (for Genome Builder to use)
        this.setOutputData(1, signal_names);

        // Convert signals dict to ordered array using provided signal names
        if (!signals || typeof signals !== 'object') {
            // No signals yet - output null
            this.setOutputData(0, null);
            this.current_vector = [];
            return;
        }

        var vector = [];
        for (var i = 0; i < signal_names.length; i++) {
            var key = signal_names[i];
            vector.push(signals[key] || 0.0);
        }

        this.current_vector = vector;
        this.setOutputData(0, vector);
    };
    
    ControllerInputNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";

            // Show vector size
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.fillText("Vector Size: " + this.current_vector.length, 10, this.size[1] - 10);

            // Show signal order status
            var hasSignalOrder = this.current_signal_order && this.current_signal_order.length > 0;
            ctx.fillStyle = hasSignalOrder ? "#2ECC71" : "#E74C3C";
            ctx.font = "9px Arial";
            var status = hasSignalOrder ? "Order: OK (" + this.current_signal_order.length + ")" : "No Signal Order";
            ctx.fillText(status, 10, this.size[1] - 25);
        }
    };
    
    LiteGraph.registerNodeType("controller/input", ControllerInputNode);
    
    // ControllerNetworkNode
    function ControllerNetworkNode() {
        // INPUTS:
        this.addInput("input_vector", "array");   // Signal vector from Controller Input
        this.addInput("signal_order", "array");   // Signal names (sets input_size)

        // OUTPUTS:
        this.addOutput("action", "number");       // Final action (0-3)
        this.addOutput("action_scores", "array"); // Raw output scores

        this.properties = {
            input_size: null,    // Will be set from signal_order
            hidden_size: 128,
            output_size: 4,

            // Weights (will be loaded from genome)
            w1: [],  // [hidden_size][input_size]
            b1: [],  // [hidden_size]
            w2: [],  // [output_size][hidden_size]
            b2: [],  // [output_size]

            weights_loaded: false
        };

        this.color = "#2980B9";
        this.bgcolor = "#1F5F8B";
        this.size = [220, 140];

        this.current_action = 0;
        this.action_scores = [0, 0, 0, 0];
        this.action_names = ["UP", "DOWN", "LEFT", "RIGHT"];
        this.current_signal_order = [];
    }
    
    ControllerNetworkNode.title = "Controller Network";
    ControllerNetworkNode.desc = "Neural network that chooses actions from signals";
    
    ControllerNetworkNode.prototype.onAdded = function() {
        // Hidden size (can be adjusted, but weights must be reloaded)
        this.addWidget("number", "Hidden Size", this.properties.hidden_size,
            function(v) {
                this.properties.hidden_size = Math.max(1, Math.floor(v));
                // If weights are loaded, they need to be reinitialized for new hidden_size
                if (this.properties.weights_loaded) {
                    this.properties.weights_loaded = false;
                }
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 512, step: 1 }
        );
        
        // Button to manually initialize/reinitialize weights
        this.addWidget("button", "Init Weights", null,
            function() {
                if (this.properties.input_size) {
                    this.initRandomWeights();
                    this.setDirtyCanvas(true);
                } else {
                    console.warn("Cannot initialize weights: input_size not set. Connect signal_order input first.");
                }
            }.bind(this)
        );
    };
    
    ControllerNetworkNode.prototype.tanh = function(x) {
        return Math.tanh(x);
    };
    
    ControllerNetworkNode.prototype.initRandomWeights = function() {
        // Xavier initialization (matching Python Controller)
        var input_size = this.properties.input_size;
        var hidden_size = this.properties.hidden_size;
        var output_size = this.properties.output_size;
        
        if (!input_size || input_size <= 0) {
            console.warn("Cannot initialize weights: input_size not set");
            return;
        }
        
        // Calculate scales for Xavier initialization
        var scale1 = Math.sqrt(2.0 / (input_size + hidden_size));
        var scale2 = Math.sqrt(2.0 / (hidden_size + output_size));
        
        // Initialize w1 [hidden_size][input_size]
        this.properties.w1 = [];
        for (var i = 0; i < hidden_size; i++) {
            var row = [];
            for (var j = 0; j < input_size; j++) {
                // Gaussian random with mean 0, stddev = scale1
                var u1 = Math.random();
                var u2 = Math.random();
                var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                row.push(z * scale1);
            }
            this.properties.w1.push(row);
        }
        
        // Initialize b1 [hidden_size] - zeros
        this.properties.b1 = [];
        for (var i = 0; i < hidden_size; i++) {
            this.properties.b1.push(0.0);
        }
        
        // Initialize w2 [output_size][hidden_size]
        this.properties.w2 = [];
        for (var i = 0; i < output_size; i++) {
            var row = [];
            for (var j = 0; j < hidden_size; j++) {
                // Gaussian random with mean 0, stddev = scale2
                var u1 = Math.random();
                var u2 = Math.random();
                var z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
                row.push(z * scale2);
            }
            this.properties.w2.push(row);
        }
        
        // Initialize b2 [output_size] - zeros
        this.properties.b2 = [];
        for (var i = 0; i < output_size; i++) {
            this.properties.b2.push(0.0);
        }
        
        this.properties.weights_loaded = true;
        console.log("Controller weights initialized randomly (Xavier init)");
    };
    
    ControllerNetworkNode.prototype.onExecute = function() {
        var inputs = this.getInputData(0);
        var signal_order = this.getInputData(1);

        // Update input_size from signal_order
        if (signal_order && Array.isArray(signal_order) && signal_order.length > 0) {
            this.current_signal_order = signal_order;
            this.properties.input_size = signal_order.length;
        }

        // Auto-initialize weights on first run if we have input_size but weights aren't loaded
        if (this.properties.input_size && !this.properties.weights_loaded) {
            console.log("[Controller Network] Auto-initializing weights (input_size:", this.properties.input_size + ")");
            this.initRandomWeights();
        }

        // REQUIRE both inputs and weights to be loaded
        if (!inputs || !this.properties.weights_loaded || !this.properties.input_size) {
            var missing = [];
            if (!inputs) missing.push("input_vector");
            if (!this.properties.weights_loaded) missing.push("weights");
            if (!this.properties.input_size) missing.push("input_size");
            console.warn("[Controller Network] Cannot compute action - missing:", missing.join(", "));
            this.setOutputData(0, 0);
            this.setOutputData(1, [0, 0, 0, 0]);
            return;
        }

        // Log input reception
        if (inputs && Array.isArray(inputs)) {
            console.log("[Controller Network] Input vector received, size:", inputs.length);
        }

        // Forward pass through neural network

        // Hidden layer
        var hidden = [];
        for (var i = 0; i < this.properties.hidden_size; i++) {
            var sum = this.properties.b1[i];
            for (var j = 0; j < this.properties.input_size; j++) {
                sum += this.properties.w1[i][j] * (inputs[j] || 0);
            }
            hidden.push(this.tanh(sum));
        }

        // Output layer
        var outputs = [];
        for (var i = 0; i < this.properties.output_size; i++) {
            var sum = this.properties.b2[i];
            for (var j = 0; j < this.properties.hidden_size; j++) {
                sum += this.properties.w2[i][j] * hidden[j];
            }
            outputs.push(sum);
        }

        // Choose action with max score (argmax)
        var max_idx = 0;
        var max_val = outputs[0];
        for (var i = 1; i < outputs.length; i++) {
            if (outputs[i] > max_val) {
                max_val = outputs[i];
                max_idx = i;
            }
        }

        this.current_action = max_idx;
        this.action_scores = outputs;

        // Log action computation and output
        var action_name = this.action_names[max_idx];
        console.log("[Controller Network] Action computed:", max_idx, "(" + action_name + "), scores:", outputs.map(function(x) { return x.toFixed(3); }).join(", "));
        console.log("[Controller Network] Outputting action to slot 0:", max_idx);

        this.setOutputData(0, max_idx);
        this.setOutputData(1, outputs);
    };
    
    ControllerNetworkNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";

            // Show current action
            var action_name = this.action_names[this.current_action];
            ctx.fillStyle = "#2ECC71";
            ctx.font = "bold 14px Arial";
            ctx.fillText("Action: " + action_name, 10, this.size[1] - 10);

            // Show weights status
            var status = this.properties.weights_loaded ? "Loaded" : "Not Loaded";
            var color = this.properties.weights_loaded ? "#2ECC71" : "#E74C3C";
            ctx.fillStyle = color;
            ctx.font = "9px Arial";
            ctx.fillText("Weights: " + status, 10, this.size[1] - 25);

            // Show signal order status
            var hasSignalOrder = this.current_signal_order && this.current_signal_order.length > 0;
            ctx.fillStyle = hasSignalOrder ? "#2ECC71" : "#E74C3C";
            ctx.font = "9px Arial";
            var orderStatus = hasSignalOrder ? "Order: OK" : "No Order";
            ctx.fillText(orderStatus, 10, this.size[1] - 40);

            // Show architecture (input_size is now dynamic)
            var inputSize = this.properties.input_size || "?";
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "9px Arial";
            var arch = inputSize + "→" + this.properties.hidden_size + "→" + this.properties.output_size;
            ctx.fillText(arch, 10, this.size[1] - 55);
        }
    };
    
    LiteGraph.registerNodeType("controller/network", ControllerNetworkNode);
}
