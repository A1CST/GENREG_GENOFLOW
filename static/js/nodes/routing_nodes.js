// Routing Node Definitions
function registerRoutingNodes() {
    // StripNode
    function StripNode() {
        this.addInput("signals", "signal");
        this.addOutput("signals", "signal");
        this.addOutput("stripped", "signal");
        
        this.properties = {
            mode: "first",
            index: 0
        };
        
        this.color = "#3498DB";
        this.bgcolor = "#2C5F7F";
        this.size = [180, 80];
    }
    
    StripNode.title = "Strip";
    StripNode.desc = "Remove index from signal stream";
    
    StripNode.prototype.onAdded = function() {
        // Add mode selection widget
        this.addWidget("combo", "Mode", this.properties.mode, 
            function(v) { 
                this.properties.mode = v;
                this.setDirtyCanvas(true);
            }.bind(this),
            { values: ["first", "last", "index"] }
        );
        
        // Add index widget (only relevant when mode is "index")
        this.addWidget("number", "Index", this.properties.index,
            function(v) { 
                this.properties.index = Math.max(0, Math.floor(v)); // Ensure integer
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 1 }
        );
    };
    
    StripNode.prototype.onExecute = function() {
        // Get input signals
        var input_signals = this.getInputData(0);
        
        // Handle different input types
        var signals_array = null;
        var is_dict = false;
        
        if (!input_signals) {
            // No input - output empty
            this.setOutputData(0, null);
            this.setOutputData(1, null);
            return;
        }
        
        // Check if input is an array
        if (Array.isArray(input_signals)) {
            signals_array = input_signals.slice(); // Copy array
        }
        // Check if input is a signals dictionary (object with string keys)
        else if (typeof input_signals === "object" && input_signals !== null) {
            is_dict = true;
            // Convert dictionary to array for processing
            var keys = Object.keys(input_signals);
            signals_array = keys.map(function(key) {
                return { key: key, value: input_signals[key] };
            });
        }
        // Single value - treat as array with one element
        else {
            signals_array = [input_signals];
        }
        
        if (!signals_array || signals_array.length === 0) {
            // Empty input
            this.setOutputData(0, is_dict ? {} : []);
            this.setOutputData(1, null);
            return;
        }
        
        // Determine which index to strip
        var strip_index = 0;
        var mode = this.properties.mode || "first";
        
        switch(mode) {
            case "first":
                strip_index = 0;
                break;
            case "last":
                strip_index = signals_array.length - 1;
                break;
            case "index":
                var idx = Math.max(0, Math.floor(this.properties.index || 0));
                strip_index = Math.min(signals_array.length - 1, idx);
                break;
        }
        
        // Extract the stripped element
        var stripped_element = signals_array[strip_index];
        
        // Remove the element from the array
        var remaining_signals = signals_array.slice(); // Copy
        remaining_signals.splice(strip_index, 1);
        
        // Convert back to appropriate output format
        if (is_dict) {
            // Reconstruct dictionary from remaining array
            var remaining_dict = {};
            remaining_signals.forEach(function(item) {
                remaining_dict[item.key] = item.value;
            });
            
            // Output remaining signals as dictionary
            this.setOutputData(0, remaining_dict);
            
            // Output stripped element (just the value, or key-value pair?)
            this.setOutputData(1, stripped_element ? stripped_element.value : null);
        } else {
            // Output remaining signals as array
            this.setOutputData(0, remaining_signals);
            
            // Output stripped element
            this.setOutputData(1, stripped_element);
        }
    };
    
    StripNode.prototype.onDrawForeground = function(ctx) {
        // Display current mode on the node
        const mode = this.properties.mode || "first";
        const index = Math.floor(this.properties.index || 0); // Ensure integer display
        
        let modeText = `Mode: ${mode}`;
        if (mode === "index") {
            modeText = `Mode: index[${index}]`;
        }
        
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.fillText(modeText, 10, this.size[1] - 10);
    };
    
    LiteGraph.registerNodeType("routing/strip", StripNode);
    
    // CloneNode
    function CloneNode() {
        this.addInput("signals", "signal");
        this.addOutput("signals", "signal");
        this.addOutput("cloned", "signal");
        
        this.properties = {
            mode: "first",
            index: 0
        };
        
        this.color = "#9B59B6";
        this.bgcolor = "#6C3483";
        this.size = [180, 80];
    }
    
    CloneNode.title = "Clone";
    CloneNode.desc = "Copy index from signal stream without modifying original";
    
    CloneNode.prototype.onAdded = function() {
        // Add mode selection widget
        this.addWidget("combo", "Mode", this.properties.mode, 
            function(v) { 
                this.properties.mode = v;
                this.setDirtyCanvas(true);
            }.bind(this),
            { values: ["first", "last", "index"] }
        );
        
        // Add index widget (only relevant when mode is "index")
        this.addWidget("number", "Index", this.properties.index,
            function(v) { 
                this.properties.index = Math.max(0, Math.floor(v)); // Ensure integer
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 1 }
        );
    };
    
    CloneNode.prototype.onExecute = function() {
        // Get input signals
        var input_signals = this.getInputData(0);
        
        // Handle different input types
        var signals_array = null;
        var is_dict = false;
        
        if (!input_signals) {
            // No input - output empty
            this.setOutputData(0, null);
            this.setOutputData(1, null);
            return;
        }
        
        // Check if input is an array
        if (Array.isArray(input_signals)) {
            signals_array = input_signals.slice(); // Copy array
        }
        // Check if input is a signals dictionary (object with string keys)
        else if (typeof input_signals === "object" && input_signals !== null) {
            is_dict = true;
            // Convert dictionary to array for processing
            var keys = Object.keys(input_signals);
            signals_array = keys.map(function(key) {
                return { key: key, value: input_signals[key] };
            });
        }
        // Single value - treat as array with one element
        else {
            signals_array = [input_signals];
        }
        
        if (!signals_array || signals_array.length === 0) {
            // Empty input
            this.setOutputData(0, is_dict ? {} : []);
            this.setOutputData(1, null);
            return;
        }
        
        // Determine which index to clone
        var clone_index = 0;
        var mode = this.properties.mode || "first";
        
        switch(mode) {
            case "first":
                clone_index = 0;
                break;
            case "last":
                clone_index = signals_array.length - 1;
                break;
            case "index":
                var idx = Math.max(0, Math.floor(this.properties.index || 0));
                clone_index = Math.min(signals_array.length - 1, idx);
                break;
        }
        
        // Extract the cloned element (without removing from original)
        var cloned_element = signals_array[clone_index];
        
        // Output original signals unchanged
        if (is_dict) {
            // Output original dictionary
            this.setOutputData(0, input_signals);
            
            // Output cloned element (just the value)
            this.setOutputData(1, cloned_element ? cloned_element.value : null);
        } else {
            // Output original array
            this.setOutputData(0, input_signals);
            
            // Output cloned element
            this.setOutputData(1, cloned_element);
        }
    };
    
    CloneNode.prototype.onDrawForeground = function(ctx) {
        // Display current mode on the node
        const mode = this.properties.mode || "first";
        const index = Math.floor(this.properties.index || 0); // Ensure integer display
        
        let modeText = `Mode: ${mode}`;
        if (mode === "index") {
            modeText = `Mode: index[${index}]`;
        }
        
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "11px Arial";
        ctx.textAlign = "left";
        ctx.fillText(modeText, 10, this.size[1] - 10);
    };
    
    LiteGraph.registerNodeType("routing/clone", CloneNode);
    
    // SignalCombinerNode
    function SignalCombinerNode() {
        // INPUTS: Start with 4, can add more dynamically
        this.addInput("signal_1", "*");
        this.addInput("signal_2", "*");
        this.addInput("signal_3", "*");
        this.addInput("signal_4", "*");
        
        // OUTPUT: Combined signals as dictionary/object
        this.addOutput("signals", "object");
        
        this.properties = {
            signal_names: ["signal_1", "signal_2", "signal_3", "signal_4"],
            auto_name: true  // Automatically name signals or use custom names
        };
        
        this.color = "#E67E22";
        this.bgcolor = "#9A5C23";
        this.size = [200, 140];
        
        this.combined_signals = {};
    }
    
    SignalCombinerNode.title = "Signal Combiner";
    SignalCombinerNode.desc = "Combine individual signals into a signal dictionary";
    
    SignalCombinerNode.prototype.onAdded = function() {
        // Auto-naming toggle
        this.addWidget("toggle", "Auto Name", this.properties.auto_name,
            function(v) { 
                this.properties.auto_name = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Add input button
        this.addWidget("button", "Add Input", null,
            function() {
                var idx = this.inputs.length;
                var signal_name = "signal_" + (idx + 1);
                this.addInput(signal_name, "*");
                this.properties.signal_names.push(signal_name);
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Remove input button
        this.addWidget("button", "Remove Input", null,
            function() {
                if (this.inputs.length > 1) {
                    this.removeInput(this.inputs.length - 1);
                    this.properties.signal_names.pop();
                    this.setDirtyCanvas(true);
                }
            }.bind(this)
        );
        
        // Clear all button
        this.addWidget("button", "Clear All", null,
            function() {
                this.combined_signals = {};
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    SignalCombinerNode.prototype.onExecute = function() {
        var signals = {};
        
        // Collect all input values
        for (var i = 0; i < this.inputs.length; i++) {
            var value = this.getInputData(i);
            
            if (value !== undefined && value !== null) {
                // Use custom name if provided, otherwise use input slot name
                var signal_name;
                if (this.properties.auto_name) {
                    signal_name = this.inputs[i].name;
                } else {
                    signal_name = this.properties.signal_names[i] || this.inputs[i].name;
                }
                
                signals[signal_name] = value;
            }
        }
        
        this.combined_signals = signals;
        this.setOutputData(0, signals);
    };
    
    SignalCombinerNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Show number of signals combined
            var count = Object.keys(this.combined_signals).length;
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("Signals: " + count + "/" + this.inputs.length, 10, this.size[1] - 10);
        }
    };
    
    LiteGraph.registerNodeType("routing/combiner", SignalCombinerNode);
}
