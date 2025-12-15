function registerUtilityNodes() {
    function ChartMonitorNode() {
        // INPUT: Signal to monitor (passthrough)
        this.addInput("signal", "signal");
        
        // OUTPUT: Passthrough (doesn't modify signal)
        this.addOutput("signal", "signal");
        
        this.properties = {
            name: "My Chart",           // Chart name
            history_length: 100,    // Number of points to display
            min_value: -5,          // Y-axis min (auto if null)
            max_value: 5,           // Y-axis max (auto if null)
            auto_scale: true,       // Auto-adjust Y-axis
            line_color: "#2ECC71",  // Chart line color
            show_value: true        // Show current value as text
        };
        
        this.color = "#3498DB";     // Blue for monitoring
        this.bgcolor = "#2C5F7F";
        this.size = [250, 180];
        
        // Data storage
        this.signal_history = [];
        this.current_value = 0;
        
        // Auto-scale tracking
        this.observed_min = 0;
        this.observed_max = 0;
    }
    
    ChartMonitorNode.title = "Chart Monitor";
    ChartMonitorNode.desc = "Visualize signal values over time in the side panel";
    
    ChartMonitorNode.prototype.onAdded = function() {
        this.addWidget("text", "Name", this.properties.name, v => {
            this.properties.name = v;
            if (ChartManager) {
                ChartManager.updateTitle();
            }
        });

        // History length
        this.addWidget("number", "History", this.properties.history_length,
            function(v) { 
                this.properties.history_length = Math.max(10, Math.min(1000, Math.floor(v)));
                if (this.signal_history.length > this.properties.history_length) {
                    this.signal_history = this.signal_history.slice(-this.properties.history_length);
                }
            }.bind(this),
            { min: 10, max: 1000, step: 10 }
        );
        
        // Auto-scale toggle
        this.addWidget("toggle", "Auto Scale", this.properties.auto_scale,
            function(v) { 
                this.properties.auto_scale = v;
                if (v) {
                    this.observed_min = this.current_value;
                    this.observed_max = this.current_value;
                }
                this.updateWidgetVisibility();
            }.bind(this)
        );
        
        // Min value
        this.min_widget = this.addWidget("number", "Y Min", this.properties.min_value,
            function(v) { this.properties.min_value = v; }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        // Max value
        this.max_widget = this.addWidget("number", "Y Max", this.properties.max_value,
            function(v) { this.properties.max_value = v; }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        this.updateWidgetVisibility();
        
        if (ChartManager) {
            ChartManager.register(this);
        }
    };

    ChartMonitorNode.prototype.onRemoved = function() {
        if (ChartManager) {
            ChartManager.unregister(this);
        }
    };
    
    ChartMonitorNode.prototype.updateWidgetVisibility = function() {
        if (this.min_widget) {
            this.min_widget.hidden = this.properties.auto_scale;
        }
        if (this.max_widget) {
            this.max_widget.hidden = this.properties.auto_scale;
        }
    };
    
    ChartMonitorNode.prototype.onExecute = function() {
        var signal = this.getInputData(0);
        
        if (signal !== undefined && signal !== null) {
            this.current_value = signal;
            this.signal_history.push(signal);
            
            if (this.signal_history.length > this.properties.history_length) {
                this.signal_history.shift();
            }
            
            if (this.properties.auto_scale) {
                if (this.signal_history.length === 1) {
                    this.observed_min = signal;
                    this.observed_max = signal;
                } else {
                    this.observed_min = Math.min(this.observed_min, signal);
                    this.observed_max = Math.max(this.observed_max, signal);
                }
            }
            this.setOutputData(0, signal);
        }
    };
    
    LiteGraph.registerNodeType("utility/chart_monitor", ChartMonitorNode);

    // DualChartMonitorNode
    function DualChartMonitorNode() {
        // INPUTS: Two signals to monitor (passthrough)
        this.addInput("signal_A", "signal");
        this.addInput("signal_B", "signal");
        
        // OUTPUTS: Passthrough (doesn't modify signals)
        this.addOutput("signal_A_out", "signal");
        this.addOutput("signal_B_out", "signal");
        
        this.properties = {
            name: "My Dual Chart",           // Chart name
            history_length: 100,    // Number of points to display
            min_value: -5,          // Y-axis min (auto if null)
            max_value: 5,           // Y-axis max (auto if null)
            auto_scale: true,       // Auto-adjust Y-axis
            line_color_A: "#2ECC71", // Chart line color for signal A
            line_color_B: "#E74C3C", // Chart line color for signal B
            show_value: true       // Show current values as text
        };
        
        this.color = "#3498DB";     // Blue for monitoring
        this.bgcolor = "#2C5F7F";
        this.size = [250, 200]; // Slightly larger than single chart
        
        // Data storage for both signals
        this.signal_history_A = [];
        this.current_value_A = 0;
        this.signal_history_B = [];
        this.current_value_B = 0;
        
        // Auto-scale tracking (combined for both signals)
        this.observed_min = 0;
        this.observed_max = 0;
    }
    
    DualChartMonitorNode.title = "Dual Chart Monitor";
    DualChartMonitorNode.desc = "Visualize two signal values over time in the side panel";
    
    DualChartMonitorNode.prototype.onAdded = function() {
        this.addWidget("text", "Name", this.properties.name, v => {
            this.properties.name = v;
            if (ChartManager) {
                ChartManager.updateTitle();
            }
        });

        // History length
        this.addWidget("number", "History", this.properties.history_length,
            function(v) { 
                this.properties.history_length = Math.max(10, Math.min(1000, Math.floor(v)));
                if (this.signal_history_A.length > this.properties.history_length) {
                    this.signal_history_A = this.signal_history_A.slice(-this.properties.history_length);
                }
                if (this.signal_history_B.length > this.properties.history_length) {
                    this.signal_history_B = this.signal_history_B.slice(-this.properties.history_length);
                }
            }.bind(this),
            { min: 10, max: 1000, step: 10 }
        );
        
        // Auto-scale toggle
        this.addWidget("toggle", "Auto Scale", this.properties.auto_scale,
            function(v) { 
                this.properties.auto_scale = v;
                if (v) {
                    // Reset observed min/max based on current values
                    var combined = [];
                    if (this.signal_history_A.length > 0) combined.push(...this.signal_history_A);
                    if (this.signal_history_B.length > 0) combined.push(...this.signal_history_B);
                    if (combined.length > 0) {
                        this.observed_min = Math.min(...combined);
                        this.observed_max = Math.max(...combined);
                    } else {
                        this.observed_min = 0;
                        this.observed_max = 0;
                    }
                }
                this.updateWidgetVisibility();
            }.bind(this)
        );
        
        // Min value
        this.min_widget = this.addWidget("number", "Y Min", this.properties.min_value,
            function(v) { this.properties.min_value = v; }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        // Max value
        this.max_widget = this.addWidget("number", "Y Max", this.properties.max_value,
            function(v) { this.properties.max_value = v; }.bind(this),
            { min: -100, max: 100, step: 0.5, precision: 1 }
        );
        
        // Line color A
        this.addWidget("text", "Color A", this.properties.line_color_A,
            function(v) { this.properties.line_color_A = v; }.bind(this)
        );
        
        // Line color B
        this.addWidget("text", "Color B", this.properties.line_color_B,
            function(v) { this.properties.line_color_B = v; }.bind(this)
        );
        
        // Show value toggle
        this.addWidget("toggle", "Show Value", this.properties.show_value,
            function(v) { 
                this.properties.show_value = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        this.updateWidgetVisibility();
        
        if (ChartManager) {
            ChartManager.register(this);
        }
    };
    
    DualChartMonitorNode.prototype.onRemoved = function() {
        if (ChartManager) {
            ChartManager.unregister(this);
        }
    };
    
    DualChartMonitorNode.prototype.updateWidgetVisibility = function() {
        if (this.min_widget) {
            this.min_widget.hidden = this.properties.auto_scale;
        }
        if (this.max_widget) {
            this.max_widget.hidden = this.properties.auto_scale;
        }
    };
    
    DualChartMonitorNode.prototype.onExecute = function() {
        var signal_A = this.getInputData(0);
        var signal_B = this.getInputData(1);
        
        if (signal_A !== undefined && signal_A !== null) {
            this.current_value_A = signal_A;
            this.signal_history_A.push(signal_A);
            
            if (this.signal_history_A.length > this.properties.history_length) {
                this.signal_history_A.shift();
            }
            this.setOutputData(0, signal_A);
        }

        if (signal_B !== undefined && signal_B !== null) {
            this.current_value_B = signal_B;
            this.signal_history_B.push(signal_B);
            
            if (this.signal_history_B.length > this.properties.history_length) {
                this.signal_history_B.shift();
            }
            this.setOutputData(1, signal_B);
        }
            
        if (this.properties.auto_scale) {
            // Combine history for auto-scaling
            var combined_history = [];
            if (this.signal_history_A.length > 0) {
                combined_history = combined_history.concat(this.signal_history_A);
            }
            if (this.signal_history_B.length > 0) {
                combined_history = combined_history.concat(this.signal_history_B);
            }

            if (combined_history.length === 0) {
                this.observed_min = 0;
                this.observed_max = 0;
            } else if (combined_history.length === 1) {
                this.observed_min = combined_history[0];
                this.observed_max = combined_history[0];
            } else {
                // Find min and max efficiently
                var min_val = combined_history[0];
                var max_val = combined_history[0];
                for (var i = 1; i < combined_history.length; i++) {
                    var val = combined_history[i];
                    if (val < min_val) min_val = val;
                    if (val > max_val) max_val = val;
                }
                this.observed_min = min_val;
                this.observed_max = max_val;
            }
        }
    };

    DualChartMonitorNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed && this.properties.show_value) {
            ctx.fillStyle = this.properties.line_color_A;
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("A: " + this.current_value_A.toFixed(3), 10, this.size[1] - 25);

            ctx.fillStyle = this.properties.line_color_B;
            ctx.fillText("B: " + this.current_value_B.toFixed(3), 10, this.size[1] - 10);
        }
    };
    
    LiteGraph.registerNodeType("utility/dual_chart_monitor", DualChartMonitorNode);
    
    // ConsoleOutputNode
    function ConsoleOutputNode() {
        // INPUT: Any type of data
        this.addInput("data", "*");  // "*" accepts any type
        
        // No outputs - this is a sink/terminal node
        
        this.properties = {
            label: "debug",           // Prefix label for console output
            log_level: "log",         // log, info, warn, error
            throttle_ms: 0,           // Throttle logging (0 = no throttle)
            show_timestamp: true,     // Include timestamp in output
            show_type: false,          // Show data type in output
            disabled: false
        };
        
        this.color = "#95A5A6";       // Gray for utility
        this.bgcolor = "#5D6D7E";
        this.size = [180, 100];
        
        // Throttling state
        this.last_log_time = 0;
        this.log_count = 0;
        
        // Unique node ID for tracking
        this.console_node_id = null;
    }
    
    ConsoleOutputNode.title = "Console Output";
    ConsoleOutputNode.desc = "Log data to browser console";
    
    ConsoleOutputNode.prototype.onAdded = function() {
        // Generate unique ID for this console node
        if (!this.console_node_id) {
            this.console_node_id = "console_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9);
        }
        
        // Label for identifying output
        this.addWidget("text", "Label", this.properties.label,
            function(v) { 
                this.properties.label = v; 
            }.bind(this)
        );
        
        // Log level selection
        this.addWidget("combo", "Level", this.properties.log_level,
            function(v) { 
                this.properties.log_level = v; 
            }.bind(this),
            { values: ["log", "info", "warn", "error"] }
        );
        
        // Throttle (milliseconds between logs)
        this.addWidget("number", "Throttle (ms)", this.properties.throttle_ms,
            function(v) { 
                this.properties.throttle_ms = Math.max(0, Math.floor(v)); 
            }.bind(this),
            { min: 0, max: 10000, step: 100 }
        );
        
        // Show timestamp toggle
        this.addWidget("toggle", "Timestamp", this.properties.show_timestamp,
            function(v) { 
                this.properties.show_timestamp = v; 
            }.bind(this)
        );
        
        // Show type toggle
                this.addWidget("toggle", "Show Type", this.properties.show_type,
            function(v) { 
                this.properties.show_type = v; 
            }.bind(this)
        );
        
        this.addWidget("toggle", "Disabled", this.properties.disabled,
            function(v) { 
                this.properties.disabled = v; 
            }.bind(this)
        );
        
        // Reset count button
        this.addWidget("button", "Reset Count", null,
            function() {
                this.log_count = 0;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    ConsoleOutputNode.prototype.onExecute = function() {
        if (this.properties.disabled) {
            return;
        }
        var data = this.getInputData(0);
        
        // Check if we should log (throttling)
        var now = Date.now();
        if (this.properties.throttle_ms > 0) {
            if (now - this.last_log_time < this.properties.throttle_ms) {
                return; // Skip this log
            }
        }
        this.last_log_time = now;
        
        // Build log message
        var parts = [];
        
        // Add label
        if (this.properties.label) {
            parts.push("[" + this.properties.label + "]");
        }
        
        // Add timestamp
        if (this.properties.show_timestamp) {
            var ms = now % 1000;
            var msStr = ms < 10 ? "00" + ms : (ms < 100 ? "0" + ms : String(ms));
            var time = new Date().toLocaleTimeString() + "." + msStr;
            parts.push(time);
        }
        
        // Add type info
        if (this.properties.show_type) {
            var type = typeof data;
            if (Array.isArray(data)) type = "array";
            if (data === null) type = "null";
            parts.push("(" + type + ")");
        }
        
        // Add data
        parts.push(data);
        
        // Log to browser console using selected level
        switch(this.properties.log_level) {
            case "info":
                console.info.apply(console, parts);
                break;
            case "warn":
                console.warn.apply(console, parts);
                break;
            case "error":
                console.error.apply(console, parts);
                break;
            case "log":
            default:
                console.log.apply(console, parts);
                break;
        }
        
        // Also log to console panel
        var timestamp = null;
        if (this.properties.show_timestamp) {
            var ms = now % 1000;
            var msStr = ms < 10 ? "00" + ms : (ms < 100 ? "0" + ms : String(ms));
            timestamp = new Date().toLocaleTimeString() + "." + msStr;
        }
        
        var type = null;
        if (this.properties.show_type) {
            type = typeof data;
            if (Array.isArray(data)) type = "array";
            if (data === null) type = "null";
        }
        
        ConsoleLogManager.addLog(
            this.properties.log_level,
            this.properties.label || null,
            timestamp,
            type,
            data,
            this.console_node_id
        );
        
        this.log_count++;
        this.setDirtyCanvas(true);
    };
    
    ConsoleOutputNode.prototype.onDrawForeground = function(ctx) {
        if (this.properties.disabled) {
            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.fillRect(0, 0, this.size[0], this.size[1]);
            ctx.fillStyle = "#F00";
            ctx.textAlign = "center";
            ctx.font = "bold 20px Arial";
            ctx.fillText("DISABLED", this.size[0] * 0.5, this.size[1] * 0.5);
        }
        else if (!this.flags.collapsed) {
            // Show log count
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("Logs: " + this.log_count, 10, this.size[1] - 10);
        }
    };
    
    LiteGraph.registerNodeType("utility/console_output", ConsoleOutputNode);
    
    // TestSignalGeneratorNode
    function TestSignalGeneratorNode() {
        // OUTPUT: Single output containing all 7 signals as an array
        this.addOutput("signals", "signal");
        
        this.properties = {
            mode: "constant",        // constant, sine, random, ramp, pulse, noise
            frequency: 1.0,          // For sine/pulse modes (Hz)
            amplitude: 1.0,          // Signal amplitude
            offset: 0.0,             // DC offset
            // Individual signal values (for constant mode)
            signal_1: 1.0,
            signal_2: 2.0,
            signal_3: 3.0,
            signal_4: 4.0,
            signal_5: 5.0,
            signal_6: 6.0,
            signal_7: 7.0,
            // For random mode
            random_min: -1.0,
            random_max: 1.0,
            // For ramp mode
            ramp_rate: 0.1
        };
        
        this.color = "#F39C12";      // Orange for test/utility
        this.bgcolor = "#9A6324";
        this.size = [220, 200];
        
        // Internal state
        this.time = 0;
        this.ramp_value = 0;
    }
    
    TestSignalGeneratorNode.title = "Test Signal Generator";
    TestSignalGeneratorNode.desc = "Generate test signals for debugging";
    
    TestSignalGeneratorNode.prototype.onAdded = function() {
        // Mode selection
        this.addWidget("combo", "Mode", this.properties.mode,
            function(v) { 
                this.properties.mode = v;
                this.updateWidgetVisibility();
                this.setDirtyCanvas(true);
            }.bind(this),
            { values: ["constant", "sine", "random", "ramp", "pulse", "noise"] }
        );
        
        // Amplitude
        this.amplitude_widget = this.addWidget("number", "Amplitude", this.properties.amplitude,
            function(v) { 
                this.properties.amplitude = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 0.1, precision: 2 }
        );
        
        // Frequency (for sine/pulse)
        this.frequency_widget = this.addWidget("number", "Frequency", this.properties.frequency,
            function(v) { 
                this.properties.frequency = Math.max(0.01, v); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0.01, max: 10, step: 0.1, precision: 2 }
        );
        
        // Offset
        this.offset_widget = this.addWidget("number", "Offset", this.properties.offset,
            function(v) { 
                this.properties.offset = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -100, max: 100, step: 0.1, precision: 2 }
        );
        
        // Individual constant values (only for constant mode)
        this.constant_widgets = [];
        for (var i = 1; i <= 7; i++) {
            var widget = this.addWidget("number", "Signal " + i, this.properties["signal_" + i],
                (function(idx) {
                    return function(v) { 
                        this.properties["signal_" + idx] = v; 
                        this.setDirtyCanvas(true);
                    }.bind(this);
                }.bind(this))(i),
                { min: -100, max: 100, step: 0.1, precision: 2 }
            );
            this.constant_widgets.push(widget);
        }
        
        // Random range (only for random mode)
        this.random_min_widget = this.addWidget("number", "Random Min", this.properties.random_min,
            function(v) { 
                this.properties.random_min = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -100, max: 100, step: 0.1, precision: 2 }
        );
        
        this.random_max_widget = this.addWidget("number", "Random Max", this.properties.random_max,
            function(v) { 
                this.properties.random_max = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -100, max: 100, step: 0.1, precision: 2 }
        );
        
        // Ramp rate (only for ramp mode)
        this.ramp_rate_widget = this.addWidget("number", "Ramp Rate", this.properties.ramp_rate,
            function(v) { 
                this.properties.ramp_rate = v; 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -10, max: 10, step: 0.01, precision: 3 }
        );
        
        // Reset button
        this.addWidget("button", "Reset", null,
            function() {
                this.time = 0;
                this.ramp_value = 0;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        this.updateWidgetVisibility();
    };
    
    TestSignalGeneratorNode.prototype.updateWidgetVisibility = function() {
        var mode = this.properties.mode;
        
        // Show/hide widgets based on mode
        var show_frequency = (mode === "sine" || mode === "pulse");
        var show_constant = (mode === "constant");
        var show_random = (mode === "random" || mode === "noise");
        var show_ramp = (mode === "ramp");
        
        if (this.frequency_widget) {
            this.frequency_widget.hidden = !show_frequency;
        }
        
        if (this.constant_widgets) {
            this.constant_widgets.forEach(function(w) {
                if (w) w.hidden = !show_constant;
            });
        }
        
        if (this.random_min_widget) {
            this.random_min_widget.hidden = !show_random;
        }
        if (this.random_max_widget) {
            this.random_max_widget.hidden = !show_random;
        }
        
        if (this.ramp_rate_widget) {
            this.ramp_rate_widget.hidden = !show_ramp;
        }
    };
    
    TestSignalGeneratorNode.prototype.onExecute = function() {
        var mode = this.properties.mode;
        var amplitude = this.properties.amplitude;
        var offset = this.properties.offset;
        var signals = [];
        
        // Increment time for time-based modes
        this.time += 1/60; // Assuming ~60 FPS execution
        
        switch(mode) {
            case "constant":
                // Each output has its own constant value
                for (var i = 1; i <= 7; i++) {
                    signals.push(this.properties["signal_" + i]);
                }
                break;
                
            case "sine":
                // Each output is a sine wave with different phase
                var freq = this.properties.frequency;
                for (var i = 0; i < 7; i++) {
                    var phase = (i / 7) * Math.PI * 2; // Phase shift per output
                    var value = amplitude * Math.sin(2 * Math.PI * freq * this.time + phase) + offset;
                    signals.push(value);
                }
                break;
                
            case "random":
                // Each output is random within range
                var min = this.properties.random_min;
                var max = this.properties.random_max;
                for (var i = 0; i < 7; i++) {
                    var value = min + Math.random() * (max - min);
                    signals.push(value);
                }
                break;
                
            case "ramp":
                // All outputs ramp up/down at same rate
                this.ramp_value += this.properties.ramp_rate;
                for (var i = 0; i < 7; i++) {
                    signals.push(this.ramp_value * amplitude + offset);
                }
                break;
                
            case "pulse":
                // Square wave pulse
                var freq = this.properties.frequency;
                var pulse = Math.sin(2 * Math.PI * freq * this.time) > 0 ? 1 : -1;
                for (var i = 0; i < 7; i++) {
                    signals.push(pulse * amplitude + offset);
                }
                break;
                
            case "noise":
                // White noise
                var min = this.properties.random_min;
                var max = this.properties.random_max;
                for (var i = 0; i < 7; i++) {
                    // Box-Muller transform for Gaussian noise
                    var u1 = Math.random();
                    var u2 = Math.random();
                    // Avoid log(0) by ensuring u1 > 0
                    if (u1 === 0) u1 = 0.0001;
                    var noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
                    var value = noise * amplitude + offset;
                    // Clamp to range
                    value = Math.max(min, Math.min(max, value));
                    signals.push(value);
                }
                break;
        }
        
        // Output all signals as a single array
        this.setOutputData(0, signals);
        
        // ALSO create a signals dictionary for Sensors to read
        var signals_dict = {
            "signal_1": signals[0],
            "signal_2": signals[1],
            "signal_3": signals[2],
            "signal_4": signals[3],
            "signal_5": signals[4],
            "signal_6": signals[5],
            "signal_7": signals[6]
        };
        
        // Store in properties so Sensors can access it
        this.properties.current_signals = signals_dict;
    };
    
    TestSignalGeneratorNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Display current mode and time
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.textAlign = "left";
            ctx.fillText("Mode: " + this.properties.mode, 10, this.size[1] - 25);
            
            if (this.properties.mode !== "constant") {
                ctx.fillText("Time: " + this.time.toFixed(2) + "s", 10, this.size[1] - 10);
            }
        }
    };
    
    LiteGraph.registerNodeType("utility/test_signal_generator", TestSignalGeneratorNode);
    
    // SignalInspectorNode
    function SignalInspectorNode() {
        // INPUT: Any signal or object (passthrough)
        this.addInput("data", "*");
        
        // OUTPUT: Passthrough unchanged
        this.addOutput("data", "*");
        
        this.properties = {
            show_count: true,        // Show number of signals
            show_list: true,         // Show list of signal names
            max_display: 10          // Max number of signal names to display
        };
        
        this.color = "#9B59B6";      // Purple for inspection
        this.bgcolor = "#6C3483";
        this.size = [200, 120];
        
        // Detected signal information
        this.signal_names = [];
        this.signal_count = 0;
        this.data_type = "none";
    }
    
    SignalInspectorNode.title = "Signal Inspector";
    SignalInspectorNode.desc = "Inspect signal names/structure (not values)";
    
    SignalInspectorNode.prototype.onAdded = function() {
        // Show count toggle
        this.addWidget("toggle", "Show Count", this.properties.show_count,
            function(v) { 
                this.properties.show_count = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Show list toggle
        this.addWidget("toggle", "Show List", this.properties.show_list,
            function(v) { 
                this.properties.show_list = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Max display limit
        this.addWidget("number", "Max Display", this.properties.max_display,
            function(v) { 
                this.properties.max_display = Math.max(1, Math.min(50, Math.floor(v))); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 50, step: 1 }
        );
        
        // Log to console button
        this.addWidget("button", "Log to Console", null,
            function() {
                var data = this.getInputData(0);
                console.log("=== Signal Inspector ===");
                console.log("Type:", this.data_type);
                console.log("Count:", this.signal_count);
                console.log("Names:", this.signal_names);
                console.log("Full Data:", data);
                console.log("========================");
            }.bind(this)
        );
    };
    
    SignalInspectorNode.prototype.onExecute = function() {
        var data = this.getInputData(0);
        
        // Passthrough data unchanged
        this.setOutputData(0, data);
        
        // Analyze the data structure
        this.signal_names = [];
        this.signal_count = 0;
        this.data_type = "none";
        
        if (data === undefined || data === null) {
            this.data_type = "null";
            return;
        }
        
        // Detect data type and extract signal names
        if (typeof data === "object") {
            if (Array.isArray(data)) {
                this.data_type = "array";
                this.signal_count = data.length;
                // For arrays, show indices as "signals"
                for (var i = 0; i < Math.min(data.length, this.properties.max_display); i++) {
                    this.signal_names.push("[" + i + "]");
                }
            } else {
                this.data_type = "object";
                // For objects/dicts, extract keys
                var keys = Object.keys(data);
                this.signal_count = keys.length;
                this.signal_names = keys.slice(0, this.properties.max_display);
            }
        } else if (typeof data === "number") {
            this.data_type = "number";
            this.signal_count = 1;
            this.signal_names = ["<single number>"];
        } else if (typeof data === "string") {
            this.data_type = "string";
            this.signal_count = 1;
            var strPreview = data.length > 20 ? data.substring(0, 20) + "..." : data;
            this.signal_names = ["<string: " + strPreview + ">"];
        } else {
            this.data_type = typeof data;
            this.signal_count = 1;
            this.signal_names = ["<" + this.data_type + ">"];
        }
    };
    
    SignalInspectorNode.prototype.onDrawForeground = function(ctx) {
        if (this.flags.collapsed) return;
        
        var y_offset = this.size[1] - 10;
        var line_height = 12;
        
        // Show data type
        ctx.fillStyle = "#F39C12";
        ctx.font = "bold 10px Arial";
        ctx.textAlign = "left";
        ctx.fillText("Type: " + this.data_type, 10, y_offset);
        y_offset -= line_height;
        
        // Show count if enabled
        if (this.properties.show_count) {
            ctx.fillStyle = "#3498DB";
            ctx.font = "10px Arial";
            ctx.fillText("Signals: " + this.signal_count, 10, y_offset);
            y_offset -= line_height;
        }
        
        // Show signal names list if enabled
        if (this.properties.show_list && this.signal_names.length > 0) {
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "9px monospace";
            
            // Draw signal names (from bottom up)
            var start_y = y_offset;
            for (var i = this.signal_names.length - 1; i >= 0; i--) {
                var name = this.signal_names[i];
                // Truncate long names
                if (name.length > 20) {
                    name = name.substring(0, 17) + "...";
                }
                ctx.fillText("• " + name, 12, y_offset);
                y_offset -= line_height;
                
                // Stop if we run out of space
                if (y_offset < 70) {
                    if (i > 0) {
                        ctx.fillText("• ... (" + (i) + " more)", 12, y_offset);
                    }
                    break;
                }
            }
            
            // Adjust node height dynamically to fit content
            var needed_height = 70 + (this.signal_names.length * line_height) + 30;
            if (this.size[1] < needed_height) {
                this.size[1] = Math.min(needed_height, 400);
            }
        }
    };
    
    LiteGraph.registerNodeType("utility/signal_inspector", SignalInspectorNode);
}
