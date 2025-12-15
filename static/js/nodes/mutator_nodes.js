// Mutator Node Definitions
function registerMutatorNodes() {
    // MutatorControllerNode
    function MutatorControllerNode() {
        // No inputs - this is a configuration node
        
        // OUTPUT: Mutation config parameters
        this.addOutput("config", "object");
        
        // Actual mutation parameters from genreg_genome.py
        this.properties = {
            base_mutation_rate: 0.1,     // Probability of mutating each parameter
            mutation_scale: 0.1,         // Magnitude of mutations (Gaussian stddev)
            trust_inheritance_rate: 0.9, // How much trust children inherit
            
            // Parameter bounds (for display/reference)
            bounds: {
                gain: [-10.0, 10.0],
                scale: [-10.0, 10.0],
                momentum: [0.0, 1.0],
                // Add other parameter bounds as needed
            }
        };
        
        // Visual properties
        this.color = "#F1C40F";  // Yellow for mutation control
        this.bgcolor = "#B7950B";
        this.size = [240, 200];
    }
    
    MutatorControllerNode.title = "Mutator Controller";
    MutatorControllerNode.desc = "Configure mutation parameters for genome evolution";
    
    MutatorControllerNode.prototype.onAdded = function() {
        // Base mutation rate
        this.addWidget("number", "Mutation Rate", this.properties.base_mutation_rate,
            function(v) { 
                this.properties.base_mutation_rate = Math.max(0, Math.min(1, v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 1, step: 0.01, precision: 2 }
        );
        
        // Mutation scale
        this.addWidget("number", "Mutation Scale", this.properties.mutation_scale,
            function(v) { 
                this.properties.mutation_scale = Math.max(0, v);
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 10, step: 0.1, precision: 2 }
        );
        
        // Trust inheritance rate
        this.addWidget("number", "Trust Inheritance", this.properties.trust_inheritance_rate,
            function(v) { 
                this.properties.trust_inheritance_rate = Math.max(0, Math.min(1, v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 1, step: 0.05, precision: 2 }
        );
    };
    
    MutatorControllerNode.prototype.onExecute = function() {
        // Output the configuration object
        this.setOutputData(0, {
            base_mutation_rate: this.properties.base_mutation_rate,
            mutation_scale: this.properties.mutation_scale,
            trust_inheritance_rate: this.properties.trust_inheritance_rate,
            bounds: this.properties.bounds
        });
    };
    
    MutatorControllerNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Display some key info
            let rate = (this.properties.base_mutation_rate * 100).toFixed(1);
            let scale = this.properties.mutation_scale.toFixed(2);
            
            ctx.fillStyle = "#FFFFFF";
            ctx.font = "11px Arial";
            ctx.textAlign = "left";
            ctx.fillText(`Rate: ${rate}%, Scale: ${scale}`, 10, this.size[1] - 10);
        }
    };
    
    LiteGraph.registerNodeType("mutator/controller", MutatorControllerNode);
}
