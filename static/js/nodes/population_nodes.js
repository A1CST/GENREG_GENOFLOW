// Population Node Definitions
function registerPopulationNodes() {
    // PopulationControllerNode
    function PopulationControllerNode() {
        // No inputs - this is a configuration node
        
        // OUTPUT: Population config parameters
        this.addOutput("config", "object");
        
        // Properties with defaults
        this.properties = {
            population_size: 100,
            elite_ratio: 0.2,      // Top 20% kept as-is
            reproduction_ratio: 0.3, // 30% via crossover
            cloning_ratio: 0.5      // 50% cloned with mutation
        };
        
        // Visual properties
        this.color = "#E67E22";  // Orange for population control
        this.bgcolor = "#9A5C23";
        this.size = [220, 180];
        
        // Validation state
        this.ratio_valid = true;
    }
    
    PopulationControllerNode.title = "Population Controller";
    PopulationControllerNode.desc = "Configure population evolution parameters";
    
    PopulationControllerNode.prototype.onAdded = function() {
        // Population size
        this.addWidget("number", "Pop Size", this.properties.population_size,
            function(v) { 
                this.properties.population_size = Math.max(1, Math.floor(v)); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 10000, step: 10 }
        );
        
        // Elite ratio
        this.addWidget("number", "Elite %", this.properties.elite_ratio * 100,
            function(v) { 
                this.properties.elite_ratio = Math.max(0, Math.min(1, v / 100));
                this.validateRatios();
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 5, precision: 1 }
        );
        
        // Reproduction ratio
        this.addWidget("number", "Repro %", this.properties.reproduction_ratio * 100,
            function(v) { 
                this.properties.reproduction_ratio = Math.max(0, Math.min(1, v / 100));
                this.validateRatios();
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 5, precision: 1 }
        );
        
        // Cloning ratio
        this.addWidget("number", "Clone %", this.properties.cloning_ratio * 100,
            function(v) { 
                this.properties.cloning_ratio = Math.max(0, Math.min(1, v / 100));
                this.validateRatios();
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 100, step: 5, precision: 1 }
        );
        
        // Validate ratios on initial creation
        this.validateRatios();
    };
    
    PopulationControllerNode.prototype.validateRatios = function() {
        var total = this.properties.elite_ratio + 
                    this.properties.reproduction_ratio + 
                    this.properties.cloning_ratio;
        
        // Store validation state
        this.ratio_valid = Math.abs(total - 1.0) < 0.01;
        
        // Trigger redraw to update visual feedback
        this.setDirtyCanvas(true);
    };
    
    PopulationControllerNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            var total = this.properties.elite_ratio + 
                        this.properties.reproduction_ratio + 
                        this.properties.cloning_ratio;
            
            var text = "Total: " + (total * 100).toFixed(1) + "%";
            var color = Math.abs(total - 1.0) < 0.01 ? "#2ECC71" : "#E74C3C";
            
            ctx.fillStyle = color;
            ctx.font = "12px Arial";
            ctx.textAlign = "left";
            ctx.fillText(text, 10, this.size[1] - 10);
        }
    };
    
    PopulationControllerNode.prototype.onExecute = function() {
        // Output the configuration object
        this.setOutputData(0, {
            population_size: this.properties.population_size,
            elite_ratio: this.properties.elite_ratio,
            reproduction_ratio: this.properties.reproduction_ratio,
            cloning_ratio: this.properties.cloning_ratio
        });
    };
    
    LiteGraph.registerNodeType("population/controller", PopulationControllerNode);
}
