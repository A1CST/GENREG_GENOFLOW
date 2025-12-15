// Genome Controller Nodes - Builder, Loader, and Saver
// These nodes handle genome creation, loading, and saving

function registerGenomeControllerNodes() {
    // GenomeBuilderNode
    function GenomeBuilderNode() {
        // INPUTS
        this.addInput("pop_config", "object");
        this.addInput("mut_config", "object");
        this.addInput("population", "object");  // New population from Generation Manager
        this.addInput("advance", "boolean");    // Signal to advance to next genome

        // OUTPUTS
        this.addOutput("genome", "object");
        this.addOutput("pop_config", "object");
        this.addOutput("mut_config", "object");
        this.addOutput("genome_index", "number");  // Current genome index

        this.properties = {
            genome_name: "genome_1",
            auto_build: true,
            population_size: 100,
            current_index: 0
        };

        this.color = "#16A085";
        this.bgcolor = "#0E6655";
        this.size = [210, 146];

        // Internal state
        this.current_pop_config = null;
        this.current_mut_config = null;
        this.current_genome = null;
        this.population = [];  // Array of genomes
        this.last_advance = false;
    }
    
    GenomeBuilderNode.title = "Genome Builder";
    GenomeBuilderNode.desc = "Build genome from current graph";
    
    GenomeBuilderNode.prototype.onAdded = function() {
        // Population size widget
        this.addWidget("number", "Pop Size", this.properties.population_size,
            function(v) {
                this.properties.population_size = Math.max(1, Math.floor(v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 1000, step: 10 }
        );

        // Current index display
        this.addWidget("number", "Current", this.properties.current_index,
            function(v) {
                this.properties.current_index = Math.max(0, Math.min(this.population.length - 1, Math.floor(v)));
                this.updateCurrentGenome();
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 999, step: 1 }
        );

        // Build population button
        this.addWidget("button", "Build Population", null,
            function() {
                this.buildPopulation();
            }.bind(this)
        );

        // Reset button
        this.addWidget("button", "Reset Index", null,
            function() {
                this.properties.current_index = 0;
                this.updateCurrentGenome();
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    GenomeBuilderNode.prototype.buildPopulation = async function() {
        var size = this.properties.population_size;
        console.log("Building population of", size, "genomes");

        this.population = [];
        this.properties.current_index = 0;

        // Request population creation from backend
        if (window.BackendConnection && window.BackendConnection.isConnected()) {
            try {
                var response = await window.BackendConnection.sendCommand("create_population", { size: size });
                if (response && response.success) {
                    // Backend created population, create local references
                    for (var i = 0; i < size; i++) {
                        this.population.push({
                            index: i,
                            name: this.properties.genome_name + "_" + i,
                            trust: 0,
                            controller: { input_size: 11, hidden_size: 16, output_size: 4 },
                            proteins: []
                        });
                    }
                    console.log("Population created via backend:", size, "genomes");
                }
            } catch (e) {
                console.error("Error creating population:", e);
                this.createLocalPopulation(size);
            }
        } else {
            this.createLocalPopulation(size);
        }

        this.updateCurrentGenome();
        this.setDirtyCanvas(true);
    };

    GenomeBuilderNode.prototype.createLocalPopulation = function(size) {
        this.population = [];
        for (var i = 0; i < size; i++) {
            this.population.push({
                index: i,
                name: this.properties.genome_name + "_" + i,
                trust: 0,
                controller: { input_size: 11, hidden_size: 16, output_size: 4 },
                proteins: []
            });
        }
        console.log("Created local population:", size, "genomes");
    };

    GenomeBuilderNode.prototype.updateCurrentGenome = function() {
        if (this.population.length > 0 && this.properties.current_index < this.population.length) {
            this.current_genome = this.population[this.properties.current_index];
        } else {
            this.current_genome = null;
        }
    };

    GenomeBuilderNode.prototype.advanceGenome = function() {
        this.properties.current_index++;
        if (this.properties.current_index >= this.population.length) {
            // Completed all genomes in population
            this.properties.current_index = this.population.length;  // Stay at end
            return false;  // Signal that we're done with population
        }
        this.updateCurrentGenome();
        this.setDirtyCanvas(true);
        return true;  // More genomes to go
    };

    GenomeBuilderNode.prototype.resetForNewGeneration = function() {
        this.properties.current_index = 0;
        this.updateCurrentGenome();
        this.setDirtyCanvas(true);
    };
    
    GenomeBuilderNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";
            ctx.font = "10px Arial";

            // Show population status
            if (this.population.length > 0) {
                ctx.fillStyle = "#2ECC71";
                ctx.fillText("Pop: " + this.population.length + " | Genome: " + this.properties.current_index, 10, this.size[1] - 10);
            } else {
                ctx.fillStyle = "#E74C3C";
                ctx.fillText("No population - click Build", 10, this.size[1] - 10);
            }
        }
    };
    
    GenomeBuilderNode.prototype.onExecute = function() {
        // Get inputs
        var pop_config = this.getInputData(0);
        var mut_config = this.getInputData(1);
        var new_population = this.getInputData(2);  // From Generation Manager
        var advance_signal = this.getInputData(3);  // Advance to next genome

        // Store configs for pass-through
        this.current_pop_config = pop_config;
        this.current_mut_config = mut_config;

        // Handle new population from Generation Manager (after evolution)
        if (new_population && new_population.genomes && new_population.genomes !== this.population) {
            console.log("Received new population from Generation Manager:", new_population.genomes.length, "genomes");
            this.population = new_population.genomes;
            this.properties.current_index = 0;
            this.updateCurrentGenome();
        }

        // Handle advance signal (edge-triggered)
        if (advance_signal && !this.last_advance) {
            this.advanceGenome();
        }
        this.last_advance = advance_signal;

        // Auto-build population if empty and auto_build is enabled
        if (this.population.length === 0 && this.properties.auto_build) {
            this.buildPopulation();
        }

        // Output configs (pass-through)
        this.setOutputData(1, pop_config);
        this.setOutputData(2, mut_config);

        // Output current genome
        this.setOutputData(0, this.current_genome || null);

        // Output current genome index
        this.setOutputData(3, this.properties.current_index);
    };
    
    LiteGraph.registerNodeType("genome/builder", GenomeBuilderNode);
    
    // GenomeLoaderNode
    function GenomeLoaderNode() {
        // INPUT: Genome file path or genome object
        this.addInput("genome", "object");
        this.addInput("pop_config", "object");  // Optional pop_config input
        this.addInput("mut_config", "object");  // Optional mut_config input
        
        // OUTPUTS: Success/failure signal and configs
        this.addOutput("loaded", "boolean");
        this.addOutput("genome", "object");  // Pass through loaded genome
        this.addOutput("pop_config", "object");  // Pass through pop_config
        this.addOutput("mut_config", "object");  // Pass through mut_config
        
        this.properties = {
            genome_file: "",
            auto_layout: true,  // Automatically arrange nodes
            clear_graph: false  // Clear existing nodes before loading
        };
        
        this.color = "#2980B9";  // Blue
        this.bgcolor = "#1F5F8B";
        this.size = [200, 120];
        
        // Store configs for pass-through
        this.current_pop_config = null;
        this.current_mut_config = null;
        this.loaded_genome = null;
    }
    
    GenomeLoaderNode.title = "Genome Loader";
    GenomeLoaderNode.desc = "Load genome and create protein nodes";
    
    GenomeLoaderNode.prototype.onAdded = function() {
        // File path widget
        this.addWidget("text", "File Path", this.properties.genome_file,
            function(v) { 
                this.properties.genome_file = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Auto-layout toggle
        this.addWidget("toggle", "Auto Layout", this.properties.auto_layout,
            function(v) { 
                this.properties.auto_layout = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Clear graph toggle
        this.addWidget("toggle", "Clear First", this.properties.clear_graph,
            function(v) { 
                this.properties.clear_graph = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Load button
        this.addWidget("button", "Load Genome", null,
            function() {
                this.loadGenome();
            }.bind(this)
        );
    };
    
    GenomeLoaderNode.prototype.loadGenome = function() {
        console.log("Loading genome from:", this.properties.genome_file);
        
        // TODO: Implementation will come later
        // Will need to:
        // 1. Load genome data (from file or input)
        // 2. Clear graph if requested
        // 3. Create protein nodes
        // 4. Connect them based on genome structure
        // 5. Auto-layout if requested
    };
    
    GenomeLoaderNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Show load status
            if (this.loaded_genome) {
                ctx.fillStyle = "#2ECC71";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Loaded", 10, this.size[1] - 10);
            } else {
                ctx.fillStyle = "#95A5A6";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Ready", 10, this.size[1] - 10);
            }
        }
    };
    
    GenomeLoaderNode.prototype.onExecute = function() {
        // Get input genome and configs
        var genome = this.getInputData(0);
        var pop_config = this.getInputData(1);
        var mut_config = this.getInputData(2);
        
        // Store for pass-through
        this.current_pop_config = pop_config;
        this.current_mut_config = mut_config;
        
        if (genome) {
            this.loaded_genome = genome;
        }
        
        // Output configs (pass-through)
        this.setOutputData(2, pop_config);
        this.setOutputData(3, mut_config);
        
        // Output loaded genome (pass-through)
        this.setOutputData(1, this.loaded_genome || genome);
        
        // Output success signal
        this.setOutputData(0, this.loaded_genome !== null);
    };
    
    LiteGraph.registerNodeType("genome/loader", GenomeLoaderNode);
    
    // GenomeSaverNode
    function GenomeSaverNode() {
        // INPUT: Genome to save
        this.addInput("genome", "object");
        this.addInput("pop_config", "object");  // Optional pop_config input
        this.addInput("mut_config", "object");  // Optional mut_config input
        
        // OUTPUTS: Success signal and configs
        this.addOutput("saved", "boolean");
        this.addOutput("pop_config", "object");  // Pass through pop_config
        this.addOutput("mut_config", "object");  // Pass through mut_config
        
        this.properties = {
            save_path: "./genomes/",
            filename: "genome.json",
            auto_save: false,  // Auto-save on every change
            save_interval: 10  // Save every N seconds if auto_save is on
        };
        
        this.color = "#E74C3C";  // Red
        this.bgcolor = "#922B21";
        this.size = [200, 140];
        
        // Save state
        this.last_save_time = 0;
        this.save_timer = null;
        
        // Store configs for pass-through
        this.current_pop_config = null;
        this.current_mut_config = null;
    }
    
    GenomeSaverNode.title = "Genome Saver";
    GenomeSaverNode.desc = "Save genome to file";
    
    GenomeSaverNode.prototype.onAdded = function() {
        // Save path widget
        this.addWidget("text", "Save Path", this.properties.save_path,
            function(v) { 
                this.properties.save_path = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Filename widget
        this.addWidget("text", "Filename", this.properties.filename,
            function(v) { 
                this.properties.filename = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Auto-save toggle
        this.addWidget("toggle", "Auto Save", this.properties.auto_save,
            function(v) { 
                this.properties.auto_save = v; 
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Save interval widget (only shown if auto_save is on)
        this.addWidget("number", "Interval (s)", this.properties.save_interval,
            function(v) { 
                this.properties.save_interval = Math.max(1, Math.floor(v)); 
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 3600, step: 1 }
        );
        
        // Manual save button
        this.addWidget("button", "Save Genome", null,
            function() {
                this.saveGenome();
            }.bind(this)
        );
    };
    
    GenomeSaverNode.prototype.saveGenome = function() {
        var genome = this.getInputData(0);
        if (!genome) {
            console.log("GenomeSaver: No genome to save");
            return;
        }
        
        console.log("Saving genome to:", this.properties.save_path + this.properties.filename);
        
        // TODO: Implementation will come later
        // Will need to:
        // 1. Serialize genome object
        // 2. Write to file (or send to backend for saving)
        // 3. Handle errors
    };
    
    GenomeSaverNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            // Show save status
            var now = Date.now();
            if (this.last_save_time > 0 && (now - this.last_save_time) < 2000) {
                ctx.fillStyle = "#2ECC71";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Saved", 10, this.size[1] - 10);
            } else {
                ctx.fillStyle = "#95A5A6";
                ctx.font = "10px Arial";
                ctx.textAlign = "left";
                ctx.fillText("Ready", 10, this.size[1] - 10);
            }
        }
    };
    
    GenomeSaverNode.prototype.onExecute = function() {
        // Get input genome and configs
        var genome = this.getInputData(0);
        var pop_config = this.getInputData(1);
        var mut_config = this.getInputData(2);
        
        // Store for pass-through
        this.current_pop_config = pop_config;
        this.current_mut_config = mut_config;
        
        // Output configs (pass-through)
        this.setOutputData(1, pop_config);
        this.setOutputData(2, mut_config);
        
        // Handle auto-save
        if (this.properties.auto_save && genome) {
            var now = Date.now();
            var interval_ms = this.properties.save_interval * 1000;
            
            if (now - this.last_save_time >= interval_ms) {
                this.saveGenome();
                this.last_save_time = now;
            }
        }
        
        // Output success signal (true if genome exists)
        this.setOutputData(0, genome !== null && genome !== undefined);
    };
    
    LiteGraph.registerNodeType("genome/saver", GenomeSaverNode);
}

// Auto-register when script loads (if LiteGraph is available)
if (typeof LiteGraph !== "undefined") {
    registerGenomeControllerNodes();
}



