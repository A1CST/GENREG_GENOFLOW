// Episode Runner Node - Executes one genome through one full Snake episode
// This node connects to the backend to run actual Python execution

function registerEpisodeRunnerNode() {
    function EpisodeRunnerNode() {
        // INPUTS (order: trust_delta, done, genome, pop_config, mut_config)
        this.addInput("trust_delta", "number");      // Trust delta from Trust Aggregator
        this.addInput("done", "boolean");            // Done signal from environment
        this.addInput("genome", "object");           // Genome configuration
        this.addInput("pop_config", "object");      // Population config
        this.addInput("mut_config", "object");       // Mutator config
        
        // OUTPUTS
        this.addOutput("episode_results", "object"); // Episode results
        this.addOutput("results", "object");         // Alias for backward compatibility
        this.addOutput("episode_done", "boolean");   // True when episode just completed (one frame)
        
        this.properties = {
            name: "episode_runner_1",
            current_episode: 0,
            current_genome: 0,
            backend_url: "ws://localhost:8000/ws",  // WebSocket URL
            auto_run: false,                         // Auto-run when play is pressed
            max_steps: 1000                          // Maximum steps per episode
        };
        
        this.color = "#16A085";
        this.bgcolor = "#0E6655";
        this.size = [280, 180];
        
        // Episode state
        this.is_running = false;
        this.episode_active = false;
        this.episode_results = null;
        this.websocket = null;
        this.step_count = 0;
        
        // Trust accumulation for current episode (from Trust Aggregator input)
        this.accumulated_trust = 0.0;
        this.trust_history = [];  // Optional: track trust over time
        
        // Episode tracking
        this.episodes_completed = 0;
        this.current_genome_index = 0;
        this.current_steps = 0;
        
        // Track episode stats
        this.food_count = 0;
        this.food_eaten = 0;
        this.steps_survived = 0;
        this.stability_score = 0;

        // Episode done flag (true for one frame after episode completes)
        this.episode_just_completed = false;
    }
    
    EpisodeRunnerNode.title = "Episode Runner";
    EpisodeRunnerNode.desc = "Execute one genome through one full Snake episode";
    
    EpisodeRunnerNode.prototype.onAdded = function() {
        // Episodes counter (read-only display)
        this.addWidget("number", "Episodes", this.properties.current_episode,
            function(v) {
                // Read-only, but allow manual setting for testing
                this.properties.current_episode = Math.max(0, Math.floor(v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 10000, step: 1, readonly: false }
        );
        
        // Current Genome display
        this.addWidget("number", "Current Genome", this.properties.current_genome,
            function(v) {
                // Read-only, but allow manual setting for testing
                this.properties.current_genome = Math.max(0, Math.min(499, Math.floor(v)));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 499, step: 1, readonly: false }
        );
        
        // Auto-run toggle
        this.addWidget("toggle", "Auto Run", this.properties.auto_run,
            function(v) {
                this.properties.auto_run = v;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Run Episode button
        this.addWidget("button", "Run Episode", null,
            function() {
                this.runEpisode();
            }.bind(this)
        );
        
        // Stop button
        this.addWidget("button", "Stop", null,
            function() {
                this.stopEpisode();
            }.bind(this)
        );
        
        // Backend URL widget
        this.addWidget("text", "Backend URL", this.properties.backend_url,
            function(v) {
                this.properties.backend_url = v;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    EpisodeRunnerNode.prototype.onExecute = function() {
        // Get trust delta from Trust Aggregator (input index 0)
        var trust_delta = this.getInputData(0);

        // If episode is running and we have a trust delta
        if (this.episode_active && trust_delta !== undefined && trust_delta !== null) {
            // Accumulate trust for current genome
            this.accumulated_trust += trust_delta;

            // Clamp to prevent overflow
            this.accumulated_trust = Math.max(Math.min(this.accumulated_trust, 100000.0), -100000.0);

            // Increment step counter for current episode
            this.current_steps++;

            // Check max steps limit
            if (this.current_steps >= this.properties.max_steps) {
                console.log("EpisodeRunner: Max steps reached, ending episode");
                this.finalizeEpisode();
            }
        }

        // Check if episode ended (done signal from environment, input index 1)
        var done = this.getInputData(1);

        if (done && this.episode_active) {
            console.log("EpisodeRunner: Done signal received, ending episode");
            this.finalizeEpisode();
        }

        // If auto_run is enabled and graph is playing, start episode
        if (this.properties.auto_run && !this.is_running && !this.episode_active) {
            var graph = this.graph;
            if (graph && graph._executionEnabled) {
                this.runEpisode();
            }
        }

        // Output current results if available
        if (this.episode_results) {
            this.setOutputData(0, this.episode_results);
            this.setOutputData(1, this.episode_results);
        }

        // Output episode_done signal (true for one frame when episode completes)
        this.setOutputData(2, this.episode_just_completed);

        // Clear the flag after outputting (edge-triggered signal)
        if (this.episode_just_completed) {
            this.episode_just_completed = false;
        }
    };
    
    EpisodeRunnerNode.prototype.runEpisode = function() {
        if (this.is_running || this.episode_active) {
            console.log("Episode already running");
            return;
        }

        // Get inputs (genome is now at index 2)
        var genome = this.getInputData(2);

        if (!genome) {
            console.error("EpisodeRunner: No genome provided");
            return;
        }

        // Start episode - no WebSocket needed, graph execution drives steps
        this.startEpisode();
        this.is_running = true;

        // Reset environment via global BackendConnection
        if (window.BackendConnection && window.BackendConnection.isConnected()) {
            window.BackendConnection.sendCommand("reset_env", { grid_size: 10 });
            console.log("EpisodeRunner: Reset env for genome", this.current_genome_index);
        }
    };
    
    EpisodeRunnerNode.prototype.startEpisode = function() {
        // Reset trust for new episode
        this.accumulated_trust = 0.0;
        this.trust_history = [];
        this.episode_active = true;
        this.current_steps = 0;
        this.step_count = 0;
        this.food_count = 0;
        this.food_eaten = 0;
        this.steps_survived = 0;
        this.stability_score = 0;
        this.episode_results = null;
        
        // Update current genome index from properties
        this.current_genome_index = this.properties.current_genome;
        
        console.log("[Episode Runner] Starting episode for genome", this.current_genome_index);
        this.setDirtyCanvas(true);
    };
    
    EpisodeRunnerNode.prototype.finalizeEpisode = function() {
        this.episode_active = false;
        this.is_running = false;

        // Set episode_just_completed flag for one frame
        this.episode_just_completed = true;

        // Prepare results
        var results = {
            genome_id: this.current_genome_index,
            genome_index: this.current_genome_index,
            trust: this.accumulated_trust,
            food_eaten: this.food_count || this.food_eaten || 0,  // From environment
            steps_survived: this.current_steps || this.step_count || 0,
            stability_score: this.stability_score || 0.0,
            episode_number: this.properties.current_episode
        };

        // Store results
        this.episode_results = results;

        // Output results
        this.setOutputData(0, results);
        this.setOutputData(1, results);

        // Update episode counter
        this.episodes_completed++;
        this.properties.current_episode++;

        console.log("[Episode Runner] Episode finished: trust=" + results.trust.toFixed(2) + ", steps=" + results.steps_survived);
        this.setDirtyCanvas(true);
    };
    
    EpisodeRunnerNode.prototype.connectToBackend = function(genome, pop_config, mut_config) {
        try {
            // Close existing connection if any
            if (this.websocket) {
                this.websocket.close();
            }
            
            // Create WebSocket connection
            this.websocket = new WebSocket(this.properties.backend_url);
            
            var self = this;
            
            this.websocket.onopen = function() {
                console.log("EpisodeRunner: Connected to backend");
                
                // Send episode request
                var request = {
                    type: "run_episode",
                    genome: genome,
                    pop_config: pop_config || {},
                    mut_config: mut_config || {},
                    genome_index: self.properties.current_genome,
                    max_steps: self.properties.max_steps
                };
                
                self.websocket.send(JSON.stringify(request));
            };
            
            this.websocket.onmessage = function(event) {
                try {
                    var response = JSON.parse(event.data);
                    self.handleBackendResponse(response);
                } catch (error) {
                    console.error("EpisodeRunner: Error parsing response:", error);
                }
            };
            
            this.websocket.onerror = function(error) {
                console.error("EpisodeRunner: WebSocket error:", error);
                self.is_running = false;
                self.episode_results = {
                    error: "WebSocket connection error",
                    trust: 0,
                    food_eaten: 0,
                    steps_survived: 0,
                    stability_score: 0
                };
                self.setDirtyCanvas(true);
            };
            
            this.websocket.onclose = function() {
                console.log("EpisodeRunner: WebSocket closed");
                self.is_running = false;
                self.websocket = null;
            };
            
        } catch (error) {
            console.error("EpisodeRunner: Error connecting to backend:", error);
            this.is_running = false;
            this.episode_results = {
                error: "Connection error: " + error.message,
                trust: 0,
                food_eaten: 0,
                steps_survived: 0,
                stability_score: 0
            };
            this.setDirtyCanvas(true);
        }
    };
    
    EpisodeRunnerNode.prototype.handleBackendResponse = function(response) {
        if (response.type === "episode_step") {
            // Update step information from backend (for compatibility)
            this.step_count = response.step || this.step_count;
            
            // Note: current_steps is incremented in onExecute when trust_delta is received
            // Backend provides step info, but we track steps via trust_delta input in onExecute
            
            // Trust is accumulated ONLY from Trust Aggregator input in onExecute
            // Backend provides step/food info, but NOT trust (trust comes from input)
            
            // Update food count if provided
            if (response.food_count !== undefined) {
                this.food_count = response.food_count;
                this.food_eaten = response.food_count;
            }
            
            this.setDirtyCanvas(true);
            
        } else if (response.type === "episode_complete") {
            // Episode finished - use finalizeEpisode if episode is active, otherwise just update
            if (this.episode_active) {
                // Update stats from response
                if (response.food_eaten !== undefined) {
                    this.food_eaten = response.food_eaten;
                    this.food_count = response.food_eaten;
                }
                if (response.steps_survived !== undefined) {
                    this.steps_survived = response.steps_survived;
                    this.current_steps = response.steps_survived;
                    this.step_count = response.steps_survived;
                }
                if (response.stability_score !== undefined) {
                    this.stability_score = response.stability_score;
                }
                // Trust comes from accumulated_trust (set in onExecute from trust_delta input)
                // Don't override it from backend response - trust should only come from input
                
                this.finalizeEpisode();
            } else {
                // Fallback if episode_active wasn't set
                this.is_running = false;
                
                // Store results
                this.episode_results = {
                    trust: response.trust || this.accumulated_trust || 0,
                    food_eaten: response.food_eaten || this.food_eaten || 0,
                    steps_survived: response.steps_survived || this.step_count || this.current_steps,
                    stability_score: response.stability_score || 0,
                    genome_index: this.properties.current_genome,
                    episode_number: this.properties.current_episode
                };
            }
            
            // Update episode counter
            this.properties.current_episode++;
            
            // Close WebSocket
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            console.log("EpisodeRunner: Episode complete", this.episode_results);
            this.setDirtyCanvas(true);
            
        } else if (response.type === "error") {
            // Error occurred
            this.is_running = false;
            this.episode_active = false;
            this.episode_results = {
                error: response.message || "Unknown error",
                trust: this.accumulated_trust || 0,
                food_eaten: 0,
                steps_survived: 0,
                stability_score: 0
            };
            
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            console.error("EpisodeRunner: Error from backend:", response.message);
            this.setDirtyCanvas(true);
        }
    };
    
    EpisodeRunnerNode.prototype.stopEpisode = function() {
        if (this.is_running || this.episode_active) {
            // Close WebSocket connection
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            this.is_running = false;
            this.episode_active = false;
            console.log("EpisodeRunner: Episode stopped");
            this.setDirtyCanvas(true);
        }
    };
    
    EpisodeRunnerNode.prototype.onDrawForeground = function(ctx) {
        if (this.flags.collapsed) return;
        
        ctx.textAlign = "left";
        
        var y = this.size[1] - 40;
        
        // Display accumulated trust (live during episode or final after)
        if (this.episode_active || this.is_running) {
            // Show running status
            ctx.fillStyle = "#2ECC71";
            ctx.font = "bold 12px Arial";
            ctx.fillText("● Running", 10, this.size[1] - 10);
            
            // Display accumulated trust
            ctx.fillStyle = this.accumulated_trust > 0 ? "#2ECC71" : "#E74C3C";
            ctx.font = "bold 12px Arial";
            ctx.fillText("Trust: " + this.accumulated_trust.toFixed(2), 10, y);
            
            // Show step count
            y += 15;
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.fillText("Steps: " + this.current_steps, 10, y);
        } else {
            // Not running
            ctx.fillStyle = "#95A5A6";
            ctx.font = "10px Arial";
            ctx.fillText("Ready", 10, this.size[1] - 10);
            
            // Show last results if available
            if (this.episode_results) {
                ctx.fillStyle = "#ECF0F1";
                ctx.font = "9px Arial";
                if (this.episode_results.error) {
                    ctx.fillStyle = "#E74C3C";
                    ctx.fillText("Error: " + this.episode_results.error, 10, y);
                } else {
                    // Display final trust
                    ctx.fillStyle = this.episode_results.trust > 0 ? "#2ECC71" : "#E74C3C";
                    ctx.font = "bold 12px Arial";
                    ctx.fillText("Trust: " + this.episode_results.trust.toFixed(2), 10, y);
                    
                    ctx.fillStyle = "#ECF0F1";
                    ctx.font = "9px Arial";
                    ctx.fillText("Food: " + this.episode_results.food_eaten, 120, y);
                    ctx.fillText("Steps: " + this.episode_results.steps_survived, 10, y - 12);
                }
            } else if (this.accumulated_trust !== 0) {
                // Show accumulated trust even if no episode_results yet
                ctx.fillStyle = this.accumulated_trust > 0 ? "#2ECC71" : "#E74C3C";
                ctx.font = "bold 12px Arial";
                ctx.fillText("Trust: " + this.accumulated_trust.toFixed(2), 10, y);
            }
        }
    };
    
    EpisodeRunnerNode.prototype.onRemoved = function() {
        // Clean up WebSocket connection
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    };
    
    LiteGraph.registerNodeType("episode/runner", EpisodeRunnerNode);
}

// Generation Manager Node - Handles full generation cycles
function registerGenerationManagerNode() {
    function GenerationManagerNode() {
        // INPUTS
        this.addInput("episode_results", "object");  // Results from Episode Runner
        this.addInput("pop_config", "object");      // Population Controller config
        this.addInput("mut_config", "object");       // Mutator Controller config
        
        // OUTPUTS
        this.addOutput("evolution_stats", "object"); // Evolution statistics
        this.addOutput("new_population", "object");  // New population after evolution
        
        this.properties = {
            name: "generation_manager_1",
            current_generation: 0,
            population_size: 500,
            genomes_completed: 0,
            backend_url: "ws://localhost:8000/ws",
            auto_evolve: true  // Automatically evolve when all genomes complete
        };
        
        this.color = "#8E44AD";
        this.bgcolor = "#5B2C6F";
        this.size = [300, 200];
        
        // Generation state
        this.is_evolving = false;
        this.episode_results_buffer = [];  // Collect results from all genomes
        this.evolution_stats = null;
        this.new_population = null;
        
        // Statistics tracking
        this.best_trust = 0;
        this.median_trust = 0;
        this.worst_trust = 0;
        this.total_food_eaten = 0;
        this.total_steps_survived = 0;
        
        // WebSocket connection
        this.websocket = null;
    }
    
    GenerationManagerNode.title = "Generation Manager";
    GenerationManagerNode.desc = "Manages full generation cycles (all genomes play, then evolve)";
    
    GenerationManagerNode.prototype.onAdded = function() {
        // Generation counter (read-only display)
        this.addWidget("number", "Generation", this.properties.current_generation,
            function(v) {
                // Read-only, but allow manual setting for testing
                this.properties.current_generation = Math.max(0, Math.floor(v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 10000, step: 1, readonly: false }
        );
        
        // Population size
        this.addWidget("number", "Population Size", this.properties.population_size,
            function(v) {
                this.properties.population_size = Math.max(1, Math.floor(v));
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 1, max: 10000, step: 1 }
        );
        
        // Genomes completed counter
        this.addWidget("number", "Genomes Complete", this.properties.genomes_completed,
            function(v) {
                // Read-only display
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: 0, max: 10000, step: 1, readonly: true }
        );
        
        // Best Trust display
        this.addWidget("number", "Best Trust", this.best_trust,
            function(v) {
                // Read-only display
                this.setDirtyCanvas(true);
            }.bind(this),
            { min: -1000, max: 1000, step: 0.01, readonly: true, precision: 2 }
        );
        
        // Auto-evolve toggle
        this.addWidget("toggle", "Auto Evolve", this.properties.auto_evolve,
            function(v) {
                this.properties.auto_evolve = v;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Evolve Now button
        this.addWidget("button", "Evolve Now", null,
            function() {
                this.triggerEvolution();
            }.bind(this)
        );
        
        // Reset Generation button
        this.addWidget("button", "Reset Generation", null,
            function() {
                this.resetGeneration();
            }.bind(this)
        );
        
        // Backend URL widget
        this.addWidget("text", "Backend URL", this.properties.backend_url,
            function(v) {
                this.properties.backend_url = v;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    GenerationManagerNode.prototype.onExecute = function() {
        // Get episode results input
        var episode_results = this.getInputData(0);
        
        if (episode_results && !this.is_evolving) {
            // Check if this is a new result (not already in buffer)
            var is_new = true;
            for (var i = 0; i < this.episode_results_buffer.length; i++) {
                if (this.episode_results_buffer[i].genome_index === episode_results.genome_index &&
                    this.episode_results_buffer[i].episode_number === episode_results.episode_number) {
                    is_new = false;
                    break;
                }
            }
            
            if (is_new) {
                this.addEpisodeResult(episode_results);
            }
        }
        
        // Output current stats
        if (this.evolution_stats) {
            this.setOutputData(0, this.evolution_stats);
        }
        
        if (this.new_population) {
            this.setOutputData(1, this.new_population);
        }
    };
    
    GenerationManagerNode.prototype.addEpisodeResult = function(result) {
        // Add result to buffer
        this.episode_results_buffer.push(result);
        this.properties.genomes_completed = this.episode_results_buffer.length;
        
        // Update best trust
        if (result.trust && result.trust > this.best_trust) {
            this.best_trust = result.trust;
        }
        
        // Check if all genomes have completed
        if (this.episode_results_buffer.length >= this.properties.population_size) {
            if (this.properties.auto_evolve) {
                this.triggerEvolution();
            }
        }
        
        this.setDirtyCanvas(true);
    };
    
    GenerationManagerNode.prototype.triggerEvolution = function() {
        if (this.is_evolving) {
            console.log("GenerationManager: Evolution already in progress");
            return;
        }
        
        if (this.episode_results_buffer.length < this.properties.population_size) {
            console.log("GenerationManager: Not all genomes completed yet (" + 
                       this.episode_results_buffer.length + "/" + this.properties.population_size + ")");
            return;
        }
        
        // Get configs
        var pop_config = this.getInputData(1);
        var mut_config = this.getInputData(2);
        
        // Calculate statistics
        this.calculateStatistics();
        
        // Start evolution
        this.is_evolving = true;
        this.evolvePopulation(pop_config, mut_config);
    };
    
    GenerationManagerNode.prototype.calculateStatistics = function() {
        if (this.episode_results_buffer.length === 0) {
            return;
        }
        
        // Extract trust scores
        var trust_scores = this.episode_results_buffer
            .map(function(r) { return r.trust || 0; })
            .filter(function(t) { return !isNaN(t); })
            .sort(function(a, b) { return a - b; });
        
        if (trust_scores.length > 0) {
            this.worst_trust = trust_scores[0];
            this.best_trust = trust_scores[trust_scores.length - 1];
            
            // Calculate median
            var mid = Math.floor(trust_scores.length / 2);
            if (trust_scores.length % 2 === 0) {
                this.median_trust = (trust_scores[mid - 1] + trust_scores[mid]) / 2;
            } else {
                this.median_trust = trust_scores[mid];
            }
        }
        
        // Calculate totals
        this.total_food_eaten = this.episode_results_buffer.reduce(function(sum, r) {
            return sum + (r.food_eaten || 0);
        }, 0);
        
        this.total_steps_survived = this.episode_results_buffer.reduce(function(sum, r) {
            return sum + (r.steps_survived || 0);
        }, 0);
    };
    
    GenerationManagerNode.prototype.evolvePopulation = function(pop_config, mut_config) {
        try {
            // Close existing connection if any
            if (this.websocket) {
                this.websocket.close();
            }
            
            // Create WebSocket connection
            this.websocket = new WebSocket(this.properties.backend_url);
            
            var self = this;
            
            this.websocket.onopen = function() {
                console.log("GenerationManager: Connected to backend for evolution");
                
                // Prepare evolution request
                var request = {
                    type: "evolve",
                    generation: self.properties.current_generation,
                    episode_results: self.episode_results_buffer,
                    pop_config: pop_config || {},
                    mut_config: mut_config || {},
                    population_size: self.properties.population_size
                };
                
                self.websocket.send(JSON.stringify(request));
            };
            
            this.websocket.onmessage = function(event) {
                try {
                    var response = JSON.parse(event.data);
                    self.handleEvolutionResponse(response);
                } catch (error) {
                    console.error("GenerationManager: Error parsing response:", error);
                }
            };
            
            this.websocket.onerror = function(error) {
                console.error("GenerationManager: WebSocket error:", error);
                self.is_evolving = false;
                self.setDirtyCanvas(true);
            };
            
            this.websocket.onclose = function() {
                console.log("GenerationManager: WebSocket closed");
                self.is_evolving = false;
                self.websocket = null;
            };
            
        } catch (error) {
            console.error("GenerationManager: Error connecting to backend:", error);
            this.is_evolving = false;
            this.setDirtyCanvas(true);
        }
    };
    
    GenerationManagerNode.prototype.handleEvolutionResponse = function(response) {
        if (response.type === "evolution_complete" || response.type === "evolve_complete") {
            // Evolution finished
            this.is_evolving = false;

            // Store evolution statistics
            this.evolution_stats = {
                generation: this.properties.current_generation,
                best_trust: this.best_trust,
                median_trust: this.median_trust,
                worst_trust: this.worst_trust,
                total_food_eaten: this.total_food_eaten,
                total_steps_survived: this.total_steps_survived,
                population_size: this.properties.population_size,
                genomes_completed: this.episode_results_buffer.length
            };

            // Create new population for next generation
            // Format: { genomes: [...], generation: N }
            var new_genomes = [];
            for (var i = 0; i < this.properties.population_size; i++) {
                new_genomes.push({
                    index: i,
                    name: "genome_" + (this.properties.current_generation + 1) + "_" + i,
                    trust: 0,
                    controller: { input_size: 11, hidden_size: 16, output_size: 4 },
                    proteins: []
                });
            }

            this.new_population = {
                genomes: new_genomes,
                generation: this.properties.current_generation + 1
            };

            // Increment generation
            this.properties.current_generation++;

            // Reset for next generation
            this.resetGeneration();

            // Close WebSocket
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }

            console.log("GenerationManager: Evolution complete, new population ready", this.evolution_stats);
            this.setDirtyCanvas(true);
            
        } else if (response.type === "error") {
            // Error occurred
            this.is_evolving = false;
            
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            console.error("GenerationManager: Error from backend:", response.message);
            this.setDirtyCanvas(true);
        }
    };
    
    GenerationManagerNode.prototype.resetGeneration = function() {
        // Clear episode results buffer
        this.episode_results_buffer = [];
        this.properties.genomes_completed = 0;
        
        // Reset statistics
        this.best_trust = 0;
        this.median_trust = 0;
        this.worst_trust = 0;
        this.total_food_eaten = 0;
        this.total_steps_survived = 0;
        
        this.setDirtyCanvas(true);
    };
    
    GenerationManagerNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";
            
            // Show evolution status
            if (this.is_evolving) {
                ctx.fillStyle = "#F39C12";
                ctx.font = "bold 12px Arial";
                ctx.fillText("● Evolving...", 10, this.size[1] - 10);
            } else {
                ctx.fillStyle = "#95A5A6";
                ctx.font = "10px Arial";
                ctx.fillText("Ready", 10, this.size[1] - 10);
            }
            
            // Show progress
            var progress = this.properties.genomes_completed + "/" + this.properties.population_size;
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "10px Arial";
            ctx.fillText("Progress: " + progress, 10, this.size[1] - 25);
            
            // Show statistics if available
            if (this.evolution_stats) {
                ctx.fillStyle = "#ECF0F1";
                ctx.font = "9px Arial";
                var y = this.size[1] - 45;
                ctx.fillText("Best: " + this.best_trust.toFixed(2), 10, y);
                ctx.fillText("Median: " + this.median_trust.toFixed(2), 100, y);
                ctx.fillText("Worst: " + this.worst_trust.toFixed(2), 200, y);
            }
        }
    };
    
    GenerationManagerNode.prototype.onRemoved = function() {
        // Clean up WebSocket connection
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    };
    
    LiteGraph.registerNodeType("generation/manager", GenerationManagerNode);
}

// Backend Connector Node - Handles WebSocket communication with Python backend
function registerBackendConnectorNode() {
    function BackendConnectorNode() {
        // INPUTS
        this.addInput("command", "string");  // Command to send
        this.addInput("data", "object");     // Data to send with command
        
        // OUTPUTS
        this.addOutput("response", "object"); // Response from backend
        this.addOutput("error", "string");    // Error messages
        
        this.properties = {
            name: "backend_connector_1",
            backend_url: "ws://localhost:8000/ws",
            auto_connect: false,
            reconnect_interval: 3000,  // 3 seconds
            max_reconnect_attempts: 5
        };
        
        this.color = "#27AE60";
        this.bgcolor = "#1E8449";
        this.size = [250, 140];
        
        // Connection state
        this.is_connected = false;
        this.websocket = null;
        this.reconnect_timer = null;
        this.reconnect_attempts = 0;
        this.message_callbacks = [];
        this.pending_commands = [];
        
        // Statistics
        this.messages_sent = 0;
        this.messages_received = 0;
        this.last_error = null;
    }
    
    BackendConnectorNode.title = "Backend Connector";
    BackendConnectorNode.desc = "WebSocket connection to Python backend";
    
    BackendConnectorNode.prototype.onAdded = function() {
        // Backend URL widget
        this.addWidget("text", "Backend URL", this.properties.backend_url,
            function(v) {
                this.properties.backend_url = v;
                if (this.is_connected) {
                    this.disconnect();
                }
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Connect/Disconnect button
        this.addWidget("button", "Connect", null,
            function() {
                if (this.is_connected) {
                    this.disconnect();
                } else {
                    this.connect();
                }
            }.bind(this)
        );
        
        // Auto-connect toggle
        this.addWidget("toggle", "Auto Connect", this.properties.auto_connect,
            function(v) {
                this.properties.auto_connect = v;
                if (v && !this.is_connected) {
                    this.connect();
                }
                this.setDirtyCanvas(true);
            }.bind(this)
        );
        
        // Send Test Command button
        this.addWidget("button", "Test Connection", null,
            function() {
                this.sendCommand("ping", { timestamp: Date.now() });
            }.bind(this)
        );
        
        // Clear Stats button
        this.addWidget("button", "Clear Stats", null,
            function() {
                this.messages_sent = 0;
                this.messages_received = 0;
                this.last_error = null;
                this.setDirtyCanvas(true);
            }.bind(this)
        );
    };
    
    BackendConnectorNode.prototype.onExecute = function() {
        // Get command and data from inputs
        var command = this.getInputData(0);
        var data = this.getInputData(1);
        
        if (command && this.is_connected) {
            this.sendCommand(command, data);
        }
    };
    
    BackendConnectorNode.prototype.connect = function() {
        if (this.is_connected) {
            console.log("BackendConnector: Already connected");
            return;
        }
        
        try {
            // Close existing connection if any
            if (this.websocket) {
                this.websocket.close();
            }
            
            // Clear reconnect timer
            if (this.reconnect_timer) {
                clearTimeout(this.reconnect_timer);
                this.reconnect_timer = null;
            }
            
            // Create WebSocket connection
            this.websocket = new WebSocket(this.properties.backend_url);
            
            var self = this;
            
            this.websocket.onopen = function() {
                console.log("BackendConnector: Connected to backend");
                self.is_connected = true;
                self.reconnect_attempts = 0;
                self.last_error = null;
                
                // Send any pending commands
                while (self.pending_commands.length > 0) {
                    var cmd = self.pending_commands.shift();
                    self.sendCommand(cmd.command, cmd.data);
                }
                
                self.setDirtyCanvas(true);
            };
            
            this.websocket.onmessage = function(event) {
                try {
                    var response = JSON.parse(event.data);
                    self.messages_received++;
                    self.handleMessage(response);
                    self.setDirtyCanvas(true);
                } catch (error) {
                    console.error("BackendConnector: Error parsing message:", error);
                    self.last_error = "Parse error: " + error.message;
                    self.setOutputData(1, self.last_error);
                    self.setDirtyCanvas(true);
                }
            };
            
            this.websocket.onerror = function(error) {
                console.error("BackendConnector: WebSocket error:", error);
                self.last_error = "Connection error";
                self.setOutputData(1, self.last_error);
                self.setDirtyCanvas(true);
            };
            
            this.websocket.onclose = function(event) {
                console.log("BackendConnector: WebSocket closed", event.code, event.reason);
                self.is_connected = false;
                self.websocket = null;
                
                // Attempt reconnection if auto_connect is enabled
                if (self.properties.auto_connect && 
                    self.reconnect_attempts < self.properties.max_reconnect_attempts) {
                    self.reconnect_attempts++;
                    console.log("BackendConnector: Attempting reconnect (" + 
                               self.reconnect_attempts + "/" + 
                               self.properties.max_reconnect_attempts + ")");
                    
                    self.reconnect_timer = setTimeout(function() {
                        self.connect();
                    }, self.properties.reconnect_interval);
                }
                
                self.setDirtyCanvas(true);
            };
            
        } catch (error) {
            console.error("BackendConnector: Error connecting:", error);
            this.last_error = "Connection error: " + error.message;
            this.setOutputData(1, this.last_error);
            this.is_connected = false;
            this.setDirtyCanvas(true);
        }
    };
    
    BackendConnectorNode.prototype.disconnect = function() {
        // Disable auto-connect
        this.properties.auto_connect = false;
        
        // Clear reconnect timer
        if (this.reconnect_timer) {
            clearTimeout(this.reconnect_timer);
            this.reconnect_timer = null;
        }
        
        // Close WebSocket
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        
        this.is_connected = false;
        this.reconnect_attempts = 0;
        this.setDirtyCanvas(true);
    };
    
    BackendConnectorNode.prototype.sendCommand = function(command, data) {
        if (!command) {
            console.error("BackendConnector: No command provided");
            return;
        }
        
        // If not connected, queue the command
        if (!this.is_connected) {
            if (this.properties.auto_connect) {
                this.pending_commands.push({ command: command, data: data || {} });
                this.connect();
                return;
            } else {
                console.error("BackendConnector: Not connected. Enable auto-connect or connect manually.");
                this.last_error = "Not connected";
                this.setOutputData(1, this.last_error);
                return;
            }
        }
        
        try {
            // Build command object
            var message = {
                cmd: command
            };
            
            // Add data if provided
            if (data) {
                if (typeof data === "object") {
                    // Merge data into message
                    for (var key in data) {
                        if (data.hasOwnProperty(key)) {
                            message[key] = data[key];
                        }
                    }
                } else {
                    message.data = data;
                }
            }
            
            // Send message
            this.websocket.send(JSON.stringify(message));
            this.messages_sent++;
            this.setDirtyCanvas(true);
            
        } catch (error) {
            console.error("BackendConnector: Error sending command:", error);
            this.last_error = "Send error: " + error.message;
            this.setOutputData(1, this.last_error);
            this.setDirtyCanvas(true);
        }
    };
    
    BackendConnectorNode.prototype.handleMessage = function(response) {
        // Output response
        this.setOutputData(0, response);
        
        // Call registered callbacks
        for (var i = 0; i < this.message_callbacks.length; i++) {
            try {
                this.message_callbacks[i](response);
            } catch (error) {
                console.error("BackendConnector: Error in callback:", error);
            }
        }
    };
    
    BackendConnectorNode.prototype.onMessage = function(callback) {
        if (typeof callback === "function") {
            this.message_callbacks.push(callback);
        }
    };
    
    BackendConnectorNode.prototype.removeMessageCallback = function(callback) {
        var index = this.message_callbacks.indexOf(callback);
        if (index !== -1) {
            this.message_callbacks.splice(index, 1);
        }
    };
    
    // Convenience methods for common commands
    BackendConnectorNode.prototype.resetEnvironment = function() {
        this.sendCommand("reset_env", {});
    };
    
    BackendConnectorNode.prototype.stepEnvironment = function(action) {
        this.sendCommand("step", { action: action });
    };
    
    BackendConnectorNode.prototype.runProteins = function(signals) {
        this.sendCommand("run_proteins", { signals: signals });
    };
    
    BackendConnectorNode.prototype.runController = function(input_vector) {
        this.sendCommand("run_controller", { input_vector: input_vector });
    };
    
    BackendConnectorNode.prototype.evolve = function(pop_config, mut_config) {
        this.sendCommand("evolve", {
            pop_config: pop_config || {},
            mut_config: mut_config || {}
        });
    };
    
    BackendConnectorNode.prototype.onDrawForeground = function(ctx) {
        if (!this.flags.collapsed) {
            ctx.textAlign = "left";
            
            // Show connection status
            if (this.is_connected) {
                ctx.fillStyle = "#2ECC71";
                ctx.font = "bold 12px Arial";
                ctx.fillText("● Connected", 10, this.size[1] - 10);
            } else {
                ctx.fillStyle = "#E74C3C";
                ctx.font = "bold 12px Arial";
                ctx.fillText("○ Disconnected", 10, this.size[1] - 10);
            }
            
            // Show statistics
            ctx.fillStyle = "#ECF0F1";
            ctx.font = "9px Arial";
            ctx.fillText("Sent: " + this.messages_sent + " | Recv: " + this.messages_received, 
                        10, this.size[1] - 25);
            
            // Show last error if any
            if (this.last_error) {
                ctx.fillStyle = "#E74C3C";
                ctx.font = "8px Arial";
                var errorText = this.last_error.length > 30 ? 
                    this.last_error.substring(0, 30) + "..." : this.last_error;
                ctx.fillText(errorText, 10, this.size[1] - 40);
            }
            
            // Show reconnect attempts if reconnecting
            if (this.reconnect_attempts > 0 && !this.is_connected) {
                ctx.fillStyle = "#F39C12";
                ctx.font = "9px Arial";
                ctx.fillText("Reconnecting (" + this.reconnect_attempts + ")", 10, this.size[1] - 55);
            }
        }
    };
    
    BackendConnectorNode.prototype.onRemoved = function() {
        // Clean up
        this.disconnect();
        this.message_callbacks = [];
    };
    
    LiteGraph.registerNodeType("backend/connector", BackendConnectorNode);
}

// Statistics Display Node - Shows live training metrics
function registerStatisticsDisplayNode() {
    function StatisticsDisplayNode() {
        this.addInput("stats", "object");
        
        // Completely empty properties
        this.properties = {};
        
        this.display_data = {
            generation: 0,
            genome_index: 0,
            episodes_completed: 0,
            trust: 0,
            food_count: 0
        };
        
        this.color = "#34495E";
        this.bgcolor = "#2C3E50";
        this.size = [200, 180];
        
        // Force no widgets
        this.widgets_up = false;
        this.clip_area = true; // Force clipping
    }
    
    StatisticsDisplayNode.title = "Statistics Display";
    StatisticsDisplayNode.desc = "Shows live training metrics";
    
    // Override serialize to prevent saving display data as properties
    StatisticsDisplayNode.prototype.serialize = function() {
        return {
            id: this.id,
            type: this.type,
            pos: this.pos,
            size: this.size,
            properties: {} // Always empty
        };
    };
    
    // Override configure to prevent loading bad data
    StatisticsDisplayNode.prototype.configure = function(o) {
        this.pos = o.pos;
        this.size = o.size;
        this.properties = {}; // Ignore saved properties
    };
    
    StatisticsDisplayNode.prototype.onAdded = function() {
        this.widgets = [];
        this.properties = {}; // Force empty again
    };
    
    StatisticsDisplayNode.prototype.onExecute = function() {
        var stats = this.getInputData(0);
        
        if (stats) {
            this.display_data.generation = stats.generation !== undefined ? stats.generation : this.display_data.generation;
            this.display_data.genome_index = stats.genome_index !== undefined ? stats.genome_index : this.display_data.genome_index;
            this.display_data.episodes_completed = stats.episodes_completed !== undefined ? stats.episodes_completed : this.display_data.episodes_completed;
            this.display_data.trust = stats.trust !== undefined ? stats.trust : this.display_data.trust;
            this.display_data.food_count = stats.food_count !== undefined ? stats.food_count : this.display_data.food_count;
        }
    };
    
    // Complete custom rendering - override the node's draw completely
    StatisticsDisplayNode.prototype.onDrawForeground = function(ctx) {
        if (this.flags.collapsed) return;
        
        // Fill background to cover any text LiteGraph tries to draw
        ctx.fillStyle = this.bgcolor || "#2C3E50";
        ctx.fillRect(0, LiteGraph.NODE_TITLE_HEIGHT, this.size[0], this.size[1] - LiteGraph.NODE_TITLE_HEIGHT);
        
        var x = 10;
        var y = 40;
        var lineHeight = 18;
        
        // Header
        ctx.font = "bold 13px Arial";
        ctx.fillStyle = "#3498DB";
        ctx.fillText("Training Statistics", x, y);
        y += lineHeight + 3;
        
        // Separator
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(this.size[0] - 10, y);
        ctx.stroke();
        y += 12;
        
        // Stats
        ctx.font = "11px Arial";
        ctx.fillStyle = "#ECF0F1";
        
        ctx.fillText("Generation: " + this.display_data.generation, x, y);
        y += lineHeight;
        
        ctx.fillText("Genome: " + this.display_data.genome_index, x, y);
        y += lineHeight;
        
        ctx.fillText("Episodes: " + this.display_data.episodes_completed, x, y);
        y += lineHeight + 3;
        
        // Separator
        ctx.strokeStyle = "#555";
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(this.size[0] - 10, y);
        ctx.stroke();
        y += 12;
        
        // Trust
        ctx.fillStyle = "#2ECC71";
        ctx.fillText("Trust: " + this.display_data.trust.toFixed(2), x, y);
        y += lineHeight;
        
        // Food
        ctx.fillStyle = "#F39C12";
        ctx.fillText("Food: " + this.display_data.food_count, x, y);
    };
    
    LiteGraph.registerNodeType("statistics/display", StatisticsDisplayNode);
}

// Combined registration function
function registerEpisodeNodes() {
    registerEpisodeRunnerNode();
    registerGenerationManagerNode();
    // Backend Connector node removed - GUI now connects directly to backend via global BackendConnection
    // registerBackendConnectorNode();
    registerStatisticsDisplayNode();
}

// Auto-register when script loads (if LiteGraph is available)
if (typeof LiteGraph !== "undefined") {
    registerEpisodeNodes();
}

