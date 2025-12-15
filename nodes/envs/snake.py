# ================================================================
# Snake Environment Node - For GENREG Visualizer
# This wraps the SnakeEnv for use in the node graph system
# ================================================================

import random


class SnakeEnvNode:
    """
    Snake environment node for the visualizer.
    Takes in action from previous step, outputs all signals.
    """
    
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.env = SnakeEnvironment(grid_size)
        
        # Store last signals for output
        self.current_signals = self.env.get_signals()
        
        # Track episode stats
        self.episode_count = 0
        self.total_food = 0
        self.best_score = 0
        
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: int (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            signals: dict of all environment signals
            done: bool indicating if episode ended
        """
        signals, done = self.env.step(action)
        self.current_signals = signals
        
        if done:
            # Track stats
            self.episode_count += 1
            self.total_food += self.env.food_eaten
            self.best_score = max(self.best_score, self.env.food_eaten)
            
        return signals, done
    
    def reset(self):
        """Reset the environment to initial state."""
        signals = self.env.reset()
        self.current_signals = signals
        return signals
    
    def get_signals(self):
        """Get current signals without stepping."""
        return self.current_signals
    
    def get_stats(self):
        """Get episode statistics."""
        return {
            'episodes': self.episode_count,
            'total_food': self.total_food,
            'best_score': self.best_score,
            'current_score': self.env.food_eaten,
            'alive': self.env.alive
        }


# ================================================================
# Core Snake Environment (same as genreg_snake_env.py)
# ================================================================

TAIL_ENABLED = False


class SnakeEnvironment:
    def __init__(self, grid_size=10, tail_enabled=None):
        self.grid_size = grid_size
        self.tail_enabled = TAIL_ENABLED if tail_enabled is None else tail_enabled

        # Action remapping state for control inversion experiments
        self.action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.controls_inverted = False

        self.reset()

    def reset(self):
        self.head_x = random.randint(0, self.grid_size - 1)
        self.head_y = random.randint(0, self.grid_size - 1)

        self.food_eaten = 0
        self.tail = [] if self.tail_enabled else []

        self.spawn_food()

        self.steps_alive = 0
        self.alive = True

        self.max_energy = 25
        self.energy = self.max_energy

        self.direction = (0, -1)  # start facing up

        self.last_death_reason = None
        return self.get_signals()

    def spawn_food(self):
        while True:
            fx = random.randint(0, self.grid_size - 1)
            fy = random.randint(0, self.grid_size - 1)
            tail_conflict = self.tail_enabled and (fx, fy) in self.tail
            if (fx != self.head_x or fy != self.head_y) and not tail_conflict:
                break

        self.food_x = fx
        self.food_y = fy

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: int (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            signals: dict of environment signals
            done: bool indicating episode end
        """
        if not self.alive:
            return self.get_signals(), True

        # Remap action through current mapping
        remapped_action = self.action_map[action]

        # Store old head position for tail
        old_head_pos = (self.head_x, self.head_y)

        # Update direction and position
        if remapped_action == 0:
            self.direction = (0, -1)  # up
            self.head_y -= 1
        elif remapped_action == 1:
            self.direction = (0, 1)   # down
            self.head_y += 1
        elif remapped_action == 2:
            self.direction = (-1, 0)  # left
            self.head_x -= 1
        elif remapped_action == 3:
            self.direction = (1, 0)   # right
            self.head_x += 1

        self.steps_alive += 1
        self.energy -= 1

        # Collision with own tail
        if self.tail_enabled and (self.head_x, self.head_y) in self.tail:
            self.alive = False
            self.last_death_reason = "self-collision"
            return self.get_signals(), True

        # Food eaten
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.food_eaten += 1
            self.max_energy += 2
            self.energy = self.max_energy
            self.spawn_food()
        else:
            if self.tail_enabled and len(self.tail) > 0:
                self.tail.pop()

        if self.tail_enabled:
            self.tail.insert(0, old_head_pos)

        # Collision with walls
        if (
            self.head_x < 0
            or self.head_x >= self.grid_size
            or self.head_y < 0
            or self.head_y >= self.grid_size
        ):
            self.alive = False
            self.last_death_reason = "wall collision"
            return self.get_signals(), True

        # Death by energy depletion
        if self.energy <= 0:
            self.alive = False
            self.last_death_reason = "energy depletion"
            return self.get_signals(), True

        self.last_death_reason = None
        return self.get_signals(), False

    def get_signals(self):
        """
        Return environment signals as a dictionary.
        These are the raw inputs that sensor proteins will read.
        """
        food_dx = self.food_x - self.head_x
        food_dy = self.food_y - self.head_y
        dist = abs(food_dx) + abs(food_dy)

        near_wall_x = 1.0 if (self.head_x <= 0 or self.head_x >= self.grid_size - 1) else 0.0
        near_wall_y = 1.0 if (self.head_y <= 0 or self.head_y >= self.grid_size - 1) else 0.0
        near_wall = max(near_wall_x, near_wall_y)

        return {
            "steps_alive": float(self.steps_alive),
            "energy": float(self.energy),
            "dist_to_food": float(dist),
            "head_x": float(self.head_x),
            "head_y": float(self.head_y),
            "food_x": float(self.food_x),
            "food_y": float(self.food_y),
            "food_dx": float(food_dx),
            "food_dy": float(food_dy),
            "near_wall": near_wall,
            "alive": 1.0 if self.alive else 0.0,
        }

    def randomize_controls(self):
        """Randomly permute action mappings."""
        actions = [0, 1, 2, 3]
        shuffled = actions.copy()

        while shuffled == actions:
            random.shuffle(shuffled)

        self.action_map = {actions[i]: shuffled[i] for i in range(4)}
        self.controls_inverted = True
        return self.action_map

    def reset_controls(self):
        """Reset to identity mapping."""
        self.action_map = {0: 0, 1: 1, 2: 2, 3: 3}
        self.controls_inverted = False
        return self.action_map

    def get_mapping_display(self):
        """Return human-readable mapping string."""
        action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        return f"UP→{action_names[self.action_map[0]]} | DOWN→{action_names[self.action_map[1]]} | LEFT→{action_names[self.action_map[2]]} | RIGHT→{action_names[self.action_map[3]]}"