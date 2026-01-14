from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import json
import uuid
import threading

# Import existing modules
from env.grid_world import GridWorld
from agent.q_learning import QLearningAgent
from utils.visualization import Visualizer

app = Flask(__name__)

# Global state
class GameState:
    def __init__(self):
        self.env = GridWorld()
        self.agent = None
        self.grid_size = 10
        self.is_training = False
        self.training_results = {}
        
        # Ensure static/images directory exists
        self.vis = Visualizer()
        self.images_dir = os.path.join(app.root_path, 'static', 'images')
        os.makedirs(self.images_dir, exist_ok=True)

state = GameState()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['GET'])
def init_game():
    """Initialize or get current state."""
    return get_env_state()

@app.route('/api/set_cell', methods=['POST'])
def set_cell():
    """Update cell type."""
    data = request.json
    row = data.get('row')
    col = data.get('col')
    type_code = data.get('type')  # 0:Empty, 1:Obstacle, 2:Start, 3:Goal
    
    # Logic to update environment
    # Note: env.grid updates, but we also need to update env.obstacles, start_pos, goal_pos lists
    
    if type_code == GridWorld.OBSTACLE:
        if (row, col) not in state.env.obstacles:
            state.env.obstacles.append((row, col))
            state.env.grid[row, col] = GridWorld.OBSTACLE
    elif type_code == GridWorld.EMPTY:
        if (row, col) in state.env.obstacles:
            state.env.obstacles.remove((row, col))
        if state.env.start_pos == (row, col):
            # Cannot clear start pos completely, must set new one or just visual clear
            # For simplicity, if clearing start/goal, we just remove the visual type, 
            # but deep logic might require start/goal to always exist.
            # Here we assume user will set new start/goal.
            pass
        state.env.grid[row, col] = GridWorld.EMPTY
    elif type_code == GridWorld.START:
        # Clear old start
        old_r, old_c = state.env.start_pos
        state.env.grid[old_r, old_c] = GridWorld.EMPTY
        # Set new start
        state.env.start_pos = (row, col)
        state.env.grid[row, col] = GridWorld.START
        # Remove from obstacles if present
        if (row, col) in state.env.obstacles:
            state.env.obstacles.remove((row, col))
            
    elif type_code == GridWorld.GOAL:
        # Clear old goal
        old_r, old_c = state.env.goal_pos
        state.env.grid[old_r, old_c] = GridWorld.EMPTY
        # Set new goal
        state.env.goal_pos = (row, col)
        state.env.grid[row, col] = GridWorld.GOAL
        # Remove from obstacles if present
        if (row, col) in state.env.obstacles:
            state.env.obstacles.remove((row, col))
            
    # Reset agent/training state logic REMOVED to allow generalization testing
    # state.agent will be preserved
    
    return get_env_state()

@app.route('/api/randomize', methods=['POST'])
def randomize():
    """Randomize environment."""
    num_obstacles = request.json.get('num_obstacles', 15)
    
    # Use existing environment methods
    state.env = GridWorld(
        grid_size=state.grid_size,
        random_obstacles=True,
        num_random_obstacles=num_obstacles,
        random_start_goal=True
    )
    
    # Agent preserved for generalization testing
    
    return get_env_state()

@app.route('/api/train', methods=['POST'])
def train():
    """Run training (Reset Agent)."""
    if state.is_training:
        return jsonify({"status": "busy", "message": "正在训练中，请勿重复提交"})
    
    episodes = request.json.get('episodes', 500)
    
    state.is_training = True
    
    try:
        # Initialize NEW Agent (Reset)
        state.agent = QLearningAgent(
            n_states=state.env.n_states,
            n_actions=state.env.n_actions,
            epsilon=1.0
        )
        
        # Training loop
        for _ in range(episodes):
            state.agent.train_episode(state.env)
            
        return generate_results(success_source="training")
        
    except Exception as e:
        state.is_training = False
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['POST'])
def test_generalization():
    """Test existing agent on current environment (Generalization)."""
    if not state.agent:
        return jsonify({"error": "请先点击'重新训练'来获取一个模型，然后再进行测试"}), 400
    
    if state.is_training:
        return jsonify({"status": "busy", "message": "正在训练中..."})

    try:
        return generate_results(success_source="testing")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_results(success_source="training"):
    """Helper to generate visualizations and JSON response."""
    # Check if goal is reachable via BFS first
    bfs_len = state.env.get_bfs_shortest_path()
    if bfs_len == -1:
        # Destination unreachable: Stop training/testing state immediately
        state.is_training = False
        return jsonify({
            "success": False,
            "is_unreachable": True,
            "steps": 0,
            "reward": 0,
            "success_rate": state.agent.training_rewards[-100:].count(100) / 100.0 if len(state.agent.training_rewards) > 0 else 0,
            "images": {}
        })

    # Get Optimal Path (Pure exploitation)
    path, total_reward, success = state.agent.get_optimal_path(state.env)
    
    # Generate Visualization Images
    run_id = str(uuid.uuid4())[:8]
    
    # 1. Optimal Path (Animation GIF)
    path_img = f"path_{run_id}.gif"
    # Use animate_path to generate GIF
    try:
        state.vis.animate_path(state.env, path=path, title="Optimal Path (Current Env)", 
                              save_path=os.path.join(state.images_dir, path_img), interval=300)
    except Exception as e:
        # Fallback to PNG if GIF generation fails
        print(f"GIF generation failed: {e}")
        path_img = f"path_{run_id}.png"
        state.vis.plot_grid(state.env, path=path, title="Optimal Path (Current Env)", show=False, 
                           save_path=os.path.join(state.images_dir, path_img))
    
    # NEW: Generate Static Route Map (Always)
    route_map_img = f"route_map_{run_id}.png"
    state.vis.plot_grid(state.env, path=path, title="Complete Route Map", show=False,
                       save_path=os.path.join(state.images_dir, route_map_img))
    
    # 2. Training Curves (Always show history of the current brain)
    curves_img = f"curves_{run_id}.png"
    state.vis.plot_training_curves(state.agent.training_rewards, state.agent.training_steps, 
                                  title="Training Curves (Model History)", show=False,
                                  save_path=os.path.join(state.images_dir, curves_img))

    # 3. Policy
    policy = state.agent.get_policy(state.env.grid_size)
    policy_img = f"policy_{run_id}.png"
    state.vis.plot_policy(state.env, policy, title="Learned Policy", show=False,
                         save_path=os.path.join(state.images_dir, policy_img))
                         
    # 4. Value Function
    values = state.agent.get_value_function(state.env.grid_size)
    value_img = f"value_{run_id}.png"
    state.vis.plot_value_function(state.env, values, title="Value Function", show=False,
                                 save_path=os.path.join(state.images_dir, value_img))
                                 
    # 5. Efficiency
    manhattan = state.env.get_manhattan_distance()
    bfs = state.env.get_bfs_shortest_path()
    actual_steps = len(path) - 1 if success else 0
    
    manhattan_eff = (manhattan / actual_steps) if actual_steps > 0 else 0
    bfs_eff = (bfs / actual_steps) if actual_steps > 0 and bfs > 0 else 0
    
    eff_img = f"efficiency_{run_id}.png"
    state.vis.plot_path_efficiency(
        labels=['Current Run'],
        avg_steps=[actual_steps],
        manhattan_efficiency=[manhattan_eff*100],
        bfs_efficiency=[bfs_eff*100],
        title="Path Efficiency",
        show=False,
        save_path=os.path.join(state.images_dir, eff_img)
    )
    
    results = {
        "success": success,
        "steps": actual_steps,
        "reward": total_reward,
        # success_rate roughly estimated from training history if training, or just 1/0 if testing? 
        # User output requirement: "Success Rate". If we are testing generalization, 'Success Rate' of the *training* helps understand the model quality.
        "success_rate": state.agent.training_rewards[-100:].count(100) / 100.0 if len(state.agent.training_rewards) > 0 else 0,
        "images": {
            "path": path_img,
            "route_map": route_map_img,
            "curves": curves_img,
            "policy": policy_img,
            "value": value_img,
            "efficiency": eff_img
        }
    }
    
    state.training_results = results
    state.is_training = False
    return jsonify(results)

def get_env_state():
    grid_list = state.env.grid.tolist()
    return jsonify({
        "grid": grid_list,
        "obstacles": [list(x) for x in state.env.obstacles],
        "start": list(state.env.start_pos),
        "goal": list(state.env.goal_pos),
        "size": state.grid_size
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
