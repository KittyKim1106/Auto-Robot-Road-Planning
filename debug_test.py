from app import app, state
from env.grid_world import GridWorld
import json

def run_test():
    with app.app_context():
        # Setup unreachable
        print("Setting up env...")
        state.env.grid[9, 8] = GridWorld.OBSTACLE
        state.env.obstacles.append((9, 8))
        state.env.grid[8, 9] = GridWorld.OBSTACLE
        state.env.obstacles.append((8, 9))
        
        state.env.goal_pos = (9, 9)
        state.env.start_pos = (0, 0)
        
        bfs = state.env.get_bfs_shortest_path()
        print(f"BFS Shortest Path: {bfs}")
        
        if bfs != -1:
            print("WARNING: BFS says reachable! Obstacles might be wrong.")
            print(f"Goal: {state.env.goal_pos}")
            print(f"Obstacles around goal: {state.env.grid[9,8]} {state.env.grid[8,9]}")
        
        # Fake an agent
        from agent.q_learning import QLearningAgent
        state.agent = QLearningAgent(state.env.n_states, state.env.n_actions)
        
        client = app.test_client()
        print("Calling /api/test...")
        rv = client.post('/api/test')
        print(f"Status Code: {rv.status_code}")
        print(f"Response Data: {rv.data.decode('utf-8')}")
        
        data = json.loads(rv.data)
        if data.get('is_unreachable'):
            print("SUCCESS: is_unreachable detected.")
        else:
            print("FAILURE: is_unreachable NOT detected.")

if __name__ == "__main__":
    run_test()
