import unittest
import json
from app import app, state
from env.grid_world import GridWorld

class TestUnreachable(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_unreachable_goal(self):
        # 1. Provide a map where goal is surrounded
        # Goal is at (9,9). Surround it with obstacles at (8,9) and (9,8).
        # Assuming grid size 10.
        
        # Reset environment first
        self.app.get('/api/init')
        
        # Manually set obstacles to block (9,9)
        # We need to block (9,8) and (8,9)
        blockers = [(9, 8), (8, 9)]
        
        with app.app_context():
            # Accessing global state directly for test setup convenience
            # In a real integration test we would use the set_cell API, but this is faster.
            for r, c in blockers:
                state.env.grid[r, c] = GridWorld.OBSTACLE
                state.env.obstacles.append((r, c))
            
            # Ensure goal is at 9,9
            state.env.goal_pos = (9, 9)
            state.env.grid[9, 9] = GridWorld.GOAL
            if (9, 9) in state.env.obstacles:
                state.env.obstacles.remove((9, 9))
                
            # Ensure start is at 0,0
            state.env.start_pos = (0, 0)
            state.env.grid[0, 0] = GridWorld.START
            if (0, 0) in state.env.obstacles:
                state.env.obstacles.remove((0, 0))

        # 2. Trigger test (which calls generate_results)
        # We need a trained agent or at least an agent exists to call /api/test
        # We can fake an agent or just trigger /api/train for 1 episode to init it.
        self.app.post('/api/train', json={'episodes': 1})
        
        # Now trigger test
        print("Testing unreachable condition...")
        response = self.app.post('/api/test')
        data = json.loads(response.data)
        
        print("Response:", data)
        
        # 3. Assertions
        self.assertTrue(data.get('is_unreachable'), "Should return is_unreachable=True")
        self.assertFalse(data.get('success'), "Success should be False")

if __name__ == '__main__':
    unittest.main()
