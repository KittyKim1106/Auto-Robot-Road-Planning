"""
网格世界环境
10x10 网格，包含起点、终点和障碍物
智能体需要从起点导航到终点，避开障碍物
"""

import numpy as np
import json
import os
from typing import Tuple, List, Optional


class GridWorld:
    """
    网格世界环境类
    
    状态空间: 10x10 = 100 个状态 (机器人位置)
    动作空间: 4 个动作 (上、下、左、右)
    """
    
    # 动作定义
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    # 动作名称映射
    ACTION_NAMES = {0: '上', 1: '下', 2: '左', 3: '右'}
    
    # 动作对应的位移 (行变化, 列变化)
    ACTION_EFFECTS = {
        0: (-1, 0),  # 上
        1: (1, 0),   # 下
        2: (0, -1),  # 左
        3: (0, 1)    # 右
    }
    
    # 地图元素
    EMPTY = 0       # 空地
    OBSTACLE = 1    # 障碍物
    START = 2       # 起点
    GOAL = 3        # 终点
    AGENT = 4       # 智能体当前位置
    
    def __init__(self, 
                 grid_size: int = 10,
                 start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (9, 9),
                 obstacles: Optional[List[Tuple[int, int]]] = None,
                 random_obstacles: bool = False,
                 num_random_obstacles: int = 15,
                 random_start_goal: bool = False,
                 min_start_goal_distance: int = 5):
        """
        初始化网格世界
        
        Args:
            grid_size: 网格大小 (grid_size x grid_size)
            start_pos: 起点位置 (行, 列)
            goal_pos: 终点位置 (行, 列)
            obstacles: 障碍物位置列表，如果为None则使用默认障碍物
            random_obstacles: 是否随机生成障碍物
            num_random_obstacles: 随机障碍物数量
            random_start_goal: 是否随机生成起点和终点
            min_start_goal_distance: 起点和终点的最小距离
        """
        self.grid_size = grid_size
        
        # 状态空间和动作空间大小
        self.n_states = grid_size * grid_size
        self.n_actions = 4
        
        # 初始化网格
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # 先设置障碍物
        if random_obstacles:
            self.obstacles = []
            self._generate_random_obstacles(num_random_obstacles)
        elif obstacles is not None:
            self.obstacles = obstacles
            for obs in obstacles:
                self.grid[obs[0], obs[1]] = self.OBSTACLE
        else:
            # 默认固定障碍物布局
            self.obstacles = [
                (1, 2), (2, 2), (3, 2), (4, 2),
                (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
                (5, 7), (6, 7), (7, 7), (8, 7),
                (1, 8), (7, 3), (8, 1)
            ]
            for obs in self.obstacles:
                self.grid[obs[0], obs[1]] = self.OBSTACLE
        
        # 设置起点和终点
        if random_start_goal:
            self.start_pos, self.goal_pos = self._generate_random_start_goal(min_start_goal_distance)
        else:
            self.start_pos = start_pos
            self.goal_pos = goal_pos
        
        # 标记起点和终点
        self.grid[self.start_pos[0], self.start_pos[1]] = self.START
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        
        # 智能体当前位置
        self.agent_pos = self.start_pos
        
        # 记录轨迹
        self.trajectory = [self.start_pos]
        
        # 步数计数
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2  # 最大步数限制
        
    def _generate_random_obstacles(self, num_obstacles: int):
        """随机生成障碍物（不包括起点终点位置，起点终点后设置）"""
        self.obstacles = []
        attempts = 0
        max_attempts = num_obstacles * 10
        
        while len(self.obstacles) < num_obstacles and attempts < max_attempts:
            row = np.random.randint(0, self.grid_size)
            col = np.random.randint(0, self.grid_size)
            pos = (row, col)
            
            # 确保不在已有障碍物位置
            if pos not in self.obstacles:
                self.obstacles.append(pos)
                self.grid[row, col] = self.OBSTACLE
            
            attempts += 1
    
    def _generate_random_start_goal(self, min_distance: int = 5) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        随机生成起点和终点
        
        Args:
            min_distance: 起点和终点的最小曼哈顿距离
            
        Returns:
            (start_pos, goal_pos): 起点和终点坐标
        """
        # 获取所有可用位置（非障碍物）
        available_positions = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.obstacles:
                    available_positions.append((i, j))
        
        max_attempts = 100
        for _ in range(max_attempts):
            # 随机选择起点
            start_idx = np.random.randint(len(available_positions))
            start = available_positions[start_idx]
            
            # 随机选择终点
            goal_idx = np.random.randint(len(available_positions))
            goal = available_positions[goal_idx]
            
            # 计算曼哈顿距离
            distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            
            # 检查距离是否满足要求
            if distance >= min_distance:
                return start, goal
        
        # 如果找不到满足条件的，选择距离最远的两个点
        best_start, best_goal = available_positions[0], available_positions[-1]
        best_distance = 0
        
        for start in available_positions[:20]:  # 只检查前20个以加快速度
            for goal in available_positions[-20:]:
                distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                if distance > best_distance:
                    best_distance = distance
                    best_start, best_goal = start, goal
        
        return best_start, best_goal

    def reset_obstacles(self, num_obstacles: Optional[int] = None):
        """
        重新随机生成障碍物（保持起点终点不变）
        
        Args:
            num_obstacles: 障碍物数量，默认使用当前数量
        """
        if num_obstacles is None:
            num_obstacles = len(self.obstacles) if self.obstacles else 15
        
        # 清空网格
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 重新生成障碍物
        self._generate_random_obstacles(num_obstacles)
        
        # 重新设置起点和终点
        self.grid[self.start_pos[0], self.start_pos[1]] = self.START
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        
        # 重置智能体位置
        self.agent_pos = self.start_pos
        self.trajectory = [self.start_pos]
        self.steps = 0
    
    def reset_with_new_obstacles(self, num_obstacles: Optional[int] = None) -> Tuple[int, int]:
        """
        重置环境并生成新的随机障碍物
        
        Args:
            num_obstacles: 障碍物数量
            
        Returns:
            初始状态 (起点位置)
        """
        self.reset_obstacles(num_obstacles)
        return self.agent_pos

    def reset(self) -> Tuple[int, int]:
        """
        重置环境，将智能体放回起点
        
        Returns:
            初始状态 (起点位置)
        """
        self.agent_pos = self.start_pos
        self.trajectory = [self.start_pos]
        self.steps = 0
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        执行一步动作
        
        Args:
            action: 动作 (0=上, 1=下, 2=左, 3=右)
            
        Returns:
            next_state: 下一个状态 (位置)
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        self.steps += 1
        
        # 获取动作效果
        dr, dc = self.ACTION_EFFECTS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        
        reward = -1  # 默认每步-1，鼓励找最短路径
        done = False
        info = {'event': 'move'}
        
        # 检查边界
        if new_row < 0 or new_row >= self.grid_size or \
           new_col < 0 or new_col >= self.grid_size:
            # 撞墙，位置不变
            reward = -5
            info['event'] = 'wall'
        # 检查障碍物
        elif self.grid[new_row, new_col] == self.OBSTACLE:
            # 撞到障碍物，位置不变
            reward = -10
            info['event'] = 'obstacle'
        else:
            # 移动成功
            self.agent_pos = (new_row, new_col)
            self.trajectory.append(self.agent_pos)
            
            # 检查是否到达终点
            if self.agent_pos == self.goal_pos:
                reward = 100
                done = True
                info['event'] = 'goal'
        
        # 检查是否超过最大步数
        if self.steps >= self.max_steps:
            done = True
            info['event'] = 'timeout'
        
        info['steps'] = self.steps
        return self.agent_pos, reward, done, info
    
    def pos_to_state(self, pos: Tuple[int, int]) -> int:
        """将位置转换为状态索引"""
        return pos[0] * self.grid_size + pos[1]
    
    def state_to_pos(self, state: int) -> Tuple[int, int]:
        """将状态索引转换为位置"""
        return (state // self.grid_size, state % self.grid_size)
    
    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """检查位置是否有效（在边界内且非障碍物）"""
        row, col = pos
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        if self.grid[row, col] == self.OBSTACLE:
            return False
        return True
    
    def get_valid_actions(self, pos: Tuple[int, int]) -> List[int]:
        """获取某位置的所有有效动作"""
        valid_actions = []
        for action in range(self.n_actions):
            dr, dc = self.ACTION_EFFECTS[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self.is_valid_pos(new_pos):
                valid_actions.append(action)
        return valid_actions
    
    def get_manhattan_distance(self, start: Optional[Tuple[int, int]] = None, 
                               goal: Optional[Tuple[int, int]] = None) -> int:
        """
        计算曼哈顿距离（不考虑障碍物的最短理论距离）
        
        Args:
            start: 起点，默认使用环境起点
            goal: 终点，默认使用环境终点
            
        Returns:
            曼哈顿距离
        """
        if start is None:
            start = self.start_pos
        if goal is None:
            goal = self.goal_pos
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    
    def get_bfs_shortest_path(self, start: Optional[Tuple[int, int]] = None,
                              goal: Optional[Tuple[int, int]] = None) -> int:
        """
        使用BFS计算考虑障碍物的最短路径长度
        
        Args:
            start: 起点，默认使用环境起点
            goal: 终点，默认使用环境终点
            
        Returns:
            最短路径长度，如果无法到达返回-1
        """
        from collections import deque
        
        if start is None:
            start = self.start_pos
        if goal is None:
            goal = self.goal_pos
        
        if start == goal:
            return 0
        
        # BFS
        queue = deque([(start, 0)])  # (位置, 步数)
        visited = {start}
        
        while queue:
            pos, steps = queue.popleft()
            
            # 尝试四个方向
            for action in range(4):
                dr, dc = self.ACTION_EFFECTS[action]
                new_pos = (pos[0] + dr, pos[1] + dc)
                
                # 检查是否到达终点
                if new_pos == goal:
                    return steps + 1
                
                # 检查是否有效且未访问
                if self.is_valid_pos(new_pos) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, steps + 1))
        
        # 无法到达
        return -1
    
    def get_grid_for_display(self) -> np.ndarray:
        """获取用于显示的网格（包含智能体位置）"""
        display_grid = self.grid.copy()
        if self.agent_pos != self.goal_pos:
            display_grid[self.agent_pos[0], self.agent_pos[1]] = self.AGENT
        return display_grid
    
    def render_text(self) -> str:
        """文本方式渲染环境"""
        symbols = {
            self.EMPTY: '.',
            self.OBSTACLE: '█',
            self.START: 'S',
            self.GOAL: 'G',
            self.AGENT: 'A'
        }
        
        display_grid = self.get_grid_for_display()
        lines = []
        lines.append('  ' + ' '.join([str(i) for i in range(self.grid_size)]))
        for i, row in enumerate(display_grid):
            line = f'{i} ' + ' '.join([symbols[cell] for cell in row])
            lines.append(line)
        
        return '\n'.join(lines)
    
    def __str__(self) -> str:
        return self.render_text()
    
    def save_config(self, filepath: str):
        """
        保存环境配置到JSON文件
        
        Args:
            filepath: 保存路径
        """
        config = {
            'grid_size': self.grid_size,
            'start_pos': list(self.start_pos),
            'goal_pos': list(self.goal_pos),
            'obstacles': [list(obs) for obs in self.obstacles]
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"环境配置已保存到: {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'GridWorld':
        """
        从JSON文件加载环境配置
        
        Args:
            filepath: 配置文件路径
            
        Returns:
            GridWorld 环境实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        env = cls(
            grid_size=config['grid_size'],
            start_pos=tuple(config['start_pos']),
            goal_pos=tuple(config['goal_pos']),
            obstacles=[tuple(obs) for obs in config['obstacles']]
        )
        print(f"环境配置已从 {filepath} 加载")
        return env


# 运行此文件生成新的随机环境
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.visualization import Visualizer
    
    # 输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, 'env_config.json')
    grid_path = os.path.join(output_dir, 'grid_world.png')
    
    print("=" * 50)
    print("生成新的随机环境")
    print("=" * 50)
    
    # 创建随机环境（随机障碍物 + 随机起点终点）
    env = GridWorld(
        random_obstacles=True, 
        num_random_obstacles=15,
        random_start_goal=True,
        min_start_goal_distance=5
    )
    
    print(f"\n网格大小: {env.grid_size}x{env.grid_size}")
    print(f"起点: {env.start_pos} (随机生成)")
    print(f"终点: {env.goal_pos} (随机生成)")
    print(f"起点到终点距离: {abs(env.start_pos[0]-env.goal_pos[0]) + abs(env.start_pos[1]-env.goal_pos[1])} 格")
    print(f"障碍物数量: {len(env.obstacles)}")
    print("\n环境地图:")
    print(env)
    
    # 保存配置
    env.save_config(config_path)
    
    # 保存图片
    vis = Visualizer()
    vis.plot_grid(env, title="随机生成的网格环境", show=False, save_path=grid_path)
    
    print(f"\n环境图片已保存到: {grid_path}")
    print("\n现在可以运行 train.py 进行训练了！")
