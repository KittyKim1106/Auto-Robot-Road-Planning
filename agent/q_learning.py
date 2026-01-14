"""
Q-Learning 智能体
使用表格型Q学习算法进行路径规划
"""

import numpy as np
from typing import Tuple, List, Optional
import pickle
import os


class QLearningAgent:
    """
    Q-Learning 智能体类
    
    使用 Q表 存储状态-动作对的价值
    通过 epsilon-greedy 策略进行探索和利用
    """
    
    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        初始化 Q-Learning 智能体
        
        Args:
            n_states: 状态空间大小
            n_actions: 动作空间大小
            learning_rate: 学习率 (alpha)
            discount_factor: 折扣因子 (gamma)
            epsilon: 初始探索率
            epsilon_min: 最小探索率
            epsilon_decay: 探索率衰减系数
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 初始化Q表为0
        self.q_table = np.zeros((n_states, n_actions))
        
        # 训练统计
        self.training_rewards = []
        self.training_steps = []
        self.episode_count = 0
    
    def pos_to_state(self, pos: Tuple[int, int], grid_size: int = 10) -> int:
        """将位置转换为状态索引"""
        return pos[0] * grid_size + pos[1]
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        选择动作 (epsilon-greedy 策略)
        
        Args:
            state: 当前状态索引
            training: 是否在训练模式 (训练时使用epsilon-greedy，否则使用贪心策略)
            
        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索: 随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用: 选择Q值最大的动作
            return int(np.argmax(self.q_table[state]))
    
    def learn(self, 
              state: int, 
              action: int, 
              reward: float, 
              next_state: int, 
              done: bool) -> float:
        """
        Q-Learning 更新规则
        Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
            
        Returns:
            TD误差 (用于调试)
        """
        # 当前Q值
        current_q = self.q_table[state, action]
        
        # 目标Q值
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])
        
        # TD误差
        td_error = target_q - current_q
        
        # 更新Q表
        self.q_table[state, action] = current_q + self.lr * td_error
        
        return td_error
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps: int = 200) -> Tuple[float, int, bool]:
        """
        训练一个episode
        
        Args:
            env: 网格环境
            max_steps: 最大步数
            
        Returns:
            total_reward: 总奖励
            steps: 步数
            success: 是否成功到达终点
        """
        state_pos = env.reset()
        state = self.pos_to_state(state_pos, env.grid_size)
        
        total_reward = 0
        steps = 0
        success = False
        
        for step in range(max_steps):
            # 选择动作
            action = self.choose_action(state, training=True)
            
            # 执行动作
            next_state_pos, reward, done, info = env.step(action)
            next_state = self.pos_to_state(next_state_pos, env.grid_size)
            
            # 学习
            self.learn(state, action, reward, next_state, done)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                if info['event'] == 'goal':
                    success = True
                break
        
        # 衰减探索率
        self.decay_epsilon()
        
        # 记录统计
        self.episode_count += 1
        self.training_rewards.append(total_reward)
        self.training_steps.append(steps)
        
        return total_reward, steps, success
    
    def get_optimal_path(self, env) -> Tuple[List[Tuple[int, int]], float, bool]:
        """
        使用训练好的Q表获取最优路径（带实时避障）
        
        Args:
            env: 网格环境
            
        Returns:
            path: 最优路径 (位置列表)
            total_reward: 总奖励
            success: 是否成功
        """
        state_pos = env.reset()
        state = self.pos_to_state(state_pos, env.grid_size)
        
        path = [state_pos]
        total_reward = 0
        visited = set()
        visited.add(state_pos)
        
        max_steps = env.grid_size * env.grid_size * 2  # 增加步数上限以允许绕行
        
        for _ in range(max_steps):
            # 使用贪心策略 (不探索)
            action = self.choose_action(state, training=False)
            
            # 检查Q-Learning建议的动作是否会撞障碍物
            dr, dc = env.ACTION_EFFECTS[action]
            next_pos = (state_pos[0] + dr, state_pos[1] + dc)
            
            # 如果会撞障碍物或撞墙，尝试用BFS绕行
            if not env.is_valid_pos(next_pos) or next_pos in visited:
                bypass_action = self._find_bypass_action(env, state_pos, visited)
                if bypass_action is not None:
                    action = bypass_action
            
            # 执行动作
            next_state_pos, reward, done, info = env.step(action)
            next_state = self.pos_to_state(next_state_pos, env.grid_size)
            
            total_reward += reward
            
            # 如果位置没变（撞墙/障碍物），尝试其他方向
            if next_state_pos == state_pos:
                bypass_action = self._find_bypass_action(env, state_pos, visited)
                if bypass_action is not None:
                    next_state_pos, reward, done, info = env.step(bypass_action)
                    next_state = self.pos_to_state(next_state_pos, env.grid_size)
                    total_reward += reward
            
            # 检测循环 - 只有在真的走不下去时才判定失败
            if next_state_pos == state_pos:
                # 尝试了绕行还是走不动，失败
                return path, total_reward, False
            
            if next_state_pos not in visited:
                path.append(next_state_pos)
                visited.add(next_state_pos)
            
            state_pos = next_state_pos
            state = next_state
            
            if done:
                return path, total_reward, info['event'] == 'goal'
        
        return path, total_reward, False
    
    def _find_bypass_action(self, env, current_pos: Tuple[int, int], 
                            visited: set) -> Optional[int]:
        """
        使用BFS找到一个可以绕行的动作
        
        Args:
            env: 网格环境
            current_pos: 当前位置
            visited: 已访问的位置集合
            
        Returns:
            绕行动作，如果没有可用动作返回None
        """
        from collections import deque
        
        goal = env.goal_pos
        best_action = None
        best_distance = float('inf')
        
        # 方案1：优先选择离终点更近且未访问的方向
        for action in range(self.n_actions):
            dr, dc = env.ACTION_EFFECTS[action]
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            
            if env.is_valid_pos(next_pos) and next_pos not in visited:
                # 计算到终点的曼哈顿距离
                dist = abs(next_pos[0] - goal[0]) + abs(next_pos[1] - goal[1])
                if dist < best_distance:
                    best_distance = dist
                    best_action = action
        
        if best_action is not None:
            return best_action
        
        # 方案2：如果所有未访问方向都不可行，用BFS找最近的未访问可达格子
        queue = deque([(current_pos, [])])  # (位置, 动作序列)
        bfs_visited = {current_pos}
        
        while queue:
            pos, actions = queue.popleft()
            
            for action in range(self.n_actions):
                dr, dc = env.ACTION_EFFECTS[action]
                next_pos = (pos[0] + dr, pos[1] + dc)
                
                if env.is_valid_pos(next_pos) and next_pos not in bfs_visited:
                    new_actions = actions + [action]
                    
                    # 如果找到未访问过的位置，返回第一步动作
                    if next_pos not in visited:
                        return new_actions[0] if new_actions else None
                    
                    bfs_visited.add(next_pos)
                    queue.append((next_pos, new_actions))
        
        # 方案3：实在没办法，返回任意可走的方向
        for action in range(self.n_actions):
            dr, dc = env.ACTION_EFFECTS[action]
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)
            if env.is_valid_pos(next_pos):
                return action
        
        return None

    
    def get_policy(self, grid_size: int = 10) -> np.ndarray:
        """
        获取当前策略 (每个状态的最优动作)
        
        Args:
            grid_size: 网格大小
            
        Returns:
            policy: 策略矩阵 (grid_size x grid_size)
        """
        policy = np.argmax(self.q_table, axis=1)
        return policy.reshape(grid_size, grid_size)
    
    def get_value_function(self, grid_size: int = 10) -> np.ndarray:
        """
        获取状态价值函数 (每个状态的最大Q值)
        
        Args:
            grid_size: 网格大小
            
        Returns:
            values: 价值函数矩阵 (grid_size x grid_size)
        """
        values = np.max(self.q_table, axis=1)
        return values.reshape(grid_size, grid_size)
    
    def save(self, filepath: str):
        """保存智能体到文件"""
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_steps': self.training_steps,
            'episode_count': self.episode_count,
            'params': {
                'n_states': self.n_states,
                'n_actions': self.n_actions,
                'lr': self.lr,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"智能体已保存到: {filepath}")
    
    def load(self, filepath: str):
        """从文件加载智能体"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.training_rewards = data['training_rewards']
        self.training_steps = data['training_steps']
        self.episode_count = data['episode_count']
        print(f"智能体已从 {filepath} 加载")
        print(f"已训练 {self.episode_count} 个episode")


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.grid_world import GridWorld
    
    # 创建环境和智能体
    env = GridWorld()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions
    )
    
    print("开始训练...")
    
    # 训练100个episode
    for episode in range(100):
        reward, steps, success = agent.train_episode(env)
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}: 奖励={reward:.1f}, 步数={steps}, 成功={success}, epsilon={agent.epsilon:.3f}")
    
    # 获取最优路径
    print("\n获取最优路径:")
    path, total_reward, success = agent.get_optimal_path(env)
    print(f"路径长度: {len(path)}")
    print(f"总奖励: {total_reward}")
    print(f"成功: {success}")
    print(f"路径: {path}")
