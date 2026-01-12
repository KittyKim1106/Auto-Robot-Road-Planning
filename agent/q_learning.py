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
        使用训练好的Q表获取最优路径
        
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
        
        max_steps = env.grid_size * env.grid_size
        
        for _ in range(max_steps):
            # 使用贪心策略 (不探索)
            action = self.choose_action(state, training=False)
            
            # 执行动作
            next_state_pos, reward, done, info = env.step(action)
            next_state = self.pos_to_state(next_state_pos, env.grid_size)
            
            total_reward += reward
            
            # 检测循环
            if next_state_pos in visited and not done:
                # 陷入循环，失败
                return path, total_reward, False
            
            if next_state_pos not in visited:
                path.append(next_state_pos)
                visited.add(next_state_pos)
            
            state = next_state
            
            if done:
                return path, total_reward, info['event'] == 'goal'
        
        return path, total_reward, False
    
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
