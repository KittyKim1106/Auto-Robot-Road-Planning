"""
评估脚本
评估训练好的智能体在不同条件下的表现
"""

import os
import sys
import numpy as np
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.grid_world import GridWorld
from agent.q_learning import QLearningAgent
from utils.visualization import Visualizer


def evaluate_single(agent, env, n_episodes=100, verbose=False):
    """
    评估智能体在单个环境上的表现
    
    Args:
        agent: 训练好的智能体
        env: 网格环境
        n_episodes: 评估轮数
        verbose: 是否打印详细信息
        
    Returns:
        success_rate: 成功率
        avg_steps: 平均步数 (仅计算成功的)
        avg_reward: 平均奖励
    """
    successes = 0
    total_steps = []
    total_rewards = []
    
    for episode in range(n_episodes):
        path, reward, success = agent.get_optimal_path(env)
        
        total_rewards.append(reward)
        
        if success:
            successes += 1
            total_steps.append(len(path) - 1)
        
        if verbose and episode < 5:
            print(f"  Episode {episode + 1}: 成功={success}, 路径长度={len(path)}, 奖励={reward:.1f}")
    
    success_rate = successes / n_episodes
    avg_steps = np.mean(total_steps) if total_steps else 0
    avg_reward = np.mean(total_rewards)
    
    return success_rate, avg_steps, avg_reward


def evaluate_generalization(agent, grid_size=10, n_tests=10, verbose=True):
    """
    评估智能体的泛化能力 (使用随机障碍物)
    
    Args:
        agent: 训练好的智能体
        grid_size: 网格大小
        n_tests: 测试环境数量
        verbose: 是否打印详细信息
        
    Returns:
        results: 评估结果列表
    """
    results = []
    
    if verbose:
        print("\n评估泛化能力 (随机障碍物环境)...")
        print("-" * 50)
    
    for i in range(n_tests):
        # 创建随机障碍物环境
        random_env = GridWorld(
            grid_size=grid_size,
            random_obstacles=True,
            num_random_obstacles=15
        )
        
        # 评估
        success_rate, avg_steps, avg_reward = evaluate_single(agent, random_env, n_episodes=1)
        
        results.append({
            'env_id': i + 1,
            'success': success_rate > 0,
            'steps': avg_steps if success_rate > 0 else 0,
            'reward': avg_reward
        })
        
        if verbose:
            status = "成功" if success_rate > 0 else "失败"
            print(f"  环境 {i + 1}: {status}, 步数: {avg_steps:.0f}, 奖励: {avg_reward:.1f}")
    
    return results


def evaluate_different_starts(agent, env, n_starts=10, verbose=True):
    """
    评估从不同起点出发的表现
    
    Args:
        agent: 训练好的智能体
        env: 原始环境
        n_starts: 测试起点数量
        verbose: 是否打印详细信息
        
    Returns:
        results: 评估结果列表
    """
    results = []
    
    if verbose:
        print("\n评估不同起点的表现...")
        print("-" * 50)
    
    # 获取所有可用的起始位置
    available_starts = []
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            if (i, j) not in env.obstacles and (i, j) != env.goal_pos:
                available_starts.append((i, j))
    
    # 随机选择起点
    if len(available_starts) > n_starts:
        selected_starts = np.random.choice(len(available_starts), n_starts, replace=False)
        test_starts = [available_starts[i] for i in selected_starts]
    else:
        test_starts = available_starts[:n_starts]
    
    for start in test_starts:
        # 创建新环境
        test_env = GridWorld(
            grid_size=env.grid_size,
            start_pos=start,
            goal_pos=env.goal_pos,
            obstacles=env.obstacles
        )
        
        # 评估
        success_rate, avg_steps, avg_reward = evaluate_single(agent, test_env, n_episodes=1)
        
        results.append({
            'start': start,
            'success': success_rate > 0,
            'steps': avg_steps if success_rate > 0 else 0,
            'reward': avg_reward
        })
        
        if verbose:
            status = "成功" if success_rate > 0 else "失败"
            print(f"  起点 {start}: {status}, 步数: {avg_steps:.0f}, 奖励: {avg_reward:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Q-Learning 智能体评估")
    
    parser.add_argument("--agent-path", type=str, default="output/agent.pkl", 
                       help="智能体模型路径")
    parser.add_argument("--grid-size", type=int, default=10, help="网格大小")
    parser.add_argument("--n-episodes", type=int, default=100, help="评估轮数")
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument("--show-plots", action="store_true", help="显示图形")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Q-Learning 智能体评估")
    print("=" * 60)
    
    # 加载智能体
    print(f"\n[1] 加载智能体: {args.agent_path}")
    if not os.path.exists(args.agent_path):
        print(f"错误: 找不到智能体文件 {args.agent_path}")
        print("请先运行 train.py 进行训练")
        return
    
    agent = QLearningAgent(n_states=args.grid_size**2, n_actions=4)
    agent.load(args.agent_path)
    
    # 环境配置文件路径
    config_path = os.path.join(args.output_dir, 'env_config.json')
    
    # 加载或创建原始环境
    print(f"\n[2] 加载环境...")
    if os.path.exists(config_path):
        env = GridWorld.load_config(config_path)
    else:
        env = GridWorld(grid_size=args.grid_size)
    
    # 评估原始环境
    print(f"\n[3] 评估原始环境 ({args.n_episodes} 轮)...")
    success_rate, avg_steps, avg_reward = evaluate_single(
        agent, env, n_episodes=args.n_episodes, verbose=True
    )
    
    # 计算原始环境的效率
    manhattan_dist = env.get_manhattan_distance()
    bfs_dist = env.get_bfs_shortest_path()
    
    manhattan_eff = (manhattan_dist / avg_steps * 100) if avg_steps > 0 else 0
    bfs_eff = (bfs_dist / avg_steps * 100) if avg_steps > 0 and bfs_dist > 0 else 0
    
    print(f"\n    原始环境结果:")
    print(f"    - 成功率: {success_rate:.1%}")
    print(f"    - 平均步数: {avg_steps:.1f}")
    print(f"    - 曼哈顿距离: {manhattan_dist}, 曼哈顿效率: {manhattan_eff:.1f}%")
    print(f"    - BFS最短路径: {bfs_dist}, BFS效率: {bfs_eff:.1f}%")
    
    # 评估不同起点
    print(f"\n[4] 评估不同起点...")
    start_results = evaluate_different_starts(agent, env, n_starts=10)
    
    start_success = sum(1 for r in start_results if r['success'])
    start_avg_steps = np.mean([r['steps'] for r in start_results if r['success']]) if start_success > 0 else 0
    
    # 计算不同起点的平均效率
    start_manhattan_effs = []
    start_bfs_effs = []
    for r in start_results:
        if r['success'] and r['steps'] > 0:
            # 计算从该起点的曼哈顿距离
            m_dist = abs(r['start'][0] - env.goal_pos[0]) + abs(r['start'][1] - env.goal_pos[1])
            start_manhattan_effs.append(m_dist / r['steps'] * 100)
            # 计算从该起点的BFS距离
            test_env = GridWorld(
                grid_size=env.grid_size,
                start_pos=r['start'],
                goal_pos=env.goal_pos,
                obstacles=env.obstacles
            )
            b_dist = test_env.get_bfs_shortest_path()
            if b_dist > 0:
                start_bfs_effs.append(b_dist / r['steps'] * 100)
    
    start_manhattan_eff = np.mean(start_manhattan_effs) if start_manhattan_effs else 0
    start_bfs_eff = np.mean(start_bfs_effs) if start_bfs_effs else 0
    
    print(f"\n    不同起点结果:")
    print(f"    - 成功率: {start_success}/{len(start_results)} ({start_success/len(start_results):.1%})")
    print(f"    - 平均步数: {start_avg_steps:.1f}")
    print(f"    - 平均曼哈顿效率: {start_manhattan_eff:.1f}%")
    print(f"    - 平均BFS效率: {start_bfs_eff:.1f}%")
    
    # 可视化评估结果
    print(f"\n[5] 生成评估结果可视化...")
    vis = Visualizer()
    labels = ['原始环境', '不同起点']
    
    # 图1: 成功率图
    success_rates = [success_rate, start_success / len(start_results)]
    success_path = os.path.join(args.output_dir, "evaluation_success.png")
    vis.plot_success_rate(
        labels,
        success_rates,
        title="成功率对比",
        show=args.show_plots,
        save_path=success_path
    )
    
    # 图2: 路径效率图
    avg_steps_list = [avg_steps, start_avg_steps]
    manhattan_effs = [manhattan_eff, start_manhattan_eff]
    bfs_effs = [bfs_eff, start_bfs_eff]
    
    efficiency_path = os.path.join(args.output_dir, "evaluation_efficiency.png")
    vis.plot_path_efficiency(
        labels,
        avg_steps_list,
        manhattan_effs,
        bfs_effs,
        title="路径效率对比",
        show=args.show_plots,
        save_path=efficiency_path
    )
    
    print(f"\n[6] 评估完成!")
    print(f"    成功率图: {success_path}")
    print(f"    效率图: {efficiency_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
