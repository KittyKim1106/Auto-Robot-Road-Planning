"""
训练脚本
训练 Q-Learning 智能体进行路径规划
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


def train(args):
    """
    训练主函数
    """
    print("=" * 60)
    print("基于 Q-Learning 的机器人路径规划训练")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 环境配置文件路径
    config_path = os.path.join(output_dir, 'env_config.json')
    
    # 加载或创建环境
    print(f"\n[1] 加载网格环境...")
    
    if os.path.exists(config_path) and not args.random_obstacles:
        # 从配置文件加载环境
        env = GridWorld.load_config(config_path)
        print("    (使用已保存的环境配置)")
    else:
        # 创建新环境
        if args.random_obstacles:
            print("    (使用随机障碍物模式)")
        else:
            print("    (使用默认固定障碍物)")
        env = GridWorld(
            grid_size=args.grid_size,
            start_pos=(0, 0),
            goal_pos=(args.grid_size - 1, args.grid_size - 1),
            random_obstacles=args.random_obstacles
        )
    
    print(f"    起点: {env.start_pos}")
    print(f"    终点: {env.goal_pos}")
    print(f"    障碍物数量: {len(env.obstacles)}")
    print("\n环境地图:")
    print(env)
    
    # 创建智能体
    print(f"\n[2] 创建 Q-Learning 智能体...")
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay
    )
    
    print(f"    学习率: {args.learning_rate}")
    print(f"    折扣因子: {args.discount_factor}")
    print(f"    初始探索率: {args.epsilon}")
    print(f"    最小探索率: {args.epsilon_min}")
    print(f"    探索率衰减: {args.epsilon_decay}")
    
    # 环境轮换训练模式
    env_change_mode = args.random_obstacles and args.env_change_interval > 0
    if env_change_mode:
        print(f"\n    [多环境训练模式]")
        print(f"    环境更换间隔: 每 {args.env_change_interval} episodes")
        print(f"    障碍物数量: {args.num_obstacles}")
    
    # 训练
    print(f"\n[3] 开始训练 ({args.episodes} episodes)...")
    print("-" * 60)
    
    success_count = 0
    recent_rewards = []
    env_change_count = 0
    
    for episode in range(1, args.episodes + 1):
        # 环境轮换：每隔一定episode更换障碍物
        if env_change_mode and episode > 1 and (episode - 1) % args.env_change_interval == 0:
            env.reset_with_new_obstacles(args.num_obstacles)
            env_change_count += 1
            if args.print_interval <= args.env_change_interval:
                print(f"         >>> 环境已更换 (第 {env_change_count} 次)，新障碍物数量: {len(env.obstacles)}")
        
        reward, steps, success = agent.train_episode(env, max_steps=args.max_steps)
        
        if success:
            success_count += 1
        
        recent_rewards.append(reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)
        
        # 打印进度
        if episode % args.print_interval == 0:
            avg_reward = np.mean(recent_rewards)
            recent_success_rate = success_count / episode
            env_info = f" | 环境#{env_change_count+1}" if env_change_mode else ""
            print(f"Episode {episode:5d} | "
                  f"奖励: {reward:7.1f} | "
                  f"步数: {steps:4d} | "
                  f"成功: {'是' if success else '否'} | "
                  f"平均奖励(100期): {avg_reward:7.1f} | "
                  f"总成功率: {recent_success_rate:.1%} | "
                  f"ε: {agent.epsilon:.4f}{env_info}")
    
    print("-" * 60)
    print(f"\n训练完成!")
    print(f"总成功次数: {success_count}/{args.episodes} ({success_count/args.episodes:.1%})")
    if env_change_mode:
        print(f"训练过程中使用了 {env_change_count + 1} 个不同的环境")
    
    # 保存智能体
    agent_path = os.path.join(output_dir, "agent.pkl")
    agent.save(agent_path)
    
    # 可视化
    print(f"\n[4] 生成可视化结果...")
    vis = Visualizer()
    
    # 保存训练曲线
    curves_path = os.path.join(output_dir, "training_curves.png")
    vis.plot_training_curves(
        agent.training_rewards,
        agent.training_steps,
        title="Q-Learning 训练曲线",
        show=args.show_plots,
        save_path=curves_path
    )
    
    # 保存环境地图
    grid_path = os.path.join(output_dir, "grid_world.png")
    vis.plot_grid(
        env,
        title="网格世界环境",
        show=args.show_plots,
        save_path=grid_path
    )
    
    # 获取并保存最优路径
    print(f"\n[5] 评估最优路径...")
    path, total_reward, success = agent.get_optimal_path(env)
    
    if success:
        print(f"    成功找到路径!")
        print(f"    路径长度: {len(path)} 步")
        print(f"    总奖励: {total_reward:.1f}")
        
        # 保存路径图
        path_fig_path = os.path.join(output_dir, "optimal_path.png")
        vis.plot_grid(
            env,
            path=path,
            title=f"最优路径 (长度: {len(path)-1} 步)",
            show=args.show_plots,
            save_path=path_fig_path
        )
        
        # 保存路径动画GIF
        anim_path = os.path.join(output_dir, "path_animation.gif")
        print(f"\n[6] 生成路径动画...")
        vis.animate_path(
            env,
            path,
            title="训练后的最优路径",
            save_path=anim_path
        )
    else:
        print(f"    未能找到有效路径，可能需要更多训练")
    
    # 保存策略图
    policy = agent.get_policy(env.grid_size)
    policy_path = os.path.join(output_dir, "policy.png")
    vis.plot_policy(
        env,
        policy,
        title="学习到的策略",
        show=args.show_plots,
        save_path=policy_path
    )
    
    # 保存价值函数图
    values = agent.get_value_function(env.grid_size)
    value_path = os.path.join(output_dir, "value_function.png")
    vis.plot_value_function(
        env,
        values,
        title="状态价值函数",
        show=args.show_plots,
        save_path=value_path
    )
    
    # 如果是多环境训练，额外评估泛化能力
    if env_change_mode:
        print(f"\n[6] 评估泛化能力 (10个随机环境)...")
        gen_success = 0
        gen_steps = []
        for i in range(10):
            test_env = GridWorld(
                grid_size=args.grid_size,
                random_obstacles=True,
                num_random_obstacles=args.num_obstacles
            )
            path, reward, success = agent.get_optimal_path(test_env)
            if success:
                gen_success += 1
                gen_steps.append(len(path) - 1)
            print(f"    测试环境 {i+1}: {'成功' if success else '失败'}", 
                  f"步数: {len(path)-1}" if success else "")
        
        print(f"\n    泛化成功率: {gen_success}/10 ({gen_success*10}%)")
        if gen_steps:
            print(f"    平均步数: {np.mean(gen_steps):.1f}")
    
    print(f"\n[7] 所有结果已保存到: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    return agent, env


def main():
    parser = argparse.ArgumentParser(description="Q-Learning 路径规划训练")
    
    # 环境参数
    parser.add_argument("--grid-size", type=int, default=10, help="网格大小")
    parser.add_argument("--random-obstacles", action="store_true", help="使用随机障碍物")
    parser.add_argument("--num-obstacles", type=int, default=15, help="随机障碍物数量")
    parser.add_argument("--env-change-interval", type=int, default=0, 
                       help="环境更换间隔(多环境训练模式)，0表示不更换")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=1000, help="训练轮数")
    parser.add_argument("--max-steps", type=int, default=200, help="每轮最大步数")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon", type=float, default=1.0, help="初始探索率")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="最小探索率")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="探索率衰减")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="output", help="输出目录")
    parser.add_argument("--print-interval", type=int, default=100, help="打印间隔")
    parser.add_argument("--show-plots", action="store_true", help="显示图形")
    parser.add_argument("--save-animation", action="store_true", help="保存路径动画GIF")
    
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
