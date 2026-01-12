"""
演示脚本
展示训练好的智能体执行路径规划
"""

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.grid_world import GridWorld
from agent.q_learning import QLearningAgent
from utils.visualization import Visualizer


def demo(args):
    """
    演示主函数
    """
    print("=" * 60)
    print("Q-Learning 路径规划演示")
    print("=" * 60)
    
    # 加载智能体
    print(f"\n[1] 加载智能体: {args.agent_path}")
    if not os.path.exists(args.agent_path):
        print(f"错误: 找不到智能体文件 {args.agent_path}")
        print("请先运行 train.py 进行训练")
        return
    
    # 环境配置文件路径
    output_dir = args.output_dir
    config_path = os.path.join(output_dir, 'env_config.json')
    
    # 加载或创建环境
    print(f"\n[2] 加载环境...")
    
    if os.path.exists(config_path) and not args.random_obstacles:
        # 从配置文件加载环境
        env = GridWorld.load_config(config_path)
    else:
        # 创建新环境
        env = GridWorld(
            grid_size=args.grid_size,
            start_pos=tuple(args.start) if args.start else (0, 0),
            goal_pos=tuple(args.goal) if args.goal else (args.grid_size - 1, args.grid_size - 1),
            random_obstacles=args.random_obstacles
        )
    
    # 创建并加载智能体
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.load(args.agent_path)
    
    print(f"\n[3] 环境信息:")
    print(f"    网格大小: {env.grid_size}x{env.grid_size}")
    print(f"    起点: {env.start_pos}")
    print(f"    终点: {env.goal_pos}")
    print(f"    障碍物数量: {len(env.obstacles)}")
    
    print("\n环境地图:")
    print(env)
    
    # 获取最优路径
    print(f"\n[3] 计算最优路径...")
    path, total_reward, success = agent.get_optimal_path(env)
    
    if success:
        print(f"    [成功] 找到路径!")
        print(f"    路径长度: {len(path) - 1} 步")
        print(f"    总奖励: {total_reward:.1f}")
        print(f"\n    路径详情:")
        for i, pos in enumerate(path):
            if i == 0:
                print(f"      {i}: {pos} (起点)")
            elif i == len(path) - 1:
                print(f"      {i}: {pos} (终点)")
            else:
                print(f"      {i}: {pos}")
    else:
        print(f"    [失败] 未能找到有效路径")
        print(f"    可能原因: 障碍物阻挡或需要更多训练")
    
    # 可视化
    print(f"\n[4] 生成可视化...")
    vis = Visualizer()
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 显示路径
    if success:
        path_fig_path = os.path.join(output_dir, "demo_path.png")
        vis.plot_grid(
            env,
            path=path,
            title=f"最优路径演示 (长度: {len(path)-1} 步)",
            show=True,
            save_path=path_fig_path
        )
        
        # 路径动画
        if args.animate:
            print(f"\n[5] 播放路径动画...")
            anim_path = os.path.join(output_dir, "demo_animation.gif")
            vis.animate_path(
                env,
                path,
                interval=args.animation_speed,
                title="路径动画演示",
                save_path=anim_path  # 总是保存GIF
            )
    else:
        vis.plot_grid(
            env,
            title="当前环境 (未找到路径)",
            show=True
        )
    
    # 显示策略
    if args.show_policy:
        policy = agent.get_policy(env.grid_size)
        policy_path = os.path.join(output_dir, "demo_policy.png")
        vis.plot_policy(
            env,
            policy,
            title="学习到的策略",
            show=True,
            save_path=policy_path
        )
    
    # 显示价值函数
    if args.show_value:
        values = agent.get_value_function(env.grid_size)
        value_path = os.path.join(output_dir, "demo_value.png")
        vis.plot_value_function(
            env,
            values,
            title="状态价值函数",
            show=True,
            save_path=value_path
        )
    
    print(f"\n演示完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Q-Learning 路径规划演示")
    
    parser.add_argument("--agent-path", type=str, default="output/agent.pkl",
                       help="智能体模型路径")
    parser.add_argument("--grid-size", type=int, default=10, help="网格大小")
    parser.add_argument("--start", type=int, nargs=2, default=None,
                       help="起点坐标 (行 列)")
    parser.add_argument("--goal", type=int, nargs=2, default=None,
                       help="终点坐标 (行 列)")
    parser.add_argument("--random-obstacles", action="store_true",
                       help="使用随机障碍物")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="输出目录")
    parser.add_argument("--animate", action="store_true",
                       help="播放路径动画")
    parser.add_argument("--animation-speed", type=int, default=500,
                       help="动画速度 (毫秒/帧)")
    parser.add_argument("--save-animation", action="store_true",
                       help="保存动画为GIF")
    parser.add_argument("--show-policy", action="store_true",
                       help="显示策略图")
    parser.add_argument("--show-value", action="store_true",
                       help="显示价值函数图")
    
    args = parser.parse_args()
    
    demo(args)


if __name__ == "__main__":
    main()
