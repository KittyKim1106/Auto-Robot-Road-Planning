"""
可视化工具
包含地图绘制、路径动画、训练曲线等功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Optional
import os


class Visualizer:
    """
    可视化工具类
    """
    
    # 颜色定义
    COLORS = {
        'empty': '#FFFFFF',      # 白色 - 空地
        'obstacle': '#2C3E50',   # 深蓝灰 - 障碍物
        'start': '#27AE60',      # 绿色 - 起点
        'goal': '#E74C3C',       # 红色 - 终点
        'agent': '#3498DB',      # 蓝色 - 智能体
        'path': '#F39C12',       # 橙色 - 路径
        'visited': '#BDC3C7'     # 灰色 - 已访问
    }
    
    # 动作箭头方向
    ACTION_ARROWS = {
        0: (0, 0.3),   # 上
        1: (0, -0.3),  # 下
        2: (-0.3, 0),  # 左
        3: (0.3, 0)    # 右
    }
    
    def __init__(self, figsize: Tuple[int, int] = (10, 10)):
        """
        初始化可视化器
        
        Args:
            figsize: 图形大小
        """
        self.figsize = figsize
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_grid(self, 
                  env, 
                  agent_pos: Optional[Tuple[int, int]] = None,
                  path: Optional[List[Tuple[int, int]]] = None,
                  title: str = "网格世界",
                  show: bool = True,
                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制网格世界
        
        Args:
            env: 网格环境
            agent_pos: 智能体位置 (可选)
            path: 路径列表 (可选)
            title: 图标题
            show: 是否显示
            save_path: 保存路径 (可选)
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grid_size = env.grid_size
        
        # 绘制网格背景
        for i in range(grid_size):
            for j in range(grid_size):
                color = self.COLORS['empty']
                
                # 检查是否是障碍物
                if (i, j) in env.obstacles:
                    color = self.COLORS['obstacle']
                # 检查是否是起点
                elif (i, j) == env.start_pos:
                    color = self.COLORS['start']
                # 检查是否是终点
                elif (i, j) == env.goal_pos:
                    color = self.COLORS['goal']
                
                rect = patches.Rectangle(
                    (j, grid_size - 1 - i), 1, 1,
                    linewidth=1,
                    edgecolor='#7F8C8D',
                    facecolor=color
                )
                ax.add_patch(rect)
        
        # 绘制路径
        if path and len(path) > 1:
            path_y = [grid_size - 0.5 - p[0] for p in path]
            path_x = [p[1] + 0.5 for p in path]
            ax.plot(path_x, path_y, 
                   color=self.COLORS['path'], 
                   linewidth=3, 
                   marker='o', 
                   markersize=8,
                   alpha=0.8,
                   label='路径')
        
        # 绘制智能体
        if agent_pos is not None:
            circle = patches.Circle(
                (agent_pos[1] + 0.5, grid_size - 0.5 - agent_pos[0]),
                0.35,
                color=self.COLORS['agent'],
                zorder=5
            )
            ax.add_patch(circle)
        
        # 添加标注
        ax.text(env.start_pos[1] + 0.5, grid_size - 0.5 - env.start_pos[0],
               'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(env.goal_pos[1] + 0.5, grid_size - 0.5 - env.goal_pos[0],
               'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # 设置坐标轴
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        # 设置刻度
        ax.set_xticks(np.arange(0.5, grid_size, 1))
        ax.set_yticks(np.arange(0.5, grid_size, 1))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size - 1, -1, -1))
        
        # 添加图例
        legend_elements = [
            patches.Patch(facecolor=self.COLORS['start'], label='起点 (S)'),
            patches.Patch(facecolor=self.COLORS['goal'], label='终点 (G)'),
            patches.Patch(facecolor=self.COLORS['obstacle'], label='障碍物'),
            patches.Patch(facecolor=self.COLORS['agent'], label='智能体'),
        ]
        if path:
            legend_elements.append(plt.Line2D([0], [0], color=self.COLORS['path'], 
                                             linewidth=3, marker='o', label='路径'))
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_training_curves(self,
                            rewards: List[float],
                            steps: List[int],
                            window_size: int = 50,
                            title: str = "训练曲线",
                            show: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制训练曲线
        
        Args:
            rewards: 每个episode的奖励
            steps: 每个episode的步数
            window_size: 平滑窗口大小
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        episodes = np.arange(1, len(rewards) + 1)
        
        # 计算滑动平均
        def moving_average(data, window):
            if len(data) < window:
                window = len(data)
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # 绘制奖励曲线
        ax1 = axes[0]
        ax1.plot(episodes, rewards, alpha=0.3, color='#3498DB', label='原始奖励')
        if len(rewards) >= window_size:
            ma_rewards = moving_average(rewards, window_size)
            ax1.plot(np.arange(window_size, len(rewards) + 1), ma_rewards, 
                    color='#E74C3C', linewidth=2, label=f'{window_size}期滑动平均')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('累积奖励', fontsize=12)
        ax1.set_title('训练奖励曲线', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # 绘制步数曲线
        ax2 = axes[1]
        ax2.plot(episodes, steps, alpha=0.3, color='#27AE60', label='原始步数')
        if len(steps) >= window_size:
            ma_steps = moving_average(steps, window_size)
            ax2.plot(np.arange(window_size, len(steps) + 1), ma_steps,
                    color='#9B59B6', linewidth=2, label=f'{window_size}期滑动平均')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('步数', fontsize=12)
        ax2.set_title('训练步数曲线', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_policy(self,
                   env,
                   policy: np.ndarray,
                   title: str = "策略可视化",
                   show: bool = True,
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制策略图（每个状态的最优动作用箭头表示）
        
        Args:
            env: 网格环境
            policy: 策略矩阵 (grid_size x grid_size)
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grid_size = env.grid_size
        
        # 绘制网格背景
        for i in range(grid_size):
            for j in range(grid_size):
                color = self.COLORS['empty']
                
                if (i, j) in env.obstacles:
                    color = self.COLORS['obstacle']
                elif (i, j) == env.start_pos:
                    color = self.COLORS['start']
                elif (i, j) == env.goal_pos:
                    color = self.COLORS['goal']
                
                rect = patches.Rectangle(
                    (j, grid_size - 1 - i), 1, 1,
                    linewidth=1,
                    edgecolor='#7F8C8D',
                    facecolor=color
                )
                ax.add_patch(rect)
        
        # 绘制策略箭头
        for i in range(grid_size):
            for j in range(grid_size):
                # 跳过障碍物、起点、终点
                if (i, j) in env.obstacles or (i, j) == env.goal_pos:
                    continue
                
                action = policy[i, j]
                dx, dy = self.ACTION_ARROWS[action]
                
                # 箭头中心位置
                cx = j + 0.5
                cy = grid_size - 0.5 - i
                
                ax.arrow(cx - dx/2, cy - dy/2, dx, dy,
                        head_width=0.15, head_length=0.1,
                        fc='#2C3E50', ec='#2C3E50')
        
        # 标注起点和终点
        ax.text(env.start_pos[1] + 0.5, grid_size - 0.5 - env.start_pos[0],
               'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(env.goal_pos[1] + 0.5, grid_size - 0.5 - env.goal_pos[0],
               'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        
        # 设置坐标轴
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        ax.set_xticks(np.arange(0.5, grid_size, 1))
        ax.set_yticks(np.arange(0.5, grid_size, 1))
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size - 1, -1, -1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"策略图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_value_function(self,
                           env,
                           values: np.ndarray,
                           title: str = "状态价值函数",
                           show: bool = True,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制状态价值函数热力图
        
        Args:
            env: 网格环境
            values: 价值函数矩阵 (grid_size x grid_size)
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grid_size = env.grid_size
        
        # 创建显示用的value矩阵
        display_values = values.copy()
        
        # 障碍物设置为NaN
        for obs in env.obstacles:
            display_values[obs[0], obs[1]] = np.nan
        
        # 绘制热力图
        im = ax.imshow(display_values, cmap='RdYlGn', aspect='equal')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('状态价值', fontsize=12)
        
        # 在每个格子中显示数值
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in env.obstacles:
                    ax.text(j, i, '█', ha='center', va='center', fontsize=12)
                elif (i, j) == env.start_pos:
                    ax.text(j, i, 'S', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='white')
                elif (i, j) == env.goal_pos:
                    ax.text(j, i, 'G', ha='center', va='center',
                           fontsize=14, fontweight='bold', color='white')
                else:
                    ax.text(j, i, f'{values[i, j]:.0f}', ha='center', va='center', fontsize=8)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('列', fontsize=12)
        ax.set_ylabel('行', fontsize=12)
        
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"价值函数图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def animate_path(self,
                    env,
                    path: List[Tuple[int, int]],
                    interval: int = 500,
                    title: str = "路径动画",
                    save_path: Optional[str] = None) -> FuncAnimation:
        """
        创建路径动画
        
        Args:
            env: 网格环境
            path: 路径列表
            interval: 帧间隔（毫秒）
            title: 动画标题
            save_path: 保存路径 (gif文件)
            
        Returns:
            FuncAnimation 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        grid_size = env.grid_size
        
        # 初始化网格
        def init():
            ax.clear()
            
            # 绘制网格背景
            for i in range(grid_size):
                for j in range(grid_size):
                    color = self.COLORS['empty']
                    
                    if (i, j) in env.obstacles:
                        color = self.COLORS['obstacle']
                    elif (i, j) == env.start_pos:
                        color = self.COLORS['start']
                    elif (i, j) == env.goal_pos:
                        color = self.COLORS['goal']
                    
                    rect = patches.Rectangle(
                        (j, grid_size - 1 - i), 1, 1,
                        linewidth=1,
                        edgecolor='#7F8C8D',
                        facecolor=color
                    )
                    ax.add_patch(rect)
            
            # 标注
            ax.text(env.start_pos[1] + 0.5, grid_size - 0.5 - env.start_pos[0],
                   'S', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            ax.text(env.goal_pos[1] + 0.5, grid_size - 0.5 - env.goal_pos[0],
                   'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            return []
        
        # 智能体圆点列表
        agent_circles = []
        path_line, = ax.plot([], [], color=self.COLORS['path'], linewidth=2, 
                             marker='o', markersize=5, alpha=0.5)
        
        def update(frame):
            # 隐藏之前的智能体圆点
            for circle in agent_circles:
                circle.set_visible(False)
            
            # 当前位置
            current_pos = path[frame]
            
            # 绘制已走过的路径
            if frame > 0:
                path_x = [p[1] + 0.5 for p in path[:frame + 1]]
                path_y = [grid_size - 0.5 - p[0] for p in path[:frame + 1]]
                path_line.set_data(path_x, path_y)
            
            # 绘制智能体
            agent_circle = patches.Circle(
                (current_pos[1] + 0.5, grid_size - 0.5 - current_pos[0]),
                0.35,
                color=self.COLORS['agent'],
                zorder=5
            )
            ax.add_patch(agent_circle)
            agent_circles.append(agent_circle)
            
            # 更新标题显示当前步数
            ax.set_title(f'{title} - 步数: {frame}/{len(path) - 1}', 
                        fontsize=16, fontweight='bold')
            
            return [path_line, agent_circle]
        
        anim = FuncAnimation(fig, update, init_func=init,
                            frames=len(path), interval=interval,
                            blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000//interval)
            print(f"动画已保存到: {save_path}")
            plt.close(fig)  # 关闭图形，避免在服务器环境中阻塞
        else:
            plt.show()  # 仅在不保存时显示
        
        return anim
    
    def plot_evaluation_results(self,
                               success_rates: List[float],
                               avg_steps: List[float],
                               labels: List[str],
                               title: str = "评估结果对比",
                               show: bool = True,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制评估结果对比图（兼容旧版本）
        
        Args:
            success_rates: 成功率列表
            avg_steps: 平均步数列表
            labels: 标签列表
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(labels))
        width = 0.6
        
        # 成功率图
        ax1 = axes[0]
        bars1 = ax1.bar(x, [r * 100 for r in success_rates], width, color='#27AE60')
        ax1.set_ylabel('成功率 (%)', fontsize=12)
        ax1.set_title('成功率对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_ylim(0, 105)
        
        # 在柱子上显示数值
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 平均步数图
        ax2 = axes[1]
        bars2 = ax2.bar(x, avg_steps, width, color='#3498DB')
        ax2.set_ylabel('平均步数', fontsize=12)
        ax2.set_title('平均步数对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar, steps in zip(bars2, avg_steps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{steps:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"评估结果图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_success_rate(self,
                         labels: List[str],
                         success_rates: List[float],
                         title: str = "成功率对比",
                         show: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制成功率对比图
        
        Args:
            labels: 标签列表
            success_rates: 成功率列表
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(labels))
        width = 0.6
        colors = ['#27AE60', '#3498DB']
        
        bars = ax.bar(x, [r * 100 for r in success_rates], width, 
                     color=colors[:len(labels)])
        ax.set_ylabel('成功率 (%)', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 110)
        
        # 在柱子上显示数值
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{rate*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"成功率图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_path_efficiency(self,
                            labels: List[str],
                            avg_steps: List[float],
                            manhattan_efficiency: List[float],
                            bfs_efficiency: List[float],
                            title: str = "路径效率对比",
                            show: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制路径效率对比图（双Y轴：左边步数，右边效率）
        
        Args:
            labels: 标签列表 (如 ['原始环境', '不同起点'])
            avg_steps: 平均步数列表
            manhattan_efficiency: 曼哈顿效率列表 (0-100%)
            bfs_efficiency: BFS效率列表 (0-100%)
            title: 图标题
            show: 是否显示
            save_path: 保存路径
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(labels))
        width = 0.25
        
        # 左Y轴：平均步数
        color_steps = '#E74C3C'
        bars1 = ax1.bar(x - width, avg_steps, width, label='平均步数', 
                       color=color_steps, alpha=0.8)
        ax1.set_xlabel('测试环境', fontsize=12)
        ax1.set_ylabel('平均步数', color=color_steps, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_steps)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        
        # 在步数柱子上显示数值
        for bar, steps in zip(bars1, avg_steps):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{steps:.1f}', ha='center', va='bottom', fontsize=10, 
                    color=color_steps, fontweight='bold')
        
        # 右Y轴：效率百分比
        ax2 = ax1.twinx()
        color_manhattan = '#27AE60'
        color_bfs = '#3498DB'
        
        bars2 = ax2.bar(x, manhattan_efficiency, width, label='曼哈顿效率', 
                       color=color_manhattan, alpha=0.8)
        bars3 = ax2.bar(x + width, bfs_efficiency, width, label='BFS效率', 
                       color=color_bfs, alpha=0.8)
        
        ax2.set_ylabel('路径效率 (%)', fontsize=12)
        ax2.set_ylim(0, 120)
        
        # 在效率柱子上显示数值
        for bar, eff in zip(bars2, manhattan_efficiency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontsize=9,
                    color=color_manhattan, fontweight='bold')
        for bar, eff in zip(bars3, bfs_efficiency):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{eff:.1f}%', ha='center', va='bottom', fontsize=9,
                    color=color_bfs, fontweight='bold')
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"路径效率图已保存到: {save_path}")
        
        if show:
            plt.show()
        
        return fig


# 测试代码
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.grid_world import GridWorld
    
    # 创建环境
    env = GridWorld()
    vis = Visualizer()
    
    # 测试绘制网格
    print("测试绘制网格...")
    vis.plot_grid(env, title="测试网格环境")
    
    # 测试绘制带路径的网格
    test_path = [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1)]
    vis.plot_grid(env, path=test_path, title="测试路径显示")
