"""
Fermat's Spiral Generator (Supporting Bluesky & Custom 2-Arm modes with Square Cropping)
費馬螺線散佈點生成器 (整合 Bluesky 模式與自訂雙臂模式，支援 range * range 正方形裁切)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

try:
    import bluesky.plan_patterns as bpp
    BLUESKY_AVAILABLE = True
except ImportError:
    BLUESKY_AVAILABLE = False

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_fermat_spiral_bluesky(step=0.2, square_range=3.0, two_arm=True, factor=1.0, tilt=0.0):
    """
    使用 Bluesky 的演算法生成費馬螺線散佈點 (支援單臂與雙臂，並自動裁剪至正方形範圍)。
    
    參數:
        step: 徑向步長 (對應 bluesky 的 dr)
        square_range: 正方形邊長 (x_range, y_range)
        two_arm: 是否生成 +/- 雙臂
        factor: 半徑縮放因子
        tilt: 旋轉角度 (rad)
    """
    phi = 137.508 * np.pi / 180.0
    half_x = square_range / 2.0
    half_y = square_range / 2.0
    diag = np.sqrt(half_x**2 + half_y**2)
    tilt_tan = np.tan(tilt + np.pi / 2.0)
    
    if two_arm:
        num_rings = int((1.5 * diag / (step / factor)) ** 2 // 2)
        x_p, y_p = [], []
        x_n, y_n = [], []
        
        for i_ring in range(1, num_rings):
            radius = np.sqrt(i_ring) * step / factor * np.sqrt(2.0)
            angle = phi * i_ring
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # 正方形裁切
            if (abs(x - y / tilt_tan) <= half_x) and (abs(y) <= half_y):
                x_p.append(x)
                y_p.append(y)
            if (abs(-x - (-y) / tilt_tan) <= half_x) and (abs(-y) <= half_y):
                x_n.append(-x)
                y_n.append(-y)
                
        x_p, y_p = np.array(x_p), np.array(y_p)
        x_n, y_n = np.array(x_n), np.array(y_n)
        x_all = np.concatenate([[0.0], x_p, x_n])
        y_all = np.concatenate([[0.0], y_p, y_n])
        return (x_p, y_p), (x_n, y_n), (x_all, y_all)
    else:
        num_rings = int((1.5 * diag / (step / factor)) ** 2)
        x_pts, y_pts = [], []
        for i_ring in range(1, num_rings):
            radius = np.sqrt(i_ring) * step / factor
            angle = phi * i_ring
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            if (abs(x - y / tilt_tan) <= half_x) and (abs(y) <= half_y):
                x_pts.append(x)
                y_pts.append(y)
        x_all = np.array(x_pts)
        y_all = np.array(y_pts)
        return (x_all, y_all), (np.array([]), np.array([])), (x_all, y_all)


def generate_fermat_spiral_trajectory_square(step=0.2, square_range=3.0, a=1.0):
    """
    軌跡等弧長模式: 先生成覆蓋外接圓的雙臂費馬螺線，再裁切成 range * range 正方形。
    """
    half_range = square_range / 2.0
    r_circle = half_range * np.sqrt(2.0) * 1.05
    
    theta_max = (r_circle / a) ** 2
    theta_list = []
    theta = max(0.01, (step / (2 * a)) ** 2)
    
    while theta <= theta_max:
        theta_list.append(theta)
        ds_dtheta = a * np.sqrt(1.0 / (4.0 * theta) + theta)
        dtheta = step / max(ds_dtheta, 1e-6)
        theta += dtheta
        
    theta_arr = np.array(theta_list)
    r_pos = a * np.sqrt(theta_arr)
    
    x_raw_p = r_pos * np.cos(theta_arr)
    y_raw_p = r_pos * np.sin(theta_arr)
    x_raw_n = -x_raw_p
    y_raw_n = -y_raw_p
    
    mask_p = (np.abs(x_raw_p) <= half_range) & (np.abs(y_raw_p) <= half_range)
    mask_n = (np.abs(x_raw_n) <= half_range) & (np.abs(y_raw_n) <= half_range)
    
    x_pos = x_raw_p[mask_p]
    y_pos = y_raw_p[mask_p]
    x_neg = x_raw_n[mask_n]
    y_neg = y_raw_n[mask_n]
    
    x_all = np.concatenate([[0.0], x_pos, x_neg])
    y_all = np.concatenate([[0.0], y_pos, y_neg])
    
    return (x_pos, y_pos), (x_neg, y_neg), (x_all, y_all)


def generate_fermat_spiral_golden_square(step=0.2, square_range=3.0):
    """
    黃金角等面積模式: 先生成覆蓋外接圓的雙臂費馬螺線，再裁切成 range * range 正方形。
    """
    half_range = square_range / 2.0
    r_circle = half_range * np.sqrt(2.0) * 1.05
    
    golden_angle = np.pi * (3 - np.sqrt(5))
    n_pts_single_arm = int(np.ceil(np.pi * (r_circle ** 2) / (2 * (step ** 2))))
    
    indices = np.arange(1, n_pts_single_arm + 1)
    r = step * np.sqrt(indices / (np.pi / 2.0))
    valid = r <= r_circle
    r = r[valid]
    indices = indices[valid]
    
    theta = indices * golden_angle
    
    x_raw_p = r * np.cos(theta)
    y_raw_p = r * np.sin(theta)
    x_raw_n = -x_raw_p
    y_raw_n = -y_raw_p
    
    mask_p = (np.abs(x_raw_p) <= half_range) & (np.abs(y_raw_p) <= half_range)
    mask_n = (np.abs(x_raw_n) <= half_range) & (np.abs(y_raw_n) <= half_range)
    
    x_pos = x_raw_p[mask_p]
    y_pos = y_raw_p[mask_p]
    x_neg = x_raw_n[mask_n]
    y_neg = y_raw_n[mask_n]
    
    x_all = np.concatenate([[0.0], x_pos, x_neg])
    y_all = np.concatenate([[0.0], y_pos, y_neg])
    
    return (x_pos, y_pos), (x_neg, y_neg), (x_all, y_all)


def plot_fermat_spiral_square(step=0.2, square_range=3.0, mode='bluesky_2arm', save_path=None):
    """
    繪製並保存裁切成正方形的雙臂費馬螺線散佈點圖
    """
    half_range = square_range / 2.0
    
    if mode == 'bluesky_2arm':
        (x_p, y_p), (x_n, y_n), (x_all, y_all) = generate_fermat_spiral_bluesky(step=step, square_range=square_range, two_arm=True)
        title_mode = "Bluesky 雙臂模式 (Bluesky 2-Arm Pattern)"
    elif mode == 'bluesky_1arm':
        (x_p, y_p), (x_n, y_n), (x_all, y_all) = generate_fermat_spiral_bluesky(step=step, square_range=square_range, two_arm=False)
        title_mode = "Bluesky 原生單臂模式 (Bluesky Single-Arm Pattern)"
    elif mode == 'trajectory':
        (x_p, y_p), (x_n, y_n), (x_all, y_all) = generate_fermat_spiral_trajectory_square(step=step, square_range=square_range)
        title_mode = "等弧長軌跡模式 (Continuous Trajectory)"
    else:
        (x_p, y_p), (x_n, y_n), (x_all, y_all) = generate_fermat_spiral_golden_square(step=step, square_range=square_range)
        title_mode = "黃金角等面積模式 (Golden Angle Sampling)"
        
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
    
    # 圖 1: 正負兩臂分別標示
    ax1 = axs[0]
    rect1 = plt.Rectangle((-half_range, -half_range), square_range, square_range,
                          linewidth=2, edgecolor='black', facecolor='none', linestyle='--', label=f'目標正方形 ({square_range} x {square_range})')
    ax1.add_patch(rect1)
    
    if len(x_n) > 0:
        ax1.scatter(x_p, y_p, color='#1f77b4', s=30, label=f'正臂 Arm (+) ({len(x_p)} 點)', zorder=3)
        ax1.scatter(x_n, y_n, color='#d62728', s=30, label=f'負臂 Arm (-) ({len(x_n)} 點)', zorder=3)
        ax1.scatter([0], [0], color='black', s=60, marker='x', label='中心原點 (0,0)', zorder=4)
    else:
        ax1.scatter(x_all, y_all, color='#1f77b4', s=30, label=f'掃描點 ({len(x_all)} 點)', zorder=3)
        
    ax1.set_title(f'雙臂費馬螺線正方形分佈\n{title_mode}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X 軸座標 (μm / mm)', fontsize=10)
    ax1.set_ylabel('Y 軸座標 (μm / mm)', fontsize=10)
    ax1.set_xlim(-half_range * 1.15, half_range * 1.15)
    ax1.set_ylim(-half_range * 1.15, half_range * 1.15)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 圖 2: 掃描點順序 (Color Gradient)
    ax2 = axs[1]
    rect2 = plt.Rectangle((-half_range, -half_range), square_range, square_range,
                          linewidth=2, edgecolor='red', facecolor='none', linestyle='-', alpha=0.7)
    ax2.add_patch(rect2)
    
    total_pts = len(x_all)
    sc = ax2.scatter(x_all, y_all, c=np.arange(total_pts), cmap='turbo', s=32, edgecolors='k', linewidths=0.3, zorder=3)
    plt.colorbar(sc, ax=ax2, label='掃描點序號 (Point Index)', fraction=0.046, pad=0.04)
    
    ax2.set_title(f'正方形結果 ({square_range} x {square_range})\n總點數: {total_pts} (step={step})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X 軸座標 (μm / mm)', fontsize=10)
    ax2.set_ylabel('Y 軸座標 (μm / mm)', fontsize=10)
    ax2.set_xlim(-half_range * 1.15, half_range * 1.15)
    ax2.set_ylim(-half_range * 1.15, half_range * 1.15)
    ax2.set_aspect('equal', 'box')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"圖片已成功保存至: {save_path}")
        
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成 range * range 正方形雙臂 (+/-) 費馬螺線散佈點")
    parser.add_argument('--step', type=float, default=0.2, help="點間距 (步長 step / dr)")
    parser.add_argument('--range', type=float, default=3.0, help="正方形邊長範圍 (range x range)")
    parser.add_argument('--mode', type=str, choices=['bluesky_2arm', 'bluesky_1arm', 'trajectory', 'golden_angle'],
                        default='bluesky_2arm', help="生成模式")
    parser.add_argument('--output', type=str, default='fermat_spiral_square.png', help="輸出圖片檔名")
    parser.add_argument('--show', action='store_true', help="是否顯示圖片")
    
    args = parser.parse_args()
    
    print(f"=== 費馬螺線正方形掃描參數 ===")
    print(f"步長 (step / dr)    : {args.step}")
    print(f"正方形邊長 (range)  : {args.range} x {args.range}")
    print(f"模式 (mode)         : {args.mode}")
    
    fig = plot_fermat_spiral_square(step=args.step, square_range=args.range, mode=args.mode, save_path=args.output)
    if args.show:
        plt.show()
    else:
        plt.close(fig)
