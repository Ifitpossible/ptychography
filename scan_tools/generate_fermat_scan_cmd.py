"""
Fermat Spiral Scan Command Generator (for Beamline Motor Controller & Detector Trigger)
基於 Bluesky 的費馬螺線掃描指令生成器

指令結構:
  STR                      : 掃描開始
  每個點循環:
    SFL,0,<axis_x>,<x_pos> : X 軸移動到指定座標
    SFL,0,<axis_z>,<z_pos> : Z 軸移動到指定座標
    RES,<rest_before>      : 移動後穩定等待 (預設 3 = 0.3 秒，單位 0.1s)
    TRG,<exposure_us>      : 觸發 Detector 曝光收光 (單位: 微秒 us)
    RES,<rest_after>       : 收光後等待 (預設 1 = 0.1 秒，單位 0.1s)
  END                      : 掃描結束
"""

import numpy as np
import bluesky.plan_patterns as bpp
import argparse
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_fermat_scan_points(cenx=50.0, cenz=50.0, range_val=1.0, step=0.1, factor=1.0):
    """
    使用 Bluesky 生成正方形範圍內的費馬螺線掃描點 (XZ 平面)
    
    返回:
        points: numpy 陣列，形狀 (N, 2)，每列為 [x, z]
    """
    cyc = bpp.spiral_fermat(
        x_motor='x',
        y_motor='z',
        x_start=cenx,
        y_start=cenz,
        x_range=range_val,
        y_range=range_val,
        dr=step,
        factor=factor
    )
    
    raw_pts = list(cyc)
    x_coords = [p['x'] for p in raw_pts]
    z_coords = [p['z'] for p in raw_pts]
    
    return np.column_stack([x_coords, z_coords])


def generate_scan_commands(points, axis_x=0, axis_z=2, rest_before=3, exposure_us=1000000, rest_after=1, units_per_um=1000000000):
    """
    將點座標列表 (單位: um) 轉換為完整的控制與收光指令序列 (SFL 單位: 1nm = 1,000,000 單位 => 1um = 10^9 單位)
    
    參數:
        points: (N, 2) 座標陣列 (單位: um)
        axis_x: X 軸編號 (預設: 0)
        axis_z: Z 軸編號 (預設: 2)
        rest_before: 移動定位後休息時間 (單位 0.1秒，預設 3 = 0.3s)
        exposure_us: Detector 曝光時間 (單位 微秒 us，預設 1000000 = 1.0s)
        rest_after: 收光後休息時間 (單位 0.1秒，預設 1 = 0.1s)
        units_per_um: 單位換算倍率 (1 nm = 10^6 units => 1 um = 10^9 units)
    """
    lines = []
    lines.append("STR")  # 掃描開始
    
    for idx, (x_um, z_um) in enumerate(points):
        # 換算為 SFL 馬達整數單位 (1 um = 10^9 units)
        x_units = int(round(x_um * units_per_um))
        z_units = int(round(z_um * units_per_um))
        
        # 1. 馬達移動至目標座標 (整數單位)
        lines.append(f"SFL,0,{axis_x},{x_units}")
        lines.append(f"SFL,0,{axis_z},{z_units}")
        
        # 2. 定位後休息 0.3 秒 (RES,3)
        if rest_before is not None:
            lines.append(f"RES,{rest_before}")
            
        # 3. 觸發 Detector 收光 (TRG,<us>)
        if exposure_us is not None:
            lines.append(f"TRG,{exposure_us}")
            
        # 4. 收完光休息 0.1 秒 (RES,1)
        if rest_after is not None:
            lines.append(f"RES,{rest_after}")
            
    # 5. 掃描結束
    lines.append("END")
    
    return lines


def save_command_file(filepath, lines):
    """
    儲存指令到文字檔案
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')
    print(f"掃描指令已成功儲存至: {filepath} (共 {len(lines)} 行指令)")


def plot_scan_points(points, cenx, cenz, range_val, step, save_img_path=None):
    """
    繪製並保存掃描點路徑圖
    """
    half_r = range_val / 2.0
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    
    rect = plt.Rectangle((cenx - half_r, cenz - half_r), range_val, range_val,
                          linewidth=2, edgecolor='red', facecolor='none', linestyle='--', label=f'掃描邊界 ({range_val} x {range_val})')
    ax.add_patch(rect)
    
    x = points[:, 0]
    z = points[:, 1]
    n_pts = len(points)
    
    ax.plot(x, z, '-', color='gray', alpha=0.4, linewidth=0.8, zorder=2)
    sc = ax.scatter(x, z, c=np.arange(n_pts), cmap='turbo', s=50, edgecolors='k', linewidths=0.5, zorder=3)
    ax.scatter([cenx], [cenz], color='black', s=90, marker='x', label=f'中心 ({cenx}, {cenz})', zorder=5)
    
    ax.text(x[0], z[0], '  0 (Start)', fontsize=9, fontweight='bold', color='green', zorder=6)
    ax.text(x[-1], z[-1], f'  {n_pts-1} (End)', fontsize=9, fontweight='bold', color='purple', zorder=6)
    
    plt.colorbar(sc, ax=ax, label='掃描點序號 (Index)', fraction=0.046, pad=0.04)
    
    ax.set_title(f'Bluesky 費馬螺線掃描路徑 (Case 1: 正方形 {range_val} x {range_val})\n中心=({cenx}, {cenz}), step={step}, 總點數={n_pts}', fontsize=11, fontweight='bold')
    ax.set_xlabel('X 軸座標 (μm)', fontsize=10)
    ax.set_ylabel('Z 軸座標 (μm)', fontsize=10)
    ax.set_xlim(cenx - half_r * 1.15, cenx + half_r * 1.15)
    ax.set_ylim(cenz - half_r * 1.15, cenz + half_r * 1.15)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    if save_img_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_img_path)), exist_ok=True)
        plt.savefig(save_img_path, bbox_inches='tight')
        print(f"預覽圖片已保存至: {save_img_path}")
    plt.close(fig)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="生成費馬螺線掃描指令 .txt 檔案 (包含 TRG 收光與 RES 休息)")
    parser.add_argument('--cenx', type=float, default=50.0, help="X 軸中心座標 (預設: 50.0)")
    parser.add_argument('--cenz', type=float, default=50.0, help="Z 軸中心座標 (預設: 50.0)")
    parser.add_argument('--range', type=float, default=1.0, help="正方形邊長範圍 (預設: 1.0)")
    parser.add_argument('--step', type=float, default=0.1, help="點間距 (步長 step, 預設: 0.1)")
    parser.add_argument('--axis_x', type=int, default=0, help="X 軸代號 (預設: 0)")
    parser.add_argument('--axis_z', type=int, default=2, help="Z 軸代號 (預設: 2)")
    parser.add_argument('--rest_before', type=int, default=3, help="定位後休息時間 (單位 0.1s, 預設: 3 = 0.3s)")
    parser.add_argument('--trg', type=int, default=1000000, help="Detector 收光曝光時間 (單位 微秒 us, 預設: 1000000 = 1.0s)")
    parser.add_argument('--rest_after', type=int, default=1, help="收光後休息時間 (單位 0.1s, 預設: 1 = 0.1s)")
    parser.add_argument('--output', type=str, default=r'Z:\TPS23A_USER\Ptychography\fermat_scan_commands.txt', help="輸出指令檔案路徑 (.txt)")
    parser.add_argument('--plot', type=str, default=r'Z:\TPS23A_USER\Ptychography\fermat_scan_path.png', help="輸出預覽圖片路徑 (.png)")
    
    args = parser.parse_args()
    
    print(f"=== 費馬螺線掃描與收光參數 ===")
    print(f"中心座標 (cenx, cenz) : ({args.cenx}, {args.cenz})")
    print(f"邊長範圍 (range)       : {args.range} x {args.range}")
    print(f"步長 (step)            : {args.step}")
    print(f"馬達軸代號 (X, Z)      : (軸 {args.axis_x}, 軸 {args.axis_z})")
    print(f"定位後等待 (RES)       : {args.rest_before} ({args.rest_before * 0.1:.1f} 秒)")
    print(f"收光時間 (TRG)         : {args.trg} us ({args.trg / 1e6:.3f} 秒)")
    print(f"收光後等待 (RES)       : {args.rest_after} ({args.rest_after * 0.1:.1f} 秒)")
    print(f"==============================")
    
    # 1. 生成點
    points = generate_fermat_scan_points(cenx=args.cenx, cenz=args.cenz, range_val=args.range, step=args.step)
    print(f"生成掃描點數: {len(points)} 點")
    
    # 2. 生成指令
    commands = generate_scan_commands(
        points,
        axis_x=args.axis_x,
        axis_z=args.axis_z,
        rest_before=args.rest_before,
        exposure_us=args.trg,
        rest_after=args.rest_after
    )
    
    # 3. 儲存指令檔案
    save_command_file(args.output, commands)
    
    # 4. 繪製預覽圖
    if args.plot:
        plot_scan_points(points, args.cenx, args.cenz, args.range, args.step, save_img_path=args.plot)
