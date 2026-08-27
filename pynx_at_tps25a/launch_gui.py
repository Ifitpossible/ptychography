import os
import sys
import argparse
import glob

# Ensure current folder is in Python search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Auto-resolve PyNX library path and matching site-packages
pynx_paths = [
    os.path.join(SCRIPT_DIR, "pynx"),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Code", "PYNX", "PyNX")),
    r"C:\Users\User\Desktop\AllenCheng\Code\PYNX\PyNX",
]
if sys.version_info >= (3, 13):
    pynx_paths = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Code", "myvenv", "Lib", "site-packages")),
        r"C:\Users\User\Desktop\AllenCheng\Code\myvenv\Lib\site-packages",
    ] + pynx_paths

for p in pynx_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Ensure MSVC and CUDA tools are in PATH
def _load_msvc_env():
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\portable\msvc\msvc-14.44.17.14_sdk-26100\setup_x64.bat"),
        os.path.expandvars(r"%LOCALAPPDATA%\portable\msvc\msvc-14.44.17.14_sdk-26100\activate.cmd"),
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    import subprocess
    for bat in candidates:
        if os.path.exists(bat) and "INCLUDE" not in os.environ:
            try:
                cmd = f'cmd /c "call \"{bat}\" && set"'
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True)
                for line in proc.stdout.splitlines():
                    if '=' in line:
                        k, v = line.split('=', 1)
                        os.environ[k] = v
                break
            except Exception:
                pass

    cuda_nvcc_bin = os.path.expandvars(r"%LOCALAPPDATA%\portable\cuda\cuda_nvcc\nvcc\bin")
    if os.path.exists(cuda_nvcc_bin) and cuda_nvcc_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_nvcc_bin + ";" + os.environ.get("PATH", "")
    cuda_root = os.path.expandvars(r"%LOCALAPPDATA%\portable\cuda\cuda_nvcc\nvcc")
    if os.path.exists(cuda_root):
        os.environ["CUDA_PATH"] = cuda_root

_load_msvc_env()

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'typeDict'):
    np.typeDict = getattr(np, 'sctypeDict', {})
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void]
    }
import pandas as pd

# Import main toolkit
import main
from main import (
    simple_gui,
    pynx_set_data,
    pynx_plot_overview,
    pynx_plot_obj,
    pynx_plot_probe,
    pynx_save,
    set_params,
    set_algorithm
)


def load_fermat_scan_commands(cmd_file):
    """
    從費馬螺線掃描指令檔 (.txt) 中解析各點的 X, Z 馬達座標 (單位: um)
    """
    if not os.path.exists(cmd_file):
        raise FileNotFoundError(f"找不到掃描指令檔: {cmd_file}")

    with open(cmd_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    pts = []
    curr_x, curr_z = None, None
    for line in lines:
        if line.startswith("SFL,0,0,"):
            curr_x = int(line.split(",")[3]) / 1e9  # 轉換為 um
        elif line.startswith("SFL,0,2,"):
            curr_z = int(line.split(",")[3]) / 1e9  # 轉換為 um
        elif line.startswith("TRG,"):
            if curr_x is not None and curr_z is not None:
                pts.append((curr_x, curr_z))

    return np.array(pts)


def build_workspace_from_series(
    series_num=35,
    date_str="20260821",
    data_dir=None,
    cmd_file=None,
    center=(572, 766),
    crop_size=300,
    pixelsize=7.5e-5,
    detectordistance=1.41,
    energy_eV=9670.0,
    wavelength=None,
    nbprobe=3,
    probe_type="disc,500e-9",
    defocus_val=0.0,
    scale_x=1.0,
    scale_z=1.0,
):
    """
    從指定 Series 編號的 HDF5 檔案與掃描指令構建 PyNX 的 ws (PtychoRunnerScanTPS25A) 物件。
    """
    if wavelength is None:
        # 由能量 (eV) 計算波長 (m)
        wavelength = 1.239841984e-6 / energy_eV

    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, date_str)
        if not os.path.exists(data_dir):
            data_dir = SCRIPT_DIR

    master_path = os.path.join(data_dir, f"{date_str}_series_{series_num}_master.h5")
    data_path = os.path.join(data_dir, f"{date_str}_series_{series_num}_data_000001.h5")

    if not os.path.exists(master_path):
        candidates = glob.glob(os.path.join(data_dir, f"*series_{series_num}*master*.h5"))
        if candidates:
            master_path = candidates[0]
        else:
            raise FileNotFoundError(f"找不到 Master HDF5 檔案: {master_path}")

    if not os.path.exists(data_path):
        candidates = glob.glob(os.path.join(data_dir, f"*series_{series_num}*data*.h5"))
        if candidates:
            data_path = candidates[0]
        else:
            raise FileNotFoundError(f"找不到 Data HDF5 檔案: {data_path}")

    print(f"\n[載入中] Series {series_num}:")
    print(f"  Master: {master_path}")
    print(f"  Data  : {data_path}")
    print(f"  能量 (Energy): {energy_eV:.1f} eV (波長: {wavelength*1e10:.5f} A)")

    # 1. 讀取與裁切 Detector 影像
    with h5py.File(master_path, 'r') as f_m, h5py.File(data_path, 'r') as f_d:
        raw_data = f_d['/entry/data/data'][:]
        raw_mask = f_m['/entry/instrument/detector/detectorSpecific/pixel_mask'][:]

    cx, cy = center
    half_sz = crop_size // 2
    x0, x1 = max(0, cx - half_sz), min(raw_data.shape[2], cx + half_sz)
    y0, y1 = max(0, cy - half_sz), min(raw_data.shape[1], cy + half_sz)

    cropped_data = raw_data[:, y0:y1, x0:x1].astype(np.float64)
    cropped_mask = (raw_mask[y0:y1, x0:x1] > 0)
    cropped_mask = cropped_mask | (cropped_data >= 0xFFFFFFFF - 10)
    cropped_data[cropped_mask] = 0

    n_frames = len(cropped_data)
    print(f"  影像形狀: {cropped_data.shape}, 裁切範圍: X=[{x0}, {x1}], Y=[{y0}, {y1}]")

    # 2. 自動匹配掃描指令檔
    if cmd_file is None:
        cmd_candidates = [
            os.path.join(SCRIPT_DIR, f"fermat_scan_commands_{n_frames}pts.txt"),
            os.path.join(SCRIPT_DIR, "fermat_scan_commands.txt")
        ]
        for cand in cmd_candidates:
            if os.path.exists(cand):
                test_pts = load_fermat_scan_commands(cand)
                if len(test_pts) == n_frames:
                    cmd_file = cand
                    break
        if cmd_file is None:
            cmd_file = os.path.join(SCRIPT_DIR, "fermat_scan_commands.txt")

    print(f"  使用掃描指令檔: {os.path.basename(cmd_file)}")
    pts = load_fermat_scan_commands(cmd_file)
    if len(pts) != len(cropped_data):
        print(f"  [警告] 指令檔點數 ({len(pts)}) 與影像張數 ({len(cropped_data)}) 不一致！")

    # 計算相對於中心的位移 (um)
    cen_x, cen_z = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    scan_x = (pts[:, 0] - cen_x) * scale_x
    scan_z = (pts[:, 1] - cen_z) * scale_z
    print(f"  掃描座標範圍 (um): X=[{scan_x.min():.3f}, {scan_x.max():.3f}], Z=[{scan_z.min():.3f}, {scan_z.max():.3f}] (scale_x={scale_x}, scale_z={scale_z})")

    # 3. 建立 PyNX Workspace
    ws = pynx_set_data(
        data=cropped_data,
        mask=cropped_mask,
        scan_x=scan_x,
        scan_z=scan_z,
        pixelsize=pixelsize,
        detectordistance=detectordistance,
        wavelength=wavelength,
        nbprobe=nbprobe,
        probe=probe_type
    )
    ws.params['defocus'] = defocus_val
    ws.params['probe'] = probe_type
    ws.scan = series_num
    print(f"[完成] 已建立 `ws` 物件 (Series {series_num}, Probe: {probe_type}, Defocus: {defocus_val}m)！\n")
    return ws


def main_cli():
    parser = argparse.ArgumentParser(description="TPS 25A PyNX Ptychography GUI 快速啟動器")
    parser.add_argument('--series', type=int, default=35, help="自動載入指定 Series 編號的 HDF5 數據 (預設: 35)")
    parser.add_argument('--date', type=str, default="20260821", help="實驗日期資料夾 (預設: 20260821)")
    parser.add_argument('--energy', type=float, default=9670.0, help="光子能量 (eV，預設: 9670.0)")
    parser.add_argument('--cx', type=int, default=572, help="繞射斑圖 X 軸中心像素 (預設: 572)")
    parser.add_argument('--cy', type=int, default=766, help="繞射斑圖 Y 軸中心像素 (預設: 766)")
    parser.add_argument('--size', type=int, default=300, help="繞射斑圖裁切邊長 (預設: 300)")
    parser.add_argument('--probe', type=str, default="disc,500e-9", help="探針初始猜測 (預設: disc,500e-9)")
    parser.add_argument('--defocus', type=float, default=0.0, help="散焦量 (預設: 0.0)")
    parser.add_argument('--cmd', type=str, default=None, help="掃描指令路徑 (.txt)")
    parser.add_argument('--scale-x', type=float, default=1.0, help="X 軸座標縮放因子 (預設: 1.0)")
    parser.add_argument('--scale-z', type=float, default=1.0, help="Z 軸座標縮放因子 (預設: 1.0)")
    parser.add_argument('--no-block', action='store_true', help="非阻塞模式啟動 (IPython 環境使用)")

    args = parser.parse_args()

    ws = None
    if args.series is not None:
        try:
            ws = build_workspace_from_series(
                series_num=args.series,
                date_str=args.date,
                cmd_file=args.cmd,
                center=(args.cx, args.cy),
                crop_size=args.size,
                energy_eV=args.energy,
                probe_type=args.probe,
                defocus_val=args.defocus,
                scale_x=args.scale_x,
                scale_z=args.scale_z,
            )
            # 注入至全域以供 GUI 讀取
            import __main__
            __main__.ws = ws
        except Exception as e:
            print(f"[錯誤] 載入 Series {args.series} 失敗: {e}")

    print("=" * 60)
    print(" 啟動 PyNX PyQt5 控制面板...")
    if ws is not None:
        print(f" 已為您預先載入 Series {args.series} (能量: {args.energy} eV, Scale: X={args.scale_x}, Z={args.scale_z}) 的 `ws` 物件。")
        print(" 點擊 GUI 上的【Set To ws】按鈕即可同步至 GUI 面板！")
    print("=" * 60)

    gui = simple_gui(blocking=not args.no_block)
    return gui


if __name__ == '__main__':
    main_cli()
