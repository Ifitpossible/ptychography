import os
import subprocess
def _load_msvc_env():
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\portable\msvc\msvc-14.44.17.14_sdk-26100\setup_x64.bat"),
        os.path.expandvars(r"%LOCALAPPDATA%\portable\msvc\msvc-14.44.17.14_sdk-26100\activate.cmd"),
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
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

    # Ensure CUDA nvcc and tools are on PATH and CUDA_PATH
    cuda_nvcc_bin = os.path.expandvars(r"%LOCALAPPDATA%\portable\cuda\cuda_nvcc\nvcc\bin")
    if os.path.exists(cuda_nvcc_bin) and cuda_nvcc_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_nvcc_bin + ";" + os.environ.get("PATH", "")
    cuda_root = os.path.expandvars(r"%LOCALAPPDATA%\portable\cuda\cuda_nvcc\nvcc")
    if os.path.exists(cuda_root):
        os.environ["CUDA_PATH"] = cuda_root

# Auto-resolve PyNX library path, matching site-packages, and CUDA DLL directory
import sys
pynx_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pynx"),
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Code", "PYNX", "PyNX")),
    r"C:\Users\User\Desktop\AllenCheng\Code\PYNX\PyNX",
]
if sys.version_info >= (3, 13):
    pynx_paths = [
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Code", "myvenv", "Lib", "site-packages")),
        r"C:\Users\User\Desktop\AllenCheng\Code\myvenv\Lib\site-packages",
    ] + pynx_paths

for p in pynx_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

for cuda_cand in [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
    os.path.expandvars(r"%LOCALAPPDATA%\portable\cuda\cuda_nvcc\nvcc\bin"),
]:
    if os.path.exists(cuda_cand):
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(cuda_cand)
            except Exception:
                pass
        if cuda_cand not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_cand + ";" + os.environ.get("PATH", "")

import matplotlib
import matplotlib.cm
if not hasattr(matplotlib.cm, 'get_cmap'):
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap

import matplotlib.pyplot as plt
plt.ion()
import pandas as pd
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pycuda")
warnings.filterwarnings("ignore", message=".*The CUDA compiler succeeded.*")
warnings.filterwarnings("ignore", message=".*creating CUBLAS context.*")
warnings.filterwarnings("ignore", message=".*cuFFT plan destruction inhibited.*")

from pprint import pprint
from pynx.ptycho import *

from pynx.ptycho.runner import PtychoRunnerScan
from pynx.ptycho.runner.tps25a import PtychoRunnerScanTPS25A, params as default_params
from pynx.utils.fourier_shell_correlation import FSCPlot
from pynx.utils.phase import unwrap_phase,shift_phase_zero,remove_phase_ramp,minimize_grad_phase
from pynx.utils.benchmark import benchmark_fft
from pynx.utils.array import center_array_2d

import h5py
import hdf5plugin
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import pickle
from pynx.ptycho.analysis import probe_propagate
import sys


path_default = '/buffer/UsersData/'


instuction_text = """
    ================================================================================
                        TPS 25A PyNX Ptychography 操作手冊
    ================================================================================

    --------------------------------------------------------------------------------
    【1. 簡單使用 (Quick Start)】
    --------------------------------------------------------------------------------
    您可以先透過 `set_params` 調整基礎參數，再將其帶入執行函數中進行重建。
    #### 1. 基礎參數設定 (Params) 範例：
    params = set_params(
        defocus=800e-6,         # 散焦距離 (公尺)
        maxsize=800,            # 影像裁切大小 (像素)
        nbprobe=3,              # Probe 混合模態數量 (處理部分同調性或震動)
        loadID='9263',          # 指定讀取前一個 Scan 的結果作為初始值
    )
    #### 2. 執行完整重建並自動存檔 (pynx_run)：
    assign_gpu()  # 自動指派閒置的 GPU
    ws = pynx_run(
        scanID='9264', 
        params=params,      # 可將自訂的參數字典吃入 run 中，若不傳入則使用全預設值
        title='first_test', # 儲存檔案時的後綴名稱 (如 ptycho_9264_pynx_result_first_test.cxi)
        path_save=''        # 存檔路徑 (留空 '' 不存檔，指定字串為儲存路徑，設為 True 則自動遞增命名)
    ) 
    #### 3. 分步執行與演算法微調 (pynx_start & pynx_algo_run)：
    ws = pynx_start(scanID='9264', params=params) # 需要中途檢查或更換演算法可使用 pynx_start 僅做資料載入與初始化

    #  - 'normal'     : 預設標準流程 (ML**300*DM**300)
    #  - 'quick'      : 快速測試流程 (ML**100*DM**100)
    #  - 'mask probe' : 鎖定 Probe 不更新，專注於更新 Object (DM**10)
    #  - 'DM only'    : 僅執行 DM (Difference Map) 演算法 (DM**100)
    #  - 'ML only'    : 僅執行 ML (Maximum Likelihood) 演算法 (ML**100)
    #  - 'PC'         : 包含位置校正 (Position Correction) 的演算法組合 (AP**20*ML**20)
    pynx_algo_run(ws, algo_setting='DM only') 

    --------------------------------------------------------------------------------
    【2. 進階使用 (Advanced Usage)】
    --------------------------------------------------------------------------------
    #### 1. 狀態重置、讀取與模態繼承：
    pynx_init(ws, init='both')  # 將 Object 或 Probe 恢復初始狀態 ('probe', 'obj', 'both')
    pynx_inherit_1st_mode(ws, init_obj=True) # 保留 Probe 的第 0 個主模態，並將其他模糊模態重新加入微小亂數初始化
    pynx_load(ws, path_cxi='/path/to/Result.cxi', load='probe') # 從外部 CXI 檔案讀取並覆寫當前的 Probe

    #### 2. 空間遮罩處理 (Masking)：
    pynx_probe_masking(ws) # 將 Probe 在有效掃描區域外的數值強制歸零
    pynx_obj_masking(ws)   # 將 Object 在有效掃描區域外的數值強制歸零

    #### 3. 變更演算法設定 (Algorithm Modification)：
    custom_algo = {
        'algo_string': 'DM**50*ML**50',
        'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
        'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20'
    }
    pynx_algo_run(ws, algo_setting=custom_algo)

    --------------------------------------------------------------------------------
    【3. 視覺化 (Visualization)】
    --------------------------------------------------------------------------------
    模組內建多種視覺化工具，支援傳入工作區物件 `ws` 或直接傳入檔案路徑/關鍵字：

    # 總覽 (同時看 Object 與 Probe 的振幅與相位)：
        pynx_plot_overview(ws) 
        
    # 檢視 Object 詳細重建影像 (支援 'original', 'coloring_lin', 'coloring_dark', 'coloring_log' 等著色模式)：
        pynx_plot_obj(ws, mode='coloring_lin')

    # 檢視 Probe 詳細波前與各模態佔比 (支援 'original', 'coloring_lin', 'coloring_dark', 'coloring_log' 等著色模式)：
        pynx_plot_probe(ws, mode='original')

    # 互動式尋找焦點 (計算 Probe 在不同 Z 軸深度的傳播變化)：
        pynx_probe_propagate(
            ws, 
            use_probe=0,                            # 指定要傳播的 Probe 模態索引 (預設為 0，即主模態)
            p_linspace=[-1000e-6, 1000e-6, 200],    # 傳播的深度範圍與步數，格式為 [起始距離, 結束距離, 切片數量] (單位: 公尺)
            interact=True                           # 設為 True 可開啟互動式滑桿，動態觀察不同深度的切面
        )

    --------------------------------------------------------------------------------
    【4. 其他便利函數 (Utility Functions)】
    --------------------------------------------------------------------------------
    # 自動指派系統中閒置的 GPU 給程式使用：
        assign_gpu()

    # 影像拼接 (Stitching)：
    # 將多個 Scan 的相位影像進行全域座標對齊與距離加權融合。
        ws_list = ['9264', '9265', '9266']
        final_phase = stitch_obj(ws_list, pos_source='baseline')

    # 檢視 HDF5 檔案結構 (包含 SoftLinks 追蹤)：
        print_h5_tree('/path/to/file.h5')
        
    # 搜尋檔案路徑 (自動在預設目錄下遞迴尋找)：
        path = find_files(['9264', 'master'], path='/buffer/UsersData/')

    # 計算 FSC 解析度估計 (Fourier Shell Correlation)：
        FSC = fsc(img1, img2, method='gradient')

    ================================================================================
    【5. 函數總覽 (Function Summary)】
    ================================================================================
    [核心流程與執行]
    * pynx_run              : 執行標準的完整 Ptychography 重建流程並自動存檔。
    * pynx_start            : 初始化重建工作區並載入資料，但不立即執行演算法。
    * pynx_algo_run         : 對已初始化的工作區套用指定的演算法組合（如純 DM 或 ML+DM）。
    * set_params            : 設定與更新 Ptychography 重建的各項參數 (Defocus, Maxsize 等)。
    * pynx_set_params       : 直接更新現有工作區 (workspace) 中的參數字典。
    * set_algorithm         : 建立自訂的迭代演算法組合字串。

    [狀態與存取]
    * pynx_save             : 將重建結果（Object 與 Probe）儲存為 CXI 檔案。
    * pynx_save_default     : 使用預設的遞增命名規則儲存結果以避免覆蓋。
    * pynx_load             : 從外部 CXI 檔案讀取並覆寫當前的 Probe 或 Object。
    * pynx_init             : 重新初始化工作區，將 Object 與 Probe 恢復到演算法執行前狀態。
    * pynx_inherit_1st_mode : 保留 Probe 第一個主模態，並重新隨機初始化其餘混合模態。
    * pynx_get_objprobe     : 取得當前工作區或 CXI 檔案中的 Object 與 Probe 陣列。
    * pynx_set_objprobe     : 手動覆寫工作區中的 Object 或 Probe NumPy 陣列。
    * pynx_probe_masking    : 將 Probe 在有效掃描區域外的數值強制歸零。
    * pynx_obj_masking      : 將 Object 在有效掃描區域外的數值強制歸零。

    [視覺化與分析]
    * pynx_plot_overview    : 繪製重建結果的總覽圖（包含 Object 與 Probe 的振幅與相位）。
    * pynx_plot_obj         : 繪製 Object 的詳細重建影像，支援多種色彩與對比度映射模式。
    * pynx_plot_probe       : 繪製 Probe 的詳細重建影像與各模態能量佔比。
    * pynx_probe_propagate  : 計算並繪製 Probe 在不同 Z 軸深度（Defocus）下的自由空間傳播。
    * interactive_probes    : 啟動互動式滑桿介面，動態觀察 Probe 在不同深度的切面變化。
    * plot_image            : 快速繪製單張 2D 影像並自動附帶 Colorbar。
    * fsc                   : 計算兩張獨立重建影像的傅立葉殼層相關 (FSC) 曲線以估計解析度。
    * stitch_obj            : 提取多個掃描的相位資料，進行距離加權平均與全域無縫拼接。

    [資料讀取與預處理]
    * get_exp_path          : 取得指定 Scan ID 的所有相關實驗檔案路徑（master, primary, data）。
    * load_exp_condition    : 從 master 檔案讀取實驗幾何物理條件（波長、偵測器距離等）。
    * read_metadata         : 讀取實驗的 Metadata 資訊檔案並轉為 Python 字典。
    * read_baseline         : 讀取 baseline CSV 檔案中的馬達初始相對座標。
    * get_info_from_scanID  : 整合讀取實驗路徑、條件與參數設定的快捷整合函數。
    * get_all_data          : 讀取並回傳繞射數據、偵測器遮罩、馬達座標等所有原始資料陣列。
    * get_data              : 讀取 HDF5 中的繞射資料，並支援自動化中心裁切與置中。
    * get_mask              : 從 master 檔案讀取偵測器的壞點 (Dead pixels) 遮罩。
    * get_scan              : 從 primary CSV 檔案讀取二維掃描網格的馬達實際記錄座標。
    * pynx_set_data         : 允許手動傳入 NumPy 陣列數據以建立自訂的 PyNX 重建工作區。

    [幾何運算與置中]
    * center_of_mass        : 計算 2D 影像數值分佈的質心位置。
    * center_from_info      : 從 master 檔案中提取記錄的光束中心點位置。
    * center_crop           : 根據給定的中心點與尺寸大小對二維影像陣列進行精確裁切。
    * match                 : 利用相位交叉相關 (Cross-correlation) 對齊並裁切兩張影像。
    * flatten               : 消除影像中的整體線性背景傾斜（支援互動式選取背景基準點）。

    [系統與工具函數]
    * assign_gpu            : 自動尋找並將環境變數指派給目前系統中閒置的 GPU。
    * print_h5_tree         : 以層級樹狀結構印出 HDF5 檔案內容，並包含外部與軟連結資訊。
    * build_h5_dict         : 將 HDF5 檔案結構轉換為嵌套字典，支援延遲讀取以最佳化記憶體。
    * extract_scan_id       : 從檔案名稱或路徑字串中利用正則表達式萃取實驗 Scan ID。
    * get_today_path        : 取得符合當天日期的實驗資料夾絕對路徑。
    * find_files            : 在指定目錄下遞迴搜尋檔名包含特定關鍵字串的檔案。
    * get_latest_path       : 取得指定目錄下最新修改時間的子資料夾路徑。
    * print_dict            : 以易讀的層次化色彩格式將複雜的字典內容輸出至終端機。
    * simple_gui            : 啟動基於 PyQt5 撰寫的圖形化使用者介面 (GUI) 控制面板。
    ================================================================================

    """



#===========================================================================
#                              Utilities                                    
#===========================================================================
def print_h5_tree(path):
    """
    Prints HDF5 structure using manual recursion to include all SoftLinks.
    (ASCII-only version to prevent Windows terminal encoding errors)
    """
    with h5py.File(path, 'r') as f:
        print(f"Structure of: {path}\n/")
        
        def traverse(obj, depth=1):
            for key in obj.keys():
                indent = "  " * depth
                
                # 先取得 Link 資訊
                link = obj.get(key, getlink=True)
                is_ext = isinstance(link, h5py.ExternalLink)
                is_soft = isinstance(link, h5py.SoftLink)
                is_link = is_ext or is_soft
                
                # 組合 Link 的顯示字串
                if is_ext:
                    link_str = f" -> [External] {link.filename}::{link.path}"
                elif is_soft:
                    link_str = f" -> [Soft] {link.path}"
                else:
                    link_str = ""

                # 嘗試讀取物件
                try:
                    item = obj[key]
                except KeyError:
                    print(f"{indent}[X] {key} (Broken Link){link_str}")
                    continue 
                
                if isinstance(item, h5py.Group):
                    print(f"{indent}[Dir] {key}/{link_str}")
                    traverse(item, depth + 1)
                else:
                    # Dataset info
                    info = f"{indent}[File] {key} {item.shape} {item.dtype}{link_str}"
                    
                    # Print value for scalars
                    if not is_link and item.size == 1:
                        val = item[()]
                        if isinstance(val, bytes): 
                            val = val.decode('utf-8', errors='ignore')
                        info += f" -> {val}"
                    print(info)

        traverse(f)


def build_h5_dict(path):
    """
    Recursively builds a nested dictionary representing the structure of an HDF5 file.
    
    Scalar datasets are loaded immediately (and decoded if they are bytes), while 
    non-scalar datasets are returned as lazy-loading functions to optimize memory usage.
    
    Args:
        path (str): The system path to the HDF5 file.
        
    Returns:
        dict: A nested dictionary where keys are group/dataset names and values 
              are either nested dictionaries, scalar values, or reader functions.
    """
    def make_reader(file_path, dataset_path):
        # This function opens the file only when called
        def reader():
            with h5py.File(file_path, 'r') as f:
                return f[dataset_path][()]
        return reader

    def recursive_scan(group, file_path):
        current_dict = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # If it is a group, recurse
                current_dict[key] = recursive_scan(item, file_path)
            
            elif isinstance(item, h5py.Dataset):
                if item.ndim == 0: 
                    val = item[()]
                    if isinstance(val, bytes): # decoding bytes to utf-8
                        val = val.decode('utf-8')
                    current_dict[key] = val
                else:
                    current_dict[key] = make_reader(file_path, item.name)
                    
        return current_dict

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, 'r') as f:
        return recursive_scan(f, path)


def extract_scan_id(path):
    """
    Extracts the scan ID from a file path.
    Supports both legacy formats (scan_id-123) and new formats (YYMMDD_ptycho_XXXX_master.h5).
    
    Args:
        path (str): The path to the file.
        
    Returns:
        str or None: The scan ID if found, otherwise None.
    """
    import re
    if path is None:
        return None
    
    # Extract only the filename to avoid interference from parent directory names
    filename = os.path.basename(path)
    
    # 1. Match New Format: YYMMDD_ptycho_(XXXXX)_master.h5
    # Look for 4 to 5 digits following 'ptycho_' and preceding an underscore
    new_pattern = r'ptycho_(\d{4,5})_'
    new_match = re.search(new_pattern, filename)
    if new_match:
        return new_match.group(1)
    
    # 2. Match Legacy Format: scan_id-123 or scanid_123
    legacy_pattern = r'(?i)scan_?id[-_](\d+)'
    legacy_match = re.search(legacy_pattern, filename)
    if legacy_match:
        return legacy_match.group(1)
    
    return None


def get_today_path(path_base=path_default):
    """
    Returns the path to the current day's data directory.
    """
    from datetime import datetime
    if os.path.exists(path_base):
        folders = [f for f in os.listdir(path_base) if f.isdigit() and len(f) == 8]
        if folders:
            now_int = int(datetime.now().strftime('%Y%m%d'))
            closest = min(folders, key=lambda x: abs(int(x[:8]) - now_int))
            print(f'Today path: {os.path.join(path_base, closest)}')
            return os.path.join(path_base, closest)
        else:
            print(f'No folders found in {path_base}')
            return None


def assign_gpu():
    """
    Automatically assigns an idle GPU to CUDA_VISIBLE_DEVICES.
    Principle: Uses 'nvidia-smi pmon' to find devices with no active processes.
    """
    import os
    import pandas as pd
    import subprocess
    from io import StringIO
    try:
        result = subprocess.run(['nvidia-smi', 'pmon' ,'-c', '1'], stdout=subprocess.PIPE, text=True).stdout.replace('#','')
        output = StringIO(result)
        df = pd.read_csv(output,sep=r'\s+', skiprows=[1], header=0)
        idle_gpu_list = df['gpu'][df['pid'] == '-'].to_numpy()
        if len(idle_gpu_list) == 0:
            print('!!! Warning: All GPU are occupied. !!!')
            print('!!! Warning: No GPU is assigned. !!!')
        else:
            print(f'Idle GPU: {idle_gpu_list}')
            print(f'GPU set to GPU_{idle_gpu_list[0]}.')
            os.environ['CUDA_VISIBLE_DEVICES'] = str(idle_gpu_list[0])
    except Exception as e:
        print(f"assign_gpu notice: {e}, falling back to default GPU 0.")
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
assign_gpu()


def find_files(keyword, path=path_default, find_all=True):
    """
    Recursively search for files containing specific keyword(s) in their filenames.

    Args:
        keyword (str or list): The substring or list of substrings to look for in the filenames.
        path (str): The root directory path where the search begins.
        find_all (bool): If True, find all matching files. If False, return the first match found.

    Returns:
        str or list: The path string if one file is found (or find_all=False), otherwise a list of paths.
    """
    keyword = str(keyword) if not isinstance(keyword, (list, str)) else keyword
    keywords = [keyword] if isinstance(keyword, str) else keyword
    found_list = []

    for root, dirs, files in os.walk(path):
        for filename in files:
            if all(str(k) in filename for k in keywords):
                full_path = os.path.normpath(os.path.join(root, filename))
                print(f'Found file: {full_path}')
                if not find_all:
                    return full_path
                found_list.append(full_path)
    
    if not found_list:
        raise FileNotFoundError(f"No files found containing keywords: {keywords} in {path}")
                
    return found_list[0] if len(found_list) == 1 else found_list


def get_latest_path(path=path_default):
    from pathlib import Path
    path = Path(path)
    subdirs = [p for p in path.iterdir() if p.is_dir()]
    if not subdirs:
        return None   
    latest_subdir = max(subdirs, key=lambda p: p.stat().st_mtime)
    return latest_subdir


def print_dict(data, indent=0):
    import re
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_YELLOW = "\033[93m"
    GREEN = "\033[32m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    if indent == 0:
        print(f"\n{BOLD}{BRIGHT_CYAN}{'━'*60}{RESET}")
        print(f"{BOLD}{BRIGHT_CYAN} 📦 DATA STRUCTURE EXPLORER (Object-Aware){RESET}")
        print(f"{BOLD}{BRIGHT_CYAN}{'━'*60}{RESET}")

    spacing = "    " * indent
    connector = f"{GRAY}├─{RESET} " if indent > 0 else ""

    def _parse_custom_obj(v):
        """ 嘗試解析 Name(attr1=..., attr2=...) 結構 """
        if not isinstance(v, str): return v
        v = v.strip()
        
        # 匹配 "名稱(內容)" 結構
        match = re.match(r"^([\w\.]+)\((.*)\)$", v, re.DOTALL)
        if match:
            obj_name = match.group(1)
            content = match.group(2)
            
            # 這裡簡單處理內容：嘗試按逗號拆分，但避免拆開內層的 [] 或 ()
            # 使用 regex 拆分不在括號內的逗號
            parts = re.split(r",\s*(?![^\[\(]*[\]\)])", content)
            return {"[OBJECT] " + obj_name: [p.strip() for p in parts if p.strip()]}
        
        # 處理單純的 (a, b) 或 [a, b] 
        if v.startswith('(') and v.endswith(')') or v.startswith('[') and v.endswith(']'):
            content = v[1:-1]
            parts = re.split(r",\s*(?![^\[\(]*[\]\)])", content)
            return [p.strip() for p in parts if p.strip()]
            
        return v

    if isinstance(data, dict):
        for k, v in data.items():
            processed_v = _parse_custom_obj(v)
            if isinstance(processed_v, (dict, list)):
                print(f"{spacing}{connector}{BRIGHT_YELLOW}{k}{RESET}")
                print_dict(processed_v, indent + 1)
            else:
                val_str = str(processed_v)
                if len(val_str) > 80: val_str = val_str[:77] + "..."
                print(f"{spacing}{connector}{BOLD}{k:<18}{RESET} : {GREEN}{val_str}{RESET}")

    elif isinstance(data, list):
        for i, item in enumerate(data):
            processed_item = _parse_custom_obj(item)
            if isinstance(processed_item, (dict, list)):
                # 如果是 object 轉換過來的 dict，我們直接印出內容，不印 Index
                print_dict(processed_item, indent)
            else:
                print(f"{spacing}{connector}{GREEN}{item}{RESET}")

    if indent == 0:
        print(f"{BOLD}{BRIGHT_CYAN}{'━'*60}{RESET}\n")


#===========================================================================
#                             Build-in pynx                                  
#===========================================================================
def load_exp_condition(path_exp,**kwargs):
    """
    Extracts experimental conditions and detector parameters from the master HDF5 file and primary CSV file.
    IMPORTANT: If the strucuter of the files is changed, please modify this function.
    
    Args:
        path_exp (str or dict): The path to the experiment data.
    Returns:
        dict: A dictionary containing the extracted experimental conditions.
    """
    path_master = path_exp['path_master']
    exp_condition = {} # Initialize an empty dictionary
    # Reading experiment condition files
    with h5py.File(path_master, 'r') as f:
        # --- Beam and Energy Parameters ---
        exp_condition['wavelength'] = f['entry/instrument/beam/incident_wavelength'][()]
        exp_condition['photon_energy'] = f['entry/instrument/detector/detectorSpecific/photon_energy'][()]
        
        # --- Detector Geometry Parameters ---
        exp_condition['detector_distance'] = f['entry/instrument/detector/detector_distance'][()]
        # Pixel size is square (x=y), storing only one value
        exp_condition['pixel_size_detector'] = f['entry/instrument/detector/x_pixel_size'][()]
        
        # --- Beam Center Parameters ---
        exp_condition['beam_center_x'] = f['entry/instrument/detector/beam_center_x'][()]
        exp_condition['beam_center_y'] = f['entry/instrument/detector/beam_center_y'][()]
        
        # --- Time and Readout Parameters ---
        exp_condition['count_time'] = f['entry/instrument/detector/count_time'][()]
        exp_condition['detector_readout_time'] = f['entry/instrument/detector/detector_readout_time'][()]
        exp_condition['detector_readout_period'] = f['entry/instrument/detector/detectorSpecific/detector_readout_period'][()]
        
        # --- Detector Metadata (Decoding bytes to string) ---
        roi_mode = f['entry/instrument/detector/detectorSpecific/roi_mode'][()]
        exp_condition['roi_mode'] = roi_mode.decode('utf-8') if isinstance(roi_mode, bytes) else roi_mode
        
        det_num = f['entry/instrument/detector/detector_number'][()]
        exp_condition['detector_number'] = det_num.decode('utf-8') if isinstance(det_num, bytes) else det_num
        
        # pixel_mask is a 2D array (e.g., 2167 x 2070)
        exp_condition['mask_detector'] = f['entry/instrument/detector/detectorSpecific/pixel_mask'][()]

    print("\nExperimental Condition:")
    pprint(exp_condition)

    return exp_condition


def get_exp_path(scanID='9264', path=path_default, key='ptycho'):
    """
    Locates and returns the file paths for the master, primary, baseline, and data files of a scan.

    Args:
        scanID (str): The unique identifier for the scan.
        path (str): The root directory to search for experimental files.

    Returns:
        dict: A dictionary mapping file types ('master', 'primary', 'baseline', 'data') to their paths.
    """
    path_dict = {}
    # Search for files with the specified keywords
    """
    Need to be modified
    find from bluesky, the name can be different
    """
    path_dict['path_master'] = find_files(['master', scanID, key], path, find_all=False)
    path_dict['path_primary'] = find_files(['primary', scanID, key], path, find_all=False)
    path_dict['path_baseline'] = find_files(['baseline', scanID, key], path, find_all=False)
    path_dict['path_data'] = find_files(['data', scanID, key], path, find_all=True)
    path_dict['path_today'] = '/'.join(path_dict['path_master'].split('/')[:-1])+'/'
    print(f'\nExperiment path for scan ID {scanID}:')
    pprint(path_dict)
    return path_dict


def read_metadata(scanID, folder=path_default):
    """
    Read metadata from the metadata file.
    
    Args:
        scanID (str): The unique identifier for the scan.
        folder (str): The root directory to search for experimental files.
    
    Returns:
        dict: A dictionary containing the metadata.
    """
    import ast
    import pandas as pd
    
    if folder == 'today': 
        folder = str(get_latest_path())

    path_metadata = find_files([scanID, 'metadata'], path=folder)
    df = pd.read_csv(path_metadata, header=None).set_index(0).T
    df.columns.name = None
    df.reset_index(drop=True, inplace=True)

    raw_metadata = df.iloc[0].to_dict()
    
    def clean_structure(v):
        if isinstance(v, str):
            v = v.strip()
            if (v.startswith('{') and v.endswith('}')) or (v.startswith('[') and v.endswith(']')):
                try:
                    v = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    pass
        
        if isinstance(v, dict):
            return {k: clean_structure(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [clean_structure(i) for i in v]
        return v

    metadata_dict = clean_structure(raw_metadata)

    return metadata_dict


def read_baseline(scanID,folder=path_default):
    """
    Read baseline from the baseline file.
    
    Args:
        scanID (str): The unique identifier for the scan.
        folder (str): The root directory to search for experimental files.
    
    Returns:
        dict: A dictionary containing the baseline data.
    """
    path_baseline = find_files([str(scanID), 'baseline'], path=folder, find_all=False)
    coord_df = pd.read_csv(path_baseline)
    data_baseline = coord_df.to_dict()
    return data_baseline


def set_params_center(center, maxsize, path_master, params={}):
    """
    Calculates the Region of Interest (ROI) and maxsize based on the center strategy.
    
    Args:
        center (str or list/tuple): 'auto', 'exp_config', or [cx, cy].
        maxsize (int): The size of the crop window.
        path_master (str or dict): Path to master file or dict from get_exp_path.
        
    Returns:
        dict: A dictionary containing updates for 'roi' and 'maxsize'.
    """
    updates = params.copy()

    # if path_master is dict, then path_master['path_master'] is the path to master file
    if isinstance(path_master, dict):
        path_master = path_master['path_master']

    if center == 'auto':
        updates['roi'] = 'auto'
        updates['maxsize'] = maxsize
    
    elif center == 'exp_config':
        try:
            with h5py.File(path_master, 'r') as f:
                cx = f['entry/instrument/detector/beam_center_x'][()].astype(int)
                cy = f['entry/instrument/detector/beam_center_y'][()].astype(int)
            print(f"\nCenter is set to be: {cx}, {cy} (from master file)")
            xmin, xmax = cx - maxsize / 2, cx + maxsize / 2
            ymin, ymax = cy - maxsize / 2, cy + maxsize / 2
            updates['roi'] = f"{xmin},{xmax},{ymin},{ymax}"
        except Exception as e:
            raise ValueError(f"Failed to read beam center from master file {path_master}:\n {e}")
    else:
        # Manual center input [cx, cy]
        try:
            cx, cy = center
            print(f"\nCenter is set to be: {cx}, {cy}")
            xmin, xmax = cx - maxsize / 2, cx + maxsize / 2
            ymin, ymax = cy - maxsize / 2, cy + maxsize / 2
            updates['roi'] = f"{xmin},{xmax},{ymin},{ymax}"
        except Exception as e:
            print(f"Center must be 'auto', 'info', or a list or tuple of two elements.\n The center input: {center}")
            raise ValueError(f"Invalid center format: {e}")

    return updates


def set_params_exp(exp_condition, path_exp, params={}):
    """
    Updates or returns experimental parameters based on experimental conditions and paths.

    Args:
        exp_condition (dict): Dictionary containing experimental metadata.
        path_exp (str or dict, optional): New unified input for experimental paths.
        params (dict, optional): Dictionary to be updated.

    Returns:
        dict: A dictionary containing updates for 'data', 'scanfile', 'nrj', etc.
    """
    updates = params.copy()
    
    # Handle path_exp to set data and scanfile
    if params.get('data') is None:
        updates['data'] = path_exp.get('path_master')
    if params.get('scanfile') is None:
        updates['scanfile'] = path_exp.get('path_primary')
        updates['scan_id'] = int(extract_scan_id(path_exp.get('path_primary')) or 0)
    if params.get('nrj') is None:
        if exp_condition.get('wavelength') is not None:
            updates['nrj'] = 12.3984 / np.array(exp_condition.get('wavelength'))
        else:
            updates['nrj'] = None
    if params.get('detectordistance') is None:
        updates['detectordistance'] = exp_condition.get('detector_distance')
    if params.get('pixelsize') is None:
        updates['pixelsize'] = exp_condition.get('pixel_size_detector')
    
    return updates


def set_params_load(loadID=None, path_load=path_default, params={}, load='probe'):
    """
    Updates or returns parameters for loading specific components like probe, mask, or all.
    Args:
        loadID      (int, str, or list, optional): Keywords to find the file if path_load is a directory.
                                                   Can be a single ID/name or a list (e.g., ['test', '9264']).
        path_load   (str, optional)              : Path to a specific file or a search directory. 
                                                   Defaults to path_default.
        params      (dict, optional)             : Dictionary to be updated.
        load        (str, optional)              : Type of component to load ('probe', 'mask', or 'all'). 
                                                   Defaults to 'probe'.
    Returns:
        dict: A dictionary containing updates for 'loadprobe', 'loadmask', or 'load'.
    """
    updates = params.copy()
    target_path = None

    if os.path.isfile(path_load):
        target_path = path_load
    else:
        if loadID is None:
            raise ValueError(f"path_load '{path_load}' is not a file, and no loadID provided for search.")
        keywords = [str(k) for k in loadID] if isinstance(loadID, list) else [str(loadID)] 
        found = find_files(keywords + ['cxi'], path=path_load, find_all=True)
        if isinstance(found, list):
            target_path = sorted(found)[-1]
            if len(found) > 1:
                print(f"\n[Alert] Multiple {load} files found with keywords {keywords} in {found}.\n")
                print(f"Loading the latest one:\n {target_path}\n")
        else:
            target_path = found
    print(f'\nLoading {load} from: {target_path}\n')
    
    if load == 'probe':
        updates['loadprobe'] = target_path
    elif load == 'mask':
        updates['loadmask'] = target_path
    elif load == 'all':
        updates['load'] = target_path

    return updates


def set_params(
    exp_condition={},
    path_exp=None,
    center=None,
    nbprobe=3,
    loadID=None,
    path_load=path_default,
    # --- Data & Beamline ---
    instrument='TPS 25A',
    data=None,
    scanfile=None,
    h5data='entry/data/data_%06d',
    nrj=None,
    detectordistance=None,
    pixelsize=None,
    rebin=1,
    detector_orientation=None,
    maxsize=800,
    roi='auto',
    # --- ROI & Object ---
    obj_max_pix=8000,
    obj_margin=32,
    obj_inertia=0.05,
    remove_obj_phase_ramp=True,
    object='random,0.8,1,0,0.5',    
    # --- Algorithm & Performance ---
    algorithm='manual',
    stack_size=96,
    verbose=50,
    gpu=None,
    mpi='scan',
    profiling=False,
    # --- Initialization & Probe ---
    probe='focus,100e-6,0.042',
    defocus=500e-6,
    probe_inertia=0.005,
    rotate=None,
    # --- Loading & External Files ---
    load=None,
    loadprobe=None,
    loadmask=None,
    cxifile=None,
    loadpixelsize=None,
    # --- Corrections & Masking ---
    flatfield=None,
    dark=None,
    dark_subtract=False,
    mask_iobs_max=None,
    # --- Saving & Output ---
    save='final',
    saveplot=False,
    saveprefix='none', # original default: 'ResultsScan{scan:04d}/Run{run:04d}'
    cxi_output='object_probe',
    # --- Advanced & Iteration Control ---
    interpolation=False,
    liveplot=False,
    livescan=False,
    data2cxi=False,
    near_field=False,
    no_rerun=False,
    nbrun=1,
    run0=None,
    maxframe=None,
    moduloframe=None,
    scan=None,
    xy=None,
    xyrange=None,
    # --- Additional Parameters ---
    fig_num=100,
    obj_smooth=0,
    probe_smooth=0,
    pos_mult=5,
    pos_max_shift=1,
    pos_min_shift=0,
    pos_threshold=0.2,
    background_smooth=3,
    center_probe_n=5,
    center_probe_max_shift=5,
    dm_loop_obj_probe=1,
    dm_alpha=0.02,
    raar_beta=0.9,
    ml_obj_regularisation=0,
    floating_intensity=False,
    orientation_round_robin=False,   
    output_format='cxi',             
    multiscan_reuse_ptycho=None,     
    padding=0,                       
    use_direct_beam=False,           
    autocenter=True,                 
    movie=None,                      
    **kwargs
):
    """
    Sets parameters for ptychographic reconstruction at TPS 25A.

    --- Data & Geometry ---
    - instrument           : Beamline name (e.g., 'TPS 25A').
    - data                 : Path to master data. (default: path_exp['master'])
    - scanfile             : Path to primary scan file. (default: path_exp['primary'])
    - h5data               : HDF5 path to raw data inside the file.
    - nrj                  : Photon energy in keV. (default: calculated from wavelength)
    - detectordistance     : Sample-to-detector distance (m). (default: exp_condition['detector_distance'])
    - pixelsize            : Detector pixel size (m). (default: exp_condition['pixel_size_detector'])
    - rebin                : Binning factor (sum n x n pixels). (default: 1)
    - detector_orientation : Orientation flags [transpose, flipud, fliplr] (e.g. '0,1,0'). (default: None)
    - xy                   : Force coordinates expression (e.g. 'x,-y'). Overrides scan data. (default: None)
    - xyrange              : Limit reconstruction to specific scan range (xmin, xmax, ymin, ymax). (default: None)

    --- ROI & Object ---
    - maxsize              : Max frame size for auto-cropping. (default: 800)
    - roi                  : 'auto' (COM), 'full', or specific coords (xmin,xmax,ymin,ymax). (default: 'auto')
    - obj_max_pix          : Max allowed object size in pixels. (default: 8000)
    - obj_margin           : Margin (pixels) around the object. (default: 32)
    - obj_inertia          : Inertia for object update (0.0-1.0). (default: 0.05)

    --- Algorithm & Performance ---
    - algorithm            : Optimization chain (e.g., 'ML**50,DM**100') or 'manual'. (default: 'manual')
    - stack_size           : Batch size for GPU processing. (default: 96)
    - verbose              : Log output frequency (every N cycles). (default: 50)
    - gpu                  : Specific GPU name or index (e.g., '0' or 'Titan'). (default: None)
    - mpi                  : MPI mode ('scan' for distinct scans, 'split' for single scan split). (default: 'scan')
    - profiling            : Enable OpenCL profiling output at end of run. (default: False)

    --- Initialization ---
    - object               : Initial object guess (e.g., 'random,0.8,1,0,0.5'). (default: 'random...')
    - probe                : Initial probe guess ('focus,size,f', 'gauss...', 'disc...').
    - defocus              : Defocus distance (m) for initial probe propagation. (default: 500e-6)
    - rotate               : Rotate initial probe by N degrees. (default: None)
    - probe_inertia        : Inertia for probe update (0.0-1.0). (default: 0.005)

    --- Loading & Resuming ---
    - load                 : Load full state (obj+probe) from .npz/.cxi. Overrides init params. (default: None)
    - loadprobe            : Load existing probe only (if 'load' is None). (default: None)
    - loadmask             : Load external mask (.h5, .edf, .npy). Merged with detector mask. (default: None)
    - cxifile              : Path to load parameters/data from CXI file. (default: None)
    - loadpixelsize        : The Loaded data pixel size. (default: None, then will read from cxifile)

    --- Corrections ---
    - flatfield            : Path for flatfield correction file. (default: None)
    - dark                 : Path for dark current correction file. (default: None)
    - dark_subtract        : Subtract dark current. (Deprecated/Discouraged by PyNX). (default: False)
    - mask_iobs_max        : Mask pixels with intensity >= this threshold. (default: None)
    - remove_obj_phase_ramp: Remove phase ramp on save. (default: True)

    --- Scan Control & Output ---
    - save                 : Save mode ('final', 'all'). (default: 'final')
    - saveplot             : Save preview plots (png). (default: False)
    - saveprefix           : Output filename pattern. (original default: 'ResultsScan{scan:04d}/Run{run:04d}', modified: None)
    - nbrun                : Number of repeated runs. (default: 1)
    - run0                 : Start run number index. If None, auto-increments. (default: None)
    - maxframe             : Limit number of frames loaded. (default: None)
    - moduloframe          : Load every Nth frame (skip frames). (default: None)
    - center               : Center logic ('auto', 'exp_config', or [cx,cy]). (default: 'auto')
    - nbprobe              : Number of modes for multi-mode probe. (default: 3)
    """
    # Copy default parameters
    params = default_params.copy()
    # =========================================
    # Data & Geometry Settings
    # =========================================
    params['instrument'] = instrument
    params['data'] = data
    params['scanfile'] = scanfile
    params['h5data'] = h5data
    params['nrj'] = nrj
    params['detectordistance'] = detectordistance
    params['pixelsize'] = pixelsize
    params['rebin'] = rebin
    params['detector_orientation'] = detector_orientation
    # =========================================
    # ROI & Object Settings
    # =========================================
    params['maxsize'] = maxsize   # [Important] Fixed missing assignment
    params['obj_max_pix'] = obj_max_pix
    params['obj_margin'] = obj_margin
    params['object'] = object
    params['obj_inertia'] = obj_inertia
    params['remove_obj_phase_ramp'] = remove_obj_phase_ramp
    # =========================================
    # Algorithm & Performance
    # =========================================
    params['algorithm'] = algorithm
    params['stack_size'] = stack_size
    params['verbose'] = verbose
    params['gpu'] = gpu
    params['mpi'] = mpi
    params['profiling'] = profiling
    # =========================================
    # Probe & Initialization
    # =========================================
    params['probe'] = probe
    params['nbprobe'] = nbprobe
    params['defocus'] = defocus
    params['probe_inertia'] = probe_inertia
    params['rotate'] = rotate
    # =========================================
    # External Files (Load/Resume)
    # =========================================
    params['load'] = load
    params['loadprobe'] = loadprobe
    params['loadmask'] = loadmask
    params['cxifile'] = cxifile
    params['loadpixelsize'] = loadpixelsize
    # =========================================
    # Corrections
    # =========================================
    params['flatfield'] = flatfield
    params['dark'] = dark
    params['dark_subtract'] = dark_subtract
    params['mask_iobs_max'] = mask_iobs_max
    # =========================================
    # Saving & Output
    # =========================================
    params['save'] = save
    params['saveplot'] = saveplot
    params['saveprefix'] = saveprefix
    params['cxi_output'] = cxi_output
    # =========================================
    # Advanced Control
    # =========================================
    params['interpolation'] = interpolation
    params['liveplot'] = liveplot
    params['livescan'] = livescan
    params['data2cxi'] = data2cxi
    params['near_field'] = near_field
    params['no_rerun'] = no_rerun
    params['nbrun'] = nbrun
    params['run0'] = run0
    params['maxframe'] = maxframe
    params['moduloframe'] = moduloframe
    params['scan'] = scan
    params['xy'] = xy
    params['xyrange'] = xyrange
    # =========================================
    # Additional Parameters
    # =========================================
    params['fig_num'] = fig_num
    params['obj_smooth'] = obj_smooth
    params['probe_smooth'] = probe_smooth
    params['pos_mult'] = pos_mult
    params['pos_max_shift'] = pos_max_shift
    params['pos_min_shift'] = pos_min_shift
    params['pos_threshold'] = pos_threshold
    params['background_smooth'] = background_smooth
    params['center_probe_n'] = center_probe_n
    params['center_probe_max_shift'] = center_probe_max_shift
    params['dm_loop_obj_probe'] = dm_loop_obj_probe
    params['dm_alpha'] = dm_alpha
    params['raar_beta'] = raar_beta
    params['ml_obj_regularisation'] = ml_obj_regularisation
    params['floating_intensity'] = floating_intensity
    params['orientation_round_robin'] = orientation_round_robin
    params['output_format'] = output_format
    params['multiscan_reuse_ptycho'] = multiscan_reuse_ptycho
    params['padding'] = padding
    params['use_direct_beam'] = use_direct_beam
    params['autocenter'] = autocenter
    params['movie'] = movie
    # =========================================
    # Dealing with Experimental Condition & Paths
    # =========================================
    if exp_condition or path_exp:
        params = set_params_exp(exp_condition, path_exp, params)
    # =========================================
    # Dealing with Center Logic (Extracted)
    # =========================================
    if center and path_exp:
        params = set_params_center(center, maxsize, path_exp['path_master'], params)
    # =========================================
    # Dealing with Load Logic (Extracted)
    # =========================================
    if loadID or os.path.isfile(path_load):
        params = set_params_load(loadID, path_load, params, load='probe')
    # =========================================
    # Update additional parameters from kwargs
    # =========================================
    for key in kwargs:
        if key not in params:
            print('#'*20 + f" Alert: Key '{key}' is not in the default parameters. " + '#'*20)
    params.update(kwargs)

    print("\nParameters:")
    print(f"You can call '{set_params.__name__}?' to get help\n")
    pprint(params)
    return params


def pynx_set_params(ws,params):
    ws.params.update(params)


def set_algorithm(
    algo_string= 'ML**100*DM**100',
    ap_string  = 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    dm_string  = 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    raar_string= 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    ml_string  = 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    ):

    ap = AP(**eval(f"dict({ap_string})"))
    dm = DM(**eval(f"dict({dm_string})"))
    raar = RAAR(**eval(f"dict({raar_string})"))
    ml = ML(**eval(f"dict({ml_string})"))

    op_dict = {'ML': ml, 'DM': dm, 'RAAR': raar, 'AP': ap}
    algorithm = eval(algo_string, op_dict)
    return algorithm


algo_normal = {
    'algo_string': 'ML**300*DM**300',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
}

algo_quick = {
    'algo_string': 'ML**100*DM**100',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
}

algo_maskProbe = {
    'algo_string': 'DM**10',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'dm_string'  : 'update_object=True, update_probe=False, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
}

algo_DM_only = {
    'algo_string': 'DM**100',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
}

algo_ML_only = {
    'algo_string': 'ML**100',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
}

algo_PC = {
    'algo_string': 'AP**20*ML**20',
    'ap_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20, update_pos=20',
    'dm_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'raar_string': 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20',
    'ml_string'  : 'update_object=True, update_probe=True, calc_llk=20, show_obj_probe=20, update_pos=20',
}


def pynx_run(
    scanID='9264',
    params={},
    algo_setting=algo_normal,
    path_save='', 
    title='ptycho_test',
    path_default=path_default,
    folder='', # find folder in path_default
    key='ptycho',
    ):

    print('#'*100)
    print(f'Run pynx for scan ID: {scanID}, as title: {title}')
    print('#'*100)

    if folder:
        path_default = os.path.join(path_default, folder)
    path_exp = get_exp_path(str(scanID),path=path_default, key=key)
    exp_condition = load_exp_condition(path_exp)
    params = set_params(exp_condition=exp_condition, path_exp=path_exp, **params)
    
    ws = PtychoRunnerScanTPS25A(params,int(scanID)) # the scanID is added

    ws.load_scan()
    ws.load_data()
    ws.load_data_post_process()

    ws.prepare_processing_unit()
    ws.center_crop_data()
    ws.prepare()

    ws.run()
    if params['algorithm']=='manual':
        ws.run_algorithm(f'nbprobe={params["nbprobe"]}')
        p = ws.p
        if isinstance(algo_setting, str):
            algo_map = {
                'normal': algo_normal,
                'quick': algo_quick,
                'mask probe': algo_maskProbe,
                'DM only': algo_DM_only,
                'ML only': algo_ML_only,
                'PC': algo_PC
            }
            algo_setting = algo_map.get(algo_setting, algo_normal)
        algorithm = set_algorithm(**algo_setting)
        print('\nThe algorithm is set to:')
        pprint(algo_setting)
        p = algorithm * p

    if path_save == True:
        today = path_exp['path_master'].split('/')[-1].split('_')[0][:]
        path_save = path_exp['path_today'] + 'Results/'
        for n in range(1,999999):
            title = f'{n:06d}'
            success = pynx_save(ws,path_save=path_save,suffix=title,prefix=today)
            if success != 'Exists':
                break
    elif path_save:
        pynx_save(ws,path_save=path_save,suffix=title)

    return ws


def pynx_start(scanID=None, folder='', params={}, path_default=path_default):
    params = set_params(**params)
    if folder:
        path_default = os.path.join(path_default, folder)
    if scanID is not None:
        path_exp = get_exp_path(str(scanID),path=path_default)
        exp_condition = load_exp_condition(path_exp)
        params = set_params(exp_condition=exp_condition, path_exp=path_exp, **params)
    
    ws = PtychoRunnerScanTPS25A(params, params.get('scan_id', int(scanID)))
    ws.load_scan()
    ws.load_data()
    ws.load_data_post_process()
    ws.prepare_processing_unit()
    ws.center_crop_data()
    ws.prepare()
    ws.run()
    if params.get('nbprobe') is not None:
        nbprobe = params['nbprobe']
    else:
        nbprobe = 1
    ws.run_algorithm(f'nbprobe={nbprobe}')
    return ws


def pynx_algo_run(ws, algo_setting='normal'):
    if algo_setting == 'normal':
        algo_setting = algo_normal
    elif algo_setting == 'quick':
        algo_setting = algo_quick
    elif algo_setting == 'mask probe':
        algo_setting = algo_maskProbe
    elif algo_setting == 'DM only':
        algo_setting = algo_DM_only
    elif algo_setting == 'ML only':
        algo_setting = algo_ML_only
    elif algo_setting == 'PC':
        algo_setting = algo_PC

    p = ws.p
    algorithm = set_algorithm(**algo_setting)
    print('\nThe algorithm is set to:')
    pprint(algo_setting)
    p = algorithm * p


def pynx_save(ws, path_save=None, suffix='test', prefix=None, overwrite=False, remove_obj_phase_ramp=True, **kwargs):
    """
    Saves the PyNX reconstruction results (object and probe) to a CXI file.

    Args:
        ws: The PtychoRunnerScanTPS25A object containing the reconstruction results.
        path_save: The directory or full path where the CXI file will be saved.
        suffix: A title suffix for the filename.
        prefix: A string to prefix the filename.
        overwrite: If True, overwrites existing files.
        remove_obj_phase_ramp: If True, removes the phase ramp from the object before saving.
        **kwargs: Additional arguments.
    """
    if path_save == True:
        pynx_save_default(ws)
        return
    from pathlib import Path
    if path_save is None:
        latest_folder_path = get_latest_path(path_default)
        if latest_folder_path:
            if prefix is None:
                prefix = latest_folder_path.name
            path_save = latest_folder_path / "Results"
        else:
            path_save = Path(path_default) / "Results"

    path_save = Path(path_save)
    if path_save.suffix: # If path_save is a file
        final_path = path_save
    else:
        prefix_str = f"{prefix}_" if prefix else ""
        suffix_str = f"_{suffix}" if suffix else ""
        default_filename = f"{prefix_str}ptycho_{ws.scan}_pynx_result{suffix_str}.cxi"
        final_path = path_save / default_filename
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and final_path.exists():
        print(f'\nFile {final_path} already exists. Skipping save.')
        return 'Exists'
    print('\n'+'*'*100)
    print(f'Saving pynx results in:')
    print(f'{final_path}')
    print('*'*100+'\n')
    ws.p.save_obj_probe_cxi(str(final_path), remove_obj_phase_ramp=remove_obj_phase_ramp, **kwargs)


def pynx_save_default(ws):
    today = ws.params['scanfile'].split('/')[-1].split('_')[0][:]
    path_save = '/'.join(ws.params['scanfile'].split('/')[:-1])+'/' + 'Results/'
    for n in range(1,999999):
        title = f'{n:06d}'
        success = pynx_save(ws,path_save=path_save,suffix=title,prefix=today)
        if success != 'Exists':
            break


#===========================================================================
#                              Plotting                                     
#===========================================================================
def pynx_plot_overview(ws_p, path_save='', path_default=path_default, figsize=(10, 8), dpi=100):
    """
    Plots reconstruction overview (Object & Probe).
    Auto-detects input: PyNX object (uses operator) or file path (manual plot).
    """
    import matplotlib.patches as patches
    from matplotlib import gridspec
    # --- Branch 1: PyNX Object (Live Memory) ---
    if hasattr(ws_p, 'get_probe') or hasattr(ws_p, 'p') or hasattr(ws_p, 'get_obj'):
        # Extract ptycho object
        p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
        scan_lbl = getattr(ws_p, 'scan', getattr(ws_p, 'scanID', ''))
        fig = plt.figure(figsize=figsize, dpi=dpi, label=f'Scan ID:{scan_lbl}')
        p = ShowObjProbe() * p
        if path_save:
            fig.savefig(path_save, bbox_inches='tight', dpi=dpi)
            print(f"[Info] Saved via ShowObjProbe: {path_save}")
        plt.gca().invert_yaxis()
        plt.show()
        return
    # --- Branch 2: File Path / Keyword (Manual Load) ---
    else:
        # 1. Resolve File Path
        target_file = ws_p
        if isinstance(ws_p, list) or (isinstance(ws_p, str) and not os.path.exists(ws_p)):
            keywords = [ws_p] if isinstance(ws_p, str) else ws_p
            found = find_files(keywords + ['cxi'], path=path_default, find_all=True)
            target_file = sorted(found)[-1] if isinstance(found, list) else found
            print(f"[Info] Loading: {target_file}")
        scan_id = extract_scan_id(target_file)
        # 2. Load HDF5 Data
        with h5py.File(target_file, 'r') as f:
            obj = f['entry_last/object/data'][()]
            probe = f['entry_last/probe/data'][()]
            mask = f['entry_last/object/mask'][()]
            coord = (f['entry_last/object/col_coords'][()],f['entry_last/object/row_coords'][()])

        # 3. Data Prep (Ensure 3D shape: mode, y, x)
        if obj.ndim == 2: obj = obj[np.newaxis, ...]
        if probe.ndim == 2: probe = probe[np.newaxis, ...]
        
        obj_m, probe_m = obj[0], probe[0] # Use primary mode
        ny, nx = obj_m.shape
        npy, npx = probe_m.shape

        # 4. Calc Extents (um)
        # Object: Map real coords
        x_map, y_map = coord[0] , coord[1] 
        ext_obj = [x_map.min(), x_map.max(), y_map.min(), y_map.max()]
        
        # Probe: Calc pixel size from object coords
        px_sz = (coord[0].max() - coord[0].min()) / (nx - 1) 
        ext_prb = [-npx * px_sz / 2, npx * px_sz / 2, -npy * px_sz / 2, npy * px_sz / 2]

        # 5. Contrast (Mask based)
        vmin, vmax = None, None
        if mask is not None:
            vals = np.abs(obj_m)[mask > 0]
            if vals.size > 0: vmin, vmax = np.percentile(vals, [0.5, 99.5])

        # 6. Plotting
        fig = plt.figure(figsize=figsize, dpi=dpi, label=f'Scan ID:{scan_id}')
        gs = gridspec.GridSpec(2, 2, height_ratios=[obj.shape[-1],probe.shape[-1]], width_ratios=[1,1])
        
        # Config: (Data, Extent, Title, Cmap, Vlims, IsObj)
        plots = [
            (np.abs(obj_m),     ext_obj, 'Obj Mod',   'gray',    (vmin, vmax),    True),
            (np.angle(obj_m),   ext_obj, 'Obj Phase', 'gray',     (-np.pi, np.pi), True),
            (np.abs(probe_m),   ext_prb, 'Prb Mod',   'gray', (None, None),    False),
            (np.angle(probe_m), ext_prb, 'Prb Phase', 'hsv',     (-np.pi, np.pi), False)
        ]

        for i, (data, ext, tit, cmap, vlim, is_obj) in enumerate(plots):
            ax = fig.add_subplot(gs[i])
            im = ax.imshow(data, extent=ext, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            ax.invert_yaxis()

            # Overlays
            if is_obj and mask is not None:
                # Scan boundary contour
                ax.contour(x_map, y_map, mask, levels=[0.5], colors='k', linewidths=1, alpha=0.8)
            elif not is_obj:
                # Center crosshair for probe
                ax.axhline(0, c='w', lw=0.5, ls='--', alpha=0.5)
                ax.axvline(0, c='w', lw=0.5, ls='--', alpha=0.5)

            ax.set_title(tit, fontweight='bold')
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            
            if i >= 2: ax.set_xlabel('x (µm)')
            if i % 2 == 0: ax.set_ylabel('y (µm)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        if path_save:
            plt.savefig(path_save, dpi=dpi, bbox_inches='tight')
            print(f"[Info] Saved: {path_save}")
        plt.show()


def pynx_plot_obj(ws_p, path_save='', path_default=path_default, figsize=(4, 4), zoom=True, mode='original',dpi=100):
    """
    Plots object reconstruction independent of 'pos', using 'coord' and 'mask' instead.
    """
    import matplotlib.patches as patches
    import matplotlib.patheffects as path_effects
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from pynx.utils.plot_utils import complex2rgbalin, complex2rgbalin_dark, complex2rgbalog
    # --- 1. Data Extraction ---
    if hasattr(ws_p, 'get_probe') or hasattr(ws_p, 'p') or hasattr(ws_p, 'get_obj'):
        p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
        obj = p.get_obj()
        mask = p.get_scan_area_obj() 
        coord = p.get_obj_coord() 
        scan_id = getattr(ws_p, 'scan', getattr(ws_p, 'scanID', ''))
    else:
        target_file = ws_p
        if not (isinstance(ws_p, str) and os.path.exists(ws_p)):
            ws_p = [ws_p] if not isinstance(ws_p, list) else ws_p
            found = find_files(ws_p + ['cxi'], path=path_default, find_all=True) # 需自行定義 find_files
            target_file = sorted(found)[-1] if isinstance(found, list) else found
            print(f"[Info] Loading: {target_file}")
        scan_id = extract_scan_id(target_file)
        with h5py.File(target_file, 'r') as f:
            obj = f['entry_last/object/data'][()][...,::-1,:] # Since PyNX saving Top origin 
            mask = f['entry_last/object/mask'][()]
            coord = (f['entry_last/object/col_coords'][()]/1e6, # Saved as um
                     f['entry_last/object/row_coords'][()]/1e6)

    if obj.ndim == 2:
        obj = obj[np.newaxis, ...]
    
    n_modes, ny, nx = obj.shape

    # --- 2. Coordinate Alignment (1D Arrays) ---
    x_map = coord[0] * 1e6 # Ploted with um
    y_map = coord[1] * 1e6
    
    ext_obj = [x_map.min(), x_map.max(), y_map.min(), y_map.max()]

    # --- 3. Zoom Logic (Fix for 1D coords) ---
    xlims, ylims = (ext_obj[0], ext_obj[1]), (ext_obj[2], ext_obj[3])
    
    if zoom and mask is not None:
        rows, cols = np.where(mask > 0)
        
        if len(rows) > 0:
            valid_x = x_map[cols]
            valid_y = y_map[rows]
            
            min_x, max_x = valid_x.min(), valid_x.max()
            min_y, max_y = valid_y.min(), valid_y.max()
            
            pad_x = (max_x - min_x) * 0.05
            pad_y = (max_y - min_y) * 0.05
            
            xlims = (min_x - pad_x, max_x + pad_x)
            ylims = (min_y - pad_y, max_y + pad_y)

    # --- 4. Plotting Setup ---
    if mode == 'original':
        n_cols = n_modes * 2
    elif mode.startswith('coloring'):
        n_cols = n_modes
        
    total_width = figsize[0] * n_cols
    total_height = figsize[1]
    
    fig, axes = plt.subplots(1, n_cols, figsize=(total_width, total_height),
                            dpi=dpi,label=f'Scan ID:{scan_id}')
    plt.suptitle(f'Scan ID:{scan_id}', fontweight='bold')
    if n_cols == 1: axes = [axes]

    # Global Contrast
    obj_abs = np.abs(obj)
    vmin, vmax = None, None
    if mask is not None:
        valid_px = obj_abs[0][mask > 0]
        if len(valid_px) > 0:
            vmin, vmax = np.percentile(valid_px, [0.5, 99.5])

    obj_phase = np.angle(obj)
    vmin_phase, vmax_phase = None, None
    if mask is not None:
        valid_px = obj_phase[0][mask > 0]
        if len(valid_px) > 0:
            vmin_phase, vmax_phase = np.percentile(valid_px, [2, 98])

    # --- 5. Plotting Loop ---
    for m in range(n_modes):
        
        target_axes = []

        # === MODE: ORIGINAL ===
        if mode == 'original':
            # Amp
            ax_amp = axes[m]
            im_amp = ax_amp.imshow(np.abs(obj[m]), extent=ext_obj, cmap='gray', 
                                   vmin=vmin, vmax=vmax,origin='lower')
            ax_amp.set_title(f'Obj Modulus (Mode {m})', fontweight='bold')
            plt.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)
            
            # Phase
            ax_phs = axes[n_modes + m]
            im_phs = ax_phs.imshow(np.angle(obj[m]), extent=ext_obj, cmap='gray', 
                                   vmin=vmin_phase, vmax=vmax_phase,origin='lower')
            ax_phs.set_title(f'Obj Phase (Mode {m})', fontweight='bold')
            plt.colorbar(im_phs, ax=ax_phs, fraction=0.046, pad=0.04)
            
            target_axes = [ax_amp, ax_phs]

        # === MODE: COLORING ===
        elif mode.startswith('coloring'):
            ax = axes[m]
            if 'log' in mode:
                rgba_img = complex2rgbalog(obj[m])
            elif 'dark' in mode:
                rgba_img = complex2rgbalin_dark(obj[m], percentile=(0.5, 99.5))
            else:
                rgba_img = complex2rgbalin(obj[m], percentile=(0.5, 99.5))

            ax.imshow(rgba_img, extent=ext_obj, origin='lower')

            title_mode = "Linear"
            if 'log' in mode: title_mode = "Log"
            if 'dark' in mode: title_mode = "Dark"
            ax.set_title(f'Complex Obj ({title_mode}, Mode {m})', fontweight='bold')
            
            # --- Inset Layout ---
            # 1. Amplitude Bar
            cax = ax.inset_axes([1.02, 0.35, 0.05, 0.65]) 
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap_cbar = 'gray' 
            
            sm = ScalarMappable(norm=norm, cmap=cmap_cbar)
            sm.set_array([])
            cbar = plt.colorbar(sm, cax=cax)
            cbar.set_label('Amplitude', rotation=270, labelpad=10, fontsize=9)
            
            # 2. Color Wheel
            wax = ax.inset_axes([1.02, 0.0, 0.3, 0.35]) 
            wax.axis('off')
            
            # Wheel Generation
            xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
            wheel_h = np.arctan2(yy, xx)
            wheel_s = np.sqrt(xx**2 + yy**2)
            wheel_v = np.ones_like(wheel_s)
            wheel_v[wheel_s > 1] = 0 
            
            from matplotlib.colors import hsv_to_rgb
            wheel_h_norm = (wheel_h + np.pi) / (2 * np.pi)
            hsv_wheel = np.dstack((wheel_h_norm, wheel_s, wheel_v))
            rgb_wheel = hsv_to_rgb(hsv_wheel)
            alpha_wheel = np.ones_like(wheel_s)
            alpha_wheel[wheel_s > 1] = 0
            rgba_wheel = np.dstack((rgb_wheel, alpha_wheel))
            
            wax.imshow(rgba_wheel, aspect='equal', extent=[-1, 1, -1, 1])
            
            # Phase Labels
            def add_wheel_label(x, y, txt, ha, va):
                t = wax.text(x, y, txt, ha=ha, va=va, 
                             fontsize=12, fontweight='bold', color='black')
                t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

            r_txt = 0.8
            add_wheel_label(r_txt, 0, r'$0$', 'center', 'center')
            add_wheel_label(0, r_txt, r'$\pi/2$', 'center', 'center')
            add_wheel_label(-r_txt, 0, r'$\pi$', 'center', 'center')
            add_wheel_label(0, -r_txt, r'$-\pi/2$', 'center', 'center')

            target_axes = [ax]

        # === Common Overlays ===
        for ax in target_axes:
            if mask is not None:
                ax.contour(x_map, y_map, mask, levels=[0.5], 
                           colors='red', linewidths=1.5, alpha=0.8)

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.set_xlabel('x (µm)')
            if ax == axes[0]: 
                ax.set_ylabel('y (µm)')
            else:
                ax.set_yticks([]) 

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save, dpi=dpi, bbox_inches='tight')
        print(f"Saved Object to: {path_save}")
    plt.show()


def pynx_plot_probe(ws_p, path_save='', path_default=path_default, figsize=(4, 4), mode='original', dpi=100):
    """
    Args:
        ws_p: PyNX object, existing filepath, or search keywords (str/list).
        mode: 'original', 'coloring_lin', 'coloring_dark', 'coloring_log'
    """
    from pynx.utils.plot_utils import complex2rgbalin, complex2rgbalin_dark, complex2rgbalog
    # --- 1. Data Extraction ---
    if hasattr(ws_p, 'get_probe') or hasattr(ws_p, 'p'):
        p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
        probe_final = p.get_probe()
        probe_coords = np.array(p.get_probe_coord())
        scan_id = getattr(ws_p, 'scan', getattr(ws_p, 'scanID', ''))
        
    else:
        if not (isinstance(ws_p, str) and os.path.exists(ws_p)):
            ws_p = [ws_p] if not isinstance(ws_p, list) else ws_p
            found = find_files(ws_p + ['cxi'], path=path_default, find_all=True)
            ws_p = sorted(found)[-1] if isinstance(found, list) else found
            print(f"[Info] Loading: {ws_p}")
        scan_id = extract_scan_id(ws_p)
        with h5py.File(ws_p, 'r') as f:
            probe_final = f['entry_last/probe/data'][()] 
            probe_coords = f['entry_last/probe/col_coords'][()]
            probe_coords = np.tile(probe_coords, [2, 1])

    # --- 2. Data Preparation ---
    total_intensity = np.sum(np.abs(probe_final))
    probe_intensity = np.sum(np.abs(probe_final), axis=(-2, -1)) / total_intensity
    n_modes = probe_final.shape[0]

    # Coordinate Alignment
    try:
        xc, yc = probe_coords[0].flatten(), probe_coords[1].flatten()
        extent = [float(xc[0]), float(xc[-1]), float(yc[0]), float(yc[-1])]
    except:
        ny, nx = probe_final.shape[-2:]
        extent = [0, nx, 0, ny]

    # --- 3. Plotting Setup ---
    n_rows, n_cols = (2, n_modes) if mode == 'original' else (1, n_modes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols, figsize[1]*n_rows), dpi=dpi, label=f'Scan ID:{scan_id}')
    
    # Reshape axes for consistent indexing [row, col]
    if n_modes == 1: axes = np.array([[axes[0]], [axes[1]]]) if n_rows == 2 else np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :] 
    
    plt.rcParams['font.size'] = 10
    # --- 4. Plotting Loop ---
    for m in range(n_modes):
        ratio = probe_intensity[m] * 100
        data = probe_final[m]
        
        if mode == 'original':
            # Amp (Top)
            ax0 = axes[0, m]
            im0 = ax0.imshow(np.abs(data), extent=extent, cmap='viridis',origin='lower')
            ax0.set_title(f'Probe Mode {m+1}\n({ratio:.2f}%)', fontweight='bold')
            plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
            # Phase (Bottom)
            ax1 = axes[1, m]
            im1 = ax1.imshow(np.angle(data), extent=extent, cmap='gray',origin='lower')
            ax1.set_title(f'Phase Mode {m+1}', fontweight='bold')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            target_axes = [ax0, ax1]
        else: # Coloring Modes
            ax = axes[0, m]
            if mode == 'coloring_lin':
                rgb = complex2rgbalin(data, percentile=(0.5, 99.5), type='float')
                name = "Complex (Lin)"
            elif mode == 'coloring_dark':
                rgb = complex2rgbalin_dark(data, percentile=(0.5, 99.5))
                name = "Complex (Dark)"
            elif mode == 'coloring_log':
                rgb = complex2rgbalog(data, dlogs=2, type='float')
                name = "Complex (Log)"
            else: raise ValueError(f"Unknown mode: {mode}")

            ax.imshow(rgb, extent=extent,origin='lower')
            ax.set_title(f'{name} Mode {m+1}\n({ratio:.2f}%)', fontweight='bold')
            target_axes = [ax]
        # Clean Axis Labels
        for ax in target_axes:
            ax.set_ylabel('y [um]' if m == 0 else '')
            is_bottom = (mode == 'original' and ax == axes[1, m]) or (mode != 'original')
            ax.set_xlabel('x [um]' if is_bottom else '')

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save, dpi=dpi)
        print(f"Saved Probe to: {path_save}")
    plt.show()


def pynx_probe_propagate(ws_p,use_probe=0,p_linspace=[-1000e-6,1000e-6,200],interact=False,mode='original'):
    p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
    p_range = np.linspace(*p_linspace)
    probe = p.get_probe()[use_probe]
    pixel_size = p.pixel_size_object
    wavelength = p.data.wavelength
    
    probe_z,z_coord,_,_ = probe_propagate(probe,p_range,pixel_size,wavelength)

    if interact:
        interactive_probes(probe_z, z_coord, mode=mode, pixel_size=pixel_size)


def interactive_probes(probe_z, z_coord, mode='original', pixel_size=None):
    """
    互動式顯示 Probe 傳遞 (含左右微調按鈕與物理單位顯示)。
    
    Args:
        probe_z: (nz, ny, nx) complex array
        z_coord: (nz,) 1D array (單位: meter)
        mode: 'original', 'coloring_lin', 'coloring_dark', 'coloring_log'
        pixel_size: float (單位: meter)，用於計算 XY 軸真實物理尺寸
    """
    from matplotlib.widgets import Slider, Button
    from pynx.utils.plot_utils import complex2rgbalin, complex2rgbalin_dark, complex2rgbalog
    
    nz, ny, nx = probe_z.shape
    init_idx = nz // 2
    
    # 單位轉換: Meter -> Microns
    z_um = z_coord * 1e6
    
    # === 計算 XY 軸的物理範圍 (單位: µm) ===
    if pixel_size is not None:
        px_um = pixel_size * 1e6
        # 將原點設在中心
        extent = [-nx * px_um / 2, nx * px_um / 2, -ny * px_um / 2, ny * px_um / 2]
        xy_label = 'µm'
    else:
        extent = None
        xy_label = 'pixels'
    
    # 建立畫布
    if mode == 'original':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # 預留底部空間給 Slider 和 Buttons
        plt.subplots_adjust(bottom=0.25) 
        
        abs_max = np.max(np.abs(probe_z))
        
        im1 = ax1.imshow(np.abs(probe_z[init_idx]), cmap='inferno', vmin=0, vmax=abs_max, extent=extent)
        ax1.invert_yaxis()
        ax1.set_title(f"Amplitude")
        ax1.set_xlabel(f"x ({xy_label})")
        ax1.set_ylabel(f"y ({xy_label})")
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        im2 = ax2.imshow(np.angle(probe_z[init_idx]), cmap='hsv', vmin=-np.pi, vmax=np.pi, extent=extent)
        ax2.invert_yaxis()
        ax2.set_title(f"Phase")
        ax2.set_xlabel(f"x ({xy_label})")
        ax2.set_ylabel(f"y ({xy_label})")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    elif mode.startswith('coloring'):
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        plt.subplots_adjust(bottom=0.25)
        
        if 'log' in mode:
            converter = lambda x: complex2rgbalog(x)
            title_mode = "Log"
        elif 'dark' in mode:
            converter = lambda x: complex2rgbalin_dark(x, percentile=(0.5, 99.5))
            title_mode = "Dark"
        else:
            converter = lambda x: complex2rgbalin(x, percentile=(0.5, 99.5))
            title_mode = "Linear"
            
        im1 = ax1.imshow(converter(probe_z[init_idx]), extent=extent)
        ax1.invert_yaxis()
        ax1.set_title(f"Complex Probe ({title_mode})")
        ax1.set_xlabel(f"x ({xy_label})")
        ax1.set_ylabel(f"y ({xy_label})")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- 1. 定義共用更新函數 ---
    def update_display(idx):
        """核心更新邏輯：負責畫圖與更新 Slider 文字"""
        idx = int(np.clip(idx, 0, nz-1)) # 確保不超出範圍
        z_val = z_um[idx]
        
        # A. 更新圖像
        if mode == 'original':
            # 取得當前切片的 Amplitude
            current_abs = np.abs(probe_z[idx])
            im1.set_data(current_abs)
            im2.set_data(np.angle(probe_z[idx]))
            
            # 動態重設 Colorbar 的範圍，適應當前切片的最大亮度
            im1.set_clim(vmin=0, vmax=np.max(current_abs))
            
            ax1.set_title(f"Amplitude\nZ = {z_val:.2f} µm")
            ax2.set_title(f"Phase\nZ = {z_val:.2f} µm")
        else:
            im1.set_data(converter(probe_z[idx]))
            ax1.set_title(f"Complex Probe ({title_mode})\nZ = {z_val:.2f} µm")

        # B. 更新 Slider 顯示文字
        slider.valtext.set_text(f"{z_val:.2f} µm")
        
        fig.canvas.draw_idle()

    # --- 2. 建立 Widgets ---
    ax_prev   = plt.axes([0.15, 0.1, 0.05, 0.04]) 
    ax_slider = plt.axes([0.25, 0.1, 0.50, 0.04], facecolor='lightgray')
    ax_next   = plt.axes([0.80, 0.1, 0.05, 0.04]) 

    # Slider: 範圍 0 ~ nz-1 (整數步進)
    slider = Slider(
        ax=ax_slider,
        label='Z Pos ', # 標籤
        valmin=0,
        valmax=nz-1,
        valinit=init_idx,
        valstep=1
    )
    # 初始化 Slider 文字為目前的 um 值
    slider.valtext.set_text(f"{z_um[init_idx]:.2f} µm")

    # 按鈕
    btn_prev = Button(ax_prev, '<') # 左箭頭
    btn_next = Button(ax_next, '>') # 右箭頭

    # --- 3. 定義 Callbacks ---
    def on_slider_change(val):
        update_display(val)

    def on_prev(event):
        current_idx = slider.val
        new_idx = max(0, current_idx - 1)
        slider.set_val(new_idx)

    def on_next(event):
        current_idx = slider.val
        new_idx = min(nz-1, current_idx + 1)
        slider.set_val(new_idx)

    # 綁定事件
    slider.on_changed(on_slider_change)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    # --- 4. 綁定物件防止回收 (GC) ---
    fig._widgets_ref = [slider, btn_prev, btn_next]
    
    # 觸發一次更新以確保標題正確
    update_display(init_idx)
    
    plt.show()  


def plot_image(image,path_save='',**kwargs):
    """Quickly plot a 2D image with a colorbar and optional saving."""
    plt.figure()
    plt.imshow(image,**kwargs)
    plt.colorbar()
    if path_save:
        plt.savefig(path_save)
    plt.show()
#===========================================================================
#                             Manual Methods                                
#===========================================================================
def get_info_from_scanID(scanID='9264',path=path_default):

    print('#'*100)
    print(f'Get info for scan ID: {scanID}')
    print('#'*100)
    path_exp = get_exp_path(scanID,path=path)
    exp_condition = load_exp_condition(path_exp['path_master'], path_exp['path_primary'])
    params = set_params(exp_condition=exp_condition, path_exp=path_exp) 
    
    return path_exp, exp_condition, params


def get_all_data(scanID='9264',path=path_default,center='center',maxsize=1500,position_key='cisamf'):
    """
    Get all experiment data for calculation.
    """
    path_exp, exp_condition, params = get_info_from_scanID(scanID,path=path)
    data = get_data(path_exp['path_data'],center=center,maxsize=maxsize,path_master=path_exp['path_master'])
    mask = get_mask(path_exp['path_master'])
    scan_x,scan_z = get_scan(path_exp['path_primary'],position_key=position_key)

    pixelsize_detector = params['pixelsize']
    detectordistance = params['detectordistance']
    nrj = params['nrj']
    return data, mask, scan_x, scan_z, pixelsize_detector, detectordistance, nrj, (path_exp, exp_condition, params)


def get_data(path_data,center=None,maxsize=800,path_master=None):
    """
    Reads diffraction data from HDF5 files and applies optional centering and cropping.

    Parameters:
    -----------
    path_data : str or list
        Path(s) to the HDF5 files containing the diffraction patterns.
    center : tuple, list, str, or None
        Determines how the data is centered/cropped:
        - None: Returns the full data without cropping.
        - (cx, cy): Crops around the specified (x, y) coordinates.
        - 'auto': Calculates center of mass using a mask from the master file.
        - 'info': Retrieves beam center from the master file metadata.
    maxsize : int
        The side length of the square crop to be applied.
    path_master : str, optional
        Path to the master file, necessary when center is 'auto' or 'info'.

    Returns:
    --------
    list of np.ndarray
        A list of numpy arrays containing the processed diffraction data.
    """
    if isinstance(path_data, (str, bytes)): 
        path_data = [path_data]
    path_data.sort()

    print('Reading data with this order:')
    pprint(path_data)

    data_list = []
    for p in path_data:
        with h5py.File(p, 'r') as f:
            if center is None:
                data = f['entry/data/data'][()]
                cx,cy = None,None
            elif isinstance(center, (list, tuple, np.ndarray)) and len(center) == 2:
                data = f['entry/data/data'][()]
                cx, cy = center
                data = center_crop(data, cy, cx, maxsize)
            else:
                if path_master is None:
                    raise ValueError("Master file path is required for 'auto' center option to read beam center from master file.")
                elif center == 'auto':
                    mask = get_mask(path_master)
                    data = f['entry/data/data'][()]
                    cy, cx = center_of_mass(data,mask)
                    data = center_crop(data, int(round(cy)), int(round(cx)),maxsize)
                elif center == 'exp_config':
                    data = f['entry/data/data'][()]
                    cx, cy = center_from_info(path_master)
                    data = center_crop(data, int(round(cy)), int(round(cx)),maxsize)
                else:
                    raise ValueError("Invalid center value. Must be None, 'auto', 'info', or (cx, cy).")
            data_list.append(data)
    data = np.vstack(data_list)
    print(f'The center is set to: {([cx,cy])}, with final data shape: {data.shape}')
    print(f'The data is: {data.nbytes/1024/1024:.2f} MB')
    return data


def get_mask(path_master):
    """Reads pixel mask from master file into exp_data['mask']."""
    with h5py.File(path_master, 'r') as f:
        mask = f['entry/instrument/detector/detectorSpecific/pixel_mask'][()].astype(bool)
    return mask


def get_scan(path_primary, position_key='cisamf'):
    """Reads scan positions from primary CSV into exp_data['x'] and exp_data['z']."""
    if path_primary is None or not os.path.exists(path_primary):
        raise ValueError(f"Primary file path is missing or invalid: {path_primary}")
    data_primary = pd.read_csv(path_primary)
    
    # Ensure position_key is iterable for the 'all' check
    keys = [position_key] if isinstance(position_key, str) else position_key
    
    col_x = [c for c in data_primary.columns if all(k in c for k in keys) and 'x' in c]
    col_z = [c for c in data_primary.columns if all(k in c for k in keys) and 'z' in c]
    
    if col_x and col_z:
        scan_x = data_primary[col_x[0]]
        scan_z = data_primary[col_z[0]]
        print(f"Using columns: x -> {col_x[0]}, z -> {col_z[0]}")
        return scan_x, scan_z
    else:
        raise ValueError(f"Warning: Could not find columns matching {position_key}.")


def center_of_mass(data, mask=None):
    """Calculates the center of mass of the summed data."""
    from scipy.ndimage import center_of_mass as CoM
    img = np.sum(data, axis=0)
    if mask is not None:
        img = img * (~mask if mask.ndim == 2 else ~np.all(mask, axis=0))
    return map(int, map(round, CoM(img)))


def center_from_info(path_master):
    """Reads beam center from master file into cx and cy."""
    with h5py.File(path_master, 'r') as f:
        cx = f['entry/instrument/detector/beam_center_x'][()]
        cy = f['entry/instrument/detector/beam_center_y'][()]
    return cy, cx


def center_crop(data, cy=None, cx=None, maxsize=800):
    """Crops data to maxsize x maxsize centered on (cy, cx)."""
    h, w = data.shape[-2:]
    if cy is None or cx is None:
        cy, cx = h//2, w//2
    y0 = int(max(0, min(cy - maxsize // 2, h - maxsize)))
    x0 = int(max(0, min(cx - maxsize // 2, w - maxsize)))
    return data[..., y0:y0 + int(maxsize), x0:x0 + int(maxsize)]


def pynx_set_data(data,mask,scan_x,scan_z,pixelsize,detectordistance,wavelength=None,nrj=None,obj=None,probe=None,nbprobe=3):
    ws = PtychoRunnerScanTPS25A(default_params,0) # the scanID is not used
    ws.params['pixelsize'] = pixelsize
    ws.params['detectordistance'] = detectordistance
    if wavelength is None and nrj is None:
        raise ValueError("Either wavelength or nrj must be provided.")
    ws.params['nrj'] = nrj if nrj is not None else 12.3984 / (wavelength * 1e10)
    ws.params['padding'] = 0
    ws.params['orientation_round_robin'] = False
    ws.params['object'] = obj if isinstance(obj, str) else 'random,0.8,1,0,0.5'
    ws.params['probe'] = probe if isinstance(probe, str) else 'disc,800e-9'
    ws.params['fig_num'] = 100
    ws.params['roi'] = 'full'
    ws.params['algorithm'] = 'manual'
    if np.all(mask):raise ValueError("""Mask is all True will make operator error! 
                                    If for testing you can give only pixel without masked. ex: mask[0,0]=0
                                    """)
    ws.iobs = np.where(mask,-1,data)
    ws.dsize = data.shape[-2]
    ws.wavelength = wavelength
    ws.x = np.array(scan_x)*1e-6 # Assuming um input
    ws.y = np.array(scan_z)*1e-6 # Assuming um input
    ws.imgn = len(scan_x)
    ws.prepare_processing_unit()
    ws.prepare()
    ws.run()
    ws.run_algorithm(f'nbprobe={nbprobe}')
    if isinstance(obj, (np.ndarray, list)):
        ws.p.set_obj(obj)
    if isinstance(probe, (np.ndarray, list)):
        ws.p.set_probe(probe)
    return ws


#===========================================================================
#                      Pynx: initialization/inheritance                                  
#===========================================================================
def pynx_set_objprobe(ws_p,obj=None,probe=None):
    p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
    if obj is not None:
        p.set_obj(obj)
    if probe is not None:
        p.set_probe(probe)


def pynx_get_objprobe(ws_p, path_default=path_default):
    if hasattr(ws_p, 'get_probe') or hasattr(ws_p, 'p') or hasattr(ws_p, 'get_obj'):
        p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
        obj = p.get_obj()
        probe = p.get_probe()
    else:
        target_file = ws_p
        if not (isinstance(ws_p, str) and os.path.exists(ws_p)):
            ws_p = [ws_p] if not isinstance(ws_p, list) else ws_p
            found = find_files(ws_p + ['cxi'], path=path_default, find_all=True)
            target_file = sorted(found)[0] if isinstance(found, list) else found
            print(f"[Info] Loading: {target_file}")
        with h5py.File(target_file, 'r') as f:
            obj = f['entry_last//object/data'][()] 
            probe = f['entry_last//probe/data'][()]
    return obj,probe


def pynx_inherit_1st_mode(ws_p,init_obj=True):
    """
    Keeps the first probe mode and re-initializes other modes with small random noise.

    The additional modes are initialized by scaling the magnitude of the first mode
    by 1% and applying random uniform values to both the amplitude and the phase.
    """
    p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
    probe_ = p.get_probe().copy()
    probe_[1:]=np.tile(np.abs(probe_[0]),[probe_.shape[0]-1,1,1])*np.random.uniform(size=probe_[1:].shape)*0.01
    probe_[1:]=probe_[1:]*np.exp(1j*np.random.uniform(size=probe_[1:].shape))
    p.set_probe(probe_)
    if init_obj:
        obj_ = ws_p.obj0.copy() # Must use ws, not p
        p.set_obj(obj_[None,:,:]) # obj0 is 2D, p expects 3D


def pynx_probe_masking(ws_p):
    p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
    probe_ = p.get_probe().copy()
    obj_area = p.get_scan_area_obj()
    probe_area = center_crop(obj_area,maxsize=probe_.shape[-1])
    probe_[:,~probe_area]=0
    p.set_probe(probe_)


def pynx_obj_masking(ws_p):
    p = ws_p.p if hasattr(ws_p, 'p') and ws_p.p is not None else ws_p
    obj_ = p.get_obj().copy()
    obj_area = p.get_scan_area_obj()
    obj_[:,~obj_area]=0
    p.set_obj(obj_)


def pynx_init(ws,init='both',nbprobe=None,nbobj=None):
    """
    probe0 and obj0 are probe and object before all algorithm, including nbprobe=3
    """
    probe_ = ws.p.get_probe().copy()
    obj_ = ws.p.get_obj().copy()

    ws.prepare()
    probe0 = ws.probe0.copy()
    obj0 = ws.obj0.copy()
    nbprobe = ws.p.get_probe().shape[0] if nbprobe is None else nbprobe
    nbobj = ws.p.get_obj().shape[0] if nbobj is None else nbobj
    pynx_set_objprobe(ws, obj0, probe0)
    params = ws.params.copy()
    ws.params['algorithm'] = 'manual'
    ws.defocus_done = False
    ws.run()
    ws.run_algorithm(f'nbprobe={nbprobe},nbobj={nbobj}')
    ws.params = params
    if init == 'probe': # set obj back
        pynx_set_objprobe(ws, obj_, None)
    elif init == 'obj': # set probe back
        pynx_set_objprobe(ws, None, probe_)
    

def pynx_load(ws_p, path_cxi, path_default=path_default, load='probe'):
    """
    Loads the object and probe from a CXI file into the provided PyNX workspace or operator.
    Args:
        ws_p: PyNX workspace object (which contains the .p operator) or the operator itself.
        path_cxi (str or list): The full file path, a keyword string, or a list of keywords to search for.
        path_default (str): The default directory to search if a full path is not provided.
    Returns:
        ws_p: The updated workspace or operator.
    """
    if not (isinstance(path_cxi, str) and os.path.exists(path_cxi)):
        path_cxi = [path_cxi] if not isinstance(path_cxi, list) else path_cxi
        found = find_files(path_cxi + ['cxi'], path=path_default, find_all=True)
        path_cxi = sorted(found)[0] if isinstance(found, list) else found
        print(f"[Info] Loading: {path_cxi}")
    if load == 'probe':
        original_params= ws_p.params.copy()
        ws_p.params['loadprobe'] = path_cxi
        pynx_init(ws_p)
        ws_p.params = original_params
    elif load == 'mask':
        original_params = ws_p.params.copy()
        ws_p.params['loadmask'] = path_cxi
        pynx_init(ws_p)
        ws_p.params = original_params
    elif load == 'all':
        original_params = ws_p.params.copy()
        ws_p.params['load'] = path_cxi
        pynx_init(ws_p)
        ws_p.params = original_params
    else:
        raise ValueError("load must be 'probe' or 'mask' or 'all'")
    

#===========================================================================
#                           FSC Ploting                                 
#===========================================================================
def match(img1, img2):
    import numpy as np
    from skimage.registration import phase_cross_correlation
    # 1. Get integer pixel shifts (dy, dx)
    # Use the minimum overlapping dimensions to compute cross-correlation
    h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    shift, _, _ = phase_cross_correlation(img1[:h, :w], img2[:h, :w], upsample_factor=1)
    dy, dx = shift.astype(int)

    # 2. Core logic: Calculate slicing ranges for both images
    # For img1: if dy > 0, start from dy; if dy < 0, start from 0
    # For img2: apply the inverse offset relative to img1
    s1_y, s1_x = max(0, dy), max(0, dx)
    s2_y, s2_x = max(0, -dy), max(0, -dx)

    # 3. Determine the dimensions of the final common area
    out_h = h - abs(dy)
    out_w = w - abs(dx)

    # 4. Perform cropping in a single step
    res1 = img1[s1_y : s1_y + out_h, s1_x : s1_x + out_w]
    res2 = img2[s2_y : s2_y + out_h, s2_x : s2_x + out_w]

    return res1, res2


def flatten(img, interact=True):
    h, w = img.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    if interact:
        # --- Interactive Mode: Selection based ---
        print("\nClick at least 3 background points, then press Enter.")
        plt.imshow(img, cmap='viridis')
        points = plt.ginput(n=-1, timeout=0)
        plt.close()
        
        x_data = np.array([p[0] for p in points])
        y_data = np.array([p[1] for p in points])
        z_data = img[y_data.astype(int), x_data.astype(int)]
    else:
        # --- Automatic Mode: Global Fit ---
        x_data, y_data, z_data = X.ravel(), Y.ravel(), img.ravel()

    # Solve z = ax + by + c using Least Squares
    # Matrix A contains columns of X coordinates, Y coordinates, and ones
    A = np.column_stack((x_data, y_data, np.ones_like(x_data)))
    coeffs, _, _, _ = np.linalg.lstsq(A, z_data, rcond=None)
    a, b, c = coeffs

    # Subtract the fitted plane from the original image
    return img - (a * X + b * Y + c)


def fsc(img1,img2,r=1,pixel_size=None,p=0,method='gradient',interact=True,check=True,path_save=None):
    """
    Calculate the Fourier Shell Correlation (FSC) between two images.

    Args:
        img1 (np.ndarray): The first image.
        img2 (np.ndarray): The second image.
        method (str): The method to use for FSC calculation.
            - 'ramp': Remove a linear ramp from the images.
            - 'gradient': Remove the gradient from the images.
            - 'interactive': Remove a linear ramp interactively.

    Returns:
        float: The FSC value.
    """
    img1 = center_crop(img1,img1.shape[-1]//2,img1.shape[-2]//2,min(img1.shape[-2:])*r)
    img2 = center_crop(img2,img2.shape[-1]//2,img2.shape[-2]//2,min(img2.shape[-2:])*r)

    if method == 'ramp':
        img1 = remove_phase_ramp(img1)[0]
        img2 = remove_phase_ramp(img2)[0]
    elif method == 'gradient':
        img1 = minimize_grad_phase(img1)[0]
        img2 = minimize_grad_phase(img2)[0]
    elif method == 'interact':
        img1_ang, img1_abs = np.angle(img1), np.abs(img1)
        img2_ang, img2_abs = np.angle(img2), np.abs(img2)
        img1_ang = flatten(img1_ang, interact=interact)
        img2_ang = flatten(img2_ang, interact=interact)
        img1_abs = flatten(img1_abs, interact=interact)
        img2_abs = flatten(img2_abs, interact=interact)
        img1 = img1_abs * np.exp(1j * img1_ang)
        img2 = img2_abs * np.exp(1j * img2_ang)
    else:
        print(f'Warning: Flattening is skiped with method:{method}')

    img1 = np.abs(img1)*np.exp(1j * unwrap_phase(img1))
    img2 = np.abs(img2)*np.exp(1j * unwrap_phase(img2))

    img1, img2 = match(img1, img2)

    if check:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes[0, 0].imshow(np.abs(img1)); axes[0, 0].set_title('Img1 Amplitude')
        axes[0, 1].imshow(np.angle(img1)); axes[0, 1].set_title('Img1 Phase')
        axes[1, 0].imshow(np.abs(img2)); axes[1, 0].set_title('Img2 Amplitude')
        axes[1, 1].imshow(np.angle(img2)); axes[1, 1].set_title('Img2 Phase')
        plt.tight_layout()
        plt.savefig('./check.png',dpi=300,bbox_inches='tight')
        plt.show()
    
    # The parameters of FSCPlot, can be referenced from the pynx documentation
    FSC = FSCPlot(img1,img2,snrt=(0.2071,0.5),ring_thick= 3,rad_apod= 60, axial_apod = 20,pixel_size=pixel_size)
    FSC.plot()
    x_nyquist = FSC.d['f_nyquist']
    fsc_b1 = FSC.d['1 bit threshold']
    fsc_b0p5 = FSC.d['1/2 bit threshold']
    fsc_result = FSC.d['fsc']

    nm_axis    = (1 / x_nyquist) * FSC.pixel_size

    idx = np.where(fsc_b1 - fsc_result > 0)[0][p]
    i0, i1 = idx - 1, idx # intersection points

    nm_res, _ = find_intersection(
        [nm_axis[i0], fsc_b1[i0]], [nm_axis[i1], fsc_b1[i1]], 
        [nm_axis[i0], fsc_result[i0]], [nm_axis[i1], fsc_result[i1]]
    )

    px, py = find_intersection(
        [x_nyquist[i0], fsc_b1[i0]], [x_nyquist[i1], fsc_b1[i1]], 
        [x_nyquist[i0], fsc_result[i0]], [x_nyquist[i1], fsc_result[i1]]
    )

    plt.figure(dpi=300)
    plt.plot(x_nyquist,fsc_result,label='FSC')
    plt.plot(x_nyquist,fsc_b1,'--k',label='1 bit threshold')
    plt.plot(x_nyquist,fsc_b0p5,':k',label='1/2 bit threshold')
    plt.legend()
    plt.scatter(px,py,s=30, facecolors='none', edgecolors='r',zorder=3)
    plt.text(px,py,f'{np.real(nm_res)*1e9:.1f}',ha='right',va='top',fontsize=13)
    plt.xlabel('Spacial Frequency/Nyquist',fontsize=16)
    plt.ylabel('FRC Magnitude',fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    secax = plt.gca().secondary_xaxis('top')
    secax.set_xlabel('Resolution in nm',fontsize=16)
    secax.set_xticklabels(['']*2+[f'{value:.1f}' for value in FSC.d['nm']])
    secax.tick_params(labelsize = 14)
    plt.tight_layout()
    if path_save:
        plt.savefig(path_save,dpi=300,bbox_inches='tight')
    plt.show()

    return FSC

  
def find_intersection(p1, p2, p3, p4):
    """Intersections of line 1 (p1,p2) and line 2 (p3,p4)"""
    det = lambda a, b: a[0] * b[1] - a[1] * b[0]
    
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    div = det(xdiff, ydiff)
    
    if div == 0: raise ValueError("Lines do not intersect")

    d = (det(p1, p2), det(p3, p4))
    return det(d, xdiff) / div, det(d, ydiff) / div
    

#===========================================================================
#                               Stitching Images
#===========================================================================
def stitch_obj(ws_list, path_default=path_default, path_save='', dpi=100, show_labels=True, pos_source='baseline', search_keyword=None, title=None):
    """
    從 PyNX 提取 Phase 數據，進行距離權重平均融合 (Distance-weighted Blending)，
    並校正馬達座標。包含：有效區域 (Support Mask) 均值歸零與百分位極值截斷 (Percentile clipping)。
    
    參數:
    - show_labels: bool, 是否在最終圖片上標示各個 scanID 的中心位置 (預設為 True)
    - pos_source: str, 拼貼位置的座標來源，可選 'baseline' (預設) 或 'cxi'。
    - search_keyword: str, 搜尋檔案時的額外指定字串，用於多檔匹配時精確指定名稱。
    - title: str, 圖片的標題，若未提供則使用預設名稱。
    """
    import os
    import h5py
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import matplotlib.patheffects as path_effects

    print(f"--- 階段 1：預先讀取所有座標並進行校正 (來源: {pos_source}) ---")
    raw_x_list = []
    raw_z_list = []
    target_files = [] # 儲存已找到的檔案路徑，避免階段 2 重複搜尋
    
    for ws in ws_list:
        # 處理檔案搜尋與指定字串
        target_file = ws
        if isinstance(ws, list) or not os.path.exists(str(ws)):
            search_list = [str(ws)]
            if search_keyword:
                search_list.append(search_keyword)
            search_list.append('cxi') # 確保找的是 cxi 檔
            
            found = find_files(search_list, path=path_default, find_all=True)
            target_file = found[-1] if isinstance(found, list) else found
            
        target_files.append(target_file)

        # 根據參數選擇座標來源
        if pos_source == 'baseline':
            path_baseline = find_files([str(ws), 'baseline'], path=path_default, find_all=False)
            coord_df = pd.read_csv(path_baseline)
            
            x_col = [col for col in coord_df.columns if 'sam' in col and 'x' in col][0]
            z_col = [col for col in coord_df.columns if 'sam' in col and 'z' in col][0]
            
            raw_x_list.append(coord_df[x_col][0])
            raw_z_list.append(coord_df[z_col][0])
            
        elif pos_source == 'cxi':
            with h5py.File(target_file, 'r') as f:
                col_coords = f['entry_last/object/col_coords'][()]
                row_coords = f['entry_last/object/row_coords'][()]
                # 直接取平均，不改變原本的量級
                raw_x_list.append(np.mean(col_coords))
                raw_z_list.append(np.mean(row_coords))
        else:
            raise ValueError("pos_source 必須是 'baseline' 或 'cxi'")
        
    raw_x = np.array(raw_x_list)
    raw_z = np.array(raw_z_list)
    
    # 找到所有座標的幾何中心
    center_x = np.mean(raw_x)
    center_z = np.mean(raw_z)
    
    # 歸零 (平移到 0,0) 並反轉 X 軸 (左右交換)
    corrected_x = -(raw_x - center_x)
    corrected_z = -(raw_z - center_z)

    print("--- 階段 2：提取影像並準備全域融合 ---")
    phase_data_list = []
    mask_list = []
    extents_list = []
    pixel_sizes = []
    labels = []
    
    # 先把所有的數據與邊界算出來存好
    for i, ws in enumerate(ws_list):
        abs_x = corrected_x[i]
        abs_z = corrected_z[i]
        labels.append(str(ws))

        # 直接取用第一階段找好的檔案
        target_file = target_files[i]
            
        with h5py.File(target_file, 'r') as f:
            obj = f['entry_last/object/data'][()] 
            mask = f['entry_last/object/mask'][()]
            local_coord = (f['entry_last/object/col_coords'][()], f['entry_last/object/row_coords'][()])

        if obj.ndim == 2:
            obj = obj[np.newaxis, ...]
            
        # 這裡直接提取 Phase
        obj = minimize_grad_phase(obj[0])[0] 
        phase = np.angle(obj)
        
        # 上下反轉以配合 PyNX 座標
        phase = phase[::-1,:] 
        
        if mask is not None:
            # Phase 有反轉，mask 也必須跟著反轉，確保遮罩與影像對齊
            mask = mask[::-1,:]
            
        # ========================================================
        # 有效區域零均值化 (Mean Shift) 與 2%-98% 截斷
        # ========================================================
        if mask is not None:
            # 確認為 Support Mask 邏輯：> 0 才是有效訊號區
            is_valid = (mask > 0)
        else:
            is_valid = np.ones_like(phase, dtype=bool)

        if np.any(is_valid):
            # 1. 取得 valid 區域的數值
            valid_phase = phase[is_valid]
            
            # 2. 將 valid 區域的平均平移到 0 
            mean_val = np.mean(valid_phase)
            phase = phase - mean_val
            
            # 3. 計算平移後的 2% 與 98% 百分位數
            p_low, p_high = np.percentile(phase[is_valid], [2, 98])
            
            # 4. 進行 Clipping (極值截斷)
            phase = np.clip(phase, p_low, p_high)
        # ========================================================

        # 計算圖片真實寬高 (µm) 與單一像素大小
        # 完全依照原版數學邏輯，不添加額外單位轉換
        x_local = local_coord[0] 
        y_local = local_coord[1] 
        img_width = x_local.max() - x_local.min()
        img_height = y_local.max() - y_local.min()
        px_size = img_width / (phase.shape[1] - 1)
        
        # 紀錄這張圖的物理邊界 [左, 右, 下, 上]
        ext_obj = [
            abs_x - (img_width / 2.0),
            abs_x + (img_width / 2.0),
            abs_z - (img_height / 2.0),
            abs_z + (img_height / 2.0)
        ]
        
        phase_data_list.append(phase)
        mask_list.append(mask)
        extents_list.append(ext_obj)
        pixel_sizes.append(px_size)

    print("--- 階段 3：建立大畫布並執行加權融合 ---")
    avg_px_size = np.mean(pixel_sizes)
    
    # 找出全域物理邊界
    global_min_x = min([ext[0] for ext in extents_list])
    global_max_x = max([ext[1] for ext in extents_list])
    global_min_z = min([ext[2] for ext in extents_list])
    global_max_z = max([ext[3] for ext in extents_list])
    
    # 計算大畫布需要的像素尺寸 (預留一點邊界防呆)
    canvas_w = int(np.ceil((global_max_x - global_min_x) / avg_px_size)) + 5
    canvas_h = int(np.ceil((global_max_z - global_min_z) / avg_px_size)) + 5
    
    # 創建全域 Phase 畫布與權重畫布
    global_phase = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    global_weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    
    for i in range(len(ws_list)):
        ph = phase_data_list[i]
        msk = mask_list[i]
        ext = extents_list[i]
        ny, nx = ph.shape
        
        # 建立金字塔型距離權重矩陣 (中心為 1，邊緣為 0)
        Y, X = np.ogrid[:ny, :nx]
        dist_x = 1.0 - np.abs(X - nx/2) / (nx/2)
        dist_y = 1.0 - np.abs(Y - ny/2) / (ny/2)
        weight = np.maximum(0, dist_x) * np.maximum(0, dist_y)
        
        if msk is not None:
            # 確保只有 Mask > 0 的地方才有權重參與融合
            is_valid = (msk > 0) 
            weight *= is_valid 
            
        # 計算這張小圖在大畫布上的像素起始位置
        start_x = int(round((ext[0] - global_min_x) / avg_px_size))
        start_y = int(round((global_max_z - ext[3]) / avg_px_size))
        end_x = start_x + nx
        end_y = start_y + ny
        
        # 將權重與 Phase 資料疊加到大畫布上
        global_phase[start_y:end_y, start_x:end_x] += ph * weight
        global_weight[start_y:end_y, start_x:end_x] += weight

    # 執行加權平均
    nonzero = global_weight > 0
    final_phase = np.zeros_like(global_phase)
    final_phase[nonzero] = global_phase[nonzero] / global_weight[nonzero]

    print("--- 階段 4：繪製最終融合結果 ---")
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    
    # 全域畫布的 Extent
    global_extent = [
        global_min_x, 
        global_min_x + canvas_w * avg_px_size,
        global_max_z - canvas_h * avg_px_size, 
        global_max_z 
    ]

    cmap = plt.colormaps['gray']
    
    # 動態對比度範圍：直接抓取 final_phase 在有資料區域的最小值與最大值
    if np.any(nonzero):
        vmin_final = np.min(final_phase[nonzero])
        vmax_final = np.max(final_phase[nonzero])
    else:
        vmin_final, vmax_final = -np.pi, np.pi
        
    norm = Normalize(vmin=vmin_final, vmax=vmax_final)
    rgba_img = cmap(norm(final_phase)) 
    
    # 將沒有數據的地方 (權重為 0) 設為透明
    rgba_img[..., 3] = np.where(nonzero, 1.0, 0.0)

    # 畫出唯一的融合大圖
    ax.imshow(rgba_img, extent=global_extent)

    # ==========================================
    # 條件控制區塊：只有 show_labels 為 True 才會繪製
    # ==========================================
    if show_labels:
        print("正在繪製中心點...")
        ax.scatter(corrected_x, corrected_z, color='red', marker='x', s=100, zorder=5)
        
        for i, (cx, cy) in enumerate(zip(corrected_x, corrected_z)):
            ax.text(cx, cy, f' {labels[i]}', color='yellow', fontsize=12, 
                    fontweight='bold', va='bottom', ha='left', zorder=6,
                    path_effects=[path_effects.withStroke(linewidth=2, foreground='black')]) 
    # ==========================================

    ax.invert_yaxis() 
    ax.set_xlabel('Sample X relative (µm)')
    ax.set_ylabel('Sample Z relative (µm)')
    
    # ==========================================
    # 標題設定邏輯
    # ==========================================
    plot_title = title if title else "Stitched Object Phase (Distance-weighted & Clipped)"
    ax.set_title(plot_title, fontweight='bold')
    
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Phase (rad)')

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save, dpi=dpi, bbox_inches='tight')
        print(f"已儲存拼接圖至: {path_save}")
        
    plt.show()
    
    return final_phase


#===========================================================================
#                             GUI Test
#===========================================================================   
import sys
import traceback
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                 QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QFrame,
                                 QComboBox, QFileDialog)
except Exception:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                                     QLabel, QLineEdit, QPushButton, QGroupBox, QMessageBox, QFrame,
                                     QComboBox, QFileDialog)
    except Exception:
        pass

if 'QFrame' in locals() or 'QFrame' in globals():
    if not hasattr(QFrame, 'HLine') and hasattr(QFrame, 'Shape'):
        QFrame.HLine = QFrame.Shape.HLine
    if not hasattr(QFrame, 'Sunken') and hasattr(QFrame, 'Shadow'):
        QFrame.Sunken = QFrame.Shadow.Sunken

if 'Qt' in locals() or 'Qt' in globals():
    if not hasattr(Qt, 'Tool') and hasattr(Qt, 'WindowType'):
        Qt.Tool = Qt.WindowType.Tool
    if not hasattr(Qt, 'AlignCenter') and hasattr(Qt, 'AlignmentFlag'):
        Qt.AlignCenter = Qt.AlignmentFlag.AlignCenter

# 確保 ws 與 params 在全域環境，這樣終端機也能讀取到
if 'ws' not in globals():
    ws = None

if 'params' not in globals():
    try:
        params = set_params()
    except Exception:
        params = None

_gui_window = None 

class PtychoGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    # ==========================================
    # 穿越 IPython 命名空間的工具
    # ==========================================
    def set_ipython_var(self, var_name, value):
        """強制把變數塞進 IPython 的全域環境中"""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.user_ns[var_name] = value
        except Exception:
            pass
        import __main__
        setattr(__main__, var_name, value)

    def get_ipython_var(self, var_name):
        """從 IPython 的全域環境中讀取變數"""
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and var_name in ipython.user_ns:
                return ipython.user_ns[var_name]
        except Exception:
            pass
        import __main__
        return getattr(__main__, var_name, None)

    def _browse_directory(self, entry_widget):
        """開啟跨平台資料夾選擇視窗 (顯示資料夾內的所有檔案以供確認)"""
        current_dir = entry_widget.text().strip()
        if not current_dir or not os.path.exists(current_dir):
            current_dir = os.getcwd()

        dialog = QFileDialog(self, "選擇資料夾 (內部檔案均可預覽確認)", current_dir)
        file_mode_dir = getattr(QFileDialog, 'Directory', getattr(getattr(QFileDialog, 'FileMode', None), 'Directory', None))
        dont_native = getattr(QFileDialog, 'DontUseNativeDialog', getattr(getattr(QFileDialog, 'Option', None), 'DontUseNativeDialog', None))

        if file_mode_dir is not None:
            dialog.setFileMode(file_mode_dir)
        if dont_native is not None:
            dialog.setOption(dont_native, True)

        exec_func = getattr(dialog, 'exec_', getattr(dialog, 'exec', None))
        if exec_func and exec_func():
            selected = dialog.selectedFiles()
            if selected:
                entry_widget.setText(selected[0])

    def initUI(self):
        self.setWindowTitle("PyNX Ptycho 控制面板 (PyQt5)")
        self.setMinimumWidth(420)  

        main_layout = QVBoxLayout()

        # ==========================================
        # 1. 參數設定區
        # ==========================================
        group_params = QGroupBox("參數設定 (Params)")
        layout_params = QVBoxLayout()

        # Path Default
        h_path_default = QHBoxLayout()
        h_path_default.addWidget(QLabel("Path Default:"))
        self.entry_path_default = QLineEdit("/buffer/UsersData/") 
        h_path_default.addWidget(self.entry_path_default)
        self.btn_browse_path_default = QPushButton("瀏覽...")
        self.btn_browse_path_default.setFixedWidth(60)
        self.btn_browse_path_default.clicked.connect(lambda: self._browse_directory(self.entry_path_default))
        h_path_default.addWidget(self.btn_browse_path_default)
        layout_params.addLayout(h_path_default)

        # Scan ID
        h_scan = QHBoxLayout()
        h_scan.addWidget(QLabel("Scan ID:"))
        self.entry_scanID = QLineEdit("9264")
        h_scan.addWidget(self.entry_scanID)
        layout_params.addLayout(h_scan)

        # Defocus
        h_defocus = QHBoxLayout()
        h_defocus.addWidget(QLabel("Defocus (m):"))
        self.entry_defocus = QLineEdit("500e-6")
        h_defocus.addWidget(self.entry_defocus)
        layout_params.addLayout(h_defocus)

        # Max Size
        h_maxsize = QHBoxLayout()
        h_maxsize.addWidget(QLabel("Max Size:"))
        self.entry_maxsize = QLineEdit("800")
        h_maxsize.addWidget(self.entry_maxsize)
        layout_params.addLayout(h_maxsize)

        # Center
        h_center = QHBoxLayout()
        h_center.addWidget(QLabel("Center:"))
        
        self.combo_center = QComboBox()
        self.combo_center.addItems(["auto", "exp_config", "manual"])
        h_center.addWidget(self.combo_center)
        
        self.entry_center_manual = QLineEdit("400,400")
        self.entry_center_manual.setPlaceholderText("cx, cy")
        self.entry_center_manual.setFixedWidth(80)
        self.entry_center_manual.setEnabled(False) 
        h_center.addWidget(self.entry_center_manual)
        self.combo_center.currentTextChanged.connect(self._toggle_center_manual)
        
        layout_params.addLayout(h_center)

        # Probe (整合了 Load ID 的功能)
        h_probe = QHBoxLayout()
        h_probe.addWidget(QLabel("Probe:"))
        
        self.combo_probe = QComboBox()
        self.combo_probe.addItems(["focus", "gauss", "disc", "load"])
        h_probe.addWidget(self.combo_probe)
        
        self.entry_probe_val = QLineEdit("100e-6,0.042") # 預設 focus 的數值
        h_probe.addWidget(self.entry_probe_val)
        
        # 綁定事件：當選單改變時切換預設值
        self.combo_probe.currentTextChanged.connect(self._update_probe_default)
        
        layout_params.addLayout(h_probe)

        # Rebin & nbprobe 在同一行
        h_rebin_nbprobe = QHBoxLayout()
        
        h_rebin_nbprobe.addWidget(QLabel("Rebin:"))
        self.entry_rebin = QLineEdit("1")
        h_rebin_nbprobe.addWidget(self.entry_rebin)

        h_rebin_nbprobe.addWidget(QLabel("nbprobe:"))
        self.entry_nbprobe = QLineEdit("3")
        h_rebin_nbprobe.addWidget(self.entry_nbprobe)
        
        layout_params.addLayout(h_rebin_nbprobe)

        # ==================== 按鈕區域 ====================
        h_btn_params = QHBoxLayout()
        
        # Create Params 按鈕
        self.btn_create_params = QPushButton("Create Params")
        self.btn_create_params.setStyleSheet("background-color: #d1e7dd; font-weight: bold;") 
        self.btn_create_params.clicked.connect(lambda: self.run_cmd('create_params'))
        h_btn_params.addWidget(self.btn_create_params)

        # Set To ws 按鈕
        self.btn_set_to_ws = QPushButton("Set To ws")
        self.btn_set_to_ws.setStyleSheet("background-color: #cfe2ff; font-weight: bold;")
        self.btn_set_to_ws.clicked.connect(lambda: self.run_cmd('set_to_ws'))
        h_btn_params.addWidget(self.btn_set_to_ws)

        layout_params.addLayout(h_btn_params)

        # ==================== Read 按鈕區域 ====================
        h_btn_read = QHBoxLayout()

        # Read Meta 按鈕
        self.btn_read_meta = QPushButton("Read Meta")
        self.btn_read_meta.setStyleSheet("background-color: #e2e3e5; font-weight: bold;")
        self.btn_read_meta.clicked.connect(lambda: self.run_cmd('read_meta'))
        h_btn_read.addWidget(self.btn_read_meta)

        # Read Base 按鈕
        self.btn_read_base = QPushButton("Read Base")
        self.btn_read_base.setStyleSheet("background-color: #e2e3e5; font-weight: bold;")
        self.btn_read_base.clicked.connect(lambda: self.run_cmd('read_base'))
        h_btn_read.addWidget(self.btn_read_base)

        layout_params.addLayout(h_btn_read)
        # ==============================================================

        group_params.setLayout(layout_params)
        main_layout.addWidget(group_params)

        # ==========================================
        # 2. 執行操作區
        # ==========================================
        group_exec = QGroupBox("工作流執行 (Workflow)")
        layout_exec = QVBoxLayout()

        btn_start = QPushButton("Run Start (準備資料不存檔)")
        btn_start.clicked.connect(lambda: self.run_cmd('pynx_start'))
        layout_exec.addWidget(btn_start)

        btn_run = QPushButton("Run All (完整執行含存檔)")
        btn_run.clicked.connect(lambda: self.run_cmd('pynx_run'))
        layout_exec.addWidget(btn_run)

        # Path Save
        h_path_save = QHBoxLayout()
        h_path_save.addWidget(QLabel("Path Save:"))
        self.entry_path_save = QLineEdit("") 
        self.entry_path_save.setPlaceholderText("留空不存, 或填 True 自動存")
        h_path_save.addWidget(self.entry_path_save)
        self.btn_browse_path_save = QPushButton("瀏覽...")
        self.btn_browse_path_save.setFixedWidth(60)
        self.btn_browse_path_save.clicked.connect(lambda: self._browse_directory(self.entry_path_save))
        h_path_save.addWidget(self.btn_browse_path_save)
        layout_exec.addLayout(h_path_save)

        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        layout_exec.addWidget(line1)

        # ------------------------------------------
        # 演算法快速設定區塊
        # ------------------------------------------
        label_algo = QLabel("── 演算法設定 (Algorithm) ──")
        label_algo.setStyleSheet("font-weight: bold; color: #555;")
        layout_exec.addWidget(label_algo)

        h_fast_algo = QHBoxLayout()

        # Run DM + 數字
        btn_run_dm = QPushButton("Run DM")
        btn_run_dm.setStyleSheet("background-color: #fff3cd; font-weight: bold;")
        btn_run_dm.clicked.connect(lambda: self.run_cmd('pynx_run_dm'))
        self.entry_dm_num = QLineEdit("100")
        self.entry_dm_num.setFixedWidth(40)

        # Run ML + 數字
        btn_run_ml = QPushButton("Run ML")
        btn_run_ml.setStyleSheet("background-color: #fff3cd; font-weight: bold;")
        btn_run_ml.clicked.connect(lambda: self.run_cmd('pynx_run_ml'))
        self.entry_ml_num = QLineEdit("100")
        self.entry_ml_num.setFixedWidth(40)

        # 細節切換按鈕
        self.btn_algo_details = QPushButton("⚙ 細節")
        self.btn_algo_details.setFixedWidth(60)
        self.btn_algo_details.clicked.connect(self._toggle_algo_window)

        h_fast_algo.addWidget(btn_run_dm)
        h_fast_algo.addWidget(self.entry_dm_num)
        h_fast_algo.addWidget(btn_run_ml)
        h_fast_algo.addWidget(self.entry_ml_num)
        h_fast_algo.addWidget(self.btn_algo_details)

        layout_exec.addLayout(h_fast_algo)

        # ==========================================
        # 建立獨立的演算法細節視窗 (預設隱藏)
        # ==========================================
        self.algo_window = QWidget()
        self.algo_window.setWindowTitle("演算法細節設定")
        self.algo_window.setWindowFlags(Qt.Tool) # 讓它成為浮動工具視窗
        
        algo_details_layout = QVBoxLayout(self.algo_window)

        # Algo String 移動到這裡
        h_algo_str = QHBoxLayout()
        h_algo_str.addWidget(QLabel("Algo String:"))
        self.entry_algo_str = QLineEdit("(ML**100*DM**100)**1")
        h_algo_str.addWidget(self.entry_algo_str)

        btn_run_algo_str = QPushButton("Run algo string")
        btn_run_algo_str.setStyleSheet("background-color: #fff3cd; font-weight: bold;")
        btn_run_algo_str.clicked.connect(lambda: self.run_cmd('pynx_algo_run'))
        h_algo_str.addWidget(btn_run_algo_str)
        algo_details_layout.addLayout(h_algo_str)

        # 共用參數：Show & LLK
        h_shared_params = QHBoxLayout()
        h_shared_params.addWidget(QLabel("Show (plot):"))
        self.entry_show = QLineEdit("20")
        h_shared_params.addWidget(self.entry_show)
        h_shared_params.addWidget(QLabel("LLK (calc):"))
        self.entry_llk = QLineEdit("20")
        h_shared_params.addWidget(self.entry_llk)
        algo_details_layout.addLayout(h_shared_params)

        def add_algo_toggles(layout, title):
            h_box = QHBoxLayout()
            label = QLabel(f"{title}:")
            label.setFixedWidth(25)
            h_box.addWidget(label)
            
            h_box.addWidget(QLabel("Obj:"))
            combo_obj = QComboBox()
            combo_obj.addItems(["True", "False"])
            combo_obj.setFixedWidth(60)
            h_box.addWidget(combo_obj)
            
            h_box.addWidget(QLabel("Prb:"))
            combo_prb = QComboBox()
            combo_prb.addItems(["True", "False"])
            combo_prb.setFixedWidth(60)
            h_box.addWidget(combo_prb)
            
            h_box.addWidget(QLabel("Pos:"))
            entry_pos = QLineEdit("False")
            entry_pos.setFixedWidth(45)
            h_box.addWidget(entry_pos)
            
            h_box.addStretch() 
            layout.addLayout(h_box)
            return combo_obj, combo_prb, entry_pos

        self.combo_dm_obj, self.combo_dm_prb, self.entry_dm_pos = add_algo_toggles(algo_details_layout, "DM")
        self.combo_ml_obj, self.combo_ml_prb, self.entry_ml_pos = add_algo_toggles(algo_details_layout, "ML")
        self.combo_ap_obj, self.combo_ap_prb, self.entry_ap_pos = add_algo_toggles(algo_details_layout, "AP")
        # ==========================================

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout_exec.addWidget(line2)

        # Initialize 與 Inherit 並列
        h_init_inherit = QHBoxLayout()
        
        btn_init = QPushButton("Initialize")
        btn_init.clicked.connect(lambda: self.run_cmd('pynx_init'))
        h_init_inherit.addWidget(btn_init)

        btn_inherit = QPushButton("Inherit (Mode 1)")
        btn_inherit.clicked.connect(lambda: self.run_cmd('pynx_inherit'))
        h_init_inherit.addWidget(btn_inherit)
        
        layout_exec.addLayout(h_init_inherit)

        # Save Obj Probe
        btn_save_obj_probe = QPushButton("Save Obj Probe (儲存當前結果)")
        btn_save_obj_probe.setStyleSheet("background-color: #cce5ff; font-weight: bold;")
        btn_save_obj_probe.clicked.connect(lambda: self.run_cmd('pynx_save_obj_probe'))
        layout_exec.addWidget(btn_save_obj_probe)

        group_exec.setLayout(layout_exec)
        main_layout.addWidget(group_exec)

        # ==========================================
        # 3. 視覺化區
        # ==========================================
        group_plot = QGroupBox("視覺化 (Plotting)")
        layout_plot = QVBoxLayout()

        # View Keys
        h_view = QHBoxLayout()
        h_view.addWidget(QLabel("View Keys:"))
        self.entry_view_keys = QLineEdit("")
        self.entry_view_keys.setPlaceholderText("空白則畫當前ws，或填寫檔名關鍵字")
        h_view.addWidget(self.entry_view_keys)
        layout_plot.addLayout(h_view)

        btn_plot_overview = QPushButton("Plot Overview")
        btn_plot_overview.clicked.connect(lambda: self.run_cmd('plot_overview'))
        layout_plot.addWidget(btn_plot_overview)

        btn_plot_obj = QPushButton("Plot Object")
        btn_plot_obj.clicked.connect(lambda: self.run_cmd('plot_obj'))
        layout_plot.addWidget(btn_plot_obj)

        btn_plot_probe = QPushButton("Plot Probe")
        btn_plot_probe.clicked.connect(lambda: self.run_cmd('plot_probe'))
        layout_plot.addWidget(btn_plot_probe)

        # Probe Propagate 與其細節按鈕並排
        h_probe_prop = QHBoxLayout()
        btn_probe_prop = QPushButton("Probe Propagate (Interact)")
        btn_probe_prop.clicked.connect(lambda: self.run_cmd('probe_propagate'))
        
        self.btn_probe_details = QPushButton("⚙ 細節")
        self.btn_probe_details.setFixedWidth(60)
        self.btn_probe_details.clicked.connect(self._toggle_probe_window)
        
        h_probe_prop.addWidget(btn_probe_prop)
        h_probe_prop.addWidget(self.btn_probe_details)
        layout_plot.addLayout(h_probe_prop)

        # Plot Position 按鈕放在最下方
        btn_plot_pos = QPushButton("Plot Position")
        btn_plot_pos.setStyleSheet("background-color: #f8d7da; font-weight: bold;") 
        btn_plot_pos.clicked.connect(lambda: self.run_cmd('plot_position'))
        layout_plot.addWidget(btn_plot_pos)

        group_plot.setLayout(layout_plot)
        main_layout.addWidget(group_plot)

        # ------------------------------------------
        # 建立獨立的 Probe Propagate 細節視窗 (預設隱藏)
        # ------------------------------------------
        self.probe_window = QWidget()
        self.probe_window.setWindowTitle("Probe Propagate 細節設定")
        self.probe_window.setWindowFlags(Qt.Tool)

        probe_details_layout = QVBoxLayout(self.probe_window)

        h_use_probe = QHBoxLayout()
        h_use_probe.addWidget(QLabel("use_probe (idx):"))
        self.entry_use_probe = QLineEdit("0")
        h_use_probe.addWidget(self.entry_use_probe)
        probe_details_layout.addLayout(h_use_probe)

        h_linspace = QHBoxLayout()
        h_linspace.addWidget(QLabel("p_linspace (mm):"))
        self.entry_p_linspace = QLineEdit("-1, 1, 200")
        self.entry_p_linspace.setPlaceholderText("start, end, steps")
        h_linspace.addWidget(self.entry_p_linspace)
        probe_details_layout.addLayout(h_linspace)

        main_layout.addStretch()

        self.setLayout(main_layout)
        self.adjustSize()  

    # ==========================================
    # 核心邏輯處理
    # ==========================================
    def _toggle_algo_window(self):
        """控制獨立演算法細節視窗的顯示與隱藏"""
        if self.algo_window.isVisible():
            self.algo_window.hide()
        else:
            geom = self.geometry()
            self.algo_window.move(geom.right() + 10, geom.top())
            self.algo_window.show()

    def _toggle_probe_window(self):
        """控制獨立 Probe Propagate 細節視窗的顯示與隱藏"""
        if self.probe_window.isVisible():
            self.probe_window.hide()
        else:
            geom = self.geometry()
            self.probe_window.move(geom.right() + 10, geom.top() + 80)
            self.probe_window.show()

    def _toggle_center_manual(self, text):
        """控制 Center 手動輸入框的解鎖與鎖定"""
        if text == "manual":
            self.entry_center_manual.setEnabled(True)
        else:
            self.entry_center_manual.setEnabled(False)

    def _update_probe_default(self, text):
        """根據選擇的 Probe 類型自動填入預設值"""
        if text == "focus":
            self.entry_probe_val.setText("100e-6,0.042")
        elif text == "gauss":
            self.entry_probe_val.setText("500e-9x500e-9")
        elif text == "disc":
            self.entry_probe_val.setText("500e-9")
        elif text == "load":
            self.entry_probe_val.setText("9443")

    def get_gui_params(self):
        """讀取 GUI 上的 Params 數值"""
        # 讀取 Center 的模式
        center_mode = self.combo_center.currentText()
        if center_mode == 'auto':
            center_val = 'auto'
        elif center_mode == 'exp_config':
            center_val = 'exp_config'
        elif center_mode == 'manual':
            center_text = self.entry_center_manual.text().strip()
            try:
                center_val = [int(x.strip()) for x in center_text.split(',')]
            except Exception:
                print(f"[警告] Center 手動格式無法解析 '{center_text}'，退回預設值 'auto'")
                center_val = 'auto'

        # 組合 Probe 或處理 Load ID 邏輯
        probe_type = self.combo_probe.currentText()
        probe_val = self.entry_probe_val.text().strip()
        
        if probe_type == "load":
            load_id = probe_val
            probe_str = None  
        else:
            load_id = None
            probe_str = f"{probe_type},{probe_val}" if probe_val else probe_type

        gui_params = {
            'defocus': float(self.entry_defocus.text()),
            'maxsize': int(self.entry_maxsize.text()),
            'center': center_val,
            'loadID': load_id,
            'rebin': int(self.entry_rebin.text()),
            'nbprobe': int(self.entry_nbprobe.text()), 
            'path_load': self.entry_path_default.text().strip(),
            'probe': probe_str
        }

        if probe_type != "load":
            gui_params['loadprobe'] = None

        return gui_params

    def get_gui_algo_setting(self):
        """動態拼接 Obj, Prb (下拉) 與 Pos (數值) 以及 Show, LLK"""
        show_val = self.entry_show.text().strip()
        llk_val = self.entry_llk.text().strip()
        common_args = f"calc_llk={llk_val}, show_obj_probe={show_val}"
        
        def build_args(obj_combo, prb_combo, pos_entry):
            obj_val = obj_combo.currentText()
            prb_val = prb_combo.currentText()
            pos_val = pos_entry.text().strip()
            base_str = f"update_object={obj_val}, update_probe={prb_val}, update_pos={pos_val}"
            return f"{base_str}, {common_args}"

        return {
            'algo_string': self.entry_algo_str.text().strip(),
            'dm_string': build_args(self.combo_dm_obj, self.combo_dm_prb, self.entry_dm_pos),
            'ml_string': build_args(self.combo_ml_obj, self.combo_ml_prb, self.entry_ml_pos),
            'ap_string': build_args(self.combo_ap_obj, self.combo_ap_prb, self.entry_ap_pos),
            'raar_string': build_args(self.combo_ap_obj, self.combo_ap_prb, self.entry_ap_pos),
        }

    def run_cmd(self, cmd_name):
        try:
            print(f"\n[GUI] 執行 {cmd_name} ...", flush=True)
            scan_id = self.entry_scanID.text().strip()
            gui_path_default = self.entry_path_default.text().strip()
            
            gui_path_save_text = self.entry_path_save.text().strip()
            if gui_path_save_text.lower() == 'true':
                gui_path_save = True
            elif not gui_path_save_text:
                gui_path_save = ''
            else:
                gui_path_save = gui_path_save_text

            # --- Create Params & Run ---
            if cmd_name == 'create_params':
                gui_kwargs = self.get_gui_params()
                new_params = set_params(**gui_kwargs)
                self.set_ipython_var('params', new_params)
                print("-" * 50)
                print("[GUI] 已成功將 params 推送至終端機全域變數！")
                print("-" * 50, flush=True)

            elif cmd_name == 'set_to_ws':
                current_ws = self.get_ipython_var('ws')
                if current_ws is None:
                    raise ValueError("終端機找不到 ws 物件，請先建立(Run Start)或載入資料！")
                
                base_params = {}
                if hasattr(current_ws, 'params'):
                    if isinstance(current_ws.params, dict):
                        base_params = current_ws.params.copy()
                    else:
                        try:
                            base_params = vars(current_ws.params).copy()
                        except Exception:
                            pass
                
                gui_kwargs = self.get_gui_params()

                if self.combo_probe.currentText() != "load":
                    base_params['loadprobe'] = None
                
                base_params.update(gui_kwargs)
                new_params = set_params(**base_params)
                current_ws.params = new_params.copy()
                
                print("-" * 50)
                print(f"[GUI] 已使用 ws.params 為初始狀態，結合 GUI 設定重新更新了 params！")
                print("-" * 50, flush=True)

            elif cmd_name == 'read_meta':
                meta = read_metadata(scanID=scan_id, folder=gui_path_default)
                self.set_ipython_var('data_meta', meta)
                print_dict(meta)
                print("-" * 50)
                print("[GUI] 已成功將 metadata 推送至全域變數 'data_meta'！")
                print("-" * 50, flush=True)

            elif cmd_name == 'read_base':
                base = read_baseline(scanID=scan_id, folder=gui_path_default)
                self.set_ipython_var('data_baseline', base)
                print_dict(base)
                print("-" * 50)
                print("[GUI] 已成功將 baseline 推送至全域變數 'data_baseline'！")
                print("-" * 50, flush=True)

            elif cmd_name == 'pynx_start':
                base_params = self.get_ipython_var('params')
                if not isinstance(base_params, dict):
                    base_params = {}
                    
                base_params.update(self.get_gui_params())
                current_params = set_params(**base_params)
                self.set_ipython_var('params', current_params)
                
                print("[GUI] 已獲取並套用 GUI 上的最新參數！", flush=True)
                new_ws = pynx_start(scanID=scan_id, params=current_params, path_default=gui_path_default)
                self.set_ipython_var('ws', new_ws)
                print("[GUI] Run Start 完成！", flush=True)

            elif cmd_name == 'pynx_run':
                base_params = self.get_ipython_var('params')
                if not isinstance(base_params, dict):
                    base_params = {}
                    
                base_params.update(self.get_gui_params())
                current_params = set_params(**base_params)
                self.set_ipython_var('params', current_params)
                
                print("[GUI] 已獲取並套用 GUI 上的最新參數，執行 Run All ...", flush=True)
                gui_algo = self.get_gui_algo_setting()

                # NOTE: run synchronously on the main (GUI) thread. A previous version ran this
                # in a background thread with manual CUDA context push/pop across threads, which
                # deadlocks: the context is already current on the main thread (from Run Start /
                # prepare_processing_unit), and pushing the same context from a second thread is
                # not supported by the CUDA driver's per-thread context stack.
                try:
                    new_ws = pynx_run(scanID=scan_id, params=current_params, algo_setting=gui_algo, path_default=gui_path_default, path_save=gui_path_save)
                    self.set_ipython_var('ws', new_ws)
                    print(f"\n[GUI] Run All 完成！儲存路徑設定為: '{gui_path_save}'", flush=True)
                except Exception:
                    import traceback
                    print(f"\n[GUI Run All 錯誤]:\n{traceback.format_exc()}", flush=True)

            elif cmd_name in ['pynx_algo_run', 'pynx_run_dm', 'pynx_run_ml']:
                current_ws = self.get_ipython_var('ws')
                if current_ws is None: 
                    raise ValueError("終端機找不到 ws 物件，請先點擊 Run Start 載入資料！")
                
                gui_algo = self.get_gui_algo_setting()
                
                if cmd_name == 'pynx_run_dm':
                    num = self.entry_dm_num.text().strip()
                    gui_algo['algo_string'] = f"DM**{num}"
                elif cmd_name == 'pynx_run_ml':
                    num = self.entry_ml_num.text().strip()
                    gui_algo['algo_string'] = f"ML**{num}"
                
                algorithm = set_algorithm(**gui_algo)
                print(f"[GUI] 正在執行演算法 {gui_algo['algo_string']} ...", flush=True)

                # NOTE: run synchronously on the main (GUI) thread, see comment in the
                # 'pynx_run' branch above for why the background-thread + context push/pop
                # version deadlocked.
                try:
                    current_ws.p = algorithm * current_ws.p
                    print(f"\n[GUI] 演算法 {gui_algo['algo_string']} 執行完成！", flush=True)
                except Exception:
                    import traceback
                    print(f"\n[GUI 演算法錯誤]:\n{traceback.format_exc()}", flush=True)

            elif cmd_name == 'pynx_init':
                current_ws = self.get_ipython_var('ws')
                if current_ws is None: raise ValueError("終端機找不到 ws，請先執行 Run Start")
                
                gui_kwargs = self.get_gui_params()
                nbprobe_val = gui_kwargs.get('nbprobe', None)
                
                pynx_init(current_ws, nbprobe=nbprobe_val)
                print(f"[GUI] pynx_init 完成！(使用 nbprobe={nbprobe_val})", flush=True)

            elif cmd_name == 'pynx_inherit':
                current_ws = self.get_ipython_var('ws')
                if current_ws is None: raise ValueError("終端機找不到 ws，請先執行 Run Start")
                
                pynx_inherit_1st_mode(current_ws, init_obj=True)
                print("[GUI] pynx_inherit_1st_mode 完成！(固定使用 init_obj=True)", flush=True)

            elif cmd_name == 'pynx_save_obj_probe':
                current_ws = self.get_ipython_var('ws')
                if current_ws is None: 
                    raise ValueError("終端機找不到 ws 物件，請先載入資料！")
                
                if gui_path_save is True:
                    pynx_save_default(current_ws)
                    print("[GUI] Save Obj Probe 完成！已使用自動命名遞增儲存。", flush=True)
                elif gui_path_save:
                    pynx_save(current_ws, path_save=gui_path_save)
                    print(f"[GUI] Save Obj Probe 完成！已儲存至: {gui_path_save}", flush=True)
                else:
                    pynx_save(current_ws)
                    print("[GUI] Save Obj Probe 完成！已使用預設路徑儲存。", flush=True)

            elif cmd_name == 'plot_position':
                current_ws = self.get_ipython_var('ws')
                if current_ws is None:
                    raise ValueError("終端機找不到 ws 物件，請先載入資料！")
                
                print("[GUI] 執行 Plot Position...", flush=True)
                from pynx.ptycho import PlotPositions
                current_ws.p = PlotPositions() * current_ws.p
                print("[GUI] Plot Position 執行完成！", flush=True)

            elif cmd_name in ['plot_overview', 'plot_obj', 'plot_probe', 'probe_propagate']:
                current_ws = self.get_ipython_var('ws')
                view_keys = self.entry_view_keys.text().strip()
                
                if cmd_name == 'probe_propagate':
                    if current_ws is None:
                        raise ValueError("找不到 ws 物件 (此功能不支援 View Keys 搜尋，請先載入資料)")
                    
                    try:
                        use_probe_val = int(self.entry_use_probe.text().strip())
                        linspace_str = self.entry_p_linspace.text().strip()
                        parts = linspace_str.split(',')
                        if len(parts) != 3:
                            raise ValueError("p_linspace 格式錯誤，請確保為 'start, end, steps' (例: -1, 1, 200)")
                        
                        start_mm = float(parts[0].strip())
                        end_mm = float(parts[1].strip())
                        steps = int(parts[2].strip())
                        
                        p_linspace_val = [start_mm * 1e-3, end_mm * 1e-3, steps]
                        
                        print(f"[GUI] 執行 Probe Propagate: use_probe={use_probe_val}, p_linspace={p_linspace_val} (m)", flush=True)
                        pynx_probe_propagate(current_ws, use_probe=use_probe_val, p_linspace=p_linspace_val, interact=True)
                        
                    except ValueError as ve:
                        raise ValueError(f"輸入參數解析錯誤: {str(ve)}")
                else:
                    target_ws = view_keys if view_keys else current_ws
                    if target_ws is None:
                        raise ValueError("找不到 ws 物件且未指定 View Keys！")
                    
                    if cmd_name == 'plot_overview':
                        pynx_plot_overview(target_ws, path_default=gui_path_default)
                    elif cmd_name == 'plot_obj':
                        pynx_plot_obj(target_ws, path_default=gui_path_default)
                    elif cmd_name == 'plot_probe':
                        pynx_plot_probe(target_ws, path_default=gui_path_default)

        except Exception as e:
            err_msg = traceback.format_exc()
            print(err_msg, flush=True)
            QMessageBox.critical(self, "執行錯誤", f"{str(e)}\n\n(詳情請見終端機)")

    def closeEvent(self, event):
        if hasattr(self, 'algo_window') and self.algo_window.isVisible():
            self.algo_window.close()
        if hasattr(self, 'probe_window') and self.probe_window.isVisible():
            self.probe_window.close()
        super().closeEvent(event)


def simple_gui(blocking=None):
    """啟動 GUI 的入口函數"""
    global _gui_window
    ip = None
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            if getattr(ip, 'active_eventloop', None) not in ('qt', 'qt5'):
                try:
                    ip.enable_gui('qt5')
                except Exception:
                    try:
                        ip.enable_gui('qt')
                    except Exception:
                        pass
    except Exception:
        ip = None

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    _gui_window = PtychoGUI()
    _gui_window.show()
    _gui_window.raise_()
    _gui_window.activateWindow()
    print("PyQt5 GUI 面板已啟動！")

    if blocking is True or (blocking is None and ip is None):
        app.exec_()
    return _gui_window



