# Agent Note: Ptychography 實驗處理、探測器控制、環境建置與工具開發總紀錄

本文件彙整於 `Z:\TPS23A_USER\Ptychography` 執行的所有工作內容，包含 **GPU (CUDA) 運算環境建置**、**EIGER 探測器連線與 API 控制規範**、**核心程式碼維護與修復**、**掃描工具開發** 以及 **PyNX 數據重構紀錄（20260819 與 20260820）**，供平行處理 Agent 與後續工作無縫銜接。

---

## 1. 運算環境與 PyNX GPU (CUDA) 加速建置

- **PyCUDA & skcuda 支援**：
  - 安裝 NVIDIA CUDA 12 核心套件（`cuda_nvcc`, `cuda_runtime`, `cublas`, `cufft`, `curand`, `nvrtc`, `nvjitlink`）。
  - 修復 `pycuda`、`skcuda`（`fft`, `misc`, `cufft`, `cublas`, `cudart`）在 Python 3.12 與 NumPy 2.0+ 下的相容性問題（`np.sctypeDict`、`np.dtype.type`、DLL 動態載入路徑、移除 `np.float`/`np.sctypes` 棄用別名）。
  - 整合可攜式 MSVC 14.44 與 CUDA 13.3 編譯環境（`nvcc.exe`, `cicc.exe`, `ptxas.exe`）。
- **硬體辨識**：
  - 成功於 **NVIDIA GeForce RTX 5070 Ti**（15 GB VRAM，711 GB/s 頻寬）啟用 CUDA 原生加速。
  - 驗證 `pynx.scripts.pynx_info` 輸出 `cuda support: True`。

---

## 2. EIGER 1M 探測器連線與 API 控制規範

### 2.1 連線端點與服務資訊
* **DCU 控制主機 IP**：`172.19.23.189`
* **SIMPLON REST API**：`http://172.19.23.189/detector/api/1.6.0`（Port 80，韌體版本 `1.6.4`，相容 API `1.6.0`）
* **Webmin 系統管理**：`http://172.19.23.189:10000/`（帳號：`eiger` / 密碼：`#EIGER_Detector#`）
* **即時產檔與暫存下載區**：`http://172.19.23.189/data/`（命名規則：`20260820_series_$id`）

### 2.2 探測器硬體與實驗設定
* **型號與序號**：DECTRIS EIGER 1M（Detector S/N: `E-02-0156`, DCU S/N: `HBH6KF2`）
* **感測器規格**：$1030 \times 1065$ 像素（像素大小 $75\,\mu\text{m} \times 75\,\mu\text{m}$，Si 感測器厚度 $450\,\mu\text{m}$）
* **當前光子能量**：`9760.00 eV`（$9.76\,\text{keV}$，波長 $\lambda = 1.2703\,\text{\AA}$），門檻能量：`4880.00 eV`
* **觸發模式 (`trigger_mode`)**：`exte`（External Enable 外部硬體脈衝致能）
* **曝光設定**：`count_time: 1.0s`, `frame_time: 1.0s`, `nimages: 34`, `ntrigger: 34`

### 2.3 HDF5 數據與 Pixel Mask 規範
* **資料壓縮**：**BSLZ4 (Bitshuffle LZ4)**。Python 讀取 HDF5 數據前，**必須執行 `import hdf5plugin`**。
* **Pixel Mask 規格**：
  * Master 檔內路徑：`entry/instrument/detector/detectorSpecific/pixel_mask`
  * `0`：正常有效像素（佔 96.53%）
  * `1`：模組間隙與缺陷壞點（佔 3.47%，共 38,110 點）
  * 在 `data_*.h5` 影像中，遮罩像素數值自動被標記為 $2^{32}-1$ (`4294967295`)。

### 2.4 API 控制指令速查 (SIMPLON API)
```python
import requests

DCU_IP = "172.19.23.189"
API_BASE = f"http://{DCU_IP}/detector/api/1.6.0"
FW_BASE = f"http://{DCU_IP}/filewriter/api/1.6.0"

# [查詢狀態]
requests.get(f"{API_BASE}/status/state").json()["value"]         # 探測器狀態 (ready/idle/acquire)
requests.get(f"{FW_BASE}/files").json()                          # DCU 暫存檔案清單

# [控制命令]
requests.put(f"{API_BASE}/command/arm")                          # 進入 Arm 待命狀態
requests.put(f"{API_BASE}/command/disarm")                       # 解除 Arm / 封裝關閉檔案
requests.put(f"{API_BASE}/command/abort")                        # 中止當前擷取

# [參數設定]
requests.put(f"{API_BASE}/config/count_time", json={"value": 1.0})
requests.put(f"{API_BASE}/config/trigger_mode", json={"value": "exte"})
requests.put(f"{API_BASE}/config/ntrigger", json={"value": 34})
requests.put(f"{API_BASE}/config/nimages", json={"value": 1000})
requests.put(f"{FW_BASE}/config/name_pattern", json={"value": "20260821_series_$id"})
```

### 2.5 探測器自動化控制工作流程與工具 (`configure_eiger.py`)

為了避免每次手動透過 HTTP 逐步設定並防止狀態卡住，已封裝完整自動化工具 [`configure_eiger.py`](file:///Z:/TPS23A_USER/Ptychography/configure_eiger.py)，標準自動化工作流程如下：

```text
[讀取掃描指令檔 fermat_scan_commands.txt]
  ├─ 自動計算 TRG 指令數 → ntrigger (例如 34)
  └─ 自動解析 TRG 脈衝寬度 (us) → count_time (例如 1000000 us = 1.0s)
                         ↓
[步驟 1: Disarm] 發送 PUT /command/disarm 釋放先前狀態
                         ↓
[步驟 2: 穩定等待] time.sleep(3.0) 確保 DCU 與 FileWriter 狀態機回到 idle
                         ↓
[步驟 3: 參數寫入]
  ├─ trigger_mode : "exte" (外部硬體脈衝致能)
  ├─ ntrigger     : 掃描點數 (例如 34)
  ├─ nimages      : 影像張數 (例如 1000 或 34)
  ├─ count_time   : 曝光時間 (例如 1.0s，frame_time 由韌體自動匹配)
  └─ name_pattern : 檔名規則 (例如 20260821_series_$id)
                         ↓
[步驟 4: 讀回驗證] 回讀 /config/* 確保所有參數正確生效
                         ↓
[步驟 5: 執行 Arm] 發送 PUT /command/arm 取得 Sequence ID
                         ↓
[就緒狀態] Detector = ready, FileWriter = acquire (隨時接收馬達硬體觸發脈衝)
```

#### 2.5.1 常用命令列範例 (CLI Usage)
```bash
# 1. 最常用：自動解析指令檔 (fermat_scan_commands.txt)，完成 Disarm -> 等待 3 秒 -> 設定參數 -> Arm
python configure_eiger.py

# 2. 手動指定點數、曝光時間與檔名規則 (例如單獨設定 nimages=1000, ntrigger=34)
python configure_eiger.py --ntrigger 34 --nimages 1000 --exposure 1.0 --pattern "20260821_series_$id"

# 3. 查詢探測器與 FileWriter 即時狀態及所有配置
python configure_eiger.py --status

# 4. 單獨執行 Disarm (掃描異常中斷或手動釋放時使用)
python configure_eiger.py --disarm-only

# 5. 僅設定參數但不執行 Arm
python configure_eiger.py --points 34 --no-arm
```

#### 2.5.2 Python 程式碼模組化調用
```python
from configure_eiger import configure_and_arm_eiger, EigerController

# 方式 A: 執行完整 Disarm -> 設定 -> Arm
seq_id = configure_and_arm_eiger(
    ip="172.19.23.189",
    n_points=34,
    count_time=1.0,
    trigger_mode="exte",
    nimages=1000,
    ntrigger=34,
    name_pattern="20260821_series_$id",
    disarm_wait_sec=3.0,
    do_arm=True
)

# 方式 B: 細部操作物件
eiger = EigerController(ip="172.19.23.189")
eiger.disarm()
det_st, fw_st = eiger.get_status()
```

#### 2.5.3 控制注意事項
1. **Disarm 必要性**：探測器若先前處於 `ready` 或 `acquire` 狀態，修改特定參數會被拒絕。因此在寫入設定前**必須先 Disarm 並等待 3 秒**。
2. **FileWriter 空回應處理**：FileWriter API 在寫入 `name_pattern` 時回傳 HTTP 200 但 Body 長度為 0，Python 處理時需避免直接 `r.json()` 導致解析錯誤。
3. **`exte` 模式特性**：在 `trigger_mode="exte"` 下，每個外部脈衝觸發 1 次曝光（持續時間由脈衝寬度或 `count_time` 決定），`ntrigger` 代表期望收到的外部脈衝總數。

---

## 3. 掃描工具開發與馬達控制器指令系統

### 3.1 費馬螺線（Fermat Spiral）演算法 (`fermat_spiral.py`)
- **演算法核心**：
  - 支援 $\pm$ 雙臂對稱螺線（Arm $+$ 與 Arm $-$）以及黃金角（$137.508^\circ$）均勻採樣。
  - **正方形裁切（Square Cropping）**：以外接圓半徑 $R = \frac{\text{range}}{2} \sqrt{2}$ 擴展生成後，自動裁剪並保留正方形範圍 $[-range/2, range/2] \times [-range/2, range/2]$ 內的點。
- **Bluesky 整合**：
  - 整合 Bluesky 官方之 `bluesky.plan_patterns.spiral_fermat` 點位生成邏輯。
  - 針對 Case 1（中心 50,50，$\text{step}=0.1\ \mu m$，$\text{range}=1.0\ \mu m$），精確計算出 **34 個掃描點**。

### 3.2 馬達控制器指令生成 (`generate_fermat_scan_cmd.py`)
- **指令協議與單位規範**：
  - `STR`：掃描開始（無 `>`）。
  - `SFL,0,<axis>,<position>`：
    - 軸編號：`0` = X 軸，`1` = Y 光軸，`2` = Z 軸。
    - 單位換算：$1\text{ nm} = 10^6\text{ 單位} \implies \mathbf{1\ \mu m = 10^9\text{ SFL 單位}}$。
  - `RES,3`：馬達移動後等待 **0.3 秒**（單位為 0.1 秒）。
  - `TRG,1000000`：觸發 Detector 曝光收光 **1.0 秒**（單位為微秒 $\mu s$）。
  - `RES,1`：收光後等待 **0.1 秒**。
  - `END`：掃描結束。
- **產出檔案**：
  - 指令檔：`fermat_scan_commands.txt`（共 34 點，172 行控制指令）。
  - 路徑圖：`fermat_scan_path.png`。
  - 詳細手冊：`SCAN_MANUAL.md`。

---

## 4. 核心代碼修復與維護 (`main.py`)

1. **CUDA Deadlock 修復**：
   - PyQt5 GUI 中的 `Run DM` / `Run ML` / `Run All` 原先使用背景執行緒搭配手動 PyCUDA context push/pop，會與主執行緒衝突導致 GPU 死鎖。已改為同步於主執行緒安全執行。
2. **免安裝環境自動配置 (`_load_msvc_env`)**：
   - 加入 Portable MSVC (`%LOCALAPPDATA%\portable\msvc`) 與 Portable CUDA 12.9 (`%LOCALAPPDATA%\portable\cuda\...\nvcc\bin`) 自動路徑載入，確保 PyCUDA JIT 編譯順利調用 `nvcc.exe` 與 `cl.exe`。
3. **Matplotlib 升級相容**：
   - 修正 Matplotlib >= 3.8 移除 `matplotlib.cm.get_cmap` 導致的崩潰問題，加入自動相容轉接。
4. **繪圖函數屬性安全 (`pynx_plot_*`)**：
   - 修正 `pynx_plot_overview`、`pynx_plot_obj`、`pynx_plot_probe` 中讀取 `ws_p.scan` 時因缺少屬性發生的 `AttributeError`（改用 `getattr` 安全讀取）。
5. **GUI 啟動模式支援 (`simple_gui` / `launch_gui.py`)**：
   - 支援一般命令列阻塞執行 (`app.exec_()`) 與 IPython 互動式非阻塞執行。
   - `launch_gui.py` 自動建立 Series 資料集的 `ws` 並注入全域命名空間，點擊 **【Set To ws】** 即可直接載入重構。
6. **Git 版本庫同步 (`Ifitpossible/ptychography.git`)**：
   - 本地倉庫位置：`C:\Users\user\ptychography`（已配置 MinGit 與 `.gitignore`）。
   - 已同步核心修復至 `pynx_at_tps25a/main.py`，並將掃描工具模組化整理至 `scan_tools/`。

---

## 5. 數據重構紀錄（2026/08/19 & 2026/08/20）

### 5.1 共同幾何與物理參數
- **Detector**：Dectris Eiger 1M（$1065 \times 1030$ 像素，像素大小 $75.0\ \mu m$）
- **能量 / 波長**：$E = 9.670\text{ keV}$（$9670.0\text{ eV}$，$\lambda \approx 1.28215\text{ \AA}$）
- **探測器距離 (SSD)**：$1.41\text{ m}$
- **CDI 視窗 (Window)**：$\frac{\lambda \cdot \text{SSD}}{75\ \mu\text{m}} = 2.410\ \mu\text{m}$ ($2410.45\text{ nm}$)
- **實空間像素大小 ($300\times 300$)**：$\Delta x_{\text{obj}} \approx 8.03\text{ nm}$

### 5.2 歷史數據集重構彙整

| 數據集 / Series | 點數 / 掃描模式 | 裁切尺寸 / 樣品解析度 | 演算法 | 收斂指標 (Poisson LLK) | 成果路徑 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **20260819 (3×3 網格)** | 9 點 (Snake 網格) | $256 \times 256$ ($9.33\text{ nm}$) | `DM**100` (1 模態) | **0.87** (793.59 $\to$ 0.87) | `20260819\Results\ptycho_3x3_DM100_*` |
| **Series 10 (20260820)** | 10 點 (0.1s 測試) | - | - | 驗證通過 (能量 8.05 keV) | DCU 暫存 / 驗證通過 |
| **Series 14 (20260820)** | 13 點 (1.0s 測試) | - | - | 驗證通過 (能量 9.76 keV) | DCU 暫存 / 驗證通過 |
| **Series 15 (20260820)** | 34 點 (Fermat 螺線) | $300 \times 300$ ($7.96\text{ nm}$) | `DM**100 * ML**100` | 0.906 / 1.418 | `20260820\Results\series_15_*` |
| **Series 17 (20260820)** | 34 點 (翻轉座標) | $300 \times 300$ ($7.96\text{ nm}$) | `DM**100 * ML**100` | **0.830** / **1.270** | `20260820\Results\series_17_*` |
| **Series 29 (20260821)** | 34 點 (翻轉座標 $\times -1$) | $300 \times 300$ ($7.96\text{ nm}$) | `DM**100 * ML**100` | **1.808** | `20260821\Results\series_29_*` |
| **Series 31 (20260821)** | 34 點 (翻轉座標 $\times -1$) | $300 \times 300$ ($7.96\text{ nm}$) | 待由 GUI 執行 | - | `20260821\20260821_series_31_*` |
| **Series 33 (20260821)** | 34 點 (4 組馬達乘數評估) | $300 \times 300$ ($7.96\text{ nm}$) | `DM**100 * ML**100` (disc 500nm, def 0) | **0.3430** (X=+1, Z=-1) | `20260821\Results\series_33_*` |
| **Series 35 (20260821)** | 287 點 (4 組馬達乘數評估) | $300 \times 300$ ($7.96\text{ nm}$) | `DM**100 * ML**100` (disc 500nm, def 0) | **0.8766** (X=-1, Z=+1) ★ | `20260821\Results\series_35_*` |

### 5.3 馬達座標乘數標定結論 (Motor Coordinate Scale Calibration)
- **實測與物理驗證結論**：
  - 在 Series 33（34 點）與 Series 35（287 點）的 4 組乘數組合（$\pm 1, \pm 1$）測試中，**`scale_x = -1.0, scale_z = +1.0`**（以及對稱組 `scale_x = +1.0, scale_z = -1.0`）收斂指標顯著優於同號組。
  - 經實測圖像細節與物理方位對比確認：**`X = -1.0, Z = +1.0` 為最可信之馬達乘數設定**（Poisson LLK = 0.8766）。
- **標準座標映射公式**：
  $$\text{scan\_x} = (x_{\text{motor}} - x_{\text{center}}) \times (-1.0)$$
  $$\text{scan\_z} = (z_{\text{motor}} - z_{\text{center}}) \times (+1.0)$$
- **系統設定**：
  - `launch_gui.py` 已將預設值更新為 `--scale-x -1.0 --scale-z 1.0`。

---

## 6. 主要檔案與目錄索引

```text
Z:\TPS23A_USER\Ptychography\
├── Agent_note.md                     # 本整合紀錄文件
├── configure_eiger.py                # EIGER 探測器 HTTP 參數設定與 Arm 工具
├── main.py                           # PyNX 核心工具包與 GUI 定義 (已修復 CUDA/Matplotlib)
├── launch_gui.py                     # GUI 快速啟動與資料集載入腳本 (支援自動點數指令匹配)
├── compare_series_33_scales.py       # Series 33 四組馬達乘數評估腳本 (34 點)
├── compare_series_35_scales.py       # Series 35 四組馬達乘數評估腳本 (287 點)
├── fermat_spiral.py                  # 費馬螺線散佈點生成器
├── generate_fermat_scan_cmd.py       # 螺線掃描指令生成器
├── fermat_scan_commands.txt          # 當前掃描指令檔 (287 點)
├── fermat_scan_commands_34pts.txt    # 34 點掃描指令檔 (range 1.0, step 0.1)
├── fermat_scan_commands_287pts.txt   # 287 點掃描指令檔 (range 3.0, step 0.1)
├── SCAN_MANUAL.md                    # 掃描協議與換算說明文件
├── 20260819\
│   ├── ptycho_000_master.h5 ~ 008    # 20260819 原始數據
│   └── Results\                      # 20260819 3x3 重構成果 (.cxi, .png)
├── 20260820\
│   ├── 20260820_series_14_master.h5  # Series 14 Master / Data
│   ├── 20260820_series_15_master.h5  # Series 15 Master / Data
│   ├── 20260820_series_17_master.h5  # Series 17 Master / Data
│   └── Results\                      # Series 15 & 17 重構成果 (.cxi, .png)
└── 20260821\
    ├── 20260821_series_29_master.h5  # Series 29 Master / Data
    ├── 20260821_series_31_master.h5  # Series 31 Master / Data
    ├── 20260821_series_33_master.h5  # Series 33 Master / Data (34 frames)
    ├── 20260821_series_35_master.h5  # Series 35 Master / Data (287 frames)
    └── Results\                      # 重構成果 (.cxi, .png)
```
