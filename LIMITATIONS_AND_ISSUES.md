# Ptychography 系統環境相容性、版本限制與已知問題排查手冊 (Troubleshooting & Limitations)

本文件詳細紀錄於 Windows 原生環境、NVIDIA CUDA 12/13、NumPy 2.x 以及 PyNX 重構環境建置與運行過程中遇到的**核心問題、版本相容性限制與對應之完整解決方案**。

---

## 1. Python 版本與二進制 C 擴充模組 (ABI) 相容性

### 1.1 問題現象：跨版本 C 擴充套件衝突 (`.pyd` ABI Mismatch)
* **錯誤訊息**：
  ```text
  ImportError: cannot import name '_errors' from partially initialized module 'h5py' (most likely due to a circular import)
  ```
* **根本原因**：
  * 使用者的虛擬環境 `Code\myvenv\Lib\site-packages` 包含針對 **Python 3.13** 編譯的 C 擴充套件（如 `_errors.cp313-win_amd64.pyd`、`scipy`、`scikit-image` 等）。
  * 若在 PowerShell 終端機預設使用 **Python 3.12** 執行腳本，直接將 Python 3.13 的 `site-packages` 注入 `sys.path` 會因為 CPython ABI 不相容而導致動態庫載入失敗。

### 1.2 解決方案與最佳實踐
* 在 [`main.py`](file:///C:/Users/User/Desktop/AllenCheng/data/Ptychography/main.py) 與 [`launch_gui.py`](file:///C:/Users/User/Desktop/AllenCheng/data/Ptychography/launch_gui.py) 入口處加入動態版本判斷：
  ```python
  import sys, os
  if sys.version_info >= (3, 13):
      myvenv_sp = r"C:\Users\User\Desktop\AllenCheng\Code\myvenv\Lib\site-packages"
      if os.path.exists(myvenv_sp) and myvenv_sp not in sys.path:
          sys.path.insert(0, myvenv_sp)
  ```
* 同時為 Python 3.12 與 Python 3.13 各自安裝原生相容的 `scikit-learn`、`pycuda`、`pyopencl`、`scikit-cuda`、`PyQt5` 套件，確保兩版本均可獨立無誤執行。

---

## 2. NumPy 2.0+ 升級帶來的相容性問題與修復

NumPy 2.0 移除了大量過去在 NumPy 1.x 標記為 Deprecated 的別名與內部屬性，導致舊版 PyNX 與 `scikit-cuda` 無法直接運行。

| 缺失/移除項目 | 觸發模組 | 錯誤訊息 | 修復方案 |
| :--- | :--- | :--- | :--- |
| **`np.typeDict`** | `skcuda.misc` | `AttributeError: module 'numpy' has no attribute 'typeDict'` | 改用 `np.sctypeDict` 或在頂層轉接：<br>`if not hasattr(np, 'typeDict'): np.typeDict = getattr(np, 'sctypeDict', {})` |
| **`np.float`, `np.complex`, `np.int`, `np.bool`** | `pynx.ptycho` 核心運算子 | `AttributeError: module 'numpy' has no attribute 'float'` | 在頂層注入別名相容轉接：<br>`np.float = float`, `np.complex = complex`, `np.int = int`, `np.bool = bool` |
| **`np.sctypes`** | `pynx` 模擬探針/物體生成 | `AttributeError: 'np.sctypes' was removed in NumPy 2.0` | 手動補齊字典結構：<br>`np.sctypes = {'int': [...], 'uint': [...], 'float': [...], 'complex': [...], 'others': [...]}` |

---

## 3. NVIDIA CUDA 12.x / 13.x 與 `scikit-cuda` 動態庫連結

### 3.1 Windows DLL 載入機制變更 (Python 3.8+)
* **現象**：系統已安裝 CUDA 12.5，但 Python 仍報錯找不到 CUDA DLL。
* **原因**：Python 3.8+ 在 Windows 上不再預設從系統 `%PATH%` 搜尋 C DLL。
* **修復**：
  ```python
  cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
  if os.path.exists(cuda_bin) and hasattr(os, 'add_dll_directory'):
      os.add_dll_directory(cuda_bin)
  ```

### 3.2 `skcuda` 僅支援舊版 CUDA 10 DLL 名稱
* **現象**：
  ```text
  OSError: cufft library not found
  OSError: CUDA runtime library not found
  ```
* **原因**：`skcuda`（`cufft.py`, `cublas.py`, `cudart.py`）內部寫死的 `_win32_version_list` 僅列至版本 `10`（如 `cufft64_10.dll`），不包含 CUDA 11/12 的 `cufft64_12.dll` 與 `cudart64_12.dll`。
* **修復**：在 `skcuda` 對應模組中的 `_win32_version_list` 開頭加入 `[12, 125, 120, 11, 110, ...]`。

### 3.3 `setuptools >= 80` 移除 `pkg_resources`
* **現象**：`skcuda` 匯入時拋出 `ModuleNotFoundError: No module named 'pkg_resources'`。
* **修復**：在 `skcuda/version.py` 中直接指定固定版本號 `__version__ = '0.5.3'`，避免調用已棄用的 `pkg_resources`。

---

## 4. PyCUDA JIT 編譯警告與 Windows 繁體中文環境 (cp950 / Big5)

### 4.1 MSVC C4819 警告與 PyCUDA UserWarning
* **現象**：執行 DM / ML 演算法時，終端機被數十行警告洗版：
  ```text
  UserWarning: The CUDA compiler succeeded, but said the following:
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\include\driver_types.h(1327): warning C4819: 該檔案含有無法在目前字碼頁 (950) 中表示的字元...
  ```
* **原因**：CUDA 官方標頭檔中的 Unicode 註解字元觸發 MSVC cp950 編碼警告。PyCUDA 即使編譯成功也會將編譯器警告以 `UserWarning` 印出。
* **修復**：在程式頂層加入過濾器：
  ```python
  import warnings
  warnings.filterwarnings("ignore", category=UserWarning, module="pycuda")
  warnings.filterwarnings("ignore", message=".*The CUDA compiler succeeded.*")
  warnings.filterwarnings("ignore", message=".*creating CUBLAS context.*")
  warnings.filterwarnings("ignore", message=".*cuFFT plan destruction inhibited.*")
  ```

---

## 5. PyQt5 GUI 多執行緒與 CUDA Context 死鎖 (CUDA Context Deadlock)

### 5.1 歷史嚴重 Bug：背景執行緒手動 push/pop CUDA Context
* **現象**：點擊 GUI 的 **【Run DM】** / **【Run ML】** / **【Run All】** 時，GPU 使用率瞬間歸零（0%），視窗反白凍結且無任何錯誤輸出。
* **原因**：
  * 舊版 GUI 在 `threading.Thread` 中嘗試手動調用 `cu_ctx.push()` 與 `cu_ctx.pop()`。
  * CUDA Driver 的 Context Stack 是**嚴格 Per-Thread** 的；在另一個執行緒重複 Push 主執行緒持有的 Context 會直接引發驅動層級死鎖。
* **修復方案**：
  * 將重構演算法改為在主執行緒（GUI 執行緒）中同步執行。雖然運算期間 GUI 會短暫進入忙碌狀態，但能 100% 確保 GPU 原生加速正常運算且無死鎖風險。

---

## 6. DECTRIS EIGER 1M 探測器連線與 API 控制規範

### 6.1 BSLZ4 壓縮濾波器
* **規範**：所有 EIGER 1M 產生的 HDF5 資料均使用 DECTRIS Bitshuffle LZ4 (BSLZ4, Filter ID 32004) 壓縮。
* **要求**：在 Python 讀取 HDF5 檔案前，**必須優先執行 `import hdf5plugin`**。

### 6.2 探測器狀態機與 Disarm 延遲要求
* **規範**：探測器若處於 `ready` 或 `acquire` 狀態，修改特定參數會被 DCU 拒絕。
* **要求**：在配置任何新參數前，必須發送 `PUT /command/disarm` 並**強制等待至少 3 秒**，確保韌體內部狀態機完全回到 `idle`。

---

## 7. 費馬螺線（Fermat Spiral）掃描與馬達控制換算

### 7.1 單位換算公式
* **馬達控制器（SFL 指令）**：
  $$\mathbf{1\ \mu m = 10^9\text{ SFL 單位}} \quad (1\text{ nm} = 10^6\text{ 單位})$$
* **曝光觸發（TRG 指令）**：
  $$\mathbf{1.0\text{ 秒} = 1,000,000\ \mu s}$$

### 7.2 馬達座標映射乘數 (Scale Calibration)
* **標準映射公式**：
  $$\text{scan\_x} = (x_{\text{motor}} - x_{\text{center}}) \times \text{scale\_x}$$
  $$\text{scan\_z} = (z_{\text{motor}} - z_{\text{center}}) \times \text{scale\_z}$$
* **預設配置**：
  * 當前系統預設值已更新為 `--scale-x 1.0 --scale-z 1.0`。
  * 經 Series 35（287 點）物理標定評估，`scale_x = -1.0, scale_z = 1.0` 亦為物理反向對稱候選設定（LLK = 0.8766）。可依實驗樣品物理特徵隨時切換。

---

## 8. 即時動態繪圖 (Live Plotting) 白畫面與重構耗時線性增加 (Linear Slowdown)

### 8.1 問題現象
1. **白畫面現象**：在 GUI 中執行 DM / ML 演算法時，每隔 N 次迭代（預設 `show_obj_probe=20`）雖然會跳出繪圖視窗，但視窗內容呈現一片白色/空白，直到所有迭代結束才一次顯示。
2. **耗時線性增加**：觀察終端機輸出，每 20 次迭代的運算週期時間 (`dt/cycle`) 呈現線性遞增（例如由 `0.064s` $\to$ `0.083s` $\to$ `0.107s`）。

### 8.2 根本原因
* **圖層持續疊加（導致線性變慢）**：
  PyNX 原生 `pynx.utils.plot_utils.show_obj_probe` 在更新圖形時，僅重複執行 `plt.subplot(gs[0])` 與 `imshow()`，**從未調用 `fig.clf()` 清空前次圖層**。導致每次更新都在同一個 Figure 上堆疊 4 張新圖與線條。當迭代達到 200 次時，底層 Canvas 已疊加超過 40 層圖片，導致 Matplotlib `canvas.draw()` 耗時越來越長。
* **Qt 繪圖事件阻塞（導致白畫面）**：
  在 PyQt5 應用程式中，演算法於主執行緒執行期間，若未主動觸發 Qt 事件循環（`QApplication.processEvents()`），作業系統分配給該視窗的 Paint / Expose 事件不會被即時處理，因而僅顯示預設的白色背景底框。

### 8.3 完整解決方案
於 `pynx/utils/plot_utils.py` 的 `show_obj_probe` 函式中：
1. 在取得 Figure 後立即加入 `fig.clf()` 清空歷史圖層，確保記憶體佔用恆定且渲染耗時不隨迭代增加。
2. 在繪圖刷新處加入 `fig.canvas.flush_events()` 與 `QApplication.processEvents()`：
   ```python
   fig.canvas.draw_idle()
   fig.canvas.flush_events()
   from PyQt5.QtWidgets import QApplication
   app = QApplication.instance()
   if app is not None:
       app.processEvents()
   ```
   如此即可在每一次迭代更新時，即時流暢呈現動態演化的 Object 與 Probe 振幅與相位圖，徹底告別白畫面與效能退化。
