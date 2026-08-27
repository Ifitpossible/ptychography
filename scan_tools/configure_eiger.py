#!/usr/bin/env python3
"""
EIGER 1M Detector HTTP Configuration & Arm Controller
用於在執行馬達掃描 (Scan Script) 前，透過 HTTP SIMPLON REST API 設定 EIGER 探測器參數並進行 Arm。

主要流程:
1. 嘗試 Disarm (確保釋放先前狀態)
2. 等待 3 秒穩定
3. 設定參數:
   - trigger_mode: "exte" (External Enable 外部觸發)
   - ntrigger: 掃描點數 (例如 34)
   - nimages: 掃描點數 (例如 34)
   - count_time: 曝光時間 (秒，例如 1.0s，自動匹配 TRG 指令)
   - (可選) name_pattern: 檔案命名格式 (例如 20260821_series_$id)
4. 驗證設定結果
5. 執行 Arm 使探測器進入 Ready 待命狀態
"""

import os
import sys
import time
import argparse
import requests
from datetime import datetime

# Windows terminal UTF-8 output support
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 預設連線資訊
DEFAULT_DCU_IP = "172.19.23.189"
DEFAULT_API_VERSION = "1.6.0"


class EigerController:
    def __init__(self, ip=DEFAULT_DCU_IP, api_version=DEFAULT_API_VERSION, timeout=5.0):
        self.ip = ip
        self.api_version = api_version
        self.timeout = timeout
        self.detector_base = f"http://{ip}/detector/api/{api_version}"
        self.filewriter_base = f"http://{ip}/filewriter/api/{api_version}"

    def get_status(self):
        """查詢探測器與 FileWriter 當前狀態"""
        try:
            det_state = requests.get(f"{self.detector_base}/status/state", timeout=self.timeout).json()
            det_val = det_state.get("value", "unknown")
        except Exception as e:
            det_val = f"error ({e})"

        try:
            fw_state = requests.get(f"{self.filewriter_base}/status/state", timeout=self.timeout).json()
            fw_val = fw_state.get("value", "unknown")
        except Exception as e:
            fw_val = f"error ({e})"

        return det_val, fw_val

    def get_config(self, key):
        """查詢特定設定參數"""
        r = requests.get(f"{self.detector_base}/config/{key}", timeout=self.timeout)
        if r.status_code == 200:
            return r.json().get("value")
        return None

    def get_filewriter_config(self, key):
        """查詢 FileWriter 設定參數"""
        r = requests.get(f"{self.filewriter_base}/config/{key}", timeout=self.timeout)
        if r.status_code == 200:
            return r.json().get("value")
        return None

    def set_config(self, key, value):
        """寫入特定設定參數"""
        r = requests.put(f"{self.detector_base}/config/{key}", json={"value": value}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"設定 {key}={value} 失敗 (HTTP {r.status_code}): {r.text}")
        if r.text and r.text.strip():
            try:
                return r.json()
            except Exception:
                return r.text
        return True

    def set_filewriter_config(self, key, value):
        """寫入 FileWriter 設定參數"""
        r = requests.put(f"{self.filewriter_base}/config/{key}", json={"value": value}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"設定 FileWriter {key}={value} 失敗 (HTTP {r.status_code}): {r.text}")
        if r.text and r.text.strip():
            try:
                return r.json()
            except Exception:
                return r.text
        return True

    def disarm(self):
        """執行 Disarm 指令"""
        r = requests.put(f"{self.detector_base}/command/disarm", timeout=self.timeout)
        if r.status_code == 200:
            if r.text and r.text.strip():
                try:
                    return r.json()
                except Exception:
                    return r.text
            return True
        raise RuntimeError(f"Disarm 失敗 (HTTP {r.status_code}): {r.text}")

    def arm(self):
        """執行 Arm 指令"""
        r = requests.put(f"{self.detector_base}/command/arm", timeout=self.timeout)
        if r.status_code == 200:
            if r.text and r.text.strip():
                try:
                    return r.json()
                except Exception:
                    return r.text
            return True
        raise RuntimeError(f"Arm 失敗 (HTTP {r.status_code}): {r.text}")

    def abort(self):
        """執行 Abort 指令"""
        r = requests.put(f"{self.detector_base}/command/abort", timeout=self.timeout)
        return r.status_code == 200


def parse_scan_command_file(filepath):
    """
    解析掃描指令檔 (.txt)，自動提取掃描點數與曝光時間
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到指令檔: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    trg_times_us = []
    for line in lines:
        if line.startswith("TRG,"):
            try:
                us = int(line.split(",")[1])
                trg_times_us.append(us)
            except Exception:
                pass

    n_points = len(trg_times_us)
    if n_points == 0:
        raise ValueError(f"指令檔 {filepath} 中未找到任何 TRG 觸發指令！")

    exposure_sec = trg_times_us[0] / 1e6
    return n_points, exposure_sec


def configure_and_arm_eiger(
    ip=DEFAULT_DCU_IP,
    n_points=34,
    count_time=1.0,
    trigger_mode="exte",
    name_pattern=None,
    disarm_wait_sec=3.0,
    nimages=None,
    ntrigger=None,
    do_arm=True
):
    """
    執行完整的 Disarm -> 等待 -> 設定參數 -> Arm 流程
    """
    eiger = EigerController(ip=ip)

    if ntrigger is None:
        ntrigger = n_points
    if nimages is None:
        nimages = n_points

    print("=" * 65)
    print(f"  EIGER 1M 探測器設定與 Arm 控制 (DCU: {ip})")
    print("=" * 65)

    # 0. 查詢初始狀態
    det_st, fw_st = eiger.get_status()
    print(f"[*] 初始狀態  : Detector = [{det_st}], FileWriter = [{fw_st}]")

    # 1. 嘗試 Disarm
    print(f"[*] 步驟 1: 執行 Disarm...")
    try:
        disarm_res = eiger.disarm()
        print(f"    -> Disarm 成功: {disarm_res}")
    except Exception as e:
        print(f"    -> [提示] Disarm 回應: {e}")

    # 2. 等待穩定
    print(f"[*] 步驟 2: 等待 {disarm_wait_sec:.1f} 秒穩定...")
    time.sleep(disarm_wait_sec)

    # 3. 設定探測器參數
    print(f"[*] 步驟 3: 寫入探測器參數:")
    print(f"    - trigger_mode : {trigger_mode}")
    print(f"    - ntrigger     : {ntrigger}")
    print(f"    - nimages      : {nimages}")
    print(f"    - count_time   : {count_time:.4f} s (frame_time 自動匹配)")

    eiger.set_config("trigger_mode", trigger_mode)
    eiger.set_config("ntrigger", ntrigger)
    eiger.set_config("nimages", nimages)
    eiger.set_config("count_time", count_time)

    # 寫入檔名規則 (若有指定)
    if name_pattern:
        print(f"    - name_pattern : {name_pattern}")
        eiger.set_filewriter_config("name_pattern", name_pattern)

    # 4. 驗證讀回參數
    print(f"[*] 步驟 4: 讀回探測器當前配置驗證:")
    read_trg_mode = eiger.get_config("trigger_mode")
    read_ntrigger = eiger.get_config("ntrigger")
    read_nimages = eiger.get_config("nimages")
    read_count_t = eiger.get_config("count_time")
    read_frame_t = eiger.get_config("frame_time")
    read_fw_pattern = eiger.get_filewriter_config("name_pattern")

    print(f"    [V] trigger_mode : {read_trg_mode}")
    print(f"    [V] ntrigger     : {read_ntrigger}")
    print(f"    [V] nimages      : {read_nimages}")
    print(f"    [V] count_time   : {read_count_t:.6f} s")
    print(f"    [V] frame_time   : {read_frame_t:.6f} s")
    print(f"    [V] name_pattern : {read_fw_pattern}")

    # 5. 執行 Arm
    if do_arm:
        print(f"[*] 步驟 5: 執行 Arm 命令...")
        arm_res = eiger.arm()
        seq_id = "N/A"
        if isinstance(arm_res, dict):
            seq_id = arm_res.get("sequence id", "N/A")
        print(f"    -> Arm 成功！Sequence ID: {seq_id}")

        time.sleep(0.5)
        final_det_st, final_fw_st = eiger.get_status()
        print(f"[*] 當前狀態  : Detector = [{final_det_st}], FileWriter = [{final_fw_st}]")
        print("=" * 65)
        print(">>> 探測器已進入 Arm 待命狀態 (Ready/Acquire)，可隨時執行馬達掃描腳本！ <<<")
        print("=" * 65)
        return seq_id
    else:
        print("=" * 65)
        print(">>> 參數設定完成 (未執行 Arm) <<<")
        print("=" * 65)
        return None


def main():
    parser = argparse.ArgumentParser(description="EIGER 1M 探測器 HTTP 參數設定與 Arm 工具")
    parser.add_argument('--ip', type=str, default=DEFAULT_DCU_IP, help=f"DCU 控制主機 IP (預設: {DEFAULT_DCU_IP})")
    parser.add_argument('--cmd-file', type=str, default=r"Z:\TPS23A_USER\Ptychography\fermat_scan_commands.txt", help="掃描指令檔路徑 (.txt)，若提供將自動解析點數與曝光時間")
    parser.add_argument('--points', type=int, default=None, help="手動指定掃描點數 (預設同時套用於 ntrigger 與 nimages)")
    parser.add_argument('--ntrigger', type=int, default=None, help="單獨指定 ntrigger 觸發數")
    parser.add_argument('--nimages', type=int, default=None, help="單獨指定 nimages 影像張數 (例如: 1000)")
    parser.add_argument('--exposure', type=float, default=None, help="手動指定曝光時間 count_time (秒)")
    parser.add_argument('--trigger-mode', type=str, default="exte", help="觸發模式 (預設: exte)")
    parser.add_argument('--pattern', type=str, default=None, help="FileWriter 檔案命名規則 (例如: 20260821_series_$id)")
    parser.add_argument('--wait', type=float, default=3.0, help="Disarm 後等待秒數 (預設: 3.0)")
    parser.add_argument('--disarm-only', action='store_true', help="僅執行 Disarm")
    parser.add_argument('--no-arm', action='store_true', help="僅設定參數，不執行 Arm")
    parser.add_argument('--status', action='store_true', help="僅查詢探測器狀態與當前設定")

    args = parser.parse_args()
    eiger = EigerController(ip=args.ip)

    if args.status:
        det_st, fw_st = eiger.get_status()
        print(f"Detector State  : {det_st}")
        print(f"FileWriter State: {fw_st}")
        for k in ['trigger_mode', 'ntrigger', 'nimages', 'count_time', 'frame_time', 'photon_energy', 'threshold_energy']:
            print(f"{k:16s}: {eiger.get_config(k)}")
        print(f"{'name_pattern':16s}: {eiger.get_filewriter_config('name_pattern')}")
        return

    if args.disarm_only:
        print(f"[*] 執行 Disarm...")
        res = eiger.disarm()
        print(f"Disarm 結果: {res}")
        return

    # 決定點數與曝光時間
    n_points = args.points
    count_time = args.exposure

    if (n_points is None or count_time is None) and args.cmd_file and os.path.exists(args.cmd_file):
        try:
            file_pts, file_exp = parse_scan_command_file(args.cmd_file)
            if n_points is None:
                n_points = file_pts
            if count_time is None:
                count_time = file_exp
            print(f"[檔案解析] 從 {os.path.basename(args.cmd_file)} 解析得到: 點數 = {n_points}, 曝光時間 = {count_time:.3f} s")
        except Exception as e:
            print(f"[警告] 解析指令檔失敗 ({e})，使用預設值")

    if n_points is None:
        n_points = 34
    if count_time is None:
        count_time = 1.0

    # 執行流程
    configure_and_arm_eiger(
        ip=args.ip,
        n_points=n_points,
        count_time=count_time,
        trigger_mode=args.trigger_mode,
        name_pattern=args.pattern,
        disarm_wait_sec=args.wait,
        nimages=args.nimages,
        ntrigger=args.ntrigger,
        do_arm=not args.no_arm
    )


if __name__ == '__main__':
    main()
