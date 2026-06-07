import os
import csv
import re
from collections import defaultdict
import statistics

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN 
# =====================================================================
INPUT_DIR = "Result/HCLS_Experiment2_Strategies" # Thư mục chứa kết quả ở bước trước
OUTPUT_DIR = "HCLS_Experiment2_Strategies"  # Thư mục mới xuất báo cáo

os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_DETAIL = os.path.join(OUTPUT_DIR, "Summary_Detail.csv")
FILE_PIVOT = os.path.join(OUTPUT_DIR, "Summary_Pivot_Table.csv")

# data[N][Strategy] = {'obj': [], 'time_ms': []}
data = defaultdict(lambda: defaultdict(lambda: {'obj': [], 'time_ms': []}))
all_strategies = set()

def main():
    print("⏳ Đang quét và tổng hợp dữ liệu Thực nghiệm 2 (Thời gian tính bằng ms)...")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_DIR}.")
        return

    # 1. QUÉT DỮ LIỆU TỪ CÁC FILE CSV HISTORY
    strategies = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for strat in strategies:
        strat_dir = os.path.join(INPUT_DIR, strat)
        all_strategies.add(strat)
        csv_files = [f for f in os.listdir(strat_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            match = re.search(r'_(\d+)_', file)
            if not match:
                continue
            N = int(match.group(1))
            
            filepath = os.path.join(strat_dir, file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    obj_val = float(last_row['Objective_Value'])
                    
                    # CHUYỂN ĐỔI SANG MILI-GIÂY (Nhân với 1000)
                    time_sec = float(last_row['Time_Seconds'])
                    time_ms = time_sec * 1000 
                    
                    data[N][strat]['obj'].append(obj_val)
                    data[N][strat]['time_ms'].append(time_ms)

    if not data:
        print("⚠️ Không tìm thấy file dữ liệu hợp lệ nào để tổng hợp!")
        return

    # 2. XUẤT FILE 1: BẢNG CHI TIẾT DỌC
    with open(FILE_DETAIL, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Đổi tên cột thành Avg_Time_ms
        writer.writerow(['Size_N', 'Strategy', 'Avg_Objective', 'Avg_Time_ms', 'Num_Testcases'])
        
        for N in sorted(data.keys()):
            for strat in sorted(data[N].keys()):
                objs = data[N][strat]['obj']
                times = data[N][strat]['time_ms']
                if objs:
                    avg_obj = statistics.mean(objs)
                    avg_time = statistics.mean(times)
                    # Giữ lại 2 chữ số thập phân cho ms là quá đủ độ chính xác
                    writer.writerow([N, strat, round(avg_obj, 2), round(avg_time, 2), len(objs)])

    # 3. XUẤT FILE 2: BẢNG MA TRẬN CHÉO (PIVOT TABLE)
    sorted_strats = sorted(list(all_strategies))
    
    with open(FILE_PIVOT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Tạo Header với chữ ms
        header = ['Size_N']
        for strat in sorted_strats:
            header.extend([f"{strat}_Avg_Obj", f"{strat}_Avg_Time_ms"])
        writer.writerow(header)
        
        # Ghi dữ liệu
        for N in sorted(data.keys()):
            row = [N]
            for strat in sorted_strats:
                objs = data[N][strat]['obj']
                times = data[N][strat]['time_ms']
                if objs:
                    row.extend([round(statistics.mean(objs), 2), round(statistics.mean(times), 2)])
                else:
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)

    print(f"✅ ĐÃ TỔNG HỢP XONG!")
    print(f"  [+] Dữ liệu thời gian đã được quy đổi toàn bộ sang Mili-giây (ms).")

if __name__ == "__main__":
    main()