import os
import csv
import re
from collections import defaultdict
import statistics

INPUT_DIR = "Result/Experiment3_NO_IMPROVE" 
OUTPUT_DIR = "HCLS_Experiment3_NO_IMPROVE"  

os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PIVOT = os.path.join(OUTPUT_DIR, "Summary_Pivot_Table.csv")

data = defaultdict(lambda: defaultdict(lambda: {'obj': [], 'time_ms': []}))
all_limits = set()

def main():
    print("⏳ Đang tổng hợp dữ liệu Thực nghiệm 3 (Max No Improve)...")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_DIR}.")
        return

    limit_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for l_dir in limit_dirs:
        strat_dir = os.path.join(INPUT_DIR, l_dir)
        all_limits.add(l_dir)
        csv_files = [f for f in os.listdir(strat_dir) if f.endswith('.csv')]
        
        for file in csv_files:
            match = re.search(r'_(\d+)_', file)
            if not match: continue
            N = int(match.group(1))
            
            with open(os.path.join(strat_dir, file), 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
                if rows:
                    last_row = rows[-1]
                    obj_val = float(last_row['Objective_Value'])
                    time_ms = float(last_row['Time_Seconds']) * 1000 
                    
                    data[N][l_dir]['obj'].append(obj_val)
                    data[N][l_dir]['time_ms'].append(time_ms)

    # XUẤT PIVOT TABLE
    # Sắp xếp limit theo thứ tự số học (2000 -> 3000 -> 5000)
    sorted_limits = sorted(list(all_limits), key=lambda x: int(x.split('_')[1]))
    with open(FILE_PIVOT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Size_N']
        for l in sorted_limits:
            header.extend([f"{l}_Avg_Obj", f"{l}_Avg_Time_ms"])
        writer.writerow(header)
        
        for N in sorted(data.keys()):
            row = [N]
            for l in sorted_limits:
                objs = data[N][l]['obj']
                times = data[N][l]['time_ms']
                if objs:
                    row.extend([round(statistics.mean(objs), 2), round(statistics.mean(times), 2)])
                else:
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)
    print(f"✅ Đã lưu bảng ma trận chéo tại: {FILE_PIVOT}")

if __name__ == "__main__":
    main()