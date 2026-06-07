import os
import csv
import re
from collections import defaultdict
import statistics

INPUT_DIR = "Result/Experiment1_Rho" 
OUTPUT_DIR = "HCLS_Experiment1_Rho"  

os.makedirs(OUTPUT_DIR, exist_ok=True)
FILE_PIVOT = os.path.join(OUTPUT_DIR, "Summary_Pivot_Table.csv")

# data[N][Rho] = {'obj': [], 'time_ms': []}
data = defaultdict(lambda: defaultdict(lambda: {'obj': [], 'time_ms': []}))
all_rhos = set()

def main():
    print("⏳ Đang tổng hợp dữ liệu Thực nghiệm 1 (Rho Tuning)...")
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_DIR}.")
        return

    rho_dirs = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    for r_dir in rho_dirs:
        strat_dir = os.path.join(INPUT_DIR, r_dir)
        all_rhos.add(r_dir)
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
                    
                    data[N][r_dir]['obj'].append(obj_val)
                    data[N][r_dir]['time_ms'].append(time_ms)

    # XUẤT PIVOT TABLE
    sorted_rhos = sorted(list(all_rhos))
    with open(FILE_PIVOT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['Size_N']
        for r in sorted_rhos:
            header.extend([f"{r}_Avg_Obj", f"{r}_Avg_Time_ms"])
        writer.writerow(header)
        
        for N in sorted(data.keys()):
            row = [N]
            for r in sorted_rhos:
                objs = data[N][r]['obj']
                times = data[N][r]['time_ms']
                if objs:
                    row.extend([round(statistics.mean(objs), 2), round(statistics.mean(times), 2)])
                else:
                    row.extend(["N/A", "N/A"])
            writer.writerow(row)
    print(f"✅ Đã lưu bảng ma trận chéo tại: {FILE_PIVOT}")

if __name__ == "__main__":
    main()