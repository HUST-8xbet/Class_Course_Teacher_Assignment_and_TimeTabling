import os
import re
import pandas as pd
import csv

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN CHUẨN THEO ẢNH CỦA BẠN
# =====================================================================
BASE_DIR = "Result/HCLS_Experiment2_Strategies"
OUTPUT_CSV = "BaoCao_SoSanh_Experiment2_Strategies.csv"

# Danh sách các thư mục chứa kịch bản
RHO_FOLDERS = ["AOS", "Only_Init", "Only_Kick", "Only_Shift", "Only_Swap", "Random_HCLS"]

def aggregate_experiment2():
    print(f"⏳ Đang tổng hợp dữ liệu từ {BASE_DIR}...")
    
    if not os.path.exists(BASE_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {BASE_DIR}")
        return

    aggregated_data = {}

    # Quét qua từng thư mục Rho
    for rho in RHO_FOLDERS:
        folder_path = os.path.join(BASE_DIR, rho)
        if not os.path.exists(folder_path):
            print(f"  [!] Cảnh báo: Không thấy thư mục {folder_path}, sẽ bỏ qua.")
            continue
            
        for file in os.listdir(folder_path):
            # Lọc chỉ lấy các file CSV lịch sử
            if not file.endswith('.csv') or not file.startswith('History_'):
                continue
                
            # Trích xuất tên bài toán (VD: History_Adversarial_200_300_80.csv -> Adversarial_200_300_80)
            base_test_name = file.replace('History_', '').replace('.csv', '')
            
            # Lấy N để sắp xếp danh sách từ bé đến lớn
            match_n = re.search(r'_(\d+)_', base_test_name)
            N = int(match_n.group(1)) if match_n else 0
            
            if base_test_name not in aggregated_data:
                aggregated_data[base_test_name] = {'N': N}
                
            filepath = os.path.join(folder_path, file)
            
            # Dùng Pandas đọc file CSV để lấy dòng cuối cùng (Kết quả lúc thuật toán dừng)
            try:
                df = pd.read_csv(filepath)
                if not df.empty:
                    final_obj = int(df['Objective_Value'].iloc[-1])
                    final_time = round(float(df['Time_Seconds'].iloc[-1]), 2)
                else:
                    final_obj, final_time = "N/A", "N/A"
            except Exception:
                final_obj, final_time = "N/A", "N/A"
                
            # Lưu dữ liệu vào dictionary tổng hợp
            aggregated_data[base_test_name][rho] = {
                'obj': final_obj,
                'time': final_time
            }

    if not aggregated_data:
        print("❌ Không có dữ liệu để tạo CSV.")
        return

    # Sắp xếp các bài toán theo kích thước N
    sorted_tests = sorted(aggregated_data.keys(), key=lambda k: (aggregated_data[k]['N'], k))
    
    # Định nghĩa Header cho file báo cáo
    csv_columns = [
        "Kích thước (N)", "Tên Bài Toán",
        "Điểm_AOS", "Time_AOS(s)",
        "Điểm_Only_Init", "Time_Only_Init(s)",
        "Điểm_Only_Kick", "Time_Only_Kick(s)",
        "Điểm_Only_Shift", "Time_Only_Shift(s)",
        "Điểm_Only_Swap", "Time_Only_Swap(s)",
        "Điểm_Random_HCLS", "Time_Random_HCLS(s)"
    ]
    
    # Ghi ra file CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        
        for test_name in sorted_tests:
            row_data = aggregated_data[test_name]
            row = [row_data['N'], test_name]
            
            # Lấy số liệu từng Rho nhét vào các cột tương ứng, nếu bài nào không có thì để N/A
            for rho in RHO_FOLDERS:
                rho_data = row_data.get(rho, {'obj': 'N/A', 'time': 'N/A'})
                row.extend([rho_data['obj'], rho_data['time']])
                
            writer.writerow(row)

    print(f"\n✅ HOÀN TẤT! File bảng chéo so sánh Experiment 2 đã được tạo tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    aggregate_experiment2()