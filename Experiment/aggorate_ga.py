import os
import re
import csv

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN 
# =====================================================================
BASE_DIR = "Result/Pure_GA_Timetabling_pmx_mut0p75_mixed_tk3_initheuristic_seed42_ga_pmx_mixed"
OUTPUT_CSV = "BaoCao_SoSanh_CacBoDuLieu.csv"

# Danh sách các thư mục chứa loại dữ liệu
DATASETS = ["Adversarial", "Exponential", "Uniform", "Poisson", "Gaussian", "hustack"]

def aggregate_txt_results():
    print(f"⏳ Đang tổng hợp dữ liệu file TXT từ {BASE_DIR}...")
    
    if not os.path.exists(BASE_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {BASE_DIR}")
        return

    aggregated_data = {}

    # Quét qua từng thư mục loại dữ liệu
    for dataset in DATASETS:
        folder_path = os.path.join(BASE_DIR, dataset)
        if not os.path.exists(folder_path):
            print(f"  [!] Cảnh báo: Không thấy thư mục {folder_path}, sẽ bỏ qua.")
            continue
            
        for file in os.listdir(folder_path):
            if not file.endswith('.txt'):
                continue
                
            # Rút trích cụm kích thước để làm mốc gom nhóm (VD: 20_30_10)
            # Bất kể tên file có chứa chữ Adversarial hay Uniform thì chung size sẽ gộp 1 dòng
            match_size = re.search(r'(\d+_\d+_\d+)', file)
            if not match_size:
                continue # Bỏ qua nếu tên file không đúng chuẩn
                
            size_key = match_size.group(1)
            N = int(size_key.split('_')[0]) # Lấy N để lát sắp xếp
            
            if size_key not in aggregated_data:
                aggregated_data[size_key] = {'N': N}
                
            filepath = os.path.join(folder_path, file)
            final_obj, final_time = "N/A", "N/A"
            
            # Đọc nội dung file txt
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Bắt dòng "Điểm tối ưu/tốt nhất tìm được: ..."
                    match_obj = re.search(r'Điểm[\s\w/]*:\s*([0-9\.]+)', content, re.IGNORECASE)
                    if match_obj:
                        final_obj = int(float(match_obj.group(1)))
                        
                    # Bắt dòng "Thời gian: ... giây"
                    match_time = re.search(r'Thời gian:\s*([0-9\.]+)\s*(giây|s|ms)', content, re.IGNORECASE)
                    if match_time:
                        val = float(match_time.group(1))
                        unit = match_time.group(2).lower()
                        if 'ms' in unit: val /= 1000.0
                        final_time = round(val, 4)
            except Exception:
                pass
                
            # Lưu dữ liệu vào dictionary tổng hợp
            aggregated_data[size_key][dataset] = {
                'obj': final_obj,
                'time': final_time
            }

    if not aggregated_data:
        print("❌ Không có dữ liệu để tạo CSV.")
        return

    # Sắp xếp các bài toán theo kích thước N
    sorted_keys = sorted(aggregated_data.keys(), key=lambda k: aggregated_data[k]['N'])
    
    # Định nghĩa Header linh hoạt theo mảng DATASETS
    csv_columns = ["Kích thước (N_M_T)"]
    for ds in DATASETS:
        csv_columns.extend([f"Điểm_{ds}", f"Time_{ds}(s)"])
    
    # Ghi ra file CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)
        
        for key in sorted_keys:
            row_data = aggregated_data[key]
            row = [key]
            
            # Lấy số liệu từng bộ dữ liệu nhét vào cột tương ứng
            for ds in DATASETS:
                ds_data = row_data.get(ds, {'obj': 'N/A', 'time': 'N/A'})
                row.extend([ds_data['obj'], ds_data['time']])
                
            writer.writerow(row)

    print(f"\n✅ HOÀN TẤT! File bảng chéo so sánh bộ dữ liệu đã được tạo tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    aggregate_txt_results()