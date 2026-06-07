import os
import re
import csv

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN 
# =====================================================================
# Thay bằng thư mục chứa tất cả các file kết quả của bạn
INPUT_DIR = "Result/AOS_LocalSearch" 
OUTPUT_CSV = "BaoCao_TongHop_AOS_TatCa.csv"

def aggregate_all_files():
    print(f"⏳ Đang quét tất cả file trong: {INPUT_DIR}...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_DIR}")
        return

    records = []

    # Quét tất cả file .txt trong thư mục
    for file in os.listdir(INPUT_DIR):
        if not file.endswith('.txt'): continue
        
        filepath = os.path.join(INPUT_DIR, file)
        
        # Mặc định các giá trị
        obj_val, time_sec, status = "N/A", "N/A", "N/A"
        
        # Đọc file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 1. Lấy Điểm
            match_obj = re.search(r'Điểm[\s\w]*:\s*([0-9\.]+)', content, re.IGNORECASE)
            if match_obj:
                obj_val = int(float(match_obj.group(1)))
                
            # 2. Lấy Thời gian
            match_time = re.search(r'Thời gian:\s*([0-9\.]+)\s*(giây|s|ms)', content, re.IGNORECASE)
            if match_time:
                val = float(match_time.group(1))
                unit = match_time.group(2).lower()
                if 'ms' in unit: val /= 1000.0
                time_sec = round(val, 4)
                
            # 3. Lấy Trạng thái
            match_status = re.search(r'Trạng thái:\s*([A-Za-z]+)', content, re.IGNORECASE)
            if match_status:
                status = match_status.group(1).upper()

        records.append({
            'File_Name': file,
            'Status': status,
            'Objective': obj_val,
            'Time': time_sec
        })

    if not records:
        print("❌ Không tìm thấy file txt nào.")
        return

    # Sắp xếp theo tên file cho dễ nhìn
    records.sort(key=lambda x: x['File_Name'])

    # Ghi ra CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Tên File", "Trạng thái", "Điểm tối ưu (Fitness)", "Thời gian (s)"])
        
        for r in records:
            writer.writerow([r['File_Name'], r['Status'], r['Objective'], r['Time']])

    print(f"✅ HOÀN TẤT! Đã tổng hợp {len(records)} file vào: {OUTPUT_CSV}")

if __name__ == "__main__":
    aggregate_all_files()