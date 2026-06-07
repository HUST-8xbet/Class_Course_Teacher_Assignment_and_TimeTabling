import os
import re

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN 
# =====================================================================
INPUT_DIR = "Result/ORTools" # Thư mục chứa các file .txt của pywraplp
OUTPUT_FILE = "Summary_Pywraplp_Results.txt"

# Từ điển dịch mã trạng thái của Pywraplp sang chữ
STATUS_MAPPING = {
    "0": "OPTIMAL",
    "1": "FEASIBLE",
    "2": "INFEASIBLE",
    "3": "UNBOUNDED",
    "4": "ABNORMAL",
    "6": "NOT_SOLVED"
}

def aggregate_summary():
    print("⏳ Đang thống kê tổng hợp từ các file TXT của pywraplp...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy thư mục {INPUT_DIR}.")
        return

    total_files = 0
    solved_count = 0
    optimal_count = 0
    total_time = 0.0
    time_records_count = 0

    # Quét tất cả các file txt trong thư mục
    for file in os.listdir(INPUT_DIR):
        if not file.endswith('.txt'): continue
            
        filepath = os.path.join(INPUT_DIR, file)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            total_files += 1
            
            # 1. Kiểm tra trạng thái
            status_code = "UNKNOWN"
            raw_code = "" 
            
            # ĐÃ SỬA: Thêm .*? để bắt được cả "Trạng thái nghiệm:" hoặc "Trạng thái:"
            match_status = re.search(r'Trạng thái.*?:\s*(\d+)', content, re.IGNORECASE)
            if match_status:
                raw_code = match_status.group(1)
                status_code = STATUS_MAPPING.get(raw_code, f"CODE_{raw_code}")
            
            # Thống kê số lượng 
            if raw_code in ["0", "1"]: 
                solved_count += 1
            if raw_code == "0":
                optimal_count += 1
                
            # 2. Cộng dồn thời gian
            match_time = re.search(r'Thời gian:\s*([0-9\.]+)\s*(giây|s|ms)', content, re.IGNORECASE)
            if match_time:
                exec_time = float(match_time.group(1))
                unit = match_time.group(2).lower()
                if 'ms' in unit:
                    exec_time /= 1000.0 # Quy đổi ms ra giây
                
                total_time += exec_time
                time_records_count += 1

    if total_files == 0:
        print("❌ Không tìm thấy dữ liệu hợp lệ trong thư mục.")
        return

    # Tính thời gian trung bình
    avg_time = (total_time / time_records_count) if time_records_count > 0 else 0.0

    # Xuất báo cáo ra file TXT
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=== BÁO CÁO TỔNG HỢP KẾT QUẢ PYWRAPLP ===\n")
        f.write(f"- Tổng số bài test đã quét: {total_files} bài\n")
        f.write(f"- Số bài giải thành công (Feasible + Optimal): {solved_count} bài\n")
        f.write(f"- Số bài đạt mức Tối ưu tuyệt đối (Optimal): {optimal_count} bài\n")
        f.write(f"- Thời gian chạy trung bình: {avg_time:.4f} giây\n")
        f.write("==========================================\n")

    print(f"✅ HOÀN TẤT! Bản tóm tắt đã được lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    aggregate_summary()