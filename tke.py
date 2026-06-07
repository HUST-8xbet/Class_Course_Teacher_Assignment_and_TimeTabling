import os
import re
import csv

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC CỦA 3 BỘ GIẢI
# =====================================================================
SOLVERS = {
    "CP_SAT": "Result/CPSAT",        # Sửa lại nếu tên thư mục của bạn khác
    "Gurobi": "Result/Gurobi",
    "pywraplp": "Result/ORTools"     # Hoặc Result/pywraplp
}

OUTPUT_CSV = "BaoCao_ExactSolvers_TongHop_ms.csv"

def parse_result_file(filepath):
    """Đọc file txt, lấy điểm và thời gian. Quy chuẩn thời gian về MILI-GIÂY (ms)."""
    obj_val = "N/A"
    exec_time = "N/A"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Lấy điểm tối ưu
            match_obj = re.search(r'Điểm tối ưu:\s*([0-9\.]+)', content, re.IGNORECASE)
            if match_obj:
                val = float(match_obj.group(1))
                # Nếu điểm = 0 tức là thuật toán sập/hết giờ mà không tìm được nghiệm nào
                obj_val = int(val) if val > 0 else "N/A"
                
            # Lấy thời gian và ép về Mili-giây (ms)
            match_time = re.search(r'Thời gian:\s*([0-9\.]+)\s*(giây|s|ms)', content, re.IGNORECASE)
            if match_time:
                val = float(match_time.group(1))
                unit = match_time.group(2).lower()
                # Nếu đơn vị là giây thì nhân 1000 ra mili-giây
                if unit in ['giây', 's']:
                    val *= 1000.0
                exec_time = round(val, 2) # Làm tròn 2 chữ số vì ms đã rất chi tiết rồi
                
    except Exception:
        pass
        
    return obj_val, exec_time

def generate_scientific_csv():
    print("⏳ Đang quét dữ liệu từ các bộ giải và tổng hợp thành bảng chéo (Đơn vị: ms)...")
    
    # Biến lưu trữ tổng hợp: aggregated_data[tên_bài_toán] = {'N': 100, 'Gurobi': {'obj': X, 'time': Y}, ...}
    aggregated_data = {}
    
    for solver_name, folder_path in SOLVERS.items():
        if not os.path.exists(folder_path):
            print(f"  [!] Cảnh báo: Không tìm thấy thư mục {folder_path}")
            continue
            
        for file in os.listdir(folder_path):
            if not file.endswith('.txt'): continue
            
            # 1. Trích xuất Tên bài toán gốc (Loại bỏ các tiền tố CPSAT_, Gurobi_, pywraplp_ bị dính ở đầu file)
            base_test_name = re.sub(r'^(CPSAT|Gurobi|pywraplp|ORTools)_', '', file, flags=re.IGNORECASE)
            base_test_name = base_test_name.replace('.txt', '')
            
            # 2. Tìm kích thước N để lát nữa sắp xếp
            match_n = re.search(r'_(\d+)_', base_test_name)
            N = int(match_n.group(1)) if match_n else 0
            
            if base_test_name not in aggregated_data:
                aggregated_data[base_test_name] = {'N': N}
                
            filepath = os.path.join(folder_path, file)
            obj, time_ms = parse_result_file(filepath)
            
            # Lưu kết quả vào dictionary
            aggregated_data[base_test_name][solver_name] = {
                'obj': obj,
                'time': time_ms
            }

    if not aggregated_data:
        print("❌ Không có dữ liệu để tạo CSV.")
        return

    # Sắp xếp các bài test theo kích thước N tăng dần, nếu N bằng nhau thì xếp theo Tên bài
    sorted_tests = sorted(aggregated_data.keys(), key=lambda k: (aggregated_data[k]['N'], k))
    
    # Định nghĩa các cột cho file CSV chuẩn khoa học (Đã sửa Header thành ms)
    csv_columns = [
        "Kich_Thuoc_N", 
        "Ten_Bai_Toan", 
        "CP_SAT_Diem", "CP_SAT_Time(ms)", 
        "Gurobi_Diem", "Gurobi_Time(ms)", 
        "pywraplp_Diem", "pywraplp_Time(ms)"
    ]
    
    # Ghi dữ liệu ra CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns) # Ghi Header
        
        for test_name in sorted_tests:
            row_data = aggregated_data[test_name]
            
            # Lấy dữ liệu từng bộ giải, nếu bộ giải không có bài test này thì mặc định là N/A
            cpsat = row_data.get("CP_SAT", {'obj': 'N/A', 'time': 'N/A'})
            gurobi = row_data.get("Gurobi", {'obj': 'N/A', 'time': 'N/A'})
            pywrap = row_data.get("pywraplp", {'obj': 'N/A', 'time': 'N/A'})
            
            row = [
                row_data['N'],
                test_name,
                cpsat['obj'], cpsat['time'],
                gurobi['obj'], gurobi['time'],
                pywrap['obj'], pywrap['time']
            ]
            writer.writerow(row)

    print(f"\n✅ HOÀN TẤT! File báo cáo Phụ lục (đơn vị ms) đã được tạo thành công tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_scientific_csv()