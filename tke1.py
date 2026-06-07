import os
import re

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# =====================================================================
DIR_GREEDY = "Result/Greedy_Heuristic"
DIR_GUROBI = "Result/Gurobi"

def get_objective_value(filepath):
    """Rút trích Điểm tối ưu từ file txt"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'Điểm tối ưu:\s*([0-9\.]+)', content, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                return int(val) if val > 0 else None
    except Exception:
        pass
    return None

def check_greedy_optimality():
    print("⏳ Đang quét dữ liệu để thống kê số bài Greedy đạt Tối ưu...")
    
    if not os.path.exists(DIR_GREEDY) or not os.path.exists(DIR_GUROBI):
        print("❌ Lỗi: Không tìm thấy thư mục Greedy hoặc Gurobi.")
        return

    total_compared = 0
    optimal_hits = 0

    # Lấy danh sách file của Gurobi làm chuẩn
    for file in os.listdir(DIR_GUROBI):
        if not file.endswith('.txt'): continue
            
        # Lấy tên bài gốc (Ví dụ: Gurobi_Test_100.txt -> Test_100)
        base_name = re.sub(r'^Gurobi_', '', file, flags=re.IGNORECASE).replace('.txt', '')
        
        gurobi_path = os.path.join(DIR_GUROBI, file)
        gurobi_score = get_objective_value(gurobi_path)
        
        # Tìm file tương ứng bên thư mục Greedy
        greedy_score = None
        for g_file in os.listdir(DIR_GREEDY):
            if base_name in g_file: # Khớp tên bài
                greedy_path = os.path.join(DIR_GREEDY, g_file)
                greedy_score = get_objective_value(greedy_path)
                break
        
        # Nếu cả 2 đều có điểm hợp lệ thì mang ra so sánh
        if gurobi_score is not None and greedy_score is not None:
            total_compared += 1
            if greedy_score == gurobi_score:
                optimal_hits += 1
        else:
            # THÊM ĐOẠN NÀY ĐỂ BẮT LỖI FILE BỊ BỎ QUA
            print(f"⚠️ Đã bỏ qua bài: {base_name}")
            if gurobi_score is None:
                print(f"   -> Lý do: Gurobi không có điểm hợp lệ (hoặc điểm = 0).")
            if greedy_score is None:
                print(f"   -> Lý do: Không tìm thấy file bên Greedy, hoặc Greedy điểm = 0.")

    # IN KẾT QUẢ RA MÀN HÌNH
    print("\n" + "="*55)
    print(" 📊 THỐNG KÊ SỐ BÀI GREEDY ĐẠT ĐỈNH TỐI ƯU")
    print("="*55)
    if total_compared > 0:
        print(f"- Tổng số bài đem ra đối chiếu: {total_compared} bài")
        print(f"- Số bài Greedy đạt TỐI ƯU (Bằng điểm Gurobi): {optimal_hits} bài")
        print(f"- Tỷ lệ Greedy giải chính xác tuyệt đối: {(optimal_hits / total_compared) * 100:.2f}%")
    else:
        print("❌ Không tìm thấy bài toán nào khớp nhau giữa 2 thư mục để so sánh.")
    print("="*55)

if __name__ == "__main__":
    check_greedy_optimality()