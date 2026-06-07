import os
import re
import statistics
import matplotlib.pyplot as plt

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN CÁC THƯ MỤC KẾT QUẢ (FILE TXT)
# =====================================================================
SOLVERS = {
    "Greedy_Heuristic": {
        "dir": "Result/Greedy_Heuristic", 
        "color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 1.5
    },
    # "Gurobi": {
    #     "dir": "Result/Gurobi", 
    #     "color": "#d62728", "marker": "o", "linestyle": "-", "linewidth": 1.5
    # },
    # "pywraplp": {
    #     "dir": "Result/pywraplp", 
    #     "color": "#2ca02c", "marker": "o", "linestyle": "-", "linewidth": 1.5
    # }
}
OUTPUT_FILE = "Figure/Greedy_Heuristic_Time_Scaling_ms.png"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
MAX_N_LIMIT = 1000 # Cố định giới hạn tối đa cho trục X

def extract_time_from_txt(filepath):
    """Rút trích thời gian và ĐỒNG BỘ TOÀN BỘ VỀ MILI-GIÂY (ms)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'Thời gian:\s*([0-9\.]+)\s*(giây|s|ms)', content, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                unit = match.group(2).lower()
                # Nếu đơn vị là giây thì nhân 1000 để ra mili-giây
                if unit in ['giây', 's']:
                    val *= 1000.0 
                return val
    except Exception:
        pass
    return None

def plot_exact_solvers_scaling():
    print(f"⏳ Đang quét dữ liệu và chuẩn bị vẽ biểu đồ (Trục Y: ms, Trục X: Cố định {MAX_N_LIMIT})...")
    
    from collections import defaultdict
    data_by_N = defaultdict(lambda: defaultdict(list))
    all_solvers = list(SOLVERS.keys())
    
    for solver_name, config in SOLVERS.items():
        folder = config["dir"]
        if not os.path.exists(folder):
            print(f"  [!] Bỏ qua {solver_name} vì không tìm thấy thư mục: {folder}")
            continue
            
        for f in os.listdir(folder):
            if not f.endswith('.txt'): continue
                
            match = re.search(r'_(\d+)_', f)
            if not match: continue
            N = int(match.group(1))
            
            # Lọc bỏ các bài test vượt quá giới hạn 300
            if N > MAX_N_LIMIT: continue
            
            filepath = os.path.join(folder, f)
            exec_time = extract_time_from_txt(filepath)
            
            if exec_time is not None:
                data_by_N[N][solver_name].append(exec_time)

    if not data_by_N:
        print("❌ Không có dữ liệu hợp lệ để vẽ biểu đồ.")
        return

    sorted_N = sorted(data_by_N.keys())
    # Lấy N lớn nhất thực tế (hoặc bằng 300) để làm mốc so sánh Time Out
    target_max_N = MAX_N_LIMIT if MAX_N_LIMIT in sorted_N else max(sorted_N)
    
    plt.figure(figsize=(10, 6.5))
    
    for solver_name in all_solvers:
        config = SOLVERS[solver_name]
        valid_N = []
        valid_times = []
        
        for N in sorted_N:
            times = data_by_N[N].get(solver_name, [])
            if times:
                valid_N.append(N)
                valid_times.append(statistics.mean(times))
        
        if not valid_N: continue
        
        plt.plot(valid_N, valid_times, label=solver_name,
                 color=config["color"], marker=config["marker"], 
                 linestyle=config["linestyle"], linewidth=config["linewidth"], markersize=4)

        # Đánh dấu Time Out nếu bộ giải dừng lại trước mốc target_max_N
        if max(valid_N) < target_max_N:
            last_n = valid_N[-1]
            last_time = valid_times[-1]
            plt.plot(last_n, last_time, marker='x', markersize=12, 
                     color=config["color"], markeredgewidth=3)
            plt.annotate('Quá tải', xy=(last_n, last_time), 
                         xytext=(10, -15), textcoords='offset points', 
                         color=config["color"], fontsize=10, fontweight='bold')

    # TINH CHỈNH GIAO DIỆN
    plt.title(f"TỐC ĐỘ THỰC THI CỦA THUẬT TOÁN THAM LAM (GREEDY)\n", 
              fontsize=14, fontweight='bold', pad=15)
    
    plt.xlabel("Kích thước bài toán (Số lớp N)", fontsize=12)
    plt.ylabel("Thời gian chạy trung bình (Mili-giây)", fontsize=12) 
    
    # Tắt log-scale nếu chỉ vẽ mỗi Greedy (để thấy rõ đường tăng thời gian thực tế)
    # plt.yscale('log') # Đã comment lại
    
    # CỐ ĐỊNH TRỤC X: Nới rộng lề tỷ lệ thuận với MAX_N_LIMIT (ví dụ 5% của 1000 = 50)
    margin = int(MAX_N_LIMIT * 0.05)
    plt.xlim(min(sorted_N) - margin, MAX_N_LIMIT + margin)
    
    # Ép trục Y luôn bắt đầu từ 0 để thấy rõ Greedy chạy nhanh cỡ nào
    plt.ylim(bottom=0)
    
    plt.grid(True, which="major", linestyle='-', alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', alpha=0.3)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., framealpha=0.95, edgecolor='black')
    plt.tight_layout()

    # Đổi tên file output cho chuẩn xác
    OUTPUT_FILE = "Figure/Greedy_Time_Scaling_ms.png"
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.close()
    print(f"✅ Đã vẽ xong! Biểu đồ Greedy được lưu tại: {OUTPUT_FILE}")

# THÊM 2 DÒNG NÀY VÀO CUỐI CÙNG ĐỂ GỌI HÀM CHẠY:
if __name__ == "__main__":
    plot_exact_solvers_scaling()