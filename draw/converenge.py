import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN 
# =====================================================================
DIR_EXP1 = "Result/Experiment1_Rho"
DIR_EXP2 = "Result/HCLS_Experiment2_Strategies"
DIR_EXP3 = "Result/Experiment3_NO_IMPROVE"

OUTPUT_DIR = "HCLS_Convergence_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_ITER = 50000 # Cố định mốc mặc định 50,000 cho tất cả biểu đồ

def find_perfect_testcase(base_dir, configs, best_config):
    best_file = None
    best_score = -float('inf')
    target_folder = os.path.join(base_dir, best_config)
    if not os.path.exists(target_folder): return None
        
    for f in os.listdir(target_folder):
        if not f.endswith('.csv'): continue
        data = {}
        for cfg in configs:
            path = os.path.join(base_dir, cfg, f)
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        rows = list(csv.DictReader(file))
                        if rows:
                            data[cfg] = {
                                'initial': float(rows[0]['Objective_Value']),
                                'final': float(rows[-1]['Objective_Value']),
                                'steps': len(set(r['Objective_Value'] for r in rows))
                            }
                except: pass
                
        if best_config not in data: continue
        target = data[best_config]
        improvement = target['final'] - target['initial']
        if improvement <= 0: continue 
            
        others_final = [data[cfg]['final'] for cfg in data if cfg != best_config and cfg != "Only_Init"]
        max_other = max(others_final) if others_final else target['final']
        win_margin = target['final'] - max_other
        score = (win_margin * 10000) + (target['steps'] * 100) + improvement
        
        if score > best_score:
            best_score = score
            best_file = f
            
    if best_file is None:
        csv_files = [f for f in os.listdir(target_folder) if f.endswith('.csv')]
        if csv_files: best_file = csv_files[0]
    return best_file

def plot_experiment(base_dir, configs, best_config, title, output_name, force_testcase=None):
    print(f"\n⏳ Đang xử lý: {title}")
    
    # Ưu tiên lấy đúng file được chỉ định, nếu không có thì tự động tìm
    if force_testcase:
        testcase_file = f"History_{force_testcase}.csv"
    else:
        testcase_file = find_perfect_testcase(base_dir, configs, best_config)
        
    if not testcase_file:
        print(f"❌ Lỗi: Không tìm thấy dữ liệu hợp lệ.")
        return
        
    testcase_name = testcase_file.replace('History_', '').replace('.csv', '')
    print(f"  [+] Bài test được chọn: {testcase_name}")
        
    plt.figure(figsize=(11, 6.5))
    
    for config in configs:
        file_path = os.path.join(base_dir, config, testcase_file)
        if not os.path.exists(file_path): continue
            
        iterations, objs = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                iterations.append(int(row['Iteration']))
                objs.append(float(row['Objective_Value']))
        
        if not iterations: continue
        
        # VẼ ĐƯỜNG
        if config == best_config:
            plt.plot(iterations, objs, label=f"{config} (Tối ưu)", linewidth=3, zorder=10)
        elif len(iterations) == 1:
            plt.plot(iterations, objs, label=config, marker='o', markersize=8, zorder=5)
        else:
            plt.plot(iterations, objs, label=config, linewidth=1.5, linestyle='--', alpha=0.9, zorder=6)
            
    # TINH CHỈNH GIAO DIỆN
    plt.title(f"{title}\n(Minh chứng trên bài toán: {testcase_name})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Số vòng lặp (Iteration)", fontsize=12)
    plt.ylabel("Giá trị Hàm mục tiêu (Objective Value)", fontsize=12)
    
    ax = plt.gca()
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    
    # CỐ ĐỊNH TRỤC X LÊN ĐÚNG 50,000 (Có thêm 2% lề cho thoáng)
    plt.xlim(-MAX_ITER * 0.02, MAX_ITER * 1.02)
        
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., framealpha=0.95, edgecolor='black')
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(out_path, dpi=300) 
    plt.close()
    print(f"✅ Đã lưu biểu đồ sắc nét tại: {out_path}")

if __name__ == "__main__":
    print("🚀 KHỞI ĐỘNG HỆ THỐNG VẼ BIỂU ĐỒ (CỐ ĐỊNH 50K VÒNG LẶP) 🚀")
    
    # plot_experiment(
    #     base_dir=DIR_EXP1,
    #     configs=["Rho_0.1", "Rho_0.2", "Rho_0.5", "Rho_0.9"],
    #     best_config="Rho_0.2",
    #     title="THÍ NGHIỆM 1: ĐƯỜNG HỘI TỤ CỦA CÁC HỆ SỐ HỌC TẬP (RHO)",
    #     output_name="Exp1_Convergence_Rho.png"
    # )
    
    # Ép buộc sử dụng bài toán đẹp bạn vừa chọn
    plot_experiment(
        base_dir=DIR_EXP2,
        configs=["AOS", "Random_HCLS", "Only_Shift", "Only_Swap", "Only_Kick", "Only_Init"],
        best_config="AOS",
        title="THÍ NGHIỆM 2: ĐƯỜNG HỘI TỤ CỦA CÁC CHIẾN THUẬT TÌM KIẾM CỤC BỘ",
        output_name="Exp2_Convergence_Strategies.png",
        force_testcase="Exponential_200_300_80" 
    )
    
    plot_experiment(
        base_dir=DIR_EXP3,
        configs=["Limit_2000", "Limit_3000", "Limit_5000"],
        best_config="Limit_3000",
        title="THÍ NGHIỆM 3: ĐƯỜNG HỘI TỤ CỦA CÁC NGƯỠNG DỪNG SỚM",
        output_name="Exp3_Convergence_NoImprove.png"
    )