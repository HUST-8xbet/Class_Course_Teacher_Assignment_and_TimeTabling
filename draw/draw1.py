import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# CẤU HÌNH ĐẦU VÀO & ĐẦU RA
# ==========================================
CSV_DIR = "Result/AOS_LocalSearch" # Thư mục chứa file CSV
OUTPUT_DIR = "Figure"              # Thư mục lưu ảnh xuất ra
TARGET_N = 500                     # Kích thước bài toán bạn muốn lọc

if not os.path.exists(CSV_DIR):
    print(f"LỖI: Không tìm thấy thư mục {CSV_DIR}")
    exit()

# Tự động tạo thư mục Figure nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Quét và lọc danh sách các file CSV thuộc nhóm N=500
csv_files = []
for file in os.listdir(CSV_DIR):
    # Dùng chuỗi "_{TARGET_N}_" để bắt chính xác định dạng tên file (VD: History_Ten_500_...)
    if file.startswith("History_") and file.endswith(".csv") and f"_{TARGET_N}_" in file:
        csv_files.append(file)

print(f"🎯 Tìm thấy {len(csv_files)} file CSV thuộc nhóm N={TARGET_N}.\n")

# 2. Vòng lặp tự động vẽ biểu đồ cho từng file
for file in csv_files:
    file_path = os.path.join(CSV_DIR, file)
    
    # Trích xuất tên gốc để làm tiêu đề và tên file ảnh (Bỏ đi chữ History_ và .csv)
    basename = file.replace("History_", "").replace(".csv", "")
    df = pd.read_csv(file_path)
    
    print(f"Đang xử lý và vẽ ảnh cho: {basename}...")

    # ==========================================
    # BIỂU ĐỒ 1: SỰ THAY ĐỔI HÀM FITNESS
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.step(df['Iteration'], df['Objective_Value'], where='post', color='#e74c3c', linewidth=2.5)
    
    plt.title(f'Sự thay đổi Hàm Fitness qua các vòng lặp\n(Testcase: {basename})', fontsize=14, fontweight='bold')
    plt.xlabel('Số vòng lặp (Iterations)', fontsize=12)
    plt.ylabel('Hàm Fitness', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Đánh dấu điểm kết thúc
    final_iter = df['Iteration'].iloc[-1]
    final_obj = df['Objective_Value'].iloc[-1]
    plt.plot(final_iter, final_obj, 'o', color='black', markersize=6)
    plt.annotate(f' Điểm dừng: {final_obj}', (final_iter, final_obj), textcoords="offset points", xytext=(-10,10), ha='right')

    # Tự động xuất file ảnh vào thư mục Figure
    out_fit = os.path.join(OUTPUT_DIR, f"Chart_Fitness_{basename}.png")
    plt.tight_layout()
    plt.savefig(out_fit, dpi=150)
    plt.close() # Đóng plot lại để giải phóng RAM

    # ==========================================
    # BIỂU ĐỒ 2: SỰ THAY ĐỔI ĐIỂM SỐ TOÁN TỬ
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.plot(df['Iteration'], df['W_Shift'], label='Shift (Trượt kíp)', color='#3498db', alpha=0.8, linewidth=2)
    plt.plot(df['Iteration'], df['W_Swap'], label='Swap (Đổi Thầy)', color='#2ecc71', alpha=0.8, linewidth=2)
    plt.plot(df['Iteration'], df['W_Kick'], label='Kick (Đá & Nhét)', color='#9b59b6', alpha=1.0, linewidth=2.5)
    
    plt.title(f'Sự thay đổi điểm số Toán tử (AOS)\n(Testcase: {basename})', fontsize=14, fontweight='bold')
    plt.xlabel('Số vòng lặp (Iterations)', fontsize=12)
    plt.ylabel('Trọng số (Weight)', fontsize=12)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Tự động xuất file ảnh vào thư mục Figure
    out_weight = os.path.join(OUTPUT_DIR, f"Chart_Weights_{basename}.png")
    plt.tight_layout()
    plt.savefig(out_weight, dpi=150)
    plt.close() # Đóng plot

print(f"\n✅ ĐÃ HOÀN TẤT! Toàn bộ biểu đồ đã được lưu gọn gàng vào thư mục: {OUTPUT_DIR}/")