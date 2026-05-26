import os
import random
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH CÁC MỐC KÍCH THƯỚC ---
test_sizes = [20, 50, 80, 100, 150, 200, 300, 500, 800, 1000]
distributions = ["Uniform", "Gaussian", "Poisson", "Exponential", "Adversarial"]

output_dir = "Datasets"
os.makedirs(output_dir, exist_ok=True)

plot_data = {dist: {"teachers_per_subject": [], "subjects_per_teacher": []} for dist in distributions}

def get_num_teachers_for_subject(dist, T):
    max_teachers = max(3, int(T * 0.1)) 
    
    if dist == "Uniform":
        return random.randint(2, max_teachers) # Ít nhất 2 GV để đảm bảo an toàn
    elif dist == "Gaussian":
        mu, sigma = max_teachers / 2, max_teachers / 6
        val = int(round(random.gauss(mu, sigma)))
        return max(2, min(val, max_teachers))
    elif dist == "Poisson":
        val = np.random.poisson(lam=max_teachers / 3)
        return max(2, min(val, max_teachers))
    elif dist == "Exponential":
        val = int(round(np.random.exponential(scale=max_teachers / 4)))
        return max(2, min(val, max_teachers))
    elif dist == "Adversarial":
        if random.random() < 0.2:
            return 2 # Môn hiếm (2 người dạy)
        else:
            return random.randint(max_teachers // 2, max_teachers)
    return 2

def generate_and_save_test(dist_name, base_size):
    N = base_size                     
    T = int(N * 1.5)                  
    M = max(10, int(N * 0.4))         
    
    # Số tiết mỗi môn (2-4 tiết)
    d = {m: random.randint(2, 4) for m in range(1, M + 1)}
    
    teacher_subjects = {t: set() for t in range(1, T + 1)}
    subject_teachers_count = [] 
    
    # --- BƯỚC 1: PHÂN CÔNG GIÁO VIÊN DẠY MÔN GÌ ---
    if dist_name == "Adversarial":
        teacher_weights = np.ones(T)
        super_teachers_count = max(1, int(T * 0.2)) 
        super_teachers_idx = random.sample(range(T), super_teachers_count)
        for idx in super_teachers_idx:
            teacher_weights[idx] = 10.0 # Hạ trọng số Super Teacher xuống x10 để tránh quá tải
    else:
        teacher_weights = np.ones(T) 
        
    teacher_probs = teacher_weights / teacher_weights.sum()

    for m in range(1, M + 1):
        num_teachers = get_num_teachers_for_subject(dist_name, T)
        
        assigned_teachers_idx = np.random.choice(
            range(T), size=num_teachers, replace=False, p=teacher_probs
        )
        for idx in assigned_teachers_idx:
            teacher_subjects[idx + 1].add(m)
            
    # --- BƯỚC 1.5: SIÊU BẪY ADVERSARIAL (ĐÃ KIỂM SOÁT TÍNH KHẢ THI) ---
    boss_subject = 1
    boss_teacher = 1
    if dist_name == "Adversarial" and M >= 5:
        # Cách ly Thầy Boss: Thầy Boss CHỈ dạy môn Boss
        teacher_subjects[boss_teacher] = {boss_subject}
        
        # Cách ly Môn Boss: Môn Boss CHỈ do Thầy Boss dạy
        for t in range(2, T + 1):
            if boss_subject in teacher_subjects[t]:
                teacher_subjects[t].remove(boss_subject)
                
    # Cập nhật mảng đếm cho biểu đồ
    for m in range(1, M + 1):
        count = sum(1 for t in range(1, T + 1) if m in teacher_subjects[t])
        subject_teachers_count.append(max(1, count)) # Đảm bảo ít nhất là 1 để ko lỗi
        
    if base_size == 1000:
        plot_data[dist_name]["teachers_per_subject"].extend(subject_teachers_count)
        plot_data[dist_name]["subjects_per_teacher"].extend([len(subs) for subs in teacher_subjects.values()])

    # --- BƯỚC 2: PHÂN CÔNG LỚP HỌC MÔN GÌ (MAX 45 TIẾT) ---
    class_subjects = {n: set() for n in range(1, N + 1)}
    for n in range(1, N + 1):
        current_slots = 0
        available_subjects = list(range(2, M + 1)) # Tạm thời không bốc môn Boss
        random.shuffle(available_subjects)
        
        for m in available_subjects:
            # CHỐT CHẶN 1: Tổng số tiết không quá 45
            if current_slots + d[m] <= 45 and len(class_subjects[n]) < 12:
                class_subjects[n].add(m)
                current_slots += d[m]
            else:
                if current_slots >= 35: # Ít nhất học 35 tiết
                    break
        if len(class_subjects[n]) == 0:
            class_subjects[n].add(random.choice(available_subjects))
            
    # --- BƯỚC 2.5: ĐẨY LỚP NẠN NHÂN VÀO BẪY ---
    if dist_name == "Adversarial" and M >= 5:
        # CHỐT CHẶN 2: Tính toán chính xác số nạn nhân để Thầy Boss không bị quá 45 tiết
        victim_count = 45 // d[boss_subject] 
        victim_classes = random.sample(range(1, N + 1), min(N, victim_count))
        
        for n in victim_classes:
            # Gỡ bớt 1 môn đang có để dọn chỗ cho môn Boss (Giữ nguyên tổng số tiết an toàn)
            if class_subjects[n]:
                removed_subject = class_subjects[n].pop()
            class_subjects[n].add(boss_subject)

    # --- BƯỚC 3: GHI FILE ---
    filename = os.path.join(output_dir, f"{dist_name}_{N}_{T}_{M}.txt")
    with open(filename, 'w') as f:
        f.write(f"{T} {N} {M}\n")
        
        for n in range(1, N + 1):
            subs = list(class_subjects[n])
            f.write(" ".join(map(str, subs)) + " 0\n")
            
        for t in range(1, T + 1):
            subs = list(teacher_subjects[t])
            if not subs:
                f.write("0\n")
            else:
                f.write(" ".join(map(str, subs)) + " 0\n")
                
        f.write(" ".join(str(d[m]) for m in range(1, M + 1)) + "\n")

# --- THỰC THI CHÍNH ---
print("Bắt đầu sinh bộ dữ liệu Test CHUẨN THỰC TẾ (Đã kiểm soát tính khả thi)...")
for dist in distributions:
    for size in test_sizes:
        generate_and_save_test(dist, size)
    print(f" -> Đã sinh xong 10 test cho phân phối: {dist}")

# --- VẼ BIỂU ĐỒ ---
print("\nĐang cập nhật biểu đồ phân phối...")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(f"Phân tích Mật độ Ràng buộc Thực tế (Bản An toàn Feasible - N=1000)", fontsize=18, fontweight='bold')

for i, dist in enumerate(distributions):
    axes[0, i].hist(plot_data[dist]["teachers_per_subject"], bins=20, color='#2874A6', edgecolor='black', alpha=0.85)
    axes[0, i].set_title(f"{dist}\nGiáo viên / Môn học", fontsize=12)
    axes[0, i].set_xlabel("Số lượng Giáo viên có thể dạy")
    
    axes[1, i].hist(plot_data[dist]["subjects_per_teacher"], bins=20, color='#117A65', edgecolor='black', alpha=0.85)
    axes[1, i].set_title(f"Môn học / Giáo viên", fontsize=12)
    axes[1, i].set_xlabel("Số môn học được phân công")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
image_dir = os.path.join("Figure", "Data")
os.makedirs(image_dir, exist_ok=True)
image_path = os.path.join(image_dir, "Data_Distributions_Analysis_V3_Safe.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"Hoàn tất! Các file đã sẵn sàng để thách thức mọi bộ giải Toán học.")