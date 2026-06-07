import os
import time
import random
import pandas as pd
import matplotlib.pyplot as plt

# Kế thừa các hàm từ file gốc của bạn
from HullClimbingLocalSearch import read_testcase, AOS_Timetabling, MAX_ITERATIONS, MAX_NO_IMPROVE, AOS_RHO

# ==========================================
# CẤU HÌNH THỰC NGHIỆM
# ==========================================
# CHÚ Ý: BẠN HÃY ĐỔI TÊN FILE TEST DƯỚI ĐÂY THÀNH 1 FILE CÓ THẬT TRONG THƯ MỤC Datasets CỦA BẠN (Nên chọn bài N=100 hoặc 150)
TESTCASE_FILE = "Datasets/Adversarial/Adversarial_800_1200_320.txt" 
OUTPUT_DIR = "Figure/Parameter_Tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Định nghĩa 4 kịch bản dựa trên công thức EMA của bạn
SCENARIOS = {
    "S1_BanGoc": { # Bộ số gốc của bạn (Được kỳ vọng sẽ tốt nhất)
        "best": 50, "better": 20, "equal": 5, "bad": 0, 
        "color": "#e74c3c", "lw": 2.5, "label": "Bản gốc (Best:+50, Better:+20)"
    },
    "S2_HamLoi": { # Thưởng quá to, dễ bị kẹt tối ưu cục bộ
        "best": 200, "better": 50, "equal": 10, "bad": -10, 
        "color": "#3498db", "lw": 1.5, "label": "Hám lợi (Best:+200, Phạt:-10)"
    },
    "S3_NhatGan": { # Phạt quá nặng khi ra nghiệm xấu, làm mất tính khám phá
        "best": 20, "better": 5, "equal": 0, "bad": -50, 
        "color": "#2ecc71", "lw": 1.5, "label": "Phạt nặng (Best:+20, Phạt:-50)"
    },
    "S4_Random": { # Không có Adaptive (Mọi trọng số đứng im)
        "best": 0, "better": 0, "equal": 0, "bad": 0, 
        "color": "#9b59b6", "lw": 1.5, "label": "Cố định (Không dùng AOS)"
    }
}

# Tạo một Class con ghi đè lại hàm solve để truyền được bộ tham số vào
class Tuned_AOS(AOS_Timetabling):
    def __init__(self, T, N, M, class_reqs, teacher_caps, d, config):
        super().__init__(T, N, M, class_reqs, teacher_caps, d)
        self.cfg = config

    def solve(self):
        schedule, unassigned = self.greedy_initialization()
        best_schedule = schedule.copy()
        current_fitness = self.calculate_fitness(schedule)
        best_fitness = current_fitness
        start_time = time.time()
        
        history = [(0, 0.0, len(schedule), 100.0, 100.0, 100.0)]
        operators = [self.op_shift, self.op_swap, self.op_ejection_chain]
        weights = {op: 100.0 for op in operators}
        
        iter_count = 0
        no_improve_count = 0 
        
        while iter_count < MAX_ITERATIONS and no_improve_count < MAX_NO_IMPROVE:
            iter_count += 1
            new_schedule = schedule.copy()
            new_unassigned = unassigned.copy()
            
            chosen_op = random.choices(operators, weights=[weights[op] for op in operators])[0]
            changed = chosen_op(new_schedule, new_unassigned)
            
            reward = self.cfg["bad"] # Mặc định là điểm phạt
            is_new_best = False
            
            if changed:
                new_fitness = self.calculate_fitness(new_schedule)
                if new_fitness > best_fitness:
                    reward = self.cfg["best"] # THƯỞNG KỶ LỤC
                    best_fitness = new_fitness
                    best_schedule = new_schedule.copy()
                    schedule = new_schedule
                    current_fitness = new_fitness
                    no_improve_count = 0  
                    is_new_best = True
                elif new_fitness >= current_fitness:
                    reward = self.cfg["better"] if new_fitness > current_fitness else self.cfg["equal"]
                    schedule = new_schedule
                    current_fitness = new_fitness
                    no_improve_count += 1 
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            # Công thức cốt lõi của bạn
            weights[chosen_op] = max(5.0, (1 - AOS_RHO) * weights[chosen_op] + AOS_RHO * reward)
            
            if is_new_best or iter_count % 10 == 0:
                history.append((iter_count, time.time() - start_time, len(best_schedule),
                                weights[self.op_shift], weights[self.op_swap], weights[self.op_ejection_chain]))
                        
            if len(best_schedule) == self.total_requests:
                break
                
        return history

def run_experiment():
    print(f"🎯 BẮT ĐẦU THỰC NGHIỆM ĐỘ NHẠY THAM SỐ (PARAMETER TUNING)\n")
    if not os.path.exists(TESTCASE_FILE):
        print(f"❌ LỖI: Vui lòng sửa lại biến TESTCASE_FILE trong code trỏ đúng vào 1 bài test có thật.")
        return

    T, N, M, class_reqs, teacher_caps, d = read_testcase(TESTCASE_FILE)
    scenarios_history = {}

    for sc_name, params in SCENARIOS.items():
        print(f"--- Đang chạy kịch bản: {sc_name} ---")
        solver = Tuned_AOS(T, N, M, class_reqs, teacher_caps, d, params)
        history_list = solver.solve()
        
        df = pd.DataFrame(history_list, columns=['Iteration', 'Time', 'Obj', 'W_Shift', 'W_Swap', 'W_Kick'])
        scenarios_history[sc_name] = df

    # VẼ BIỂU ĐỒ
    plt.figure(figsize=(12, 7))
    for sc_name, df in scenarios_history.items():
        config = SCENARIOS[sc_name]
        plt.step(df['Iteration'], df['Obj'], where='post', 
                 color=config['color'], linewidth=config['lw'], label=config['label'])

    test_basename = os.path.basename(TESTCASE_FILE)
    plt.title(f'TỐC ĐỘ HỘI TỤ VỚI CÁC BỘ THAM SỐ AOS KHÁC NHAU\n(Testcase: {test_basename})', fontsize=15, fontweight='bold')
    plt.xlabel('Số vòng lặp (Iterations)', fontsize=13)
    plt.ylabel('Hàm Fitness (Số kíp xếp được)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
    
    current_ymin, current_ymax = plt.ylim()
    plt.ylim(current_ymin - 2, current_ymax + 2)

    output_plot = os.path.join(OUTPUT_DIR, "AOS_Parameters_Comparison.png")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    print(f"\n🎉 HOÀN TẤT! Biểu đồ so sánh đã được lưu tại: {output_plot}")

if __name__ == "__main__":
    run_experiment()