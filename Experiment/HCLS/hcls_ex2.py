import os
import time
import re
import random
import csv
from collections import defaultdict

# =====================================================================
# CẤU HÌNH THỰC NGHIỆM 2: SO SÁNH CHIẾN THUẬT TÌM KIẾM
# =====================================================================
INPUT_DIR = "/kaggle/input/datasets/m1nh12345/dataset2/Datasets" # Đã cấu hình cho Kaggle
OUTPUT_DIR = "/kaggle/working/Result/Experiment2_Strategies" # Lưu ý: Đổi tên tránh nhầm với thư mục CPSAT
TARGET_SIZES = [200, 800, 1000] 

# Các chiến thuật cần so sánh (Đã sửa "Only init" thành "Only_Init" tránh lỗi thư mục)
STRATEGIES = ["AOS", "Random_HCLS", "Only_Shift", "Only_Swap", "Only_Kick", "Only_Init"]

MAX_ITERATIONS = 50000   
MAX_NO_IMPROVE = 3000    
AOS_RHO = 0.2

# (Giữ nguyên các hàm read_testcase, get_valid_slots)
def read_testcase(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    T, N, M = map(int, lines[0].split())
    class_reqs = {n: list(map(int, lines[n].split()))[:-1] for n in range(1, N + 1)}
    offset = N + 1
    teacher_caps = {t: (list(map(int, lines[offset + t - 1].split()))[:-1] if lines[offset + t - 1] != "0" else []) for t in range(1, T + 1)}
    durations = list(map(int, lines[offset + T].split()))
    return T, N, M, class_reqs, teacher_caps, {m: durations[m - 1] for m in range(1, M + 1)}

def get_valid_slots(duration):
    return [day * 12 + session * 6 + start for day in range(5) for session in range(2) for start in range(6 - duration + 1)]

class AOS_Timetabling_Exp2:
    def __init__(self, T, N, M, class_reqs, teacher_caps, d):
        self.T, self.N, self.M = T, N, M
        self.class_reqs = class_reqs
        self.d = d
        self.teachers_for_m = defaultdict(list)
        for t, caps in teacher_caps.items():
            for m in caps: self.teachers_for_m[m].append(t)
        self.valid_slots = {m: get_valid_slots(d[m]) for m in range(1, M + 1)}
        self.total_requests = sum(len(reqs) for reqs in class_reqs.values())

    def greedy_initialization(self):
        schedule = {}
        unassigned = []
        all_reqs = []
        for n, reqs in self.class_reqs.items():
            for m in reqs:
                difficulty = self.d[m] * 100 - len(self.teachers_for_m[m])
                all_reqs.append((difficulty, n, m))
        all_reqs.sort(reverse=True) 
        
        for _, n, m in all_reqs:
            valid_starts = self.valid_slots[m]
            possible_moves = []
            busy_slots_n = set()
            busy_slots_t = defaultdict(set)
            for (sn, sm), (s_start, s_t) in schedule.items():
                if sn == n: busy_slots_n.update(range(s_start, s_start + self.d[sm]))
                busy_slots_t[s_t].update(range(s_start, s_start + self.d[sm]))
                
            dur = self.d[m]
            for start in valid_starts:
                target_slots = set(range(start, start + dur))
                if not target_slots.intersection(busy_slots_n):
                    for t in self.teachers_for_m[m]:
                        if not target_slots.intersection(busy_slots_t[t]):
                            possible_moves.append((start, t))
            if possible_moves: schedule[(n, m)] = random.choice(possible_moves)
            else: unassigned.append((n, m))
        return schedule, unassigned

    def calculate_fitness(self, schedule): 
        return 1000 * len(schedule) + sum(60 - v[0] for v in schedule.values())

    def op_shift(self, schedule, unassigned):
        if not schedule: return False
        n, m = random.choice(list(schedule.keys()))
        old_start, t = schedule[(n, m)]
        valid_starts = list(self.valid_slots[m])
        random.shuffle(valid_starts)
        del schedule[(n, m)] 
        busy_slots = set(s_start for (sn, sm), (s_start, s_t) in schedule.items() if sn == n or s_t == t for s_start in range(s_start, s_start + self.d[sm]))
        dur = self.d[m]
        for new_start in valid_starts:
            if new_start != old_start and not any(slot in busy_slots for slot in range(new_start, new_start + dur)):
                schedule[(n, m)] = (new_start, t)
                return True
        schedule[(n, m)] = (old_start, t) 
        return False

    def op_swap(self, schedule, unassigned):
        if not schedule: return False
        n, m = random.choice(list(schedule.keys()))
        start, old_t = schedule[(n, m)]
        teachers = list(self.teachers_for_m[m])
        random.shuffle(teachers)
        del schedule[(n, m)]
        dur = self.d[m]
        busy_teachers = set(s_t for (sn, sm), (s_start, s_t) in schedule.items() if max(start, s_start) < min(start + dur, s_start + self.d[sm]))
        for new_t in teachers:
            if new_t != old_t and new_t not in busy_teachers:
                schedule[(n, m)] = (start, new_t)
                return True
        schedule[(n, m)] = (start, old_t)
        return False

    def check_conflict(self, schedule, n, m, start, t):
        end = start + self.d[m]
        return [(sn, sm) for (sn, sm), (s_start, s_t) in schedule.items() if (sn == n or s_t == t) and max(start, s_start) < min(end, s_start + self.d[sm])]

    def op_ejection_chain(self, schedule, unassigned):
        if not unassigned: return False
        idx = random.randint(0, len(unassigned) - 1)
        n, m = unassigned[idx]
        if not self.teachers_for_m[m] or not self.valid_slots[m]:
            del unassigned[idx]
            return False
        start = random.choice(self.valid_slots[m])
        t = random.choice(self.teachers_for_m[m])
        conflicts = self.check_conflict(schedule, n, m, start, t)
        if len(conflicts) <= 2:
            for conf_n, conf_m in conflicts:
                del schedule[(conf_n, conf_m)]
                unassigned.append((conf_n, conf_m))
            schedule[(n, m)] = (start, t)
            del unassigned[idx]
            return True
        return False

    def solve(self, strategy):
        # 1. KHỞI TẠO LUÔN ĐƯỢC CHẠY
        schedule, unassigned = self.greedy_initialization()
        best_schedule = schedule.copy()
        current_fitness = best_fitness = self.calculate_fitness(schedule)
        start_time = time.time()
        
        # SỬA LỖI Ở ĐÂY: Nếu chiến thuật là "Only_Init", Dừng ngay lập tức!
        if strategy == "Only_Init":
            exec_time = time.time() - start_time
            # Ghi lại kết quả duy nhất ở iteration 0
            history = [(0, exec_time, len(best_schedule), 0.0, 0.0, 0.0)]
            return len(best_schedule), exec_time, history
        
        # BẬT TẮT TOÁN TỬ THEO CHIẾN THUẬT (Các vòng lặp dưới chỉ chạy khi KHÔNG phải Only_Init)
        if strategy == "Only_Shift": operators = [self.op_shift]
        elif strategy == "Only_Swap": operators = [self.op_swap]
        elif strategy == "Only_Kick": operators = [self.op_ejection_chain]
        else: operators = [self.op_shift, self.op_swap, self.op_ejection_chain]
            
        weights = {op: 100.0 for op in operators}
        
        # Hàm get_safe() để xuất file CSV không bị lỗi khi toán tử bị tắt
        def w(op): return weights.get(op, 0.0) 
        
        history = [(0, 0.0, len(schedule), w(self.op_shift), w(self.op_swap), w(self.op_ejection_chain))]
        
        iter_count = 0
        no_improve_count = 0 
        
        while iter_count < MAX_ITERATIONS and no_improve_count < MAX_NO_IMPROVE:
            iter_count += 1
            new_schedule, new_unassigned = schedule.copy(), unassigned.copy()
            chosen_op = random.choices(operators, weights=[weights[op] for op in operators])[0]
            
            changed = chosen_op(new_schedule, new_unassigned)
            reward = 0 
            is_new_best = False
            
            if changed:
                new_fitness = self.calculate_fitness(new_schedule)
                if new_fitness > best_fitness:
                    reward = 50  
                    best_fitness, current_fitness = new_fitness, new_fitness
                    best_schedule, schedule = new_schedule.copy(), new_schedule
                    no_improve_count = 0  
                    is_new_best = True
                elif new_fitness >= current_fitness:
                    reward = 20 if new_fitness > current_fitness else 5
                    schedule, current_fitness = new_schedule, new_fitness
                    no_improve_count += 1 
                else: no_improve_count += 1
            else: no_improve_count += 1
                
            # CẬP NHẬT TRỌNG SỐ (CHỈ CẬP NHẬT KHI LÀ AOS)
            if strategy == "AOS":
                weights[chosen_op] = max(5.0, (1 - AOS_RHO) * weights[chosen_op] + AOS_RHO * reward)
            
            if is_new_best or iter_count % 10 == 0:
                history.append((iter_count, time.time() - start_time, len(best_schedule),
                                w(self.op_shift), w(self.op_swap), w(self.op_ejection_chain)))
                        
            if len(best_schedule) == self.total_requests: break
                
        exec_time = time.time() - start_time
        return len(best_schedule), exec_time, history

if __name__ == "__main__":
    print("🚀 BẮT ĐẦU THỰC NGHIỆM 2: SO SÁNH CÁC CHIẾN THUẬT HCLS 🚀\n")
    if not os.path.exists(INPUT_DIR): exit(f"Lỗi: Không tìm thấy {INPUT_DIR}")
    
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")], 
                   key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

    for file in files:
        filepath = os.path.join(INPUT_DIR, file)
        basename = os.path.splitext(file)[0]
        
        try: N = int(re.findall(r'_(\d+)_', basename)[0])
        except: continue
        if N not in TARGET_SIZES: continue
        
        T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
        print(f"\n==================================================")
        print(f"▶️ BÀI TOÁN: {basename} | N={N}")
        
        for strat in STRATEGIES:
            out_dir = os.path.join(OUTPUT_DIR, strat)
            os.makedirs(out_dir, exist_ok=True)
            csv_file = os.path.join(out_dir, f"History_{basename}.csv")
            
            if os.path.exists(csv_file):
                print(f"  [{strat:<12}] ⏩ Đã có kết quả. Bỏ qua.")
                continue
                
            print(f"  [{strat:<12}] Đang giải...", end="", flush=True)
            solver = AOS_Timetabling_Exp2(T, N, M, class_reqs, teacher_caps, d)
            obj_val, exec_time, history = solver.solve(strategy=strat)
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Time_Seconds', 'Objective_Value', 'W_Shift', 'W_Swap', 'W_Kick'])
                writer.writerows(history)
            print(f" Xếp: {obj_val}/{solver.total_requests} | Time: {exec_time:.2f}s")