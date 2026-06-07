import os
import time
import re
import random
import csv
from collections import defaultdict

# =====================================================================
# CẤU HÌNH THƯ MỤC VÀ HỆ THỐNG
# =====================================================================
INPUT_DIR = "Datasets" 
OUTPUT_DIR = "Result/AOS_LocalSearch"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_ITERATIONS = 50000   
MAX_NO_IMPROVE = 3000    
AOS_RHO = 0.2            

def read_testcase(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    T, N, M = map(int, lines[0].split())
    class_reqs = {n: list(map(int, lines[n].split()))[:-1] for n in range(1, N + 1)}
    offset = N + 1
    teacher_caps = {t: (list(map(int, lines[offset + t - 1].split()))[:-1] if lines[offset + t - 1] != "0" else []) for t in range(1, T + 1)}
    durations = list(map(int, lines[offset + T].split()))
    d = {m: durations[m - 1] for m in range(1, M + 1)}
    return T, N, M, class_reqs, teacher_caps, d

def get_valid_slots(duration):
    return [day * 12 + session * 6 + start for day in range(5) for session in range(2) for start in range(6 - duration + 1)]

class AOS_Timetabling:
    def __init__(self, T, N, M, class_reqs, teacher_caps, d):
        self.T, self.N, self.M = T, N, M
        self.class_reqs = class_reqs
        self.d = d
        self.teachers_for_m = defaultdict(list)
        for t, caps in teacher_caps.items():
            for m in caps: self.teachers_for_m[m].append(t)
        self.valid_slots = {m: get_valid_slots(d[m]) for m in range(1, M + 1)}
        self.total_requests = sum(len(reqs) for reqs in class_reqs.values())

    def check_conflict(self, schedule, n, m, start, t):
        duration = self.d[m]
        end = start + duration
        conflicts = []
        for (sched_n, sched_m), (sched_start, sched_t) in schedule.items():
            if sched_n == n or sched_t == t:
                sched_end = sched_start + self.d[sched_m]
                if max(start, sched_start) < min(end, sched_end):
                    conflicts.append((sched_n, sched_m))
        return conflicts

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
            
            # TỐI ƯU 2: Gom toàn bộ lịch rảnh/bận của lớp n và các thầy vào Set (Tra cứu O(1))
            busy_slots_n = set()
            busy_slots_t = defaultdict(set)
            for (sn, sm), (s_start, s_t) in schedule.items():
                if sn == n:
                    busy_slots_n.update(range(s_start, s_start + self.d[sm]))
                busy_slots_t[s_t].update(range(s_start, s_start + self.d[sm]))
                
            dur = self.d[m]
            for start in valid_starts:
                target_slots = set(range(start, start + dur))
                # Tra cứu O(1) thay vì loop lại toàn bộ schedule bằng hàm check_conflict
                if not target_slots.intersection(busy_slots_n):
                    for t in self.teachers_for_m[m]:
                        if not target_slots.intersection(busy_slots_t[t]):
                            possible_moves.append((start, t))
                            
            if possible_moves:
                start, t = random.choice(possible_moves)
                schedule[(n, m)] = (start, t)
            else:
                unassigned.append((n, m))
        return schedule, unassigned

    def calculate_fitness(self, schedule):
        # Viết gọn lại unpack tuple để chạy nhanh hơn trong vòng lặp lớn
        return 1000 * len(schedule) + sum(60 - v[0] for v in schedule.values())

    def op_shift(self, schedule, unassigned):
        if not schedule: return False
        n, m = random.choice(list(schedule.keys()))
        old_start, t = schedule[(n, m)]
        valid_starts = list(self.valid_slots[m]) # Khắc phục lỗi shuffle đè mảng gốc
        random.shuffle(valid_starts)
        del schedule[(n, m)] 
        
        # TỐI ƯU 3: Lọc trước lịch bận của Thầy t và Lớp n
        busy_slots = set()
        for (sn, sm), (s_start, s_t) in schedule.items():
            if sn == n or s_t == t:
                busy_slots.update(range(s_start, s_start + self.d[sm]))
                
        dur = self.d[m]
        for new_start in valid_starts:
            if new_start != old_start:
                # Kiểm tra cực nhanh xem kíp có trống không
                if not any(slot in busy_slots for slot in range(new_start, new_start + dur)):
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
        
        # TỐI ƯU 4: Lọc trước xem giáo viên nào đang bận đúng vào cái kíp này
        dur = self.d[m]
        end = start + dur
        busy_teachers = set()
        for (sn, sm), (s_start, s_t) in schedule.items():
            s_end = s_start + self.d[sm]
            if max(start, s_start) < min(end, s_end):
                busy_teachers.add(s_t)
                
        for new_t in teachers:
            if new_t != old_t and new_t not in busy_teachers:
                schedule[(n, m)] = (start, new_t)
                return True
                
        schedule[(n, m)] = (start, old_t)
        return False

    def op_ejection_chain(self, schedule, unassigned):
        if not unassigned: return False
        idx = random.randint(0, len(unassigned) - 1)
        n, m = unassigned[idx]
        if not self.teachers_for_m[m] or not self.valid_slots[m]:
            # Xóa vĩnh viễn môn này khỏi danh sách chờ để các vòng lặp sau không mất thời gian bốc trúng nó nữa
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

    def solve(self):
        print(f"  [1. Khởi tạo] Chạy Smart Greedy...", end="")
        schedule, unassigned = self.greedy_initialization()
        best_schedule = schedule.copy()
        current_fitness = self.calculate_fitness(schedule)
        best_fitness = current_fitness
        
        start_time = time.time()
        
        # SỬA LỖI 1: Cập nhật history ban đầu đủ 6 cột (kèm 3 trọng số 100.0)
        history = [(0, 0.0, len(schedule), 100.0, 100.0, 100.0)]
        
        print(f" Xếp được: {len(schedule)}/{self.total_requests}")
        print(f"  [2. Tìm kiếm] Bắt đầu AOS (Max = {MAX_ITERATIONS}, Trì trệ = {MAX_NO_IMPROVE})...")
        
        operators = [self.op_shift, self.op_swap, self.op_ejection_chain]
        op_names = {self.op_shift: "Shift", self.op_swap: "Swap", self.op_ejection_chain: "Kick"}
        weights = {op: 100.0 for op in operators}
        usage_counts = {op: 0 for op in operators} 
        
        iter_count = 0
        no_improve_count = 0 
        
        while iter_count < MAX_ITERATIONS and no_improve_count < MAX_NO_IMPROVE:
            iter_count += 1
            new_schedule = schedule.copy()
            new_unassigned = unassigned.copy()
            
            current_weights_list = [weights[op] for op in operators]
            chosen_op = random.choices(operators, weights=current_weights_list)[0]
            usage_counts[chosen_op] += 1
            
            changed = chosen_op(new_schedule, new_unassigned)
            reward = 0 
            
            # SỬA LỖI 2: Thêm biến cờ để kiểm soát việc lưu history
            is_new_best = False
            
            if changed:
                new_fitness = self.calculate_fitness(new_schedule)
                if new_fitness > best_fitness:
                    reward = 50  
                    best_fitness = new_fitness
                    best_schedule = new_schedule.copy()
                    schedule = new_schedule
                    current_fitness = new_fitness
                    no_improve_count = 0  
                    
                    is_new_best = True # Bật cờ khi có kỷ lục mới
                    
                elif new_fitness >= current_fitness:
                    reward = 20 if new_fitness > current_fitness else 5
                    schedule = new_schedule
                    current_fitness = new_fitness
                    no_improve_count += 1 
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            weights[chosen_op] = max(5.0, (1 - AOS_RHO) * weights[chosen_op] + AOS_RHO * reward)
            
            # SỬA LỖI 3: Lưu history (đủ 6 cột) khi có kỷ lục mới HOẶC mỗi 10 vòng lặp
            if is_new_best or iter_count % 10 == 0:
                history.append((
                    iter_count, 
                    time.time() - start_time, 
                    len(best_schedule),
                    weights[self.op_shift],
                    weights[self.op_swap],
                    weights[self.op_ejection_chain]
                ))
                        
            if len(best_schedule) == self.total_requests:
                # Đảm bảo điểm kết thúc cũng được lưu đầy đủ 6 cột
                history.append((
                    iter_count, 
                    time.time() - start_time, 
                    len(best_schedule),
                    weights[self.op_shift],
                    weights[self.op_swap],
                    weights[self.op_ejection_chain]
                ))
                break
                
        exec_time = time.time() - start_time
        stop_reason = "Tối ưu tuyệt đối (100%)" if len(best_schedule) == self.total_requests else \
                      "Đạt giới hạn vòng lặp" if iter_count == MAX_ITERATIONS else \
                      f"Trì trệ {MAX_NO_IMPROVE} vòng liên tiếp"
                      
        print(f"  [3. Kết quả] ")
        print(f"    + Lý do dừng: {stop_reason}")
        print(f"    + Kỷ lục xếp: {len(best_schedule)}/{self.total_requests} môn")
        print(f"    + Vòng lặp: {iter_count:,} lần | Thời gian: {exec_time:.2f} s")
        
        assignments = []
        for (n, m), (start, t) in best_schedule.items():
            assignments.append((n, m, start + 1, t)) 
            
        return assignments, len(best_schedule), exec_time, history

if __name__ == "__main__":
    print("🚀 KHỞI ĐỘNG HỆ THỐNG AOS TIMETABLING 🚀\n")
    if not os.path.exists(INPUT_DIR):
        print(f"LỖI: Không tìm thấy thư mục {INPUT_DIR}.")
    else:
        files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")], 
                       key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

        for file in files:
            filepath = os.path.join(INPUT_DIR, file)
            basename = os.path.splitext(file)[0]
            
            out_file = os.path.join(OUTPUT_DIR, f"AOS_{basename}.txt")
            if os.path.exists(out_file):
                print(f"⏩ Đã có kết quả cho {basename}. Tự động bỏ qua.")
                continue

            T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
            
            print(f"\n==================================================")
            print(f"▶️ BÀI TOÁN: [{basename}] | N={N}, M={M}, T={T}")
            
            solver = AOS_Timetabling(T, N, M, class_reqs, teacher_caps, d)
            assignments, obj_val, exec_time, history = solver.solve()
            
            # Ghi kết quả Txt
            status_str = "FEASIBLE" if obj_val == solver.total_requests else "PARTIAL"
            with open(out_file, 'w', encoding='utf-8') as f:
                f.write(f"{len(assignments)}\n")
                for n, m, u, t in assignments:
                    f.write(f"{n} {m} {u} {t}\n")
                f.write(f"Điểm tối ưu: {obj_val}\n")
                f.write(f"Thời gian: {exec_time:.4f} giây\n")
                f.write(f"Trạng thái: {status_str}\n")
                
            # Ghi lịch sử CSV
            csv_file = os.path.join(OUTPUT_DIR, f"History_{basename}.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Iteration', 'Time_Seconds', 'Objective_Value', 'W_Shift', 'W_Swap', 'W_Kick'])
                writer.writerows(history)
            print(f"    [+] Đã xuất file báo cáo lịch sử hội tụ: {csv_file}")


