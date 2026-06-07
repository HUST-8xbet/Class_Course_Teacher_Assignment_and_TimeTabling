import os
import time
import re
from collections import defaultdict
from ortools.sat.python import cp_model

# Cấu hình đường dẫn cho Kaggle
INPUT_DIR = "/kaggle/input/datasets/m1nh12345/dataset2/Datasets" # Thay đổi tương ứng với tên Dataset bạn up lên
OUTPUT_DIR = "/kaggle/working/Result/CPSAT"
TIME_LIMIT_SEC = 1000

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    """Trả về các kíp bắt đầu hợp lệ (không nhảy qua ranh giới buổi học)"""
    return [day * 12 + session * 6 + start for day in range(5) for session in range(2) for start in range(6 - duration + 1)]

def write_result(filename, assignments, obj_val, exec_time, status):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time * 1000:<.2f} ms\n")
        f.write(f"Trạng thái: {status}\n")

# --- CHƯƠNG TRÌNH CHÍNH ---
print(f"Khởi động CP-SAT Solver trên Kaggle...")

# Lấy danh sách file và sắp xếp theo số N tăng dần
files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")], 
               key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

for file in files:
    filepath = os.path.join(INPUT_DIR, file)
    basename = os.path.splitext(file)[0]
    T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
    
    print(f"\n[{basename}] Đang build model...", flush=True)
    model = cp_model.CpModel()
    
    teachers_for_m = defaultdict(list)
    for t, caps in teacher_caps.items():
        for m in caps: teachers_for_m[m].append(t)
            
    task_vars = {}
    class_intervals = defaultdict(list)
    teacher_intervals = defaultdict(list)
    task_id = 0
    
    # 1. TẠO BIẾN INTERVAL (TIẾT KIỆM RAM)
    for n, reqs in class_reqs.items():
        for m in reqs:
            duration = d[m]
            valid_slots = get_valid_slots(duration)
            
            is_sched = model.NewBoolVar(f'is_sched_{task_id}')
            start_var = model.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_slots), f'start_{task_id}')
            end_var = model.NewIntVar(0, 60, f'end_{task_id}')
            
            # Interval của lớp
            c_int = model.NewOptionalIntervalVar(start_var, duration, end_var, is_sched, f'c_int_{task_id}')
            class_intervals[n].append(c_int)
            
            assign_t_vars = {}
            for t in teachers_for_m[m]:
                assign_t = model.NewBoolVar(f'assign_t_{task_id}_{t}')
                assign_t_vars[t] = assign_t
                
                # Interval của giáo viên (Dùng chung start/end với lớp)
                t_int = model.NewOptionalIntervalVar(start_var, duration, end_var, assign_t, f't_int_{task_id}_{t}')
                teacher_intervals[t].append(t_int)
            
            model.Add(sum(assign_t_vars.values()) == is_sched)
            
            task_vars[task_id] = {'n': n, 'm': m, 'is_sched': is_sched, 'start': start_var, 'assign_t': assign_t_vars}
            task_id += 1

    # 2. RÀNG BUỘC KHÔNG CHỒNG LẤP (NO OVERLAP)
    for n, intervals in class_intervals.items():
        model.AddNoOverlap(intervals)
    for t, intervals in teacher_intervals.items():
        model.AddNoOverlap(intervals)

    # 3. HÀM MỤC TIÊU
    model.Maximize(sum(task['is_sched'] for task in task_vars.values()))

    # 4. CẤU HÌNH SOLVER DÀNH RIÊNG CHO KAGGLE
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SEC
    
    # Tận dụng 4 nhân CPU của Kaggle để chạy song song nhiều Heuristics
    solver.parameters.num_search_workers = 4 
    solver.parameters.log_search_progress = True 

    start_time = time.time()
    status_code = solver.Solve(model)
    exec_time = time.time() - start_time
    
    # 5. XUẤT KẾT QUẢ
    status_str = "OPTIMUM" if status_code == cp_model.OPTIMAL else "FEASIBLE" if status_code == cp_model.FEASIBLE else "UNKNOWN"
    obj_val = int(solver.ObjectiveValue()) if status_code in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0
    
    assignments = []
    if obj_val > 0:
        for tid, vars_dict in task_vars.items():
            if solver.Value(vars_dict['is_sched']) == 1:
                start_val = solver.Value(vars_dict['start'])
                assigned_t = next(t for t, var in vars_dict['assign_t'].items() if solver.Value(var) == 1)
                assignments.append((vars_dict['n'], vars_dict['m'], start_val + 1, assigned_t))
    
    out_file = os.path.join(OUTPUT_DIR, f"CPSAT_{basename}.txt")
    write_result(out_file, assignments, obj_val, exec_time, status_str)
    
    print(f"[{basename}] Kết quả: {status_str} | Xếp được: {obj_val}/{task_id} | Time: {exec_time * 1000:.2f}ms")

print(f"\nĐã hoàn thành toàn bộ testcase. Kết quả lưu tại: {OUTPUT_DIR}")