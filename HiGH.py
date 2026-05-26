import os
import time
import re
from collections import defaultdict
from ortools.linear_solver import pywraplp

INPUT_DIR = "/kaggle/input/datasets/m1nh12345/dataset2/Datasets" # Đổi thành "/kaggle/input/tên-dataset" nếu chạy Kaggle
OUTPUT_DIR = "Result/HiGHS_Solver"
TIME_LIMIT_SEC = 600 

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
    return [day * 12 + session * 6 + start for day in range(5) for session in range(2) for start in range(6 - duration + 1)]

print("Khởi động HiGHS MILP Solver (Google OR-Tools Wrapper)...")

files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")], 
               key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

for file in files:
    filepath = os.path.join(INPUT_DIR, file)
    basename = os.path.splitext(file)[0]
    T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
    
    # --- LỚP PHÒNG NGỰ RAM TRÊN KAGGLE ---
    if N >= 300:
        print(f"\n[{basename}] CẢNH BÁO: BỎ QUA N={N}. Tổ hợp Big-M sẽ gây sập RAM (OOM).")
        continue
    
    print(f"\n[{basename}] Đang nạp ma trận Toán học vào HiGHS...", flush=True)
    
    solver = pywraplp.Solver.CreateSolver('HIGHS')
    if not solver:
        print("Lỗi: Không tìm thấy HiGHS solver!")
        break

    S, X, U, V = {}, {}, {}, {}
    BigM = 100 
    
    teachers_for_m = defaultdict(list)
    for t, caps in teacher_caps.items():
        for m in caps: teachers_for_m[m].append(t)
            
    # 1. KHỞI TẠO BIẾN
    for n, reqs in class_reqs.items():
        for m in reqs:
            S[n, m] = solver.IntVar(0, 1, f"S_{n}_{m}")
            U[n, m] = solver.IntVar(0, 59, f"U_{n}_{m}")
            
            valid_slots = get_valid_slots(d[m])
            for u in valid_slots:
                V[n, m, u] = solver.IntVar(0, 1, f"V_{n}_{m}_{u}")
                
            for t in teachers_for_m[m]:
                X[n, m, t] = solver.IntVar(0, 1, f"X_{n}_{m}_{t}")

    # 2. RÀNG BUỘC
    for n, reqs in class_reqs.items():
        for m in reqs:
            valid_slots = get_valid_slots(d[m])
            
            # Logic Chọn giờ
            solver.Add(solver.Sum([V[n, m, u] for u in valid_slots]) == S[n, m])
            solver.Add(U[n, m] == solver.Sum([u * V[n, m, u] for u in valid_slots]))
            
            # Phân công Giáo viên
            solver.Add(solver.Sum([X[n, m, t] for t in teachers_for_m[m]]) == S[n, m])
            
        # Lớp không đè lịch (Big-M)
        req_list = list(reqs)
        for i in range(len(req_list)):
            for j in range(i + 1, len(req_list)):
                m1, m2 = req_list[i], req_list[j]
                Z = solver.IntVar(0, 1, f"Z_C_{n}_{m1}_{m2}")
                solver.Add(U[n, m1] + d[m1] <= U[n, m2] + BigM * (1 - Z) + BigM * (2 - S[n, m1] - S[n, m2]))
                solver.Add(U[n, m2] + d[m2] <= U[n, m1] + BigM * Z + BigM * (2 - S[n, m1] - S[n, m2]))

    # Giáo viên không phân thân (Big-M)
    teacher_tasks = defaultdict(list)
    for n, reqs in class_reqs.items():
        for m in reqs:
            for t in teachers_for_m[m]:
                teacher_tasks[t].append((n, m))
                
    for t, tasks in teacher_tasks.items():
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                n1, m1 = tasks[i]
                n2, m2 = tasks[j]
                Y = solver.IntVar(0, 1, f"Y_T_{t}_{n1}{m1}_{n2}{m2}")
                solver.Add(U[n1, m1] + d[m1] <= U[n2, m2] + BigM * (1 - Y) + BigM * (2 - X[n1, m1, t] - X[n2, m2, t]))
                solver.Add(U[n2, m2] + d[m2] <= U[n1, m1] + BigM * Y + BigM * (2 - X[n1, m1, t] - X[n2, m2, t]))

    # 3. HÀM MỤC TIÊU VÀ RUN
    objective = solver.Objective()
    for n in class_reqs:
        for m in class_reqs[n]:
            objective.SetCoefficient(S[n, m], 1)
    objective.SetMaximization()

    solver.SetTimeLimit(TIME_LIMIT_SEC * 1000) # Mili-giây
    
    print("Bắt đầu Solve...")
    start_time = time.time()
    status = solver.Solve()
    exec_time = time.time() - start_time
    
    # 4. KIỂM TRA KẾT QUẢ
    status_str = "UNKNOWN"
    if status == pywraplp.Solver.OPTIMAL: status_str = "OPTIMUM"
    elif status == pywraplp.Solver.FEASIBLE: status_str = "FEASIBLE"
    elif status == pywraplp.Solver.INFEASIBLE: status_str = "INFEASIBLE"
    
    obj_val = int(objective.Value()) if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE] else 0
    
    assignments = []
    if obj_val > 0:
        for n, reqs in class_reqs.items():
            for m in reqs:
                if S[n, m].solution_value() > 0.5:
                    start_val = int(round(U[n, m].solution_value()))
                    assigned_t = next(t for t in teachers_for_m[m] if X[n, m, t].solution_value() > 0.5)
                    assignments.append((n, m, start_val + 1, assigned_t))
                    
    out_file = os.path.join(OUTPUT_DIR, f"HiGHS_{basename}.txt")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:* 1000:<.2f} ms\n")
        f.write(f"Trạng thái: {status_str}\n")
        
    print(f"[{basename}] KQ: {status_str} | Điểm: {obj_val} | Time: {exec_time:* 1000:<.2f} ms")