import os
import time
import re
from collections import defaultdict
from ortools.linear_solver import pywraplp

INPUT_DIR = "/kaggle/input/datasets/m1nh12345/dataset2/Datasets"  
OUTPUT_DIR = "/kaggle/working/Result/ORTools_TimeIndexed"
TIME_LIMIT_SEC = 1000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ... (Hàm read_testcase giữ nguyên như bản trên) ...
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

if __name__ == "__main__":
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")], 
                   key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

    for file in files:
        filepath = os.path.join(INPUT_DIR, file)
        basename = os.path.splitext(file)[0]
        out_file = os.path.join(OUTPUT_DIR, f"ORTools_{basename}.txt")
        
        if os.path.exists(out_file): continue
            
        T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
        print(f"\n▶️ [{basename}] N={N}. Đang dựng Mô hình OR-Tools (SCIP)...", flush=True)
        
        # Khởi tạo Solver mã nguồn mở SCIP (cùng họ với HiGHS)
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("Lỗi: Không tìm thấy SCIP solver.")
            continue
            
        solver.SetTimeLimit(TIME_LIMIT_SEC * 1000) # milliseconds
        
        teachers_for_m = defaultdict(list)
        for t, caps in teacher_caps.items():
            for m in caps: teachers_for_m[m].append(t)
                
        valid_slots = {m: [u for u in range(60) if (u % 6) + d[m] <= 6] for m in range(1, M + 1)}
        valid_tasks = [(n, m) for n, reqs in class_reqs.items() for m in reqs if teachers_for_m[m]]
        
        # KHỞI TẠO BIẾN
        S = {}
        for (n, m) in valid_tasks:
            S[n, m] = solver.BoolVar(f'S_{n}_{m}')
            
        V = {}
        for (n, m) in valid_tasks:
            for t in teachers_for_m[m]:
                for u in valid_slots[m]:
                    V[n, m, t, u] = solver.BoolVar(f'V_{n}_{m}_{t}_{u}')

        # RÀNG BUỘC 1: Phân công duy nhất
        for (n, m) in valid_tasks:
            terms = [V[n, m, t, u] for t in teachers_for_m[m] for u in valid_slots[m]]
            solver.Add(solver.Sum(terms) == S[n, m])

        # RÀNG BUỘC 2: Chống đè lịch Lớp
        for n in range(1, N + 1):
            reqs = [m for m in class_reqs[n] if (n, m) in valid_tasks]
            for s in range(60):
                terms = [V[n, m, t, u] for m in reqs for t in teachers_for_m[m] 
                         for u in valid_slots[m] if u <= s < u + d[m]]
                if terms:
                    solver.Add(solver.Sum(terms) <= 1)

        # RÀNG BUỘC 3: Chống đè lịch Thầy
        for t in range(1, T + 1):
            for s in range(60):
                terms = [V[n, m, t, u] for (n, m) in valid_tasks if t in teachers_for_m[m] 
                         for u in valid_slots[m] if u <= s < u + d[m]]
                if terms:
                    solver.Add(solver.Sum(terms) <= 1)

        # HÀM MỤC TIÊU
        solver.Maximize(solver.Sum(S[n, m] for (n, m) in valid_tasks))

        # GIẢI VÀ XUẤT KẾT QUẢ
        start_time = time.time()
        status = solver.Solve()
        exec_time = time.time() - start_time
        
        assignments = []
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for (n, m) in valid_tasks:
                if S[n, m].solution_value() > 0.5:
                    for t in teachers_for_m[m]:
                        for u in valid_slots[m]:
                            if V[n, m, t, u].solution_value() > 0.5:
                                assignments.append((n, m, u + 1, t))
                                break
                                
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(assignments)}\n")
            for n, m, u, t in assignments:
                f.write(f"{n} {m} {u} {t}\n")
            f.write(f"Trạng thái nghiệm: {status}\n")
            f.write(f"Điểm tối ưu: {int(solver.Objective().Value()) if assignments else 0}\n")
            f.write(f"Thời gian: {exec_time:.4f} giây\n")