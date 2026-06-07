import os
import time
import re
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

INPUT_DIR = "/kaggle/input/datasets/m1nh12345/dataset2/Datasets"  # Sửa lại theo tên thư mục data của bạn
OUTPUT_DIR = "/kaggle/working/Result/Gurobi_Kaggle"
TIME_LIMIT_SEC = 1000 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# THIẾT LẬP BẢN QUYỀN GUROBI WLS (BẮT BUỘC TRÊN KAGGLE)

WLS_ACCESS_ID = ""
WLS_SECRET = ""
LICENSE_ID = 123456 # Điền số License ID của bạn bạn

try:
    wls_env = gp.Env(empty=True)
    wls_env.setParam('WLSACCESSID', WLS_ACCESS_ID)
    wls_env.setParam('WLSSECRET', WLS_SECRET)
    wls_env.setParam('LICENSEID', LICENSE_ID)
    wls_env.start()
except:
    wls_env = gp.Env()

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
        out_file = os.path.join(OUTPUT_DIR, f"Gurobi_{basename}.txt")
        
        if os.path.exists(out_file): continue
            
        T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
        print(f"\n▶️ [{basename}] N={N}. Đang dựng Mô hình Gurobi Time-Indexed...", flush=True)
        
        model = gp.Model(f"Timetabling_{basename}", env=wls_env)
        
        teachers_for_m = defaultdict(list)
        for t, caps in teacher_caps.items():
            for m in caps: teachers_for_m[m].append(t)
                
        # RÀNG BUỘC 4: Lọc kíp an toàn ngay từ đầu (Chống vắt chéo buổi)
        # Trong code dùng 0-indexed nên công thức là: (u % 6) + d <= 6
        valid_slots = {m: [u for u in range(60) if (u % 6) + d[m] <= 6] for m in range(1, M + 1)}
        
        # TẬP HỢP OMEGA
        valid_tasks = [(n, m) for n, reqs in class_reqs.items() for m in reqs if teachers_for_m[m]]
        
        # KHỞI TẠO BIẾN
        S = model.addVars(valid_tasks, vtype=GRB.BINARY, name="S")
        
        V_keys = [(n, m, t, u) for (n, m) in valid_tasks for t in teachers_for_m[m] for u in valid_slots[m]]
        V = model.addVars(V_keys, vtype=GRB.BINARY, name="V")

        # RÀNG BUỘC 1: Phân công duy nhất
        for (n, m) in valid_tasks:
            model.addConstr(
                gp.quicksum(V[n, m, t, u] for t in teachers_for_m[m] for u in valid_slots[m]) == S[n, m]
            )

        # RÀNG BUỘC 2: Chống đè lịch Lớp (Quét qua các kíp s)
        for n in range(1, N + 1):
            reqs = [m for m in class_reqs[n] if (n, m) in valid_tasks]
            for s in range(60):
                terms = [V[n, m, t, u] for m in reqs for t in teachers_for_m[m] 
                         for u in valid_slots[m] if u <= s < u + d[m]]
                if terms: 
                    model.addConstr(gp.quicksum(terms) <= 1)

        # RÀNG BUỘC 3: Chống đè lịch Thầy (Quét qua các kíp s)
        for t in range(1, T + 1):
            for s in range(60):
                terms = [V[n, m, t, u] for (n, m) in valid_tasks if t in teachers_for_m[m] 
                         for u in valid_slots[m] if u <= s < u + d[m]]
                if terms:
                    model.addConstr(gp.quicksum(terms) <= 1)

        # HÀM MỤC TIÊU
        model.setObjective(gp.quicksum(S[n, m] for (n, m) in valid_tasks), GRB.MAXIMIZE)

        model.Params.TimeLimit = TIME_LIMIT_SEC
        model.Params.MIPFocus = 1      
        model.optimize()
        
        # TRÍCH XUẤT KẾT QUẢ
        assignments = []
        if model.SolCount > 0:
            for (n, m) in valid_tasks:
                if S[n, m].X > 0.5:
                    for t in teachers_for_m[m]:
                        for u in valid_slots[m]:
                            if V[n, m, t, u].X > 0.5:
                                assignments.append((n, m, u + 1, t)) # Cộng 1 để trả về 1-indexed
                                break
                                
        # KIỂM TRA TRẠNG THÁI NGHIỆM CỦA GUROBI
        if model.Status == GRB.OPTIMAL:
            solution_status = "OPTIMUM"
        elif model.SolCount > 0:
            solution_status = "FEASIBLE"
        else:
            solution_status = "NO_SOLUTION"
                                
        # GHI KẾT QUẢ RA FILE
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"{len(assignments)}\n")
            for n, m, u, t in assignments:
                f.write(f"{n} {m} {u} {t}\n")
            
            # Ghi thêm dòng trạng thái vào cuối file
            f.write(f"Trạng thái: {solution_status}\n")
            f.write(f"Điểm tối ưu: {int(model.ObjVal) if model.SolCount > 0 else 0}\n")
            f.write(f"Thời gian: {model.Runtime:.4f} giây\n")
                                