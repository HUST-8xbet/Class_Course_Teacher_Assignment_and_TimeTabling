"""
Pure Genetic Algorithm for Class-Course-Teacher Assignment and Timetabling.

Ý tưởng chính:
- GA thuần trên không gian hoán vị các lớp-môn.
- Chromosome = thứ tự ưu tiên xếp các cặp (class, course).
- Decode chromosome bằng randomized greedy để tạo thời khóa biểu hợp lệ.
- Có ghi log hội tụ từng thế hệ ra CSV để vẽ biểu đồ convergence.
- Hỗ trợ thử nhiều kiểu crossover và mutation.
"""

import os
import re
import csv
import time
import random
import argparse
from collections import defaultdict, Counter

DAYS = 5
SESSIONS_PER_DAY = 2
PERIODS_PER_SESSION = 6
TOTAL_SLOTS = DAYS * SESSIONS_PER_DAY * PERIODS_PER_SESSION  # 60

INPUT_DIR = "Datasets"
SOLVER_NAME = "Pure_GA_Timetabling"
SEED = 42
SUBFOLDERS = None
RESULT_DIR = os.path.join("Result", SOLVER_NAME)

TIME_LIMIT_BY_SIZE = [
    (200, 20),
    (500, 45),
    (1000, 90),
    (2000, 180),
    (5000, 360),
    (10000, 600),
    (float("inf"), 900),
]


def get_time_limit(total_cc):
    for threshold, limit in TIME_LIMIT_BY_SIZE:
        if total_cc <= threshold:
            return limit
    return 900


def fmt_param(x):
    if isinstance(x, float):
        s = f"{x:.4g}"
    else:
        s = str(x)
    return s.replace(".", "p").replace("-", "m")


def make_run_name(args):
    parts = [
        args.solver_name,
        f"pop{fmt_param(args.pop_size) if args.pop_size is not None else 'auto'}",
        f"elite{fmt_param(args.elite) if args.elite is not None else 'auto'}",
        f"cx{fmt_param(args.crossover_rate)}",
        f"{args.crossover_type}",
        f"mut{fmt_param(args.mutation_rate)}",
        f"{args.mutation_type}",
        f"tk{fmt_param(args.tournament_k)}",
        f"init{args.init_mode}",
        f"seed{fmt_param(args.seed)}",
    ]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Pure Genetic Algorithm timetabling solver with convergence CSV logging"
    )

    # Input/output
    parser.add_argument("filepath", nargs="?", default=None,
                        help="File dataset .txt. Nếu bỏ trống thì chạy batch toàn bộ input_dir.")
    parser.add_argument("--input-dir", default=INPUT_DIR,
                        help="Folder chứa các bộ dataset khi chạy batch.")
    parser.add_argument("--result-root", default="Result",
                        help="Folder gốc để lưu kết quả.")
    parser.add_argument("--solver-name", default=SOLVER_NAME,
                        help="Tên thuật toán/phiên bản dùng trong tên file kết quả.")
    parser.add_argument("--tag", default="",
                        help="Gắn nhãn thêm cho folder thí nghiệm.")

    # Dataset filter
    parser.add_argument("--subfolders", default=None,
                        help="Chỉ chạy một số bộ, cách nhau bởi dấu phẩy. Ví dụ: Uniform,Gaussian")

    # Runtime
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Ghi đè time limit cho mọi file.")

    # GA hyperparameters
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Số cá thể quần thể. Bỏ trống = auto-scale theo size bài.")
    parser.add_argument("--elite", type=int, default=None,
                        help="Số cá thể elite giữ lại. Bỏ trống = auto-scale theo size bài.")
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.85)
    parser.add_argument("--mutation-rate", type=float, default=0.75)
    parser.add_argument("--crossover-type", choices=["ox", "pmx", "position"], default="ox",
                        help="Kiểu lai ghép: ox=Order Crossover, pmx=Partially Mapped Crossover, position=Position-based crossover.")
    parser.add_argument("--mutation-type", choices=["mixed", "swap", "insert", "reverse", "scramble"], default="mixed",
                        help="Kiểu đột biến trên permutation.")
    parser.add_argument("--mutation-ops", type=int, default=None,
                        help="Số phép biến đổi mỗi lần mutation. Bỏ trống = tự scale theo size.")
    parser.add_argument("--init-mode", choices=["heuristic", "random"], default="heuristic",
                        help="heuristic = có greedy/noisy/hard-first trong quần thể; random = không đưa greedy vào quần thể, dùng để quan sát hội tụ rõ hơn.")

    # Logging
    parser.add_argument("--log-every", type=int, default=1,
                        help="Ghi log hội tụ mỗi bao nhiêu thế hệ. Mặc định 1 = ghi mọi thế hệ.")

    return parser


# ============================================================
# Parser / Writer
# ============================================================

def parse_input(text):
    tokens = list(map(int, text.split()))
    idx = 0

    def nx():
        nonlocal idx
        val = tokens[idx]
        idx += 1
        return val

    T, N, M = nx(), nx(), nx()

    class_courses = [[] for _ in range(N + 1)]
    for cls in range(1, N + 1):
        while True:
            c = nx()
            if c == 0:
                break
            class_courses[cls].append(c)

    teacher_courses = [set() for _ in range(T + 1)]
    for t in range(1, T + 1):
        while True:
            c = nx()
            if c == 0:
                break
            teacher_courses[t].add(c)

    durations = [0] * (M + 1)
    for m in range(1, M + 1):
        durations[m] = nx()

    return T, N, M, class_courses, teacher_courses, durations


def write_result(filename, assignments, obj_val, exec_time, status="DONE"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(assignments)}\n")
        for cls, crs, start, teacher in assignments:
            f.write(f"{cls} {crs} {start} {teacher}\n")
        f.write(f"Điểm tối ưu/tốt nhất tìm được: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.6f} giây\n")
        f.write(f"Trạng thái: {status}\n")


# ============================================================
# Schedule structure
# ============================================================

class Schedule:
    __slots__ = ("N", "T", "class_busy", "teacher_busy", "assignments")

    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.class_busy = [bytearray(TOTAL_SLOTS) for _ in range(N + 1)]
        self.teacher_busy = [bytearray(TOTAL_SLOTS) for _ in range(T + 1)]
        self.assignments = {}  # (cls, crs) -> (teacher, start_1_based)

    def can_place(self, cls, teacher, start, dur):
        base = start - 1
        cb = self.class_busy[cls]
        tb = self.teacher_busy[teacher]
        for k in range(dur):
            slot = base + k
            if cb[slot] or tb[slot]:
                return False
        return True

    def place(self, cls, crs, teacher, start, dur):
        base = start - 1
        for k in range(dur):
            slot = base + k
            self.class_busy[cls][slot] = 1
            self.teacher_busy[teacher][slot] = 1
        self.assignments[(cls, crs)] = (teacher, start)

    def __len__(self):
        return len(self.assignments)


# ============================================================
# Pure Genetic Algorithm solver
# ============================================================

class GeneticTimetablingSolver:
    def __init__(self, T, N, M, class_courses, teacher_courses, durations,
                 time_limit=60, seed=42, verbose=True,
                 pop_size=None, elite=None, tournament_k=3,
                 crossover_rate=0.85, mutation_rate=0.75,
                 crossover_type="ox", mutation_type="mixed", mutation_ops=None,
                 init_mode="heuristic", log_every=1):
        self.T = T
        self.N = N
        self.M = M
        self.class_courses = class_courses
        self.teacher_courses = teacher_courses
        self.durations = durations
        self.time_limit = time_limit
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.seed = seed
        self.start_time = None

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_ops = mutation_ops
        self.tournament_k = tournament_k
        self.init_mode = init_mode
        self.log_every = max(1, log_every)
        self.history = []

        self.all_cc = []
        for cls in range(1, N + 1):
            for crs in class_courses[cls]:
                self.all_cc.append((cls, crs))
        self.total_cc = len(self.all_cc)

        self.course_teachers = defaultdict(list)
        for t in range(1, T + 1):
            for c in teacher_courses[t]:
                self.course_teachers[c].append(t)

        self.teacher_load_score = [0] * (T + 1)
        for t in range(1, T + 1):
            self.teacher_load_score[t] = len(teacher_courses[t])

        # Kíp bắt đầu hợp lệ: không vắt qua buổi 6 tiết
        self.valid_slots = {}
        for m in range(1, M + 1):
            d = durations[m]
            slots = []
            if 1 <= d <= PERIODS_PER_SESSION:
                for day in range(DAYS):
                    for ses in range(SESSIONS_PER_DAY):
                        base = (day * SESSIONS_PER_DAY + ses) * PERIODS_PER_SESSION
                        for p in range(PERIODS_PER_SESSION - d + 1):
                            slots.append(base + p + 1)
            self.valid_slots[m] = slots

        # Thứ tự heuristic cơ sở để tạo cá thể greedy/noisy/hard-first
        self.base_order = sorted(
            self.all_cc,
            key=lambda x: (
                len(self.course_teachers[x[1]]),
                -self.durations[x[1]],
                -len(self.class_courses[x[0]]),
                x[0],
                x[1],
            )
        )

        self.base_teacher_order = {}
        for crs, teachers in self.course_teachers.items():
            self.base_teacher_order[crs] = sorted(
                teachers,
                key=lambda t: (self.teacher_load_score[t], t)
            )

        n = self.total_cc
        if n <= 200:
            auto_pop_size = 40
            auto_elite = 5
        elif n <= 1000:
            auto_pop_size = 32
            auto_elite = 4
        elif n <= 3000:
            auto_pop_size = 22
            auto_elite = 3
        elif n <= 8000:
            auto_pop_size = 14
            auto_elite = 2
        else:
            auto_pop_size = 8
            auto_elite = 2

        self.pop_size = pop_size if pop_size is not None else auto_pop_size
        self.elite = elite if elite is not None else auto_elite
        self.elite = min(self.elite, self.pop_size)

    # ---------------- Time helpers ----------------

    def elapsed(self):
        return time.perf_counter() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def timeout(self):
        return self.elapsed() >= self.time_limit

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------- Chromosome initialization ----------------

    def make_chromosome(self, mode="noisy"):
        order = list(self.base_order)

        if mode == "greedy":
            return order

        if mode == "random":
            self.rng.shuffle(order)
            return order

        if mode == "hard_first":
            def score(cc):
                cls, crs = cc
                few_teacher = len(self.course_teachers[crs])
                dur = self.durations[crs]
                cls_degree = len(self.class_courses[cls])
                jitter = self.rng.random() * 2.5
                return few_teacher * 8 - dur * 3 - cls_degree + jitter
            order.sort(key=score)
            return order

        # noisy: giữ bias heuristic nhưng đảo vị trí một phần
        n = len(order)
        max_noise = max(3, int(n * 0.20))
        pos = {cc: i for i, cc in enumerate(order)}
        order.sort(key=lambda cc: pos[cc] + self.rng.randint(-max_noise, max_noise))
        return order

    # ---------------- Crossover operators ----------------

    def order_crossover(self, a, b):
        """OX - Order Crossover for permutations."""
        n = len(a)
        if n <= 2:
            return list(a)
        i = self.rng.randint(0, n - 2)
        j = self.rng.randint(i + 1, n - 1)
        child = [None] * n
        used = set()
        for k in range(i, j + 1):
            child[k] = a[k]
            used.add(a[k])
        fill = [x for x in b if x not in used]
        p = 0
        for k in range(n):
            if child[k] is None:
                child[k] = fill[p]
                p += 1
        return child

    def pmx_crossover(self, a, b):
        """PMX - Partially Mapped Crossover for permutations."""
        n = len(a)
        if n <= 2:
            return list(a)
        i = self.rng.randint(0, n - 2)
        j = self.rng.randint(i + 1, n - 1)
        child = [None] * n

        # Copy segment from parent a
        child[i:j + 1] = a[i:j + 1]
        used = set(child[i:j + 1])

        # Mapping from b segment to a segment
        mapping = {b[k]: a[k] for k in range(i, j + 1)}

        for k in list(range(0, i)) + list(range(j + 1, n)):
            gene = b[k]
            while gene in used:
                gene = mapping.get(gene, gene)
                # Safety fallback for unusual cycles
                if gene in used and gene not in mapping:
                    break
            if gene not in used:
                child[k] = gene
                used.add(gene)

        # Fill any remaining None positions by parent b order
        fill = [x for x in b if x not in used]
        p = 0
        for k in range(n):
            if child[k] is None:
                child[k] = fill[p]
                used.add(fill[p])
                p += 1
        return child

    def position_crossover(self, a, b):
        """Position-based crossover: giữ ngẫu nhiên một số vị trí từ cha a, còn lại điền theo cha b."""
        n = len(a)
        if n <= 2:
            return list(a)
        child = [None] * n
        keep_count = self.rng.randint(max(1, n // 4), max(1, n // 2))
        positions = set(self.rng.sample(range(n), keep_count))
        used = set()
        for pos in positions:
            child[pos] = a[pos]
            used.add(a[pos])
        fill = [x for x in b if x not in used]
        p = 0
        for k in range(n):
            if child[k] is None:
                child[k] = fill[p]
                p += 1
        return child

    def crossover(self, a, b):
        if self.crossover_type == "pmx":
            return self.pmx_crossover(a, b)
        if self.crossover_type == "position":
            return self.position_crossover(a, b)
        return self.order_crossover(a, b)

    # ---------------- Mutation operators ----------------

    def mutate_swap(self, arr):
        n = len(arr)
        i, j = self.rng.sample(range(n), 2)
        arr[i], arr[j] = arr[j], arr[i]

    def mutate_insert(self, arr):
        n = len(arr)
        i, j = self.rng.sample(range(n), 2)
        item = arr.pop(i)
        arr.insert(j, item)

    def mutate_reverse(self, arr):
        n = len(arr)
        i = self.rng.randint(0, n - 2)
        j = self.rng.randint(i + 1, min(n - 1, i + max(2, n // 10)))
        arr[i:j + 1] = reversed(arr[i:j + 1])

    def mutate_scramble(self, arr):
        n = len(arr)
        i = self.rng.randint(0, n - 2)
        j = self.rng.randint(i + 1, min(n - 1, i + max(2, n // 8)))
        segment = arr[i:j + 1]
        self.rng.shuffle(segment)
        arr[i:j + 1] = segment

    def mutate_order(self, order):
        n = len(order)
        if n <= 1:
            return list(order)
        arr = list(order)

        ops = self.mutation_ops
        if ops is None:
            ops = 1 + min(12, max(1, n // 300))

        for _ in range(ops):
            mtype = self.mutation_type
            if mtype == "mixed":
                r = self.rng.random()
                if r < 0.35:
                    mtype = "swap"
                elif r < 0.65:
                    mtype = "insert"
                elif r < 0.85:
                    mtype = "reverse"
                else:
                    mtype = "scramble"

            if mtype == "swap":
                self.mutate_swap(arr)
            elif mtype == "insert":
                self.mutate_insert(arr)
            elif mtype == "reverse":
                self.mutate_reverse(arr)
            elif mtype == "scramble":
                self.mutate_scramble(arr)

        return arr

    # ---------------- Decode / constructive heuristic ----------------

    def teacher_order_for_decode(self, crs, random_level):
        teachers = list(self.base_teacher_order.get(crs, []))
        if random_level <= 0:
            return teachers
        teachers.sort(key=lambda t: self.teacher_load_score[t] + self.rng.random() * random_level * 5.0)
        return teachers

    def slot_order_for_decode(self, crs, random_level):
        slots = list(self.valid_slots.get(crs, []))
        if random_level <= 0:
            return slots
        if self.rng.random() < random_level:
            self.rng.shuffle(slots)
        else:
            slots.sort(key=lambda s: s + self.rng.random() * random_level * 20.0)
        return slots

    def decode(self, order, random_level=0.0):
        sched = Schedule(self.N, self.T)
        for cls, crs in order:
            if (cls, crs) in sched.assignments:
                continue
            dur = self.durations[crs]
            teachers = self.teacher_order_for_decode(crs, random_level)
            slots = self.slot_order_for_decode(crs, random_level)
            placed = False
            for t in teachers:
                if placed:
                    break
                for s in slots:
                    if sched.can_place(cls, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        placed = True
                        break
        return sched

    # ---------------- Selection / fitness ----------------

    def fitness(self, item):
        return len(item["schedule"])

    def select(self, population):
        k = min(self.tournament_k, len(population))
        sample = self.rng.sample(population, k)
        return max(sample, key=self.fitness)

    def make_individual(self, mode, random_level):
        order = self.make_chromosome(mode)
        sched = self.decode(order, random_level=random_level)
        return {"order": order, "schedule": sched}

    def record_history(self, gen, population, best, stagnant, phase="evolution"):
        fits = [self.fitness(x) for x in population] if population else [self.fitness(best)]
        best_fit = self.fitness(best)
        avg_fit = sum(fits) / len(fits)
        worst_fit = min(fits)
        self.history.append({
            "generation": gen,
            "phase": phase,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "worst_fitness": worst_fit,
            "total_tasks": self.total_cc,
            "best_rate": best_fit / self.total_cc if self.total_cc else 0,
            "avg_rate": avg_fit / self.total_cc if self.total_cc else 0,
            "elapsed_sec": self.elapsed(),
            "remaining_sec": self.remaining(),
            "stagnant": stagnant,
            "population_size": len(population),
            "elite": self.elite,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "crossover_type": self.crossover_type,
            "mutation_type": self.mutation_type,
            "tournament_k": self.tournament_k,
            "init_mode": self.init_mode,
            "seed": self.seed,
        })

    def solve(self):
        self.start_time = time.perf_counter()
        self.log(
            f"T={self.T} | N={self.N} | M={self.M} | total={self.total_cc} | "
            f"limit={self.time_limit}s | pop={self.pop_size} | elite={self.elite} | "
            f"cx={self.crossover_rate}({self.crossover_type}) | "
            f"mut={self.mutation_rate}({self.mutation_type}) | tk={self.tournament_k} | "
            f"init={self.init_mode}"
        )

        # Greedy baseline: chỉ decode greedy, không local search
        greedy_order = self.make_chromosome("greedy")
        greedy_sched = self.decode(greedy_order, random_level=0.0)
        best = {"order": greedy_order, "schedule": greedy_sched}
        greedy_val = len(greedy_sched)
        self.log(f"Greedy baseline: {greedy_val}/{self.total_cc}")

        if greedy_val == self.total_cc and self.init_mode == "heuristic":
            self.record_history(0, [best], best, 0, phase="greedy_full")
            return greedy_sched, greedy_val, greedy_val

        # Init population: không dùng local search.
        # heuristic: đưa greedy baseline vào quần thể để kết quả mạnh.
        # random: không đưa greedy vào quần thể, dùng để vẽ hội tụ rõ hơn.
        if self.init_mode == "random":
            population = []
            best = None
            modes = ["random"]
        else:
            population = [best]
            modes = ["noisy", "hard_first", "random"]

        init_deadline = self.start_time + min(self.time_limit * 0.20, max(1.0, self.time_limit - 0.5))
        while len(population) < self.pop_size and time.perf_counter() < init_deadline and not self.timeout():
            mode = modes[len(population) % len(modes)]
            if self.init_mode == "random":
                rl = 0.75 + 0.25 * self.rng.random()
            else:
                rl = 0.15 + 0.70 * self.rng.random()
            ind = self.make_individual(mode, rl)
            population.append(ind)
            if best is None or self.fitness(ind) > self.fitness(best):
                best = {"order": list(ind["order"]), "schedule": ind["schedule"]}

        if not population:
            # Trường hợp time-limit quá nhỏ: fallback để không lỗi.
            population = [best if best is not None else {"order": greedy_order, "schedule": greedy_sched}]
            best = population[0]

        self.log(f"Init: pop={len(population)} | best={self.fitness(best)}/{self.total_cc} | elapsed={self.elapsed():.2f}s")
        self.record_history(0, population, best, 0, phase="init")

        gen = 0
        stagnant = 0
        last_best = self.fitness(best)

        while not self.timeout():
            gen += 1
            if self.remaining() < 0.05:
                break

            population.sort(key=self.fitness, reverse=True)
            new_pop = population[:self.elite]

            while len(new_pop) < self.pop_size and not self.timeout():
                p1 = self.select(population)
                p2 = self.select(population)

                if self.rng.random() < self.crossover_rate:
                    child_order = self.crossover(p1["order"], p2["order"])
                else:
                    child_order = list(p1["order"])

                if self.rng.random() < self.mutation_rate:
                    child_order = self.mutate_order(child_order)

                random_level = 0.10 + 0.55 * self.rng.random()
                child_sched = self.decode(child_order, random_level=random_level)
                child = {"order": child_order, "schedule": child_sched}
                new_pop.append(child)

                if self.fitness(child) > self.fitness(best):
                    best = {"order": list(child_order), "schedule": child_sched}
                    self.log(f"  Gen {gen}: new best {self.fitness(best)}/{self.total_cc} | {self.elapsed():.2f}s")
                    if self.fitness(best) == self.total_cc:
                        population = new_pop
                        self.record_history(gen, population, best, stagnant, phase="evolution")
                        return best["schedule"], greedy_val, self.fitness(best)

            population = new_pop
            cur = self.fitness(best)
            if cur > last_best:
                stagnant = 0
                last_best = cur
            else:
                stagnant += 1

            if gen % self.log_every == 0:
                self.record_history(gen, population, best, stagnant, phase="evolution")

            if gen <= 3 or gen % 5 == 0:
                fits = [self.fitness(x) for x in population]
                avg = sum(fits) / len(fits)
                self.log(f"Gen {gen}: best={cur}/{self.total_cc} | avg={avg:.1f} | stagnant={stagnant} | remain={self.remaining():.1f}s")

            # Restart GA: vẫn là GA, chỉ giữ elite và sinh cá thể mới; không dùng local search
            if stagnant >= 10 and not self.timeout():
                population.sort(key=self.fitness, reverse=True)
                keep = population[:self.elite]
                population = keep[:]
                while len(population) < self.pop_size and not self.timeout():
                    mode = "hard_first" if self.rng.random() < 0.65 else "random"
                    rl = 0.35 + 0.60 * self.rng.random()
                    ind = self.make_individual(mode, rl)
                    population.append(ind)
                    if self.fitness(ind) > self.fitness(best):
                        best = {"order": list(ind["order"]), "schedule": ind["schedule"]}
                stagnant = 0
                self.record_history(gen, population, best, stagnant, phase="restart")

        ga_best = self.fitness(best)
        # Với init_mode=heuristic, greedy baseline nằm trong quần thể nên nghiệm cuối thường không kém greedy.
        # Với init_mode=random, greedy chỉ dùng để so sánh, không đưa vào quần thể.
        return best["schedule"], greedy_val, ga_best

    def format_assignments(self, sched):
        return [
            (cls, crs, start, teacher)
            for (cls, crs), (teacher, start) in sorted(sched.assignments.items())
        ]


# ============================================================
# Validation
# ============================================================

def validate_solution(T, N, M, class_courses, teacher_courses, durations, assignments):
    seen = set()
    class_busy = [bytearray(TOTAL_SLOTS) for _ in range(N + 1)]
    teacher_busy = [bytearray(TOTAL_SLOTS) for _ in range(T + 1)]

    class_req_set = [set() for _ in range(N + 1)]
    for cls in range(1, N + 1):
        class_req_set[cls] = set(class_courses[cls])

    for cls, crs, start, teacher in assignments:
        if not (1 <= cls <= N and 1 <= crs <= M and 1 <= teacher <= T):
            return False, f"ID ngoài phạm vi: {(cls, crs, start, teacher)}"
        if crs not in class_req_set[cls]:
            return False, f"Lớp {cls} không yêu cầu môn {crs}"
        if crs not in teacher_courses[teacher]:
            return False, f"GV {teacher} không dạy được môn {crs}"
        if (cls, crs) in seen:
            return False, f"Trùng lớp-môn {(cls, crs)}"
        seen.add((cls, crs))

        dur = durations[crs]
        if dur < 1 or dur > PERIODS_PER_SESSION:
            return False, f"duration không hợp lệ môn {crs}: {dur}"
        if start < 1 or start > TOTAL_SLOTS:
            return False, f"start ngoài phạm vi: {start}"
        start0 = start - 1
        if start0 % PERIODS_PER_SESSION + dur > PERIODS_PER_SESSION:
            return False, f"Môn {(cls, crs)} vắt qua buổi: start={start}, dur={dur}"

        for k in range(dur):
            slot = start0 + k
            if class_busy[cls][slot]:
                return False, f"Trùng lịch lớp {cls} tại slot {slot + 1}"
            if teacher_busy[teacher][slot]:
                return False, f"Trùng lịch GV {teacher} tại slot {slot + 1}"
            class_busy[cls][slot] = 1
            teacher_busy[teacher][slot] = 1

    return True, "OK"


# ============================================================
# CSV logging helpers
# ============================================================

HISTORY_FIELDS = [
    "dataset", "file", "generation", "phase",
    "best_fitness", "avg_fitness", "worst_fitness", "total_tasks",
    "best_rate", "avg_rate", "elapsed_sec", "remaining_sec", "stagnant",
    "population_size", "elite", "crossover_rate", "mutation_rate",
    "crossover_type", "mutation_type", "tournament_k", "init_mode", "seed",
]


def write_history_csv(filename, history, dataset, file_name):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    rows = []
    for row in history:
        r = dict(row)
        r["dataset"] = dataset
        r["file"] = file_name
        rows.append(r)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_history_csv(filename, history, dataset, file_name):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        for row in history:
            r = dict(row)
            r["dataset"] = dataset
            r["file"] = file_name
            writer.writerow(r)


def write_run_summary_csv(filename, rows):
    fields = [
        "dataset", "file", "obj", "total_cc", "success_rate", "greedy_baseline",
        "ga_best", "diff_vs_greedy", "time_sec", "status",
        "pop_size", "elite", "crossover_rate", "mutation_rate",
        "crossover_type", "mutation_type", "tournament_k", "init_mode", "seed", "time_limit"
    ]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main modes
# ============================================================

def solve_one_file(filepath, verbose=True, args=None):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    T, N, M, class_courses, teacher_courses, durations = parse_input(text)
    total_cc = sum(len(class_courses[cls]) for cls in range(1, N + 1))
    tl = args.time_limit if args and args.time_limit is not None else get_time_limit(total_cc)

    start = time.perf_counter()
    solver = GeneticTimetablingSolver(
        T, N, M, class_courses, teacher_courses, durations,
        time_limit=tl,
        seed=args.seed if args else SEED,
        verbose=verbose,
        pop_size=args.pop_size if args else None,
        elite=args.elite if args else None,
        tournament_k=args.tournament_k if args else 3,
        crossover_rate=args.crossover_rate if args else 0.85,
        mutation_rate=args.mutation_rate if args else 0.75,
        crossover_type=args.crossover_type if args else "ox",
        mutation_type=args.mutation_type if args else "mixed",
        mutation_ops=args.mutation_ops if args else None,
        init_mode=args.init_mode if args else "heuristic",
        log_every=args.log_every if args else 1,
    )
    best_sched, greedy_val, ga_val = solver.solve()
    exec_time = time.perf_counter() - start

    assignments = solver.format_assignments(best_sched)
    ok, msg = validate_solution(T, N, M, class_courses, teacher_courses, durations, assignments)
    status = "VALID" if ok else f"INVALID: {msg}"

    return {
        "assignments": assignments,
        "obj": len(assignments),
        "greedy_baseline": greedy_val,
        "ga_best": ga_val,
        "total_cc": total_cc,
        "time": exec_time,
        "status": status,
        "history": solver.history,
        "params": {
            "time_limit": tl,
            "seed": args.seed if args else SEED,
            "pop_size": solver.pop_size,
            "elite": solver.elite,
            "tournament_k": args.tournament_k if args else 3,
            "crossover_rate": args.crossover_rate if args else 0.85,
            "mutation_rate": args.mutation_rate if args else 0.75,
            "crossover_type": args.crossover_type if args else "ox",
            "mutation_type": args.mutation_type if args else "mixed",
            "init_mode": args.init_mode if args else "heuristic",
        },
    }


def main():
    global INPUT_DIR, RESULT_DIR, SOLVER_NAME, SEED, SUBFOLDERS

    args = build_arg_parser().parse_args()

    INPUT_DIR = args.input_dir
    SOLVER_NAME = args.solver_name
    SEED = args.seed
    SUBFOLDERS = args.subfolders.split(",") if args.subfolders else None

    run_name = make_run_name(args)
    RESULT_DIR = os.path.join(args.result_root, run_name)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(os.path.join(RESULT_DIR, "Run_Config.txt"), "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}: {v}\n")

    print(f"RUN: {run_name}")
    print(f"RESULT_DIR: {RESULT_DIR}")

    summary_rows = []
    all_history_file = os.path.join(RESULT_DIR, "All_History.csv")

    # Chạy 1 file
    if args.filepath is not None:
        filepath = args.filepath
        result = solve_one_file(filepath, verbose=True, args=args)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        dataset = os.path.basename(os.path.dirname(filepath)) or "."

        out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, result["assignments"], result["obj"], result["time"], result["status"])

        hist_file = os.path.join(RESULT_DIR, "History", f"{SOLVER_NAME}_{basename}_history.csv")
        write_history_csv(hist_file, result["history"], dataset, basename)
        append_history_csv(all_history_file, result["history"], dataset, basename)

        summary_rows.append(make_summary_row(dataset, basename, result))
        write_run_summary_csv(os.path.join(RESULT_DIR, "Run_Summary.csv"), summary_rows)

        print("\n=== RESULT ===")
        print(f"File: {filepath}")
        print(f"Greedy baseline: {result['greedy_baseline']}/{result['total_cc']}")
        print(f"Pure GA:         {result['obj']}/{result['total_cc']}")
        print(f"Status:          {result['status']}")
        print(f"Time:            {result['time']:.3f}s")
        print(f"Saved result:    {out_file}")
        print(f"Saved history:   {hist_file}")
        print(f"All history:     {all_history_file}")
        return

    # Chạy batch toàn bộ Datasets
    def sort_key(fname):
        nums = re.findall(r"\d+", fname)
        return [int(x) for x in nums] if nums else [0]

    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()
        if root == INPUT_DIR and SUBFOLDERS is not None:
            dirs[:] = [d for d in dirs if d in SUBFOLDERS]
        txts = sorted([f for f in files if f.endswith(".txt")], key=sort_key)
        for fname in txts:
            rel_dir = os.path.relpath(root, INPUT_DIR)
            all_files.append((rel_dir, os.path.join(root, fname)))

    if not all_files:
        print(f"Không tìm thấy file .txt trong {INPUT_DIR}")
        return

    group_count = Counter(rel for rel, _ in all_files)
    print(f"Tìm thấy {len(all_files)} file trong {len(group_count)} bộ dataset:")
    for grp, cnt in sorted(group_count.items()):
        print(f"  {grp}: {cnt} file")

    runs = 0
    total_time = 0.0
    group_stats = {}
    current_group = None

    for rel_dir, filepath in all_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        dataset = rel_dir

        if rel_dir != current_group:
            current_group = rel_dir
            group_stats[rel_dir] = {
                "runs": 0,
                "time": 0.0,
                "sum_obj": 0,
                "sum_total": 0,
                "sum_greedy": 0,
                "improved": 0,
                "equal": 0,
                "worse": 0,
                "invalid": 0,
            }
            print(f"\n{'=' * 70}")
            print(f"BỘ: {rel_dir}")
            print(f"{'=' * 70}")

        result = solve_one_file(filepath, verbose=False, args=args)
        out_dir = os.path.join(RESULT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, result["assignments"], result["obj"], result["time"], result["status"])

        hist_file = os.path.join(RESULT_DIR, "History", rel_dir, f"{SOLVER_NAME}_{basename}_history.csv")
        write_history_csv(hist_file, result["history"], dataset, basename)
        append_history_csv(all_history_file, result["history"], dataset, basename)

        diff = result["obj"] - result["greedy_baseline"]
        if diff > 0:
            mark = f"+{diff} BETTER"
            group_stats[rel_dir]["improved"] += 1
        elif diff == 0:
            mark = "= greedy"
            group_stats[rel_dir]["equal"] += 1
        else:
            mark = f"{diff} WORSE"
            group_stats[rel_dir]["worse"] += 1

        if not result["status"].startswith("VALID"):
            group_stats[rel_dir]["invalid"] += 1

        rate = result["obj"] / result["total_cc"] * 100 if result["total_cc"] else 0
        print(
            f"[{basename}] {result['obj']}/{result['total_cc']} ({rate:.1f}%) | "
            f"greedy={result['greedy_baseline']} | {mark} | {result['time']:.2f}s | {result['status']}"
        )

        runs += 1
        total_time += result["time"]
        st = group_stats[rel_dir]
        st["runs"] += 1
        st["time"] += result["time"]
        st["sum_obj"] += result["obj"]
        st["sum_total"] += result["total_cc"]
        st["sum_greedy"] += result["greedy_baseline"]
        summary_rows.append(make_summary_row(dataset, basename, result))

        write_overall(group_stats, runs, total_time)
        write_run_summary_csv(os.path.join(RESULT_DIR, "Run_Summary.csv"), summary_rows)

    write_overall(group_stats, runs, total_time)
    write_run_summary_csv(os.path.join(RESULT_DIR, "Run_Summary.csv"), summary_rows)
    print(f"\nHoàn thành {runs} file.")
    print(f"Kết quả lưu tại: {RESULT_DIR}")
    print(f"Log hội tụ tổng: {all_history_file}")
    print(f"Tổng hợp kết quả: {os.path.join(RESULT_DIR, 'Run_Summary.csv')}")


def make_summary_row(dataset, basename, result):
    p = result["params"]
    return {
        "dataset": dataset,
        "file": basename,
        "obj": result["obj"],
        "total_cc": result["total_cc"],
        "success_rate": result["obj"] / result["total_cc"] if result["total_cc"] else 0,
        "greedy_baseline": result["greedy_baseline"],
        "ga_best": result["ga_best"],
        "diff_vs_greedy": result["obj"] - result["greedy_baseline"],
        "time_sec": result["time"],
        "status": result["status"],
        "pop_size": p["pop_size"],
        "elite": p["elite"],
        "crossover_rate": p["crossover_rate"],
        "mutation_rate": p["mutation_rate"],
        "crossover_type": p["crossover_type"],
        "mutation_type": p["mutation_type"],
        "tournament_k": p["tournament_k"],
        "seed": p["seed"],
        "time_limit": p["time_limit"],
    }


def write_overall(group_stats, runs, total_time):
    overall_file = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
    with open(overall_file, "w", encoding="utf-8") as f:
        f.write(f"Thuật toán: {SOLVER_NAME}\n")
        f.write("Loại: Pure Genetic Algorithm with Randomized Greedy Decoder\n")
        f.write("Local search: Không sử dụng\n")
        f.write("Destroy-Repair/LNS: Không sử dụng\n")
        f.write(f"Tổng file: {runs}\n")
        f.write(f"Thời gian TB: {total_time / runs if runs else 0:.6f} giây\n\n")
        f.write(
            f"{'Bộ dataset':<25} {'Files':>6} {'Obj/Total':>18} {'Greedy':>10} "
            f"{'Better':>8} {'Equal':>8} {'Worse':>8} {'Invalid':>8} {'AvgTime':>10}\n"
        )
        f.write("-" * 115 + "\n")
        for grp, st in sorted(group_stats.items()):
            avg_time = st["time"] / st["runs"] if st["runs"] else 0
            f.write(
                f"{grp:<25} {st['runs']:>6} "
                f"{st['sum_obj']:>8}/{st['sum_total']:<8} "
                f"{st['sum_greedy']:>10} "
                f"{st['improved']:>8} {st['equal']:>8} {st['worse']:>8} {st['invalid']:>8} "
                f"{avg_time:>10.2f}\n"
            )


if __name__ == "__main__":
    main()
