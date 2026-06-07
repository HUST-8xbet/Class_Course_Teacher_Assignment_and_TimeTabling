"""
Cách dùng tự động chọn config có overall_success_rate cao nhất làm GA chính trong hình so sánh Greedy:
    python plot_ga_experiments.py --result-root Result --out-dir Figure/GA_Figures
    
# Nếu muốn ép cấu hình chính là bản base heuristic OX + mixed:
python plot_ga_experiments.py --result-root Result --out-dir Figure/GA_Figures --best-config "Heuristic OX + Mixed"



Output:
    Figure/GA_Figures/
        01_ga_vs_greedy_by_dataset.png/pdf
        02_gain_vs_greedy_by_config.png/pdf
        03_mutation_rate_comparison.png/pdf
        04_ox_vs_pmx_comparison.png/pdf
        05_init_mode_comparison.png/pdf
        06_runtime_by_dataset.png/pdf
        07_convergence_selected_files.png/pdf
        08_success_rate_heatmap.png/pdf
        Tables/
            config_summary.csv
            dataset_config_summary.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAG_ALIASES = {
    "ga_base_ox_mixed": "Heuristic OX + Mixed",
    "ga_low_mutation_0p4": "Heuristic Mut=0.40",
    "ga_high_mutation_0p9": "Heuristic Mut=0.90",
    "ga_pmx_mixed": "Heuristic PMX + Mixed",
    "r_base_ox_mixed": "Random OX + Mixed",
    "r_low_mutation_0p4": "Random Mut=0.40",
    "r_high_mutation_0p9": "Random Mut=0.90",
    "r_pmx_mixed": "Random PMX + Mixed",
}

CONFIG_ORDER = [
    "Heuristic OX + Mixed",
    "Heuristic Mut=0.40",
    "Heuristic Mut=0.90",
    "Heuristic PMX + Mixed",
    "Random OX + Mixed",
    "Random Mut=0.40",
    "Random Mut=0.90",
    "Random PMX + Mixed",
]

DATASET_ORDER = ["Adversarial", "Exponential", "Gaussian", "hustack", "Poisson", "Uniform"]


# ----------------------------- helpers -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(out_dir: Path, name: str) -> None:
    ensure_dir(out_dir)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()
    print(f"[saved] {png}")
    print(f"[saved] {pdf}")


def parse_config_from_folder(folder_name: str) -> Dict[str, object]:
    """Lấy thông tin cấu hình từ tên folder do genetic_timetabling_solver_cli.py tạo."""
    tag = None
    for key in TAG_ALIASES:
        if key in folder_name:
            tag = key
            break

    init_mode = "random" if "initrandom" in folder_name or (tag or "").startswith("r_") else "heuristic"

    crossover_type = "pmx" if "pmx" in folder_name else "ox" if "ox" in folder_name else "unknown"

    mutation_rate = None
    m = re.search(r"mut([0-9]+p?[0-9]*)_", folder_name)
    if m:
        raw = m.group(1).replace("p", ".")
        try:
            mutation_rate = float(raw)
        except ValueError:
            mutation_rate = None

    crossover_rate = None
    m = re.search(r"cx([0-9]+p?[0-9]*)_", folder_name)
    if m:
        raw = m.group(1).replace("p", ".")
        try:
            crossover_rate = float(raw)
        except ValueError:
            crossover_rate = None

    display_name = TAG_ALIASES.get(tag, folder_name)

    return {
        "run_folder": folder_name,
        "tag": tag or folder_name,
        "config": display_name,
        "init_mode": init_mode,
        "crossover_type": crossover_type,
        "mutation_rate": mutation_rate,
        "crossover_rate": crossover_rate,
    }


def sort_configs(labels: List[str]) -> List[str]:
    known = [x for x in CONFIG_ORDER if x in labels]
    unknown = sorted([x for x in labels if x not in known])
    return known + unknown


def sort_datasets(labels: List[str]) -> List[str]:
    known = [x for x in DATASET_ORDER if x in labels]
    unknown = sorted([x for x in labels if x not in known])
    return known + unknown


def extract_size_from_file_name(file_name: str) -> Optional[int]:
    """Cố gắng lấy N/size từ tên file dạng xxx_20_30_10 hoặc có số."""
    nums = re.findall(r"\d+", str(file_name))
    if not nums:
        return None
    return int(nums[0])


def load_results(result_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    history_frames = []

    for run_dir in sorted(result_root.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_file = run_dir / "Run_Summary.csv"
        history_file = run_dir / "All_History.csv"
        if not summary_file.exists():
            continue

        cfg = parse_config_from_folder(run_dir.name)
        summary = pd.read_csv(summary_file)
        for k, v in cfg.items():
            summary[k] = v
        summary["size_hint"] = summary["file"].apply(extract_size_from_file_name)
        summary_frames.append(summary)

        if history_file.exists():
            hist = pd.read_csv(history_file)
            for k, v in cfg.items():
                hist[k] = v
            history_frames.append(hist)

    if not summary_frames:
        raise FileNotFoundError(
            f"Không tìm thấy Run_Summary.csv trong {result_root}. "
            "Hãy chạy các lệnh trong run_pure_ga_experiments.txt trước."
        )

    all_summary = pd.concat(summary_frames, ignore_index=True)
    all_history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()

    # Chuẩn hóa một số cột số
    for col in ["obj", "total_cc", "success_rate", "greedy_baseline", "diff_vs_greedy", "time_sec"]:
        if col in all_summary.columns:
            all_summary[col] = pd.to_numeric(all_summary[col], errors="coerce")

    if not all_history.empty:
        for col in ["generation", "best_fitness", "avg_fitness", "worst_fitness", "total_tasks", "best_rate", "avg_rate", "elapsed_sec"]:
            if col in all_history.columns:
                all_history[col] = pd.to_numeric(all_history[col], errors="coerce")

    return all_summary, all_history


def build_summary_tables(df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    table_dir = out_dir / "Tables"
    ensure_dir(table_dir)

    config_summary = (
        df.groupby("config", as_index=False)
        .agg(
            files=("file", "count"),
            obj=("obj", "sum"),
            total_cc=("total_cc", "sum"),
            greedy=("greedy_baseline", "sum"),
            mean_success_rate=("success_rate", "mean"),
            mean_gain=("diff_vs_greedy", "mean"),
            total_gain=("diff_vs_greedy", "sum"),
            better=("diff_vs_greedy", lambda x: int((x > 0).sum())),
            equal=("diff_vs_greedy", lambda x: int((x == 0).sum())),
            worse=("diff_vs_greedy", lambda x: int((x < 0).sum())),
            mean_time_sec=("time_sec", "mean"),
        )
    )
    config_summary["overall_success_rate"] = config_summary["obj"] / config_summary["total_cc"]
    config_summary["overall_greedy_rate"] = config_summary["greedy"] / config_summary["total_cc"]
    config_summary["gain_rate_vs_greedy"] = (config_summary["obj"] - config_summary["greedy"]) / config_summary["total_cc"]
    config_summary["config"] = pd.Categorical(config_summary["config"], categories=sort_configs(config_summary["config"].tolist()), ordered=True)
    config_summary = config_summary.sort_values("config")

    dataset_config_summary = (
        df.groupby(["dataset", "config"], as_index=False)
        .agg(
            files=("file", "count"),
            obj=("obj", "sum"),
            total_cc=("total_cc", "sum"),
            greedy=("greedy_baseline", "sum"),
            mean_success_rate=("success_rate", "mean"),
            mean_gain=("diff_vs_greedy", "mean"),
            total_gain=("diff_vs_greedy", "sum"),
            mean_time_sec=("time_sec", "mean"),
        )
    )
    dataset_config_summary["overall_success_rate"] = dataset_config_summary["obj"] / dataset_config_summary["total_cc"]
    dataset_config_summary["overall_greedy_rate"] = dataset_config_summary["greedy"] / dataset_config_summary["total_cc"]
    dataset_config_summary["gain_rate_vs_greedy"] = (dataset_config_summary["obj"] - dataset_config_summary["greedy"]) / dataset_config_summary["total_cc"]

    config_summary.to_csv(table_dir / "config_summary.csv", index=False)
    dataset_config_summary.to_csv(table_dir / "dataset_config_summary.csv", index=False)
    return config_summary, dataset_config_summary


# ----------------------------- plotting functions -----------------------------

def plot_ga_vs_greedy_by_dataset(df: pd.DataFrame, out_dir: Path, best_config: Optional[str]) -> None:
    if best_config is None:
        best_config = (
            df.groupby("config")["success_rate"].mean().sort_values(ascending=False).index[0]
        )
    sub = df[df["config"] == best_config].copy()
    g = sub.groupby("dataset", as_index=False).agg(obj=("obj", "sum"), total=("total_cc", "sum"), greedy=("greedy_baseline", "sum"))
    g["GA"] = g["obj"] / g["total"] * 100
    g["Greedy"] = g["greedy"] / g["total"] * 100
    datasets = sort_datasets(g["dataset"].tolist())
    g = g.set_index("dataset").loc[datasets].reset_index()

    x = np.arange(len(g))
    width = 0.36
    plt.figure(figsize=(11, 5.8))
    plt.bar(x - width / 2, g["Greedy"], width, label="Greedy baseline")
    plt.bar(x + width / 2, g["GA"], width, label=f"GA: {best_config}")
    plt.xticks(x, g["dataset"], rotation=20, ha="right")
    plt.ylabel("Success rate (%)")
    plt.xlabel("Dataset")
    plt.title("GA vs Greedy baseline theo từng bộ dữ liệu")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "01_ga_vs_greedy_by_dataset")


def plot_gain_by_config(config_summary: pd.DataFrame, out_dir: Path) -> None:
    cs = config_summary.copy()
    labels = cs["config"].astype(str).tolist()
    x = np.arange(len(cs))
    y = cs["gain_rate_vs_greedy"] * 100
    plt.figure(figsize=(12, 5.8))
    plt.bar(x, y)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Gain vs Greedy (% tổng lớp-môn)")
    plt.xlabel("Cấu hình GA")
    plt.title("Mức cải thiện của từng cấu hình GA so với Greedy")
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "02_gain_vs_greedy_by_config")


def plot_mutation_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    # Chỉ so sánh OX mixed, tách heuristic và random
    sub = df[(df["crossover_type"] == "ox") & (df["mutation_rate"].isin([0.4, 0.75, 0.9]))].copy()
    if sub.empty:
        return
    g = sub.groupby(["init_mode", "mutation_rate"], as_index=False).agg(success=("success_rate", "mean"), gain=("diff_vs_greedy", "mean"))

    rates = sorted(g["mutation_rate"].dropna().unique())
    inits = [x for x in ["heuristic", "random"] if x in g["init_mode"].unique()]
    x = np.arange(len(rates))
    width = 0.36 if len(inits) > 1 else 0.55

    plt.figure(figsize=(9, 5.6))
    for i, init in enumerate(inits):
        vals = []
        for r in rates:
            row = g[(g["init_mode"] == init) & (g["mutation_rate"] == r)]
            vals.append(row["success"].iloc[0] * 100 if not row.empty else np.nan)
        offset = (i - (len(inits) - 1) / 2) * width
        plt.bar(x + offset, vals, width, label=f"init={init}")
    plt.xticks(x, [str(r) for r in rates])
    plt.xlabel("Mutation rate")
    plt.ylabel("Mean success rate (%)")
    plt.title("Ảnh hưởng của mutation rate đến chất lượng nghiệm")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "03_mutation_rate_comparison")


def plot_ox_vs_pmx(df: pd.DataFrame, out_dir: Path) -> None:
    sub = df[(df["mutation_rate"] == 0.75) & (df["crossover_type"].isin(["ox", "pmx"]))].copy()
    if sub.empty:
        return
    g = sub.groupby(["init_mode", "crossover_type"], as_index=False).agg(success=("success_rate", "mean"), gain=("diff_vs_greedy", "mean"))
    crosses = [x for x in ["ox", "pmx"] if x in g["crossover_type"].unique()]
    inits = [x for x in ["heuristic", "random"] if x in g["init_mode"].unique()]
    x = np.arange(len(crosses))
    width = 0.36 if len(inits) > 1 else 0.55

    plt.figure(figsize=(8.5, 5.4))
    for i, init in enumerate(inits):
        vals = []
        for c in crosses:
            row = g[(g["init_mode"] == init) & (g["crossover_type"] == c)]
            vals.append(row["success"].iloc[0] * 100 if not row.empty else np.nan)
        offset = (i - (len(inits) - 1) / 2) * width
        plt.bar(x + offset, vals, width, label=f"init={init}")
    plt.xticks(x, [c.upper() for c in crosses])
    plt.xlabel("Crossover type")
    plt.ylabel("Mean success rate (%)")
    plt.title("So sánh OX và PMX trên chromosome dạng hoán vị")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "04_ox_vs_pmx_comparison")


def plot_init_mode_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    # So sánh base OX mixed mutation 0.75 giữa heuristic và random
    sub = df[(df["crossover_type"] == "ox") & (df["mutation_rate"] == 0.75)].copy()
    if sub.empty:
        return
    g = sub.groupby(["dataset", "init_mode"], as_index=False).agg(success=("success_rate", "mean"))
    datasets = sort_datasets(g["dataset"].unique().tolist())
    inits = [x for x in ["heuristic", "random"] if x in g["init_mode"].unique()]
    x = np.arange(len(datasets))
    width = 0.36 if len(inits) > 1 else 0.55

    plt.figure(figsize=(11, 5.8))
    for i, init in enumerate(inits):
        vals = []
        for d in datasets:
            row = g[(g["dataset"] == d) & (g["init_mode"] == init)]
            vals.append(row["success"].iloc[0] * 100 if not row.empty else np.nan)
        offset = (i - (len(inits) - 1) / 2) * width
        plt.bar(x + offset, vals, width, label=f"init={init}")
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Mean success rate (%)")
    plt.title("Vai trò của khởi tạo heuristic so với random")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "05_init_mode_comparison")


def plot_runtime_by_dataset(df: pd.DataFrame, out_dir: Path, best_config: Optional[str]) -> None:
    if best_config is None:
        best_config = df.groupby("config")["success_rate"].mean().sort_values(ascending=False).index[0]
    sub = df[df["config"] == best_config].copy()
    g = sub.groupby("dataset", as_index=False).agg(mean_time=("time_sec", "mean"), mean_total=("total_cc", "mean"))
    datasets = sort_datasets(g["dataset"].tolist())
    g = g.set_index("dataset").loc[datasets].reset_index()

    plt.figure(figsize=(10.5, 5.6))
    plt.bar(np.arange(len(g)), g["mean_time"])
    plt.xticks(np.arange(len(g)), g["dataset"], rotation=20, ha="right")
    plt.xlabel("Dataset")
    plt.ylabel("Average runtime (s)")
    plt.title(f"Thời gian chạy trung bình của cấu hình {best_config}")
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "06_runtime_by_dataset")


def choose_representative_files(df: pd.DataFrame, max_files: int = 3) -> List[Tuple[str, str]]:
    """Chọn vài file đại diện có nhiều cấu hình GA nhất."""
    counts = df.groupby(["dataset", "file"])["config"].nunique().reset_index(name="n_cfg")
    counts = counts.sort_values(["n_cfg", "dataset", "file"], ascending=[False, True, True])
    return list(counts[["dataset", "file"]].head(max_files).itertuples(index=False, name=None))


def plot_convergence(history: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    if history.empty:
        return
    selected = choose_representative_files(summary, max_files=3)
    configs = sort_configs(history["config"].unique().tolist())

    for idx, (dataset, file_name) in enumerate(selected, start=1):
        sub = history[(history["dataset"] == dataset) & (history["file"] == file_name) & (history["config"].isin(configs))].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(9.5, 5.6))
        for cfg in configs:
            h = sub[sub["config"] == cfg].sort_values("generation")
            if h.empty:
                continue
            y_col = "best_rate" if "best_rate" in h.columns else None
            if y_col and h[y_col].notna().any():
                y = h[y_col] * 100
                ylabel = "Best success rate (%)"
            else:
                y = h["best_fitness"]
                ylabel = "Best fitness"
            plt.plot(h["generation"], y, marker="o", markersize=2.5, linewidth=1.5, label=cfg)
        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.title(f"Đường hội tụ GA: {dataset}/{file_name}")
        plt.legend()
        plt.grid(alpha=0.25)
        save_fig(out_dir, f"07_convergence_selected_file_{idx}")


def plot_success_heatmap(dataset_config_summary: pd.DataFrame, out_dir: Path) -> None:
    pivot = dataset_config_summary.pivot(index="dataset", columns="config", values="overall_success_rate") * 100
    datasets = sort_datasets(pivot.index.tolist())
    configs = sort_configs(pivot.columns.tolist())
    pivot = pivot.loc[datasets, configs]

    plt.figure(figsize=(13, 5.8))
    data = pivot.to_numpy()
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label="Success rate (%)")
    plt.xticks(np.arange(len(configs)), configs, rotation=30, ha="right")
    plt.yticks(np.arange(len(datasets)), datasets)
    plt.xlabel("Cấu hình GA")
    plt.ylabel("Dataset")
    plt.title("Heatmap success rate theo dataset và cấu hình")

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8)
    save_fig(out_dir, "08_success_rate_heatmap")


def plot_gain_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    configs = sort_configs(df["config"].unique().tolist())
    values = [df[df["config"] == cfg]["diff_vs_greedy"].dropna().values for cfg in configs]
    if not values:
        return
    plt.figure(figsize=(12, 5.8))
    plt.boxplot(values, labels=configs, showmeans=True)
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Diff vs Greedy trên từng file")
    plt.xlabel("Cấu hình GA")
    plt.title("Phân phối mức cải thiện so với Greedy trên từng file")
    plt.grid(axis="y", alpha=0.25)
    save_fig(out_dir, "09_gain_boxplot_by_config")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="Result", help="Thư mục chứa các kết quả chạy GA")
    parser.add_argument("--out-dir", default="Figure/GA_Figures", help="Thư mục lưu hình")
    parser.add_argument("--best-config", default=None, help="Tên cấu hình muốn dùng làm GA chính trong hình so sánh Greedy")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    summary, history = load_results(result_root)
    config_summary, dataset_config_summary = build_summary_tables(summary, out_dir)

    best_config = args.best_config
    if best_config is None:
        best_config = config_summary.sort_values("overall_success_rate", ascending=False)["config"].astype(str).iloc[0]
        print(f"[info] best_config tự chọn theo overall_success_rate: {best_config}")

    plot_ga_vs_greedy_by_dataset(summary, out_dir, best_config)
    plot_gain_by_config(config_summary, out_dir)
    plot_mutation_comparison(summary, out_dir)
    plot_ox_vs_pmx(summary, out_dir)
    plot_init_mode_comparison(summary, out_dir)
    plot_runtime_by_dataset(summary, out_dir, best_config)
    plot_convergence(history, summary, out_dir)
    plot_success_heatmap(dataset_config_summary, out_dir)
    plot_gain_boxplot(summary, out_dir)

    print("\nHoàn tất. Hãy chèn các file PNG/PDF trong thư mục:", out_dir)
    print("Bảng tổng hợp:", out_dir / "Tables" / "config_summary.csv")
    print("Bảng theo dataset:", out_dir / "Tables" / "dataset_config_summary.csv")


if __name__ == "__main__":
    main()
