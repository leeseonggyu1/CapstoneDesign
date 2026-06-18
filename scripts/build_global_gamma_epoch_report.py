import csv
from pathlib import Path


EVAL_CSV = Path(r"D:\KDG\global_gamma_64to120_eval_logs\global_gamma_64to120_eval_summary.csv")
GAMMA_CSV = Path(r"D:\KDG\global_gamma_64to120_gamma_visuals\global_gamma_by_checkpoint.csv")
OUT_DIR = Path(r"D:\KDG\global_gamma_64to120_report")


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_checkpoint(name):
    name = str(name).strip()
    if name.endswith(".pth"):
        name = name[:-4]
    return name


def checkpoint_sort_key(name):
    name = normalize_checkpoint(name)
    if name == "model_last":
        return 10**9
    try:
        return int(name.split("_")[1])
    except Exception:
        return 10**8


def fmt(value, digits=3):
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def make_rows(eval_rows, gamma_rows):
    eval_by_ckpt = {normalize_checkpoint(row["checkpoint"]): row for row in eval_rows}
    gamma_by_ckpt = {normalize_checkpoint(row["checkpoint"]): row for row in gamma_rows}
    checkpoints = sorted(set(eval_by_ckpt) | set(gamma_by_ckpt), key=checkpoint_sort_key)
    rows = []
    for ckpt in checkpoints:
        erow = eval_by_ckpt.get(ckpt, {})
        grow = gamma_by_ckpt.get(ckpt, {})
        relative_epoch = "last" if ckpt == "model_last" else ckpt.replace("model_", "")
        total_epoch = ""
        if relative_epoch.isdigit():
            total_epoch = str(64 + int(relative_epoch))
        elif grow.get("saved_epoch", "").isdigit():
            total_epoch = str(64 + int(grow["saved_epoch"]))
        rows.append({
            "checkpoint": ckpt,
            "relative_epoch": relative_epoch,
            "total_epoch_from_64": total_epoch,
            "gamma_mean": grow.get("gamma_mean", ""),
            "gamma_std": grow.get("gamma_std", ""),
            "gamma_min": grow.get("gamma_min", ""),
            "gamma_p10": grow.get("gamma_p10", ""),
            "gamma_p50": grow.get("gamma_p50", ""),
            "gamma_p90": grow.get("gamma_p90", ""),
            "gamma_max": grow.get("gamma_max", ""),
            "AP": erow.get("AP", ""),
            "AP50": erow.get("AP50", ""),
            "AP75": erow.get("AP75", ""),
            "AR100": erow.get("AR100", ""),
        })
    return rows


def write_outputs(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "global_gamma_epoch_report.csv"
    md_path = OUT_DIR / "global_gamma_epoch_report.md"

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Checkpoint | Total Epoch | Gamma Mean | Gamma P10 | Gamma P90 | AP | AP50 | AP75 | AR100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {ckpt} | {total} | {gmean} | {gp10} | {gp90} | {ap} | {ap50} | {ap75} | {ar100} |".format(
                ckpt=row["checkpoint"],
                total=row["total_epoch_from_64"] or row["relative_epoch"],
                gmean=fmt(row["gamma_mean"], 4),
                gp10=fmt(row["gamma_p10"], 4),
                gp90=fmt(row["gamma_p90"], 4),
                ap=fmt(row["AP"]),
                ap50=fmt(row["AP50"]),
                ap75=fmt(row["AP75"]),
                ar100=fmt(row["AR100"]),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_charts(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"matplotlib unavailable; skipped charts: {exc}")
        return []

    plot_rows = [row for row in rows if row.get("gamma_mean") not in ("", None)]
    if not plot_rows:
        print("no gamma rows available; skipped charts")
        return []

    labels = [
        row["total_epoch_from_64"] if row["total_epoch_from_64"] else row["relative_epoch"]
        for row in plot_rows
    ]
    x = np.arange(len(labels))

    gamma_mean = np.array([float(row["gamma_mean"]) for row in plot_rows])
    gamma_p10 = np.array([float(row["gamma_p10"]) for row in plot_rows])
    gamma_p90 = np.array([float(row["gamma_p90"]) for row in plot_rows])

    paths = []

    gamma_path = OUT_DIR / "gamma_change_by_epoch.png"
    plt.figure(figsize=(10, 5.2))
    plt.fill_between(x, gamma_p10, gamma_p90, color="#9ecae1", alpha=0.35, label="Gamma P10-P90")
    plt.plot(x, gamma_mean, marker="o", linewidth=2.4, color="#08306b", label="Gamma Mean")
    plt.xticks(x, labels)
    plt.xlabel("Total epoch")
    plt.ylabel("Predicted gamma")
    plt.title("Predicted Global Gamma by Epoch")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(gamma_path, dpi=180)
    plt.close()
    paths.append(gamma_path)

    metric_rows = [row for row in rows if row.get("AP") not in ("", None)]
    if metric_rows:
        metric_labels = [
            row["total_epoch_from_64"] if row["total_epoch_from_64"] else row["relative_epoch"]
            for row in metric_rows
        ]
        mx = np.arange(len(metric_labels))
        ap = np.array([float(row["AP"]) for row in metric_rows])
        ap50 = np.array([float(row["AP50"]) for row in metric_rows])
        ap75 = np.array([float(row["AP75"]) for row in metric_rows])
        ar100 = np.array([float(row["AR100"]) for row in metric_rows])

        metric_path = OUT_DIR / "ap_change_by_epoch.png"
        plt.figure(figsize=(10, 5.2))
        plt.plot(mx, ap, marker="o", linewidth=2.2, label="AP")
        plt.plot(mx, ap50, marker="o", linewidth=2.2, label="AP50")
        plt.plot(mx, ap75, marker="o", linewidth=2.2, label="AP75")
        plt.plot(mx, ar100, marker="o", linewidth=2.2, label="AR100")
        plt.xticks(mx, metric_labels)
        plt.xlabel("Total epoch")
        plt.ylabel("Score")
        plt.title("Detection Metrics by Epoch")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(metric_path, dpi=180)
        plt.close()
        paths.append(metric_path)

        joined = []
        gamma_by_ckpt = {row["checkpoint"]: row for row in plot_rows}
        for row in metric_rows:
            if row["checkpoint"] in gamma_by_ckpt:
                joined.append((row, gamma_by_ckpt[row["checkpoint"]]))
        if joined:
            combo_labels = [
                row["total_epoch_from_64"] if row["total_epoch_from_64"] else row["relative_epoch"]
                for row, _ in joined
            ]
            cx = np.arange(len(combo_labels))
            combo_ap = np.array([float(row["AP"]) for row, _ in joined])
            combo_gamma = np.array([float(grow["gamma_mean"]) for _, grow in joined])

            combo_path = OUT_DIR / "gamma_mean_vs_ap_by_epoch.png"
            fig, ax1 = plt.subplots(figsize=(10, 5.2))
            ax1.plot(cx, combo_gamma, marker="o", linewidth=2.4, color="#08306b", label="Gamma Mean")
            ax1.set_xlabel("Total epoch")
            ax1.set_ylabel("Gamma Mean", color="#08306b")
            ax1.tick_params(axis="y", labelcolor="#08306b")
            ax1.set_xticks(cx)
            ax1.set_xticklabels(combo_labels)
            ax1.grid(True, axis="y", alpha=0.22)

            ax2 = ax1.twinx()
            ax2.plot(cx, combo_ap, marker="s", linewidth=2.4, color="#d94801", label="AP")
            ax2.set_ylabel("AP", color="#d94801")
            ax2.tick_params(axis="y", labelcolor="#d94801")
            plt.title("Gamma Mean and AP by Epoch")
            fig.tight_layout()
            plt.savefig(combo_path, dpi=180)
            plt.close()
            paths.append(combo_path)

    return paths


def main():
    eval_rows = load_csv(EVAL_CSV)
    gamma_rows = load_csv(GAMMA_CSV)
    rows = make_rows(eval_rows, gamma_rows)
    if not rows:
        print("no rows to report")
        return
    csv_path, md_path = write_outputs(rows)
    chart_paths = write_charts(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for path in chart_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
