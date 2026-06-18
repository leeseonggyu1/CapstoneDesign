import csv
import re
from pathlib import Path

import numpy as np


EXP = "global_gamma_from0_bs12_50ep_trace"
TRAIN_LOG = Path(r"D:\KDG\global_gamma_trace_runs\_train_logs") / f"{EXP}.txt"
EVAL_DIR = Path(r"D:\KDG\global_gamma_from0_trace_eval_logs")
GAMMA_CSV = Path(r"D:\KDG\global_gamma_from0_trace_gamma_visuals\global_gamma_by_checkpoint.csv")
OUT_DIR = Path(r"D:\KDG\global_gamma_from0_trace_report")


PATTERNS = [
    ("AP", re.compile(r"Average Precision\s+\(AP\).*IoU=0\.50:0\.95.*area=\s+all.*=\s+([0-9.]+)")),
    ("AP50", re.compile(r"Average Precision\s+\(AP\).*IoU=0\.50\s+.*area=\s+all.*=\s+([0-9.]+)")),
    ("AP75", re.compile(r"Average Precision\s+\(AP\).*IoU=0\.75\s+.*area=\s+all.*=\s+([0-9.]+)")),
    ("AR100", re.compile(r"Average Recall\s+\(AR\).*IoU=0\.50:0\.95.*maxDets=100.*=\s+([0-9.]+)")),
]


TRAIN_RE = re.compile(
    r"train:\s+\[(?P<epoch>\d+)\]\[(?P<iter>\d+)/(?P<total>\d+)\].*?"
    r"\|loss\s+(?P<loss>[0-9.]+).*?"
    r"\|hm_loss\s+(?P<hm>[0-9.]+).*?"
    r"\|wh_loss\s+(?P<wh>[0-9.]+).*?"
    r"\|off_loss\s+(?P<off>[0-9.]+).*?"
    r"\|gamma_loss\s+(?P<gamma_loss>[0-9.]+).*?"
    r"\|gamma_mean\s+(?P<gamma_mean>[0-9.]+).*?"
    r"\|gamma_min\s+(?P<gamma_min>[0-9.]+).*?"
    r"\|gamma_max\s+(?P<gamma_max>[0-9.]+)"
)


def clean_text(path):
    return path.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ckpt_to_epoch(name):
    name = str(name).strip()
    if name.endswith(".pth"):
        name = name[:-4]
    try:
        return int(name.split("_")[1])
    except Exception:
        return None


def parse_eval_logs():
    rows = {}
    for path in EVAL_DIR.glob(f"eval_{EXP}_model_*.txt"):
        epoch = ckpt_to_epoch(path.stem.split(f"eval_{EXP}_")[-1])
        if epoch is None:
            continue
        text = clean_text(path)
        row = {"epoch": epoch}
        for key, pattern in PATTERNS:
            match = pattern.search(text)
            row[key] = float(match.group(1)) if match else ""
        rows[epoch] = row
    return rows


def parse_gamma_rows():
    rows = {}
    for row in read_csv(GAMMA_CSV):
        epoch = ckpt_to_epoch(row.get("checkpoint", ""))
        if epoch is None:
            continue
        rows[epoch] = row
    return rows


def parse_train_loss():
    rows = {}
    if not TRAIN_LOG.exists():
        return rows
    for match in TRAIN_RE.finditer(clean_text(TRAIN_LOG)):
        epoch = int(match.group("epoch"))
        item = {
            "loss": float(match.group("loss")),
            "hm_loss": float(match.group("hm")),
            "wh_loss": float(match.group("wh")),
            "off_loss": float(match.group("off")),
            "gamma_loss_train": float(match.group("gamma_loss")),
            "gamma_mean_train": float(match.group("gamma_mean")),
            "gamma_min_train": float(match.group("gamma_min")),
            "gamma_max_train": float(match.group("gamma_max")),
        }
        rows.setdefault(epoch, []).append(item)
    agg = {}
    for epoch, items in rows.items():
        agg[epoch] = {"epoch": epoch}
        for key in items[0].keys():
            agg[epoch][key] = float(np.mean([item[key] for item in items]))
    return agg


def fmt(value, digits=3):
    if value in ("", None):
        return ""
    return f"{float(value):.{digits}f}"


def build_rows():
    eval_rows = parse_eval_logs()
    gamma_rows = parse_gamma_rows()
    loss_rows = parse_train_loss()
    epochs = sorted(set(eval_rows) | set(gamma_rows) | set(loss_rows))
    rows = []
    for epoch in epochs:
        erow = eval_rows.get(epoch, {})
        grow = gamma_rows.get(epoch, {})
        lrow = loss_rows.get(epoch, {})
        rows.append({
            "epoch": epoch,
            "train_loss": lrow.get("loss", ""),
            "hm_loss": lrow.get("hm_loss", ""),
            "wh_loss": lrow.get("wh_loss", ""),
            "off_loss": lrow.get("off_loss", ""),
            "gamma_loss_train": lrow.get("gamma_loss_train", ""),
            "gamma_mean_train": lrow.get("gamma_mean_train", ""),
            "gamma_mean_eval": grow.get("gamma_mean", ""),
            "gamma_p10": grow.get("gamma_p10", ""),
            "gamma_p90": grow.get("gamma_p90", ""),
            "AP": erow.get("AP", ""),
            "AP50": erow.get("AP50", ""),
            "AP75": erow.get("AP75", ""),
            "AR100": erow.get("AR100", ""),
        })
    return rows


def write_table(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "global_gamma_from0_trace_report.csv"
    md_path = OUT_DIR / "global_gamma_from0_trace_report.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Epoch | Train Loss | Gamma Mean | AP | AP50 | AP75 | AR100 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {loss} | {gamma} | {ap} | {ap50} | {ap75} | {ar100} |".format(
                epoch=row["epoch"],
                loss=fmt(row["train_loss"], 4),
                gamma=fmt(row["gamma_mean_eval"] or row["gamma_mean_train"], 4),
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
    except Exception as exc:
        print(f"matplotlib unavailable; skipped charts: {exc}")
        return []

    paths = []
    loss_rows = [row for row in rows if row["train_loss"] not in ("", None)]
    if loss_rows:
        x = [row["epoch"] for row in loss_rows]
        path = OUT_DIR / "loss_change_by_epoch.png"
        plt.figure(figsize=(10, 5.2))
        plt.plot(x, [float(row["train_loss"]) for row in loss_rows], marker="o", linewidth=2.3, label="Total Loss")
        plt.plot(x, [float(row["hm_loss"]) for row in loss_rows], marker="o", linewidth=1.8, label="HM Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("Training Loss by Epoch")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    gamma_rows = [row for row in rows if row["gamma_mean_eval"] not in ("", None)]
    if gamma_rows:
        x = [row["epoch"] for row in gamma_rows]
        path = OUT_DIR / "gamma_change_by_epoch.png"
        plt.figure(figsize=(10, 5.2))
        plt.fill_between(
            x,
            [float(row["gamma_p10"]) for row in gamma_rows],
            [float(row["gamma_p90"]) for row in gamma_rows],
            color="#9ecae1",
            alpha=0.35,
            label="P10-P90",
        )
        plt.plot(x, [float(row["gamma_mean_eval"]) for row in gamma_rows], marker="o", linewidth=2.3, label="Gamma Mean")
        plt.xlabel("Epoch")
        plt.ylabel("Predicted gamma")
        plt.title("Predicted Global Gamma by Epoch")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    metric_rows = [row for row in rows if row["AP"] not in ("", None)]
    if metric_rows:
        x = [row["epoch"] for row in metric_rows]
        path = OUT_DIR / "ap_change_by_epoch.png"
        plt.figure(figsize=(10, 5.2))
        for key in ["AP", "AP50", "AP75", "AR100"]:
            plt.plot(x, [float(row[key]) for row in metric_rows], marker="o", linewidth=2.2, label=key)
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Detection Metrics by Epoch")
        plt.grid(True, axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)
    return paths


def main():
    rows = build_rows()
    if not rows:
        print("no rows to report")
        return
    csv_path, md_path = write_table(rows)
    charts = write_charts(rows)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    for chart in charts:
        print(f"wrote {chart}")


if __name__ == "__main__":
    main()
