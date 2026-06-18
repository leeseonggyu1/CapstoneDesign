from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("D:/KDG")
OUT = ROOT / "outputs" / "sharpness_detection_examples"


RUNS = [
    {
        "name": "baseline",
        "label": "Baseline",
        "log": ROOT / "paper_strict_runs" / "paper_strict_yeoju_baseline_bs2_10ep_seed317" / "logs_2026-05-31-04-13" / "log.txt",
        "ap_csv": None,
    },
    {
        "name": "paper_proposed_10ep",
        "label": "Paper-condition Proposed",
        "log": ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_bs2_10ep_seed317" / "logs_2026-05-31-05-21" / "log.txt",
        "ap_csv": ROOT / "paper_strict_runs" / "paper_strict_summary.csv",
    },
    {
        "name": "proposed_10to50_finetune",
        "label": "Proposed finetune 10-50",
        "log": ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_10to50_lr5e5_ckpt5_seed317" / "logs_2026-06-01-05-55" / "log.txt",
        "ap_csv": ROOT / "paper_strict_runs" / "paper_strict_proposed_finetune_10to50_summary.csv",
    },
    {
        "name": "active_filter_original_lr",
        "label": "Active filter original LR",
        "log": ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base0p000125_flr1_g3_s5_reset_seed317" / "logs_2026-06-03-00-05" / "log.txt",
        "ap_csv": ROOT / "paper_strict_runs" / "paper_active_filter_both_proposed10_extra30_base0p000125_flr1_g3_s5_reset_summary.csv",
    },
    {
        "name": "active_filter_aggressive",
        "label": "Active filter aggressive",
        "log": ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base5Em05_flr50_g5_s8_reset_seed317" / "logs_2026-06-02-17-14" / "log.txt",
        "ap_csv": ROOT / "paper_strict_runs" / "paper_active_filter_yeoju_proposed10_extra30_base5em05_flr50_g5_s8_reset_summary.csv",
    },
]


STRICT_ANN = ROOT / "paper_strict_data" / "yeoju_7to3_seed317" / "coco_llvip_rgb" / "annotations" / "instances_val2017.json"
STRICT_IMG_DIR = ROOT / "paper_strict_data" / "yeoju_7to3_seed317" / "coco_llvip_rgb" / "val2017"
GAP_ANN = ROOT / "Yeoju_rain_filtering" / "coco_llvip_rgb" / "annotations" / "instances_val2017.json"
GAP_IMG_DIR = ROOT / "Yeoju_rain_filtering" / "coco_llvip_rgb" / "images" / "val2017"

RESULTS = {
    "baseline": ROOT / "paper_strict_runs" / "paper_strict_yeoju_baseline_bs2_10ep_seed317_eval" / "results.json",
    "paper_proposed_10ep": ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_bs2_10ep_seed317_eval" / "results.json",
    "active_filter_e5": ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base0p000125_flr1_g3_s5_reset_seed317_eval_model_5" / "results.json",
}


def find_first(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern))
    return matches[0] if matches else None


GAP_RESULTS = {
    "gap_last": find_first("CenterNet_Origin - */exp/ctdet/predecessor_proposed_yeoju_gap_bs2_10ep_20260604_125142_eval/results.json"),
    "gap_best": find_first("CenterNet_Origin - */exp/ctdet/predecessor_proposed_yeoju_gap_bs2_10ep_20260604_125142_eval_best/results.json"),
}


METRIC_RE = re.compile(
    r"epoch:\s*(?P<epoch>\d+).*?"
    r"loss\s+(?P<loss>[0-9.]+).*?"
    r"gamma_loss\s+(?P<gamma_loss>[0-9.]+).*?"
    r"gamma_mean\s+(?P<gamma_mean>[0-9.]+).*?"
    r"gamma_min\s+(?P<gamma_min>[0-9.]+).*?"
    r"gamma_max\s+(?P<gamma_max>[0-9.]+).*?"
    r"gamma_night_mean\s+(?P<gamma_night_mean>[0-9.]+).*?"
    r"sharpness_mean\s+(?P<sharpness_mean>[0-9.]+).*?"
    r"sharpness_min\s+(?P<sharpness_min>[0-9.]+).*?"
    r"sharpness_max\s+(?P<sharpness_max>[0-9.]+).*?"
    r"night_prob_mean\s+(?P<night_prob_mean>[0-9.]+)"
)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_log(run: dict) -> list[dict]:
    rows = []
    for line in read_text(run["log"]).splitlines():
        m = METRIC_RE.search(line)
        if not m:
            continue
        row = {"run": run["name"], "label": run["label"]}
        for k, v in m.groupdict().items():
            row[k] = int(v) if k == "epoch" else float(v)
        rows.append(row)
    return rows


def load_ap_rows() -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}

    strict_path = ROOT / "paper_strict_runs" / "paper_strict_summary.csv"
    if strict_path.exists():
        with strict_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                dataset = row.get("Dataset") or row.get("dataset") or ""
                model = row.get("Model") or row.get("model") or ""
                if dataset == "Yeoju/CCTV" and model == "CenterNet":
                    out[("baseline", 10)] = row
                if dataset == "Yeoju/CCTV" and model == "Proposed CenterNet":
                    out[("paper_proposed_10ep", 10)] = row

    finetune = ROOT / "paper_strict_runs" / "paper_strict_proposed_finetune_10to50_summary.csv"
    if finetune.exists():
        with finetune.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") == "Yeoju/CCTV":
                    epoch = int(row["epoch"])
                    out[("proposed_10to50_finetune", epoch)] = row

    active_files = [
        ROOT / "paper_strict_runs" / "paper_active_filter_both_proposed10_extra30_base0p000125_flr1_g3_s5_reset_summary.csv",
        ROOT / "paper_strict_runs" / "paper_active_filter_yeoju_proposed10_extra30_base5em05_flr50_g5_s8_reset_summary.csv",
    ]
    for path in active_files:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("dataset") != "Yeoju/CCTV":
                    continue
                exp = row.get("exp", "")
                epoch = int(row["checkpoint_epoch"])
                if "flr1" in exp:
                    out[("active_filter_original_lr", epoch)] = row
                elif "flr50" in exp:
                    out[("active_filter_aggressive", epoch)] = row
    return out


def add_ap(rows: list[dict]) -> list[dict]:
    ap_rows = load_ap_rows()
    for row in rows:
        ap = ap_rows.get((row["run"], int(row["epoch"])))
        if ap:
            row["AP"] = float(ap["AP"])
            row["AP50"] = float(ap["AP50"])
            row["AP75"] = float(ap["AP75"])
            row["AR100"] = float(ap["AR100"])
        else:
            row["AP"] = ""
            row["AP50"] = ""
            row["AP75"] = ""
            row["AR100"] = ""
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "run",
        "label",
        "epoch",
        "loss",
        "gamma_loss",
        "gamma_mean",
        "gamma_min",
        "gamma_max",
        "gamma_night_mean",
        "sharpness_mean",
        "sharpness_min",
        "sharpness_max",
        "night_prob_mean",
        "AP",
        "AP50",
        "AP75",
        "AR100",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def safe_float(value, default=math.nan):
    try:
        return float(value)
    except Exception:
        return default


def make_chart(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    colors = {
        "baseline": "#6b7280",
        "paper_proposed_10ep": "#2563eb",
        "proposed_10to50_finetune": "#0f766e",
        "active_filter_original_lr": "#ca8a04",
        "active_filter_aggressive": "#dc2626",
    }
    plt.figure(figsize=(11, 6.2))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["run"]].append(row)
    for run, group in grouped.items():
        group = sorted(group, key=lambda x: x["epoch"])
        xs = [r["epoch"] for r in group]
        ys = [r["sharpness_mean"] for r in group]
        label = group[0]["label"]
        plt.plot(xs, ys, marker="o", linewidth=2.1, markersize=4, label=label, color=colors.get(run))
    plt.title("Yeoju/CCTV sharpening value movement")
    plt.xlabel("Epoch")
    plt.ylabel("sharpness_mean")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_ap_sharpness_chart(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    pts = [r for r in rows if r.get("AP") != ""]
    if not pts:
        return
    plt.figure(figsize=(8, 5.8))
    for run in sorted({r["run"] for r in pts}):
        group = [r for r in pts if r["run"] == run]
        x = [r["sharpness_mean"] for r in group]
        y = [safe_float(r["AP50"]) for r in group]
        plt.scatter(x, y, s=52, label=group[0]["label"])
        for r in group:
            plt.text(r["sharpness_mean"], safe_float(r["AP50"]) + 0.001, str(r["epoch"]), fontsize=7)
    plt.title("Yeoju/CCTV AP50 vs sharpening")
    plt.xlabel("sharpness_mean")
    plt.ylabel("AP50")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_coco_index(ann_path: Path) -> tuple[dict, dict, dict]:
    data = load_json(ann_path)
    images = {img["id"]: img for img in data["images"]}
    cats = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
    anns: dict[int, list] = defaultdict(list)
    for ann in data.get("annotations", []):
        anns[ann["image_id"]].append(ann)
    return images, cats, anns


def load_detections(path: Path) -> dict[int, list[dict]]:
    dets = load_json(path)
    by_img: dict[int, list[dict]] = defaultdict(list)
    for d in dets:
        by_img[int(d["image_id"])].append(d)
    for img_id in by_img:
        by_img[img_id].sort(key=lambda d: d.get("score", 0), reverse=True)
    return by_img


def pick_examples(det_maps: dict[str, dict[int, list]], images: dict, anns: dict, count=3) -> list[int]:
    scores = []
    for image_id, img in images.items():
        name = img.get("file_name", "")
        night_bonus = 1 if ("_03" in name or "_04" in name) else 0
        gt_bonus = min(len(anns.get(image_id, [])), 8) * 0.2
        base = [d for d in det_maps["baseline"].get(image_id, []) if d.get("score", 0) >= 0.35]
        active = [d for d in det_maps["active_filter_e5"].get(image_id, []) if d.get("score", 0) >= 0.35]
        paper = [d for d in det_maps["paper_proposed_10ep"].get(image_id, []) if d.get("score", 0) >= 0.35]
        if not (base or active or paper):
            continue
        active_sum = sum(d["score"] for d in active[:8])
        diff = abs(len(active) - len(base)) * 0.3
        score = active_sum + len(active) * 0.5 + diff + night_bonus + gt_bonus
        scores.append((score, image_id))
    scores.sort(reverse=True)
    return [image_id for _, image_id in scores[:count]]


def get_font(size=18):
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_boxes(
    image_path: Path,
    detections: list[dict],
    gt_anns: list[dict],
    cats: dict,
    title: str,
    out_size=(720, 405),
    threshold=0.35,
):
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size
    panel = img.resize(out_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    sx = out_size[0] / src_w
    sy = out_size[1] / src_h
    font = get_font(16)
    small = get_font(13)

    draw.rectangle([0, 0, out_size[0], 32], fill=(10, 18, 30))
    draw.text((10, 7), title, fill=(255, 255, 255), font=font)

    for ann in gt_anns[:80]:
        x, y, w, h = ann["bbox"]
        box = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
        draw.rectangle(box, outline=(180, 180, 180), width=1)

    dets = [d for d in detections if d.get("score", 0) >= threshold][:12]
    for d in dets:
        x, y, w, h = d["bbox"]
        box = [x * sx, y * sy, (x + w) * sx, (y + h) * sy]
        color = (255, 90, 70) if d["category_id"] == 1 else (0, 210, 130)
        draw.rectangle(box, outline=color, width=3)
        label = f"{cats.get(d['category_id'], d['category_id'])} {d.get('score', 0):.2f}"
        tw = draw.textlength(label, font=small)
        tx, ty = box[0], max(34, box[1] - 18)
        draw.rectangle([tx, ty, tx + tw + 6, ty + 17], fill=color)
        draw.text((tx + 3, ty + 1), label, fill=(0, 0, 0), font=small)

    return panel, len(dets)


def make_comparison_examples():
    images, cats, anns = load_coco_index(STRICT_ANN)
    det_maps = {k: load_detections(v) for k, v in RESULTS.items() if v.exists()}
    chosen = pick_examples(det_maps, images, anns, count=3)

    outputs = []
    for rank, image_id in enumerate(chosen, 1):
        info = images[image_id]
        image_path = STRICT_IMG_DIR / info["file_name"]
        panels = []
        titles = {
            "baseline": "Baseline CenterNet",
            "paper_proposed_10ep": "Proposed 10ep",
            "active_filter_e5": "Active filter epoch 5",
        }
        counts = {}
        for key in ["baseline", "paper_proposed_10ep", "active_filter_e5"]:
            panel, cnt = draw_boxes(
                image_path,
                det_maps[key].get(image_id, []),
                anns.get(image_id, []),
                cats,
                titles[key],
            )
            panels.append(panel)
            counts[key] = cnt
        w = sum(p.width for p in panels)
        h = panels[0].height + 58
        canvas = Image.new("RGB", (w, h), (245, 247, 250))
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width
        draw = ImageDraw.Draw(canvas)
        font = get_font(18)
        caption = (
            f"image_id={image_id}, file={info['file_name']} | "
            f"GT={len(anns.get(image_id, []))}, "
            f"det>=0.35 baseline/proposed/active="
            f"{counts['baseline']}/{counts['paper_proposed_10ep']}/{counts['active_filter_e5']}"
        )
        draw.text((12, panels[0].height + 17), caption, fill=(20, 30, 45), font=font)
        out_path = OUT / f"yeoju_centernet_compare_{rank}_{Path(info['file_name']).stem}.png"
        canvas.save(out_path)
        outputs.append(out_path)
    return outputs


def make_gap_example():
    gap_paths = {k: v for k, v in GAP_RESULTS.items() if v and v.exists()}
    if not gap_paths:
        return []
    images, cats, anns = load_coco_index(GAP_ANN)
    det_maps = {k: load_detections(v) for k, v in gap_paths.items()}
    base_key = next(iter(det_maps))
    candidates = []
    for image_id, img in images.items():
        dets = [d for d in det_maps[base_key].get(image_id, []) if d.get("score", 0) >= 0.35]
        if dets:
            night_bonus = 1 if ("_03" in img.get("file_name", "") or "_04" in img.get("file_name", "")) else 0
            candidates.append((sum(d["score"] for d in dets[:8]) + night_bonus, image_id))
    candidates.sort(reverse=True)
    if not candidates:
        return []
    image_id = candidates[0][1]
    info = images[image_id]
    image_path = GAP_IMG_DIR / info["file_name"]
    panels = []
    for key, title in [("gap_last", "GAP proposed last"), ("gap_best", "GAP proposed best")]:
        if key not in det_maps:
            continue
        panel, _ = draw_boxes(
            image_path,
            det_maps[key].get(image_id, []),
            anns.get(image_id, []),
            cats,
            title,
        )
        panels.append(panel)
    if not panels:
        return []
    w = sum(p.width for p in panels)
    h = panels[0].height + 58
    canvas = Image.new("RGB", (w, h), (245, 247, 250))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)
    draw.text((12, panels[0].height + 17), f"GAP filtered image_id={image_id}, file={info['file_name']}", fill=(20, 30, 45), font=font)
    out_path = OUT / f"yeoju_gap_repro_detection_{Path(info['file_name']).stem}.png"
    canvas.save(out_path)
    return [out_path]


def write_summary(rows: list[dict], image_paths: list[Path]):
    summary_path = OUT / "summary.md"
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["run"]].append(row)

    lines = []
    lines.append("# Sharpness and Detection Example Summary")
    lines.append("")
    lines.append("## Sharpness Movement")
    lines.append("")
    lines.append("| Run | Epoch range | Sharpness mean range | Last sharpness | Best AP50 epoch | Best AP50 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for run, group in by_run.items():
        group = sorted(group, key=lambda r: r["epoch"])
        vals = [r["sharpness_mean"] for r in group]
        ap_rows = [r for r in group if r.get("AP50") != ""]
        best = max(ap_rows, key=lambda r: safe_float(r["AP50"], -1)) if ap_rows else None
        lines.append(
            "| {label} | {e0}-{e1} | {vmin:.4f}-{vmax:.4f} | {last:.4f} | {be} | {ba} |".format(
                label=group[0]["label"],
                e0=group[0]["epoch"],
                e1=group[-1]["epoch"],
                vmin=min(vals),
                vmax=max(vals),
                last=group[-1]["sharpness_mean"],
                be=best["epoch"] if best else "",
                ba=f"{safe_float(best['AP50']):.3f}" if best else "",
            )
        )
    lines.append("")
    lines.append("Predecessor GAP uses fixed pre-filtering, so there is no checkpoint-wise learned sharpness trace. The script settings are sigma=1.5, USM detail weight=1.5, blend=0.8 sharpened + 0.2 blurred.")
    lines.append("")
    lines.append("## Generated Images")
    lines.append("")
    for p in image_paths:
        lines.append(f"- {p}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in RUNS:
        rows.extend(parse_log(run))
    rows = add_ap(rows)
    csv_path = OUT / "yeoju_sharpness_movement.csv"
    write_csv(rows, csv_path)
    make_chart(rows, OUT / "yeoju_sharpness_movement.png")
    make_ap_sharpness_chart(rows, OUT / "yeoju_ap50_vs_sharpness.png")
    image_paths = []
    image_paths.extend(make_comparison_examples())
    image_paths.extend(make_gap_example())
    summary_path = write_summary(rows, image_paths)

    print(f"CSV: {csv_path}")
    print(f"Chart: {OUT / 'yeoju_sharpness_movement.png'}")
    print(f"AP50 chart: {OUT / 'yeoju_ap50_vs_sharpness.png'}")
    print(f"Summary: {summary_path}")
    for p in image_paths:
        print(f"Image: {p}")


if __name__ == "__main__":
    main()
