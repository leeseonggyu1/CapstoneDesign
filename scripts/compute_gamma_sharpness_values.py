from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path("D:/KDG")
OUT = ROOT / "outputs" / "sharpness_detection_examples"
OUT.mkdir(parents=True, exist_ok=True)

PROJECT = next(
    p for p in sorted(ROOT.glob("CenterNet_Origin - *"))
    if (p / "src" / "lib" / "models" / "paper_filter_centernet.py").exists()
)
LIB = PROJECT / "src" / "lib"
sys.path.insert(0, str(LIB))

from models.paper_filter_centernet import IlluminationClassifier, FilterParameterEstimator  # noqa: E402
from utils.image import get_affine_transform  # noqa: E402


MEAN = np.array([0.40789655, 0.44719303, 0.47026116], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.2886383, 0.27408165, 0.27809834], dtype=np.float32).reshape(1, 1, 3)

STRICT_DIR = ROOT / "paper_strict_data" / "yeoju_7to3_seed317" / "coco_llvip_rgb" / "val2017"
RAW_YEOJU_DIR = ROOT / "Yeoju_rain" / "coco_llvip_rgb" / "val2017"
RAW_YEOJU_TRAIN_DIR = ROOT / "Yeoju_rain" / "coco_llvip_rgb" / "train2017"

EXAMPLE_IMAGES = [
    "001836_03.jpg",
    "001895_03.jpg",
    "001845_03.jpg",
    "002028_04.jpg",
]

PAPER_FILTER_MODELS = [
    {
        "name": "Paper-condition Proposed 10ep",
        "checkpoint": ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_bs2_10ep_seed317" / "model_last.pth",
        "gamma_range": 3.0,
        "sharpness_max": 5.0,
    },
    {
        "name": "Active filter original LR 5ep",
        "checkpoint": ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base0p000125_flr1_g3_s5_reset_seed317" / "model_5.pth",
        "gamma_range": 3.0,
        "sharpness_max": 5.0,
    },
    {
        "name": "Active filter aggressive 5ep",
        "checkpoint": ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base5Em05_flr50_g5_s8_reset_seed317" / "model_5.pth",
        "gamma_range": 5.0,
        "sharpness_max": 8.0,
    },
]


METRIC_RE = re.compile(
    r"epoch:\s*(?P<epoch>\d+).*?"
    r"gamma_mean\s+(?P<gamma_mean>[0-9.]+).*?"
    r"gamma_min\s+(?P<gamma_min>[0-9.]+).*?"
    r"gamma_max\s+(?P<gamma_max>[0-9.]+).*?"
    r"gamma_night_mean\s+(?P<gamma_night_mean>[0-9.]+).*?"
    r"sharpness_mean\s+(?P<sharpness_mean>[0-9.]+).*?"
    r"sharpness_min\s+(?P<sharpness_min>[0-9.]+).*?"
    r"sharpness_max\s+(?P<sharpness_max>[0-9.]+).*?"
    r"night_prob_mean\s+(?P<night_prob_mean>[0-9.]+)"
)

RUN_LOGS = [
    (
        "Baseline",
        ROOT / "paper_strict_runs" / "paper_strict_yeoju_baseline_bs2_10ep_seed317" / "logs_2026-05-31-04-13" / "log.txt",
    ),
    (
        "Paper-condition Proposed",
        ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_bs2_10ep_seed317" / "logs_2026-05-31-05-21" / "log.txt",
    ),
    (
        "Proposed finetune 10-50",
        ROOT / "paper_strict_runs" / "paper_strict_yeoju_proposed_10to50_lr5e5_ckpt5_seed317" / "logs_2026-06-01-05-55" / "log.txt",
    ),
    (
        "Active filter original LR",
        ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base0p000125_flr1_g3_s5_reset_seed317" / "logs_2026-06-03-00-05" / "log.txt",
    ),
    (
        "Active filter aggressive",
        ROOT / "paper_strict_runs" / "paper_active_yeoju_proposed10_extra30_base5Em05_flr50_g5_s8_reset_seed317" / "logs_2026-06-02-17-14" / "log.txt",
    ),
]


def fmt(value: float | str, digits: int = 4) -> str:
    if value == "" or value is None:
        return ""
    return f"{float(value):.{digits}f}"


def load_filter_parts(checkpoint: Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["state_dict"]
    illum = IlluminationClassifier()
    estimator = FilterParameterEstimator()
    illum_sd = {
        k.replace("illumination.", "", 1): v
        for k, v in state.items()
        if k.startswith("illumination.")
    }
    estimator_sd = {
        k.replace("parameter_estimator.", "", 1): v
        for k, v in state.items()
        if k.startswith("parameter_estimator.")
    }
    illum.load_state_dict(illum_sd, strict=True)
    estimator.load_state_dict(estimator_sd, strict=True)
    illum.to(device).eval()
    estimator.to(device).eval()
    return illum, estimator


def preprocess_for_centernet(image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(str(image_path))

    height, width = image.shape[:2]
    inp_h = inp_w = 512
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = max(height, width) * 1.0
    trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
    warped = cv2.warpAffine(image, trans, (inp_w, inp_h), flags=cv2.INTER_LINEAR)
    normalized = ((warped / 255.0 - MEAN) / STD).astype(np.float32)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor


def paper_filter_values(image_path: Path, model_conf: dict, device: torch.device) -> dict:
    illum, estimator = load_filter_parts(model_conf["checkpoint"], device)
    x = preprocess_for_centernet(image_path).to(device)
    mean = torch.tensor(MEAN.reshape(3), dtype=x.dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD.reshape(3), dtype=x.dtype, device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        img = torch.clamp(x * std + mean, 0.0, 1.0)
        day_night = illum(img)
        day_prob = day_night[:, 0:1]
        night_prob = day_night[:, 1:2]
        raw = estimator(img)
        gamma_raw = raw[:, 0:1]
        sharp_raw = raw[:, 1:2]

        gamma_night = torch.exp(torch.tanh(gamma_raw) * math.log(model_conf["gamma_range"]))
        gamma = day_prob * 1.0 + night_prob * gamma_night
        sharpness = torch.sigmoid(sharp_raw) * model_conf["sharpness_max"]
        sharpness = night_prob * sharpness

    return {
        "gamma": float(gamma.item()),
        "gamma_night": float(gamma_night.item()),
        "sharpness": float(sharpness.item()),
        "day_prob": float(day_prob.item()),
        "night_prob": float(night_prob.item()),
    }


def load_gap_info() -> dict[str, tuple[float, str]]:
    info = {}
    path = ROOT / "all_image_info.txt"
    if not path.exists():
        return info
    pattern = re.compile(r"\[(?P<file>[^,\]]+),\s*(?P<gamma>[0-9.]+),\s*(?P<label>[^\]]+)\]")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if m:
            info[m.group("file")] = (float(m.group("gamma")), m.group("label").strip())
    return info


def find_raw_image(fname: str) -> Path:
    for base in [RAW_YEOJU_DIR, RAW_YEOJU_TRAIN_DIR, STRICT_DIR]:
        path = base / fname
        if path.exists():
            return path
    return RAW_YEOJU_DIR / fname


def gap_local_values(image_path: Path, gap_gamma: float) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(str(image_path))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    _, _, h, w = tensor.shape
    grid_h = grid_w = 16
    block_h, block_w = h // grid_h, w // grid_w
    distance = gap_gamma - 1.0
    vals = []
    for i in range(grid_h):
        for j in range(grid_w):
            block = tensor[:, :, i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            luminance = 0.27 * block[:, 0] + 0.67 * block[:, 1] + 0.06 * block[:, 2]
            brightness = float(luminance.mean().item())
            multiplier = 1.2 if brightness < 0.5 else 0.8
            vals.append(1.0 + distance * multiplier)
    arr = np.array(vals, dtype=np.float32)
    return {
        "gap_gamma": gap_gamma,
        "lap_gamma_mean": float(arr.mean()),
        "lap_gamma_min": float(arr.min()),
        "lap_gamma_max": float(arr.max()),
        "sharpen_sigma": 1.5,
        "usm_detail_weight": 1.5,
        "blend_sharpened": 0.8,
        "effective_detail_weight": 1.2,
    }


def parse_gamma_movement() -> list[dict]:
    rows = []
    for label, log in RUN_LOGS:
        if not log.exists():
            continue
        parsed = []
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = METRIC_RE.search(line)
            if not m:
                continue
            row = {k: (int(v) if k == "epoch" else float(v)) for k, v in m.groupdict().items()}
            parsed.append(row)
        if not parsed:
            if label == "Baseline":
                rows.append({
                    "run": label,
                    "epoch_range": "1-10",
                    "gamma_mean_min": 1.0,
                    "gamma_mean_max": 1.0,
                    "last_gamma_mean": 1.0,
                    "last_gamma_min": 1.0,
                    "last_gamma_max": 1.0,
                    "last_sharpness_mean": 0.0,
                    "last_sharpness_min": 0.0,
                    "last_sharpness_max": 0.0,
                })
            continue
        last = sorted(parsed, key=lambda r: r["epoch"])[-1]
        rows.append({
            "run": label,
            "epoch_range": f"{min(r['epoch'] for r in parsed)}-{max(r['epoch'] for r in parsed)}",
            "gamma_mean_min": min(r["gamma_mean"] for r in parsed),
            "gamma_mean_max": max(r["gamma_mean"] for r in parsed),
            "last_gamma_mean": last["gamma_mean"],
            "last_gamma_min": last["gamma_min"],
            "last_gamma_max": last["gamma_max"],
            "last_sharpness_mean": last["sharpness_mean"],
            "last_sharpness_min": last["sharpness_min"],
            "last_sharpness_max": last["sharpness_max"],
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    movement = parse_gamma_movement()
    movement_csv = OUT / "yeoju_gamma_movement_summary.csv"
    write_csv(movement_csv, movement)

    rows = []
    for fname in EXAMPLE_IMAGES:
        image_path = STRICT_DIR / fname
        if not image_path.exists():
            image_path = RAW_YEOJU_DIR / fname
        if not image_path.exists():
            continue
        rows.append({
            "file": fname,
            "model": "Baseline",
            "gamma": 1.0,
            "gamma_night": "",
            "sharpness": 0.0,
            "day_prob": "",
            "night_prob": "",
            "note": "No gamma/sharpening filter",
        })
        for conf in PAPER_FILTER_MODELS:
            if not conf["checkpoint"].exists():
                continue
            vals = paper_filter_values(image_path, conf, device)
            rows.append({
                "file": fname,
                "model": conf["name"],
                "gamma": vals["gamma"],
                "gamma_night": vals["gamma_night"],
                "sharpness": vals["sharpness"],
                "day_prob": vals["day_prob"],
                "night_prob": vals["night_prob"],
                "note": f"paper_filter gamma_range={conf['gamma_range']}, sharpness_max={conf['sharpness_max']}",
            })

    image_csv = OUT / "example_image_gamma_sharpness.csv"
    write_csv(image_csv, rows)

    gap_rows = []
    gap_info = load_gap_info()
    for fname in EXAMPLE_IMAGES:
        image_path = find_raw_image(fname)
        if not image_path.exists() or fname not in gap_info:
            continue
        gap_gamma, label = gap_info[fname]
        vals = gap_local_values(image_path, gap_gamma)
        vals.update({"file": fname, "label": label})
        gap_rows.append(vals)
    gap_csv = OUT / "gap_prefilter_gamma_sharpness_values.csv"
    write_csv(gap_csv, gap_rows)

    md = OUT / "gamma_sharpness_values.md"
    lines = []
    lines.append("# Gamma and Sharpness Values")
    lines.append("")
    lines.append("## Gamma Movement")
    lines.append("")
    lines.append("| Run | Epoch range | Gamma mean min-max | Last gamma mean | Last gamma min-max | Last sharpness mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in movement:
        lines.append(
            "| {run} | {epoch} | {gmin}-{gmax} | {last} | {lmin}-{lmax} | {sharp} |".format(
                run=r["run"],
                epoch=r["epoch_range"],
                gmin=fmt(r["gamma_mean_min"]),
                gmax=fmt(r["gamma_mean_max"]),
                last=fmt(r["last_gamma_mean"]),
                lmin=fmt(r["last_gamma_min"]),
                lmax=fmt(r["last_gamma_max"]),
                sharp=fmt(r["last_sharpness_mean"]),
            )
        )
    lines.append("")
    lines.append("## Example Image Values")
    lines.append("")
    lines.append("| File | Model | Gamma | Gamma night | Sharpness | Night prob |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {file} | {model} | {gamma} | {gn} | {sharp} | {night} |".format(
                file=r["file"],
                model=r["model"],
                gamma=fmt(r["gamma"]),
                gn=fmt(r["gamma_night"]) if r["gamma_night"] != "" else "",
                sharp=fmt(r["sharpness"]),
                night=fmt(r["night_prob"]) if r["night_prob"] != "" else "",
            )
        )
    lines.append("")
    lines.append("## GAP Pre-filter Values")
    lines.append("")
    lines.append("| File | GAP gamma | LAP gamma mean | LAP gamma min-max | Fixed sharpening |")
    lines.append("|---|---:|---:|---:|---|")
    for r in gap_rows:
        lines.append(
            "| {file} | {gap} | {mean} | {minv}-{maxv} | sigma={sigma}, USM=1.5, blend=0.8 |".format(
                file=r["file"],
                gap=fmt(r["gap_gamma"]),
                mean=fmt(r["lap_gamma_mean"]),
                minv=fmt(r["lap_gamma_min"]),
                maxv=fmt(r["lap_gamma_max"]),
                sigma=fmt(r["sharpen_sigma"], 1),
            )
        )
    md.write_text("\n".join(lines), encoding="utf-8")

    print("movement:", movement_csv)
    print("image values:", image_csv)
    print("gap values:", gap_csv)
    print("summary:", md)


if __name__ == "__main__":
    main()
