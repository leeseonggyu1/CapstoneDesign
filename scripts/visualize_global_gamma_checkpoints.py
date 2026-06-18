import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class GammaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return 0.5 + x * 1.5


MEAN = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)


def checkpoint_sort_key(path):
    name = path.stem
    if name == "model_last":
        return 10**9
    try:
        return int(name.split("_")[1])
    except Exception:
        return 10**8


def find_checkpoints(run_dir):
    candidates = sorted(run_dir.glob("model_*.pth"), key=checkpoint_sort_key)
    model_last = run_dir / "model_last.pth"
    if model_last.exists() and model_last not in candidates:
        candidates.append(model_last)
    return sorted(candidates, key=checkpoint_sort_key)


def load_gamma_net(checkpoint_path, device):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["state_dict"]
    gamma_state = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("gamma_net."):
            gamma_state[key[len("gamma_net."):]] = value
    if not gamma_state:
        raise RuntimeError(f"No gamma_net parameters in {checkpoint_path}")
    model = GammaNet().to(device)
    model.load_state_dict(gamma_state, strict=True)
    model.eval()
    return model, int(checkpoint.get("epoch", -1))


def load_images(image_dir, sample_count):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]
    paths = sorted(paths)
    if sample_count > 0:
        paths = paths[:sample_count]
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    return paths


def preprocess_batch(paths):
    batch = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = img.transpose(2, 0, 1)
        batch.append(img)
    if not batch:
        raise RuntimeError("No readable images in batch")
    return torch.from_numpy(np.stack(batch, axis=0))


def evaluate_gamma(model, image_paths, device, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch = preprocess_batch(batch_paths).to(device)
            gamma = model(batch).detach().cpu().numpy().reshape(-1)
            values.extend([float(x) for x in gamma])
    return np.array(values, dtype=np.float32)


def write_chart(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable; skipped chart: {exc}")
        return

    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    mean = np.array([row["gamma_mean"] for row in rows], dtype=np.float32)
    p10 = np.array([row["gamma_p10"] for row in rows], dtype=np.float32)
    p90 = np.array([row["gamma_p90"] for row in rows], dtype=np.float32)
    gmin = np.array([row["gamma_min"] for row in rows], dtype=np.float32)
    gmax = np.array([row["gamma_max"] for row in rows], dtype=np.float32)

    plt.figure(figsize=(10, 5.5))
    plt.fill_between(x, p10, p90, color="#8ecae6", alpha=0.35, label="P10-P90")
    plt.plot(x, mean, marker="o", color="#023047", linewidth=2.2, label="Mean gamma")
    plt.plot(x, gmin, linestyle="--", color="#2a9d8f", alpha=0.75, label="Min")
    plt.plot(x, gmax, linestyle="--", color="#e76f51", alpha=0.75, label="Max")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Predicted gamma")
    plt.xlabel("Checkpoint")
    plt.title("Global Gamma Change by Checkpoint")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=r"D:\KDG\global_gamma_runs")
    parser.add_argument("--exp", default="global_gamma_bs8_64to120_lr5e5")
    parser.add_argument("--image-dir", default=r"D:\KDG\Yeoju_rain\coco_llvip_rgb\val2017")
    parser.add_argument("--out-dir", default=r"D:\KDG\global_gamma_64to120_gamma_visuals")
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_dir = Path(args.run_root) / args.exp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {run_dir}")

    image_paths = load_images(args.image_dir, args.sample_count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for ckpt in checkpoints:
        print(f"gamma visualize: {ckpt.name}")
        model, saved_epoch = load_gamma_net(ckpt, device)
        values = evaluate_gamma(model, image_paths, device, args.batch_size)
        label = ckpt.stem.replace("model_", "ep")
        if ckpt.stem == "model_last" and saved_epoch > 0:
            label = f"last({saved_epoch})"
        rows.append({
            "checkpoint": ckpt.name,
            "saved_epoch": saved_epoch,
            "label": label,
            "num_images": len(values),
            "gamma_mean": float(np.mean(values)),
            "gamma_std": float(np.std(values)),
            "gamma_min": float(np.min(values)),
            "gamma_p10": float(np.percentile(values, 10)),
            "gamma_p50": float(np.percentile(values, 50)),
            "gamma_p90": float(np.percentile(values, 90)),
            "gamma_max": float(np.max(values)),
        })

    csv_path = out_dir / "global_gamma_by_checkpoint.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    chart_path = out_dir / "global_gamma_by_checkpoint.png"
    write_chart(rows, chart_path)
    print(f"wrote {csv_path}")
    print(f"wrote {chart_path}")


if __name__ == "__main__":
    main()
