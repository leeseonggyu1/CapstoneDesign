import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="CenterNet src directory")
    parser.add_argument("--input-root", required=True, help="COCO-style input root")
    parser.add_argument("--output-root", required=True, help="COCO-style output root")
    parser.add_argument("--checkpoint", required=True, help="Trained paper-filter model")
    parser.add_argument("--illumination-ckpt", required=True, help="LLVIP illumination checkpoint")
    parser.add_argument("--arch", default="hourglass")
    parser.add_argument("--head-conv", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--gamma-range", type=float, default=3.0)
    parser.add_argument("--sharpness-max", type=float, default=5.0)
    parser.add_argument("--night-weight-sharpness", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def iter_images(split_dir):
    for path in sorted(split_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def build_model(args):
    os.environ["KDG_GAMMA_WRAPPER"] = "paper_filter"
    os.environ["KDG_ILLUMINATION_CKPT"] = args.illumination_ckpt
    os.environ["KDG_FREEZE_ILLUMINATION"] = "1"
    os.environ["KDG_PAPER_GAMMA_RANGE"] = str(args.gamma_range)
    os.environ["KDG_PAPER_SHARPNESS_MAX"] = str(args.sharpness_max)
    os.environ["KDG_PAPER_SHARPNESS_NIGHT_WEIGHT"] = str(args.night_weight_sharpness)

    sys.path.insert(0, args.src)
    from lib.models.model import create_model, load_model

    heads = {"hm": args.num_classes, "wh": 2, "reg": 2}
    model = create_model(args.arch, heads, args.head_conv)
    model = load_model(model, args.checkpoint)
    model.eval()
    return model


def enhance_image(wrapper, image, device):
    tensor = pil_to_tensor(image).float().div(255.0).unsqueeze(0).to(device)

    with torch.no_grad():
        day_night = wrapper.illumination(tensor)
        day_prob = day_night[:, 0:1]
        night_prob = day_night[:, 1:2]

        raw_params = wrapper.parameter_estimator(tensor)
        gamma_raw = raw_params[:, 0:1]
        sharpness_raw = raw_params[:, 1:2]

        log_gamma_range = math.log(wrapper.gamma_range)
        gamma_night = torch.exp(torch.tanh(gamma_raw) * log_gamma_range)
        gamma = day_prob * 1.0 + night_prob * gamma_night

        sharpness = torch.sigmoid(sharpness_raw) * wrapper.sharpness_max
        if wrapper.use_night_weight_for_sharpness:
            sharpness = night_prob * sharpness

        enhanced = torch.pow(
            torch.clamp(tensor, min=1e-4, max=1.0),
            gamma.view(-1, 1, 1, 1),
        )
        enhanced = wrapper.apply_usm(enhanced, sharpness)

    stats = {
        "day_prob": float(day_prob.item()),
        "night_prob": float(night_prob.item()),
        "gamma": float(gamma.item()),
        "gamma_night": float(gamma_night.item()),
        "sharpness": float(sharpness.item()),
    }
    out = to_pil_image(torch.clamp(enhanced.squeeze(0).cpu(), 0.0, 1.0))
    return out, stats


def copy_annotations(input_root, output_root):
    src = input_root / "annotations"
    dst = output_root / "annotations"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if input_root.resolve() == output_root.resolve():
        raise RuntimeError("input-root and output-root must be different")
    if not input_root.exists():
        raise RuntimeError(f"Missing input root: {input_root}")
    if not Path(args.checkpoint).exists():
        raise RuntimeError(f"Missing checkpoint: {args.checkpoint}")
    if not Path(args.illumination_ckpt).exists():
        raise RuntimeError(f"Missing illumination checkpoint: {args.illumination_ckpt}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    torch.backends.cudnn.benchmark = True
    model = build_model(args).to(device)

    output_root.mkdir(parents=True, exist_ok=True)
    copy_annotations(input_root, output_root)
    stats_path = output_root / "paper_filter_values.csv"

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "file_name",
                "day_prob",
                "night_prob",
                "gamma",
                "gamma_night",
                "sharpness",
            ],
        )
        writer.writeheader()

        for split in ["train", "val"]:
            split_dir = input_root / f"{split}2017"
            if not split_dir.exists():
                continue
            out_dir = output_root / f"{split}2017"
            out_dir.mkdir(parents=True, exist_ok=True)

            images = list(iter_images(split_dir))
            print(f"{split}: filtering {len(images)} images", flush=True)
            for index, path in enumerate(images, start=1):
                image = Image.open(path).convert("RGB")
                enhanced, stats = enhance_image(model, image, device)
                enhanced.save(out_dir / path.name)
                row = {"split": split, "file_name": path.name}
                row.update(stats)
                writer.writerow(row)
                if index % 100 == 0 or index == len(images):
                    print(f"{split}: {index}/{len(images)}", flush=True)

    print(f"filtered data: {output_root}", flush=True)
    print(f"filter values: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
