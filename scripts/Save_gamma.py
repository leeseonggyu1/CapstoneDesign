import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

class Config:
    def __init__(self, gamma_dark=1.3908, gamma_bright=1.0, brightness_threshold=0.5):
        self.gamma_dark = gamma_dark          # 어두운 영역에 적용할 감마값
        self.gamma_bright = gamma_bright      # 밝은 영역에 적용할 감마값
        self.brightness_threshold = brightness_threshold  # 밝기 임계값
        self.feature_extractor_dims = 64
        self.grid_size = (16, 16)
        self.sharpen_sigma = 3.0
        self.usm_amount = 1.3

def calculate_adaptive_gamma_map(img_tensor, config):
    """
    이미지의 각 블록 밝기에 따라 적응형 감마 맵을 계산합니다.
    밝은 영역: gamma = 1.0 (변화 없음)
    어두운 영역: gamma = gamma_dark (밝게 만듦)
    """
    _, _, h, w = img_tensor.shape
    grid_h, grid_w = config.grid_size
    block_h, block_w = h // grid_h, w // grid_w
    
    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    gamma_map = np.zeros((grid_h, grid_w))
    brightness_map = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            h_start, h_end = i * block_h, (i + 1) * block_h
            w_start, w_end = j * block_w, (j + 1) * block_w
            
            block = img_np[h_start:h_end, w_start:w_end, :]
            # Luminance 계산 (ITU-R BT.601 기준)
            luminance = 0.299 * block[:, :, 0] + 0.587 * block[:, :, 1] + 0.114 * block[:, :, 2]
            avg_brightness = luminance.mean()
            brightness_map[i, j] = avg_brightness
            
            # 밝기에 따라 감마값 할당
            if avg_brightness >= config.brightness_threshold:
                gamma_map[i, j] = config.gamma_bright  # 밝은 영역은 1.0
            else:
                gamma_map[i, j] = config.gamma_dark    # 어두운 영역은 높은 감마값
    
    return gamma_map, brightness_map

def apply_adaptive_gamma(img_tensor, gamma_map, config):
    """
    블록별로 다른 감마값을 적용합니다.
    """
    _, _, h, w = img_tensor.shape
    grid_h, grid_w = config.grid_size
    block_h, block_w = h // grid_h, w // grid_w
    
    result = img_tensor.clone()
    
    for i in range(grid_h):
        for j in range(grid_w):
            h_start, h_end = i * block_h, (i + 1) * block_h
            w_start, w_end = j * block_w, (j + 1) * block_w
            
            gamma_value = gamma_map[i, j]
            block = result[:, :, h_start:h_end, w_start:w_end]
            
            # 감마 보정 적용
            block_clamped = torch.clamp(block, min=0.0001)
            block_corrected = torch.pow(block_clamped, gamma_value)
            result[:, :, h_start:h_end, w_start:w_end] = block_corrected
    
    return torch.clamp(result, 0, 1)

def save_gamma_heatmap(img_tensor, gamma_map, brightness_map, config, output_directory, image_name):
    """
    감마 맵을 16x16 그리드 히트맵으로 저장합니다.
    """
    grid_h, grid_w = config.grid_size
    
    # 감마 히트맵 생성 (단일 플롯)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 감마 맵 히트맵 (origin='upper'가 기본값이므로 명시 안 해도 됨)
    im = ax.imshow(gamma_map, cmap='RdYlGn', vmin=0.5, vmax=2.0)
    
    # x축, y축 틱 설정
    ax.set_xticks(range(grid_w))
    ax.set_yticks(range(grid_h))
    ax.set_xticklabels(range(1, grid_w + 1))
    ax.set_yticklabels(range(grid_h, 0, -1))  # 16, 15, 14, ..., 1 (역순)
    
    plt.colorbar(im, ax=ax, label='Gamma Value')
    
    # 각 블록에 감마값 표시
    for i in range(grid_h):
        for j in range(grid_w):
            gamma_val = gamma_map[i, j]
            text_color = 'black' if gamma_val > 0.8 else 'white'
            ax.text(j, i, f'{gamma_val:.2f}', 
                   ha='center', va='center', color=text_color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_directory, f"{image_name}_adaptive_gamma_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Gamma heatmap saved: {heatmap_path}")
    
    return heatmap_path

def save_result_comparison(original_tensor, lap_tensor, config, output_directory, image_name):
    """
    원본과 LAP 결과를 비교하여 저장합니다.
    """
    _, _, h, w = original_tensor.shape
    grid_h, grid_w = config.grid_size
    block_h, block_w = h // grid_h, w // grid_w
    
    original_np = original_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    lap_np = lap_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 밝기 계산
    original_brightness = np.zeros((grid_h, grid_w))
    lap_brightness = np.zeros((grid_h, grid_w))
    
    for i in range(grid_h):
        for j in range(grid_w):
            h_start, h_end = i * block_h, (i + 1) * block_h
            w_start, w_end = j * block_w, (j + 1) * block_w
            
            # Original
            block_orig = original_np[h_start:h_end, w_start:w_end, :]
            lum_orig = 0.299 * block_orig[:, :, 0] + 0.587 * block_orig[:, :, 1] + 0.114 * block_orig[:, :, 2]
            original_brightness[i, j] = lum_orig.mean()
            
            # LAP
            block_lap = lap_np[h_start:h_end, w_start:w_end, :]
            lum_lap = 0.299 * block_lap[:, :, 0] + 0.587 * block_lap[:, :, 1] + 0.114 * block_lap[:, :, 2]
            lap_brightness[i, j] = lum_lap.mean()
    
    # 밝기 변화
    brightness_change = lap_brightness - original_brightness
    
    # 비교 그래프
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # 원본 밝기
    im1 = axes[0].imshow(original_brightness, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original Brightness', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Brightness')
    
    # LAP 밝기
    im2 = axes[1].imshow(lap_brightness, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Adaptive LAP Brightness', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Brightness')
    
    # 변화량
    im3 = axes[2].imshow(brightness_change, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Brightness Change (LAP - Original)', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Change')
    
    plt.suptitle(f'Brightness Comparison for {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comparison_path = os.path.join(output_directory, f"{image_name}_adaptive_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison saved: {comparison_path}")
    
    return original_brightness, lap_brightness, brightness_change

def apply_usm_sharpening(img_tensor, usm_amount, sigma=5.0):
    """
    USM 샤프닝을 적용합니다.
    """
    # Gaussian blur
    radius = int(3 * sigma)
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img_tensor.device)
    kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel = kernel_2d[None, None, :, :].expand(3, 1, size, size)
    
    pad_w = (size - 1) // 2
    blurred = F.conv2d(F.pad(img_tensor, (pad_w, pad_w, pad_w, pad_w), mode='reflect'),
                       kernel, groups=3)
    
    # USM: output = img + (img - blurred) * amount
    detail = img_tensor - blurred
    sharpened = img_tensor + detail * usm_amount
    
    return torch.clamp(sharpened, 0, 1)

def process_image_adaptive_gamma(image_path, gamma_dark, gamma_bright, brightness_threshold, 
                                 output_directory, usm_amount=1.3):
    """
    적응형 감마 보정으로 이미지를 처리합니다.
    
    Args:
        image_path: 입력 이미지 경로
        gamma_dark: 어두운 영역에 적용할 감마값 (예: 1.3908)
        gamma_bright: 밝은 영역에 적용할 감마값 (예: 1.0)
        brightness_threshold: 밝기 임계값 (0~1 사이, 예: 0.5)
        output_directory: 출력 폴더
        usm_amount: USM 샤프닝 강도 (기본값: 1.3)
    """
    config = Config(gamma_dark=gamma_dark, gamma_bright=gamma_bright, 
                   brightness_threshold=brightness_threshold)
    config.usm_amount = usm_amount
    
    img_name = os.path.basename(image_path)
    image_name = os.path.splitext(img_name)[0]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist.")
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    print(f"\n{'='*70}")
    print(f"Processing: {image_name}")
    print(f"Gamma (dark regions): {gamma_dark:.4f}")
    print(f"Gamma (bright regions): {gamma_bright:.4f}")
    print(f"Brightness threshold: {brightness_threshold:.4f}")
    print(f"USM amount: {usm_amount:.4f}")
    print(f"{'='*70}\n")
    
    # 1. 적응형 감마 맵 계산
    gamma_map, brightness_map = calculate_adaptive_gamma_map(img_tensor, config)
    
    # 2. 감마 히트맵 생성 및 저장
    save_gamma_heatmap(img_tensor, gamma_map, brightness_map, config, output_directory, image_name)
    
    # 3. 블록별 감마 적용
    gamma_applied = apply_adaptive_gamma(img_tensor, gamma_map, config)
    
    # 4. USM 샤프닝 적용
    lap_tensor = apply_usm_sharpening(gamma_applied, usm_amount)
    
    # 5. 결과 비교 저장
    orig_bright, lap_bright, brightness_change = save_result_comparison(
        img_tensor, lap_tensor, config, output_directory, image_name
    )
    
    # 6. LAP 이미지 저장
    lap_np = lap_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    lap_np = np.clip(lap_np * 255.0, 0, 255).astype(np.uint8)
    lap_np = cv2.cvtColor(lap_np, cv2.COLOR_RGB2BGR)
    
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"{image_name}_LAP_adaptive.jpg")
    cv2.imwrite(output_path, lap_np)
    print(f"✓ LAP image saved: {output_path}")
    
    # 7. 통계 정보 저장
    stats = {
        "image_name": image_name,
        "gamma_dark": float(gamma_dark),
        "gamma_bright": float(gamma_bright),
        "brightness_threshold": float(brightness_threshold),
        "usm_amount": float(usm_amount),
        "gamma_statistics": {
            "min": float(gamma_map.min()),
            "max": float(gamma_map.max()),
            "mean": float(gamma_map.mean()),
            "dark_blocks": int(np.sum(gamma_map == gamma_dark)),
            "bright_blocks": int(np.sum(gamma_map == gamma_bright))
        },
        "original_brightness": {
            "min": float(orig_bright.min()),
            "max": float(orig_bright.max()),
            "mean": float(orig_bright.mean())
        },
        "lap_brightness": {
            "min": float(lap_bright.min()),
            "max": float(lap_bright.max()),
            "mean": float(lap_bright.mean())
        },
        "brightness_change": {
            "min": float(brightness_change.min()),
            "max": float(brightness_change.max()),
            "mean": float(brightness_change.mean()),
            "ratio": float(lap_bright.mean() / orig_bright.mean()) if orig_bright.mean() > 0 else 0
        }
    }
    
    stats_path = os.path.join(output_directory, f"{image_name}_adaptive_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print(f"✓ Statistics saved: {stats_path}")
    
    print(f"\n{'='*70}")
    print(f"Processing complete!")
    print(f"Dark blocks: {np.sum(gamma_map == gamma_dark)} (γ={gamma_dark:.4f})")
    print(f"Bright blocks: {np.sum(gamma_map == gamma_bright)} (γ={gamma_bright:.4f})")
    print(f"Original mean brightness: {orig_bright.mean():.4f}")
    print(f"LAP mean brightness: {lap_bright.mean():.4f}")
    print(f"Brightness increase: {brightness_change.mean():.4f} ({(lap_bright.mean()/orig_bright.mean()-1)*100:+.1f}%)")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    # 사용 예시 1: 단일 이미지
    image_path = "E:/KDG/Yeoju_rain/coco_llvip_rgb/train2017/000105_03.jpg"
    output_directory = "E:/KDG/Gamma"
    
    # 적응형 감마 설정
    gamma_dark = 1.4435      # 어두운 영역에 적용할 감마 (밝게 만듦)
    gamma_bright = 1.0       # 밝은 영역에 적용할 감마 (변화 없음)
    brightness_threshold = 0.5  # 밝기 임계값 (0~1)
    usm_amount = 1.1450
    
    if os.path.exists(image_path):
        process_image_adaptive_gamma(
            image_path, 
            gamma_dark, 
            gamma_bright, 
            brightness_threshold,
            output_directory, 
            usm_amount
        )
    else:
        print(f"Image not found: {image_path}")
    
    # 사용 예시 2: 여러 이미지 일괄 처리
    """
    image_configs = [
        # (이미지명, gamma_dark, gamma_bright, threshold, usm_amount)
        ("000267_03.jpg", 1.3908, 1.0, 0.5, 1.3338),
        ("000026_03.jpg", 1.5947, 1.0, 0.5, 1.3142),
        ("000042_03.jpg", 1.3515, 1.0, 0.5, 1.3336),
    ]
    
    input_folder = "E:/KDG/Yeoju_rain/coco_llvip_rgb/val2017"
    output_directory = "E:/KDG/Gamma_Adaptive"
    
    for img_name, gamma_d, gamma_b, threshold, usm in image_configs:
        img_path = os.path.join(input_folder, img_name)
        if os.path.exists(img_path):
            process_image_adaptive_gamma(img_path, gamma_d, gamma_b, threshold, 
                                        output_directory, usm)
    """