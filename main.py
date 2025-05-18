import torch
import numpy as np
import os
import argparse
from PIL import Image

from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# 水平・垂直方向のクロップ率計算関数
def calculate_optimal_crop_rates(image_width, image_height, window_size=7, upscale_factor=2):
    """
    画像サイズとウィンドウサイズの関係から右側と下側のクロップ率を計算する関数
    
    Args:
        image_width (int): 画像の幅
        image_height (int): 画像の高さ
        window_size (int): Swin Transformerのウィンドウサイズ
        upscale_factor (int): 超解像の倍率
    
    Returns:
        tuple: (右側のクロップ率, 下側のクロップ率)
    """
    # 水平方向のパディング量の計算
    mod_pad_w = (window_size - image_width % window_size) % window_size
    padding_ratio_w = mod_pad_w / image_width if image_width > 0 else 0
    
    # 垂直方向のパディング量の計算
    mod_pad_h = (window_size - image_height % window_size) % window_size
    padding_ratio_h = mod_pad_h / image_height if image_height > 0 else 0
    
    # シフトウィンドウの影響を考慮
    horizontal_shift_effect = (window_size / 2) / image_width
    vertical_shift_effect = (window_size / 2) / image_height
    
    # 基本クロップ率の計算
    base_crop_rate_w = max(padding_ratio_w, horizontal_shift_effect, 0.01)
    base_crop_rate_h = max(padding_ratio_h, vertical_shift_effect, 0.015)  # 下側は少し大きめの最小値
    
    # アップスケール係数の考慮
    if upscale_factor > 2:
        base_crop_rate_w *= (upscale_factor / 2)
        base_crop_rate_h *= (upscale_factor / 2)
    
    # 安全係数の適用と範囲制限
    crop_rate_w = base_crop_rate_w * 1.3  # 安全マージン
    crop_rate_h = base_crop_rate_h * 1.5  # 下側は大きめの安全マージン
    
    crop_rate_w = min(max(crop_rate_w, 0.01), 0.05)  # 1%〜5%の範囲に制限
    crop_rate_h = min(max(crop_rate_h, 0.015), 0.05)  # 1.5%〜5%の範囲に制限
    
    return crop_rate_w, crop_rate_h

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='画像超解像処理を行います')
parser.add_argument('input', type=str, default="images/input.png",
                    help='入力画像のパス')
args = parser.parse_args()

# モデルのロード
model_name = "caidas/swin2SR-classical-sr-x2-64"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
model = Swin2SRForImageSuperResolution.from_pretrained(model_name)

# モデルのウィンドウサイズとアップスケール倍率
window_size = 7  # 固定値
# モデル名からアップスケール倍率を抽出（"x2"の部分から2を取得）
upscale_factor = int(model_name.split("x")[1].split("-")[0])

file_path = args.input
image = Image.open(file_path).convert("RGB")
width, height = image.size

# クロップ率の計算
crop_rate_w, crop_rate_h = calculate_optimal_crop_rates(width, height, window_size, upscale_factor)
print(f"計算されたクロップ率: 右={crop_rate_w:.4f}, 下={crop_rate_h:.4f}")
print(f"(元画像サイズ: {width}x{height}, ウィンドウサイズ: {window_size}, アップスケール: x{upscale_factor})")

# prepare image for the model
inputs = processor(image, return_tensors="pt")

# forward pass
with torch.no_grad():
    outputs = model(**inputs)

# 拡張画像の生成
output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
output = np.moveaxis(output, source=0, destination=-1)
output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
enhanced_image = Image.fromarray(output)

# 拡張画像の右側と下側をクロップして、アーティファクトを除去する
width, height = enhanced_image.size
right_crop = int(width * crop_rate_w)
bottom_crop = int(height * crop_rate_h)
enhanced_image = enhanced_image.crop((
    0,                    # 左端はクロップしない
    0,                    # 上端はクロップしない
    width - right_crop,   # 右端からクロップ
    height - bottom_crop  # 下端からクロップ
))

print(f"適用されたクロップ: 右={right_crop}px, 下={bottom_crop}px")

# 元の画像をクロップ後の拡張画像と同じサイズにリサイズする
# viewerでリサイズしたのをsimulateするため，nearest neighborでリサイズ
enhanced_size = enhanced_image.size
resized_original = image.resize(enhanced_size, Image.NEAREST)

# 画像のアスペクト比を計算
aspect_ratio = enhanced_size[0] / enhanced_size[1]

# アスペクト比に基づいて結合方法を決定
if aspect_ratio > 1:  # 横長画像の場合：縦に結合
    combined_width = enhanced_size[0]
    combined_height = enhanced_size[1] * 2
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(resized_original, (0, 0))
    combined_image.paste(enhanced_image, (0, enhanced_size[1]))
else:  # 縦長画像または正方形の場合：横に結合
    combined_width = enhanced_size[0] * 2
    combined_height = enhanced_size[1]
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(resized_original, (0, 0))
    combined_image.paste(enhanced_image, (enhanced_size[0], 0))

# 出力ファイル名の生成と保存
filename = os.path.basename(file_path)
name, ext = os.path.splitext(filename)
output_filename = f"results/{name}_enhanced{ext}"

# results ディレクトリがなければ作成
os.makedirs("results", exist_ok=True)

combined_image.save(output_filename)

# 画像を表示
combined_image.show()