import torch
import numpy as np
import os
import argparse
from PIL import Image

from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='画像超解像処理を行います')
parser.add_argument('input', type=str, default="images/input.png",
                    help='入力画像のパス')
parser.add_argument('--crop_rate', type=float, default=0.03,
                    help='出力画像の端をクロップする割合（デフォルト: 0.03 = 3%）')
args = parser.parse_args()

processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64", use_fast=True)
model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

file_path = args.input
image = Image.open(file_path).convert("RGB")
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

# 拡張画像の端をクロップして、アーティファクトを除去する
width, height = enhanced_image.size
crop_width = int(width * args.crop_rate)
crop_height = int(height * args.crop_rate)
enhanced_image = enhanced_image.crop((
    crop_width,           # 左端からクロップ
    crop_height,          # 上端からクロップ
    width - crop_width,   # 右端からクロップ
    height - crop_height  # 下端からクロップ
))

# 元の画像をクロップ後の拡張画像と同じサイズにリサイズする
enhanced_size = enhanced_image.size
resized_original = image.resize(enhanced_size, Image.LANCZOS)

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
combined_image.save(output_filename)

# 画像を表示
combined_image.show()