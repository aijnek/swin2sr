# 信号処理ベース
- Nearest Neighbor
- Bilinear
- Bicubic
- Lanczos
- このあたりはすべてPIL等で使える

# deep learningベース
- SRCNN以降，多くのCNNベースのモデルが提案された
- その後SRGANやESRGANなどGANベースのものが増えた
- Transformerベースのものも提案されている．
- Swin2IRはTransformersに組み込まれている．このリポジトリでもそれを使っている

# diffusionベース
- SR3(https://research.google/blog/high-fidelity-image-generation-using-diffusion-models/)がdiffusionベースのSRの草分け的存在．GANベースのものをしのぐ性能がでる
- 商用モデルとしてimagegenの機能でupscaleが提供されている(https://cloud.google.com/vertex-ai/generative-ai/docs/image/upscale-image?hl=ja)．おそらくdiffusion model．
- diffusionベースが最も精度がよいが，パフォーマンスに課題がある．