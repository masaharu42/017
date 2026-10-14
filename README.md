# EBSD 成分条件扩散图像生成项目

本工程根据 22 种镍基合金的实际成分表，为 750×450 RGB IPF+GB 图实现连续成分条件的潜空间扩散模型。

**第一次打开项目，请从 [00_新手先看.md](./00_新手先看.md) 开始。**

方案和审核文件：

- [01_项目审核清单.md](./01_项目审核清单.md)
- [02_扩散模型完整方案.md](./02_扩散模型完整方案.md)
- [03_主流程图与关键参数说明.md](./03_主流程图与关键参数说明.md)

当前处于用户审核阶段，不运行正式训练或批量采样。

## 1. 当前数据结论

- 1–21 号使用表中确定成分。
- 22 号按用户确认，采用 GH706 标准范围中位数作为名义建模条件并参与条件扩散。
- 默认条件为 `Ni, Cr, Nb, Mo, Al, Ti`。
- Fe 因闭合/余额约束剔除；Ta、Co 因各只有一个非零样本剔除。
- 热处理制度固定，不进入条件向量。
- 统一热处理为 980 ℃ × 1 h 空冷；730 ℃ × 8 h；以 55 ℃/h 炉冷至 620 ℃；620 ℃ × 8 h 后空冷，并已保存为固定元数据。

## 2. 环境

Windows + RTX 5060 Ti 建议先安装 CUDA 12.8 PyTorch wheel：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

检查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

## 3. 准备成分

仓库中的 `data/compositions.csv` 已从用户工作簿生成。需要重新生成时：

```powershell
python scripts/prepare_compositions.py `
  "D:\预测应力应变曲线数据\22种材料的组织成分信息\合金成分与热处理工艺汇总.xlsx" `
  --output data/compositions.csv
```

## 4. 建立图像清单

当前 22 张图已经放入项目的 `data/原始EBSD导出数据`，并按编号划分子目录，可自动发现：

```powershell
python A2_维护数据_重新建立图像清单.py "data\原始EBSD导出数据" `
  --discover `
  --output data/manifest.csv
```

如果图片直接命名为 `1.png` 到 `22.png`：

```powershell
python scripts/build_manifest.py "D:\你的IPFGB图目录" `
  --pattern "{alloy_id}.png" `
  --output data/manifest.csv

python scripts/validate_data.py `
  --manifest data/manifest.csv `
  --compositions data/compositions.csv `
  --patch-size 192
```

若文件名不同或每种合金有多个独立视场，可复制 `data/manifest.example.csv` 为 `data/manifest.csv` 后编辑。相同 `alloy_id` 可以出现多行，但每行必须是不同的独立图像路径。

原图若带图例、标尺或文字，在 `configs/base.yaml` 的 `crop_margins` 设置固定裁边。

## 5. 训练 VAE

```powershell
python train_vae.py --config configs/base.yaml
```

8 GB 显存默认 patch 192、batch 4。若显存不足，先把 `train_vae.batch_size` 改为 2。

严格 LOO 时：

```powershell
python train_vae.py --config configs/base.yaml --exclude-alloy-id 14
```

## 6. 训练连续条件扩散

```powershell
python train_diffusion.py --config configs/base.yaml
```

输出目录：`08_训练与生成结果/02_扩散模型/final/`，包含：

- `unet/`
- `condition_encoder/`
- `scheduler/`
- `composition_conditioner.json`
- `run_metadata.json`

断点恢复：

```powershell
python train_diffusion.py --config configs/base.yaml `
  --resume-from "08_训练与生成结果/02_扩散模型/checkpoint-0005000"
```

## 7. 生成图像

单条件命令默认生成 8 张预览图：

按已有合金成分生成：

```powershell
python sample.py `
  --checkpoint "08_训练与生成结果/02_扩散模型/final" `
  --alloy-id 14 `
  --count 8 `
  --batch-size 1 `
  --output-dir "08_训练与生成结果/03_单次预览图/合金14"
```

按新成分生成：

```powershell
python sample.py `
  --checkpoint "08_训练与生成结果/02_扩散模型/final" `
  --composition "Ni=47,Cr=17,Nb=4,Mo=1,Al=0.5,Ti=1.8" `
  --count 8 `
  --output-dir "08_训练与生成结果/03_单次预览图/自定义成分"
```

超出训练范围或离训练点过远时默认报错。仅做探索时可加 `--allow-extrapolation`，生成文件名会带 `EXTRAPOLATION`。

正式批量生成默认每种合金 32 张，共 `22 × 32 = 704` 张：

```powershell
python 09_批量生成22种合金704张图.py `
  --checkpoint "08_训练与生成结果/02_扩散模型/final" `
  --count-per-alloy 32 `
  --output-dir "08_训练与生成结果/04_22种合金正式生成图"
```

训练和生成过程会在对应输出目录持续写入 `*_progress.json` 和 `*_progress_history.jsonl`，记录当前阶段、已用时间、预计剩余时间和预计完成时刻。完整说明见 `03_主流程图与关键参数说明.md`。

## 8. 评估

```powershell
python scripts/evaluate_generated.py `
  --real "data\原始EBSD导出数据\14\14-IPF + GB.bmp" `
  --generated-dir "08_训练与生成结果/03_单次预览图/合金14" `
  --output "08_训练与生成结果/06_评价报告/合金14.json"
```

RGB 代理指标不能替代原始 EBSD 取向数据。若要评价 Σ3、取向差或织构 J，需要额外接入原始欧拉角/ODF/边界统计文件。

## 9. 留一合金验证

先跑代表折：

```powershell
python scripts/run_loo.py --folds 3 14 18 21 --retrain-vae --max-steps 5000
```

`5000` 只用于流水线与早期趋势检查，不是正式结果。正式训练使用配置中的完整步数，并在模型稳定后扩展到 1–22 全部折。22 号应标记为“标准范围中位数名义条件”，21 号单列为外推压力测试。

## 10. PyCharm 项目设置

推荐使用 PyCharm，直接打开本项目根目录。解释器选择：

```text
$PROJECT_DIR$/.venv/Scripts/python.exe
```

把项目根目录标记为 Sources Root，运行脚本时 Working directory 使用 `$PROJECT_DIR$`。`.idea/`、虚拟环境、训练输出和本机绝对路径清单不会提交到 Git。

## 11. 冒烟测试

安装依赖后：

```powershell
python scripts/smoke_test.py
python -m unittest discover -s tests -v
```
