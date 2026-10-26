# 第一阶段：Fold 03 统一采样与评价

主线只使用已有 A20k＋B2/C2，在 256×256 训练 patch 尺度比较。
Mac 负责代码、配置、说明和推送；Windows 负责预检、轻量采样与评价。
本阶段不启动新训练，不重训纯 epsilon，不叠 loss，不做 512 patch，不接力学。
Min-SNR、B3/C3 和其他折不进入本轮主表；本轮不需要 AutoDL 任务。

## 配置用途与执行顺序

| 配置 | 用途 | 执行条件 |
| --- | --- | --- |
| `configs/sample_fold03_ab2c_256.yaml` | 256 主评价，A/B2/C2 各 raw、EMA，共六组 | Windows 预检通过 → 单种子 smoke 通过 → 正式五种子 |
| `configs/sample_fold03_ab2c_original.yaml` | 原图尺度附表；768×512 生成后中心裁剪为 750×450 | 先归档；256 主表验收前不执行 |
| `configs/evaluate_fold03_ab2c_original.yaml` | legacy：历史三组原图尺度图片复评，每种子 5 张 | 先归档；256 主表验收前不执行；不能接到新采样之后 |

新采样脚本会同时生成图片、`summary.csv` 和 `summary.json`，主表直接使用这些产物。
legacy 配置读取历史三组目录，并非新采样的六组 raw/EMA 目录。
原图附表中的 512 是推理画布高度，不代表新增 512 patch 训练；其结果单独报告。

## Windows 根目录与相对路径

当前配置按以下目录布局编写：

```text
D:\017_IPF_GB_RGB扩散项目_代码数据权重四折结果_20261014\
  configs\
  scripts\
  ebsd_diffusion\
  data\
  01_基线留一验证_同一配置\
  02_256空间匹配实验\
  03_结构loss消融\
  04_统一对比图\
```

脚本以配置文件的 `parent.parent` 为项目根目录；配置必须直接位于 `configs/` 下。
不要将运行副本放入 `configs/runs/`，也不要靠切换当前目录改变资源路径基准。
若代码 checkout 与上述数据根目录分离，先在 P1 明确路径方案，再执行。
这些 Windows 路径不同于仓库历史的 `08_训练与生成结果/` 布局，Mac 无需补齐数据和权重。

## Windows 预检与 smoke 门槛

P0 仅归档配置和说明；以下是 P1 待落实的检查要求，当前没有新增自动预检入口。
完成同一提交的同步后，Windows 必须先核对：

- 所有输入文件存在、可读且为实际资源；包括 U-Net、条件编码器、EMA shadow、VAE、scheduler、成分表和真实图。
- A/B2/C2 均对应目标 20k，核对 `checkpoint-0020000`、`training_state.json`、`export_metadata.json` 和训练记录，不能只看目录名或 `final`。
- A 的导出脚本可能从参考 `final` 复制 `run_metadata.json`，其中步数不能单独作为 A20k 的证明。
- 三组训练记录与配置指定的共用 VAE 一致；核对 `latent_scaling`、成分归一化、256 patch、留出合金及 scheduler 的 epsilon 设置。
- A raw 与导出参考、B2/C2 EMA shadow 与导出、条件编码器与对应 checkpoint 的差异符合预期；预期完全相同的权重差异应为 0。当前采样脚本只记录差异，不会据此中止，非零必须先查清。
- CUDA 可用。当前脚本无 CUDA 时会回退 CPU，不能把回退运行当作 smoke。
- 输出指向一个全新的运行目录；保留实际配置、代码提交号、运行日志和检查结论。当前脚本没有覆盖保护。

预检通过后，先从 256 主配置复制一份单种子 smoke 配置，保留首个种子 `20261012`、每种子 1 张，其余采样参数和资源路径不变。
smoke 应产出六组各 1 张 256×256 图片、真实参考、VAE 重建以及完整汇总文件。
检查通过后，正式运行使用主配置原有的五个种子，每种子 1 张；不能混用 smoke 图片出正式表。

## 每次运行使用独立配置与产物目录

配置中的 `output_dir` 是模板位置。执行前创建本地副本，例如：

```text
configs/sample_fold03_ab2c_256.smoke_<运行编号>.local.yaml
configs/sample_fold03_ab2c_256.full_<运行编号>.local.yaml
```

分别将副本的 `output_dir` 设置为尚不存在的目录：

```text
04_统一对比图/阶段A_Fold03_AB2C2_256_raw_ema/smoke_<运行编号>
04_统一对比图/阶段A_Fold03_AB2C2_256_raw_ema/full_<运行编号>
```

运行编号建议包含时间和代码短提交号；每次重跑换新编号，不复用旧目录。
本地副本与整个 `04_统一对比图/` 已被 `.gitignore` 排除；将实际执行副本另存于对应产物目录，便于追溯。

在预检通过后，先执行 smoke；通过后再执行 full。以下命令中的编号应替换为实际创建的副本名：

```powershell
Set-Location 'D:\017_IPF_GB_RGB扩散项目_代码数据权重四折结果_20261014'
python -m scripts.sample_fold03_ab2c_256 --config configs/sample_fold03_ab2c_256.smoke_<运行编号>.local.yaml
# smoke 验收通过后，单独执行下一条。
python -m scripts.sample_fold03_ab2c_256 --config configs/sample_fold03_ab2c_256.full_<运行编号>.local.yaml
```

## 正式 256 主表验收与回传

- 六组固定为 `A_raw`、`A_ema`、`B2_raw`、`B2_ema`、`C2_raw`、`C2_ema`，每组 5 张，共 30 张生成图；尺寸全部为 256×256，无缺图、重复或旧图混入。
- 种子固定为 `20261012`～`20261016`，DDIM 50 步，CFG 1.8，GB 阈值 0.18；三组共用配置指定的 VAE 和 scheduler。
- 保留 `real_reference.png`、`vae_reconstruction.png`、`comparison_grid.png`、`summary.csv`、`summary.json`、`per_image_metrics.json`、`run_metadata.json`，以及执行配置、日志和预检结论。
- `summary.csv` 应有六组生成结果和两组参考，共 8 条数据行；比较 raw 对 raw、EMA 对 EMA，不按各组最优权重挑选主表。
- 指标沿用当前图像描述符口径，不新增评价算法或预设胜者。GB 与晶粒统计是 RGB 图像近似，面积单位为像素；一张真实中心裁剪和五个种子只支持本轮小样本对照。
- Windows 回传上述表格、JSON、拼图、配置与日志，由 Mac 整理主表；本阶段不将生成目录、权重或 Finder 文件加入提交。

只有 256 主表完成验收后，才讨论原图附表或 legacy 复评的执行；两者都不合并进主表。
