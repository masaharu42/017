# Fold03：小编码器局部五维条件实验

目的：在已知局部组织统计时，raw 生成器能否响应粗细条件。不是成分泛化验收，
不覆盖 Stage A 主表，不评价 ID03、不接 GPR、不输入不确定度、不接力学、不叠 loss。
Mac 只改代码与运行合成数据软件测试；准备标签、训练、采样在 AutoDL。

## 实验与对照

复用上一轮已经完成的 `A_mse`（从 raw50k 新增10k的末步 raw 权重与图片）。
只新增 `B_local5`，从同一 raw50k 和原成分编码器起步，完整继承 A 的10k训练设置。
成分编码512维，加上 `5→32→512` 的描述符残差；末层零初始化，初始化不消耗训练 RNG。
VAE/成分归一化/latent_scaling冻结。普通 epsilon MSE，无新增结构损失，无 Min-SNR。
新分支与 U-Net、原成分编码器一起训练。CFG dropout 同时屏蔽成分与描述符。

旧 `crops.csv` 不作为训练标签：它在 Lanczos 缩放 RGB 后重新阈值分割，可能造成晶粒合并。
本实验保持上一轮原图、原随机裁剪和增强，不导入016潜变量、28维条件或5376张窗口图。
新分支的加入不得改变原图/噪声/t等随机流；逐次核对旧 A 已记录的前5批及每25步指纹。
任一不匹配会在该批更新前停止。匹配范围不冒充全步穷尽证明。

## 五维定义与边界

五维为**可观测窗口的 RGB 分区代理**：像素域 D50、对数尺寸宽度、面积加权长宽比、
直径≥20像素的面积比例、单位像素面积的标签界面长度。
原尺寸RGB先提黑线，固定膨胀1像素，四连通内部标记，边界分配最近区域；不缩放。
每个256窗口重新分割，生成图采用完全相同算法。窗口截断后的可见连通片作为统计对象，
不使用窗外完整母晶粒尺寸，也不把裁剪框计入晶界长度。
过滤面积<10像素的区域；不足3个区域时五项无效。
无效训练窗口仍参与噪声学习，只关闭描述符残差，不删图、不重抽、不用零充当真实标签。

这些定义**不等同于AZtec/CTF真实晶粒五维**，单位不是已标定μm，20像素不是20μm。
原图整体分割出现不足3区域或单连通域>90%时准备阶段拒绝继续，需查看QA，
不能把此质量检查当作真实组织必须满足的物理标准，也不自动改阈值绕过。
校准均值/std仅使用排除ID03的训练合金固定随机窗口，私有NumPy RNG不影响训练抽样。
旧 GPR std 不是标签误差或窗口波动，本轮不使用。

## 服务器需要什么

需要上一轮**完整运行目录**：manifest、configs、source_snapshot、A_mse末步/初始化记录、
input_fingerprints.csv、evaluation的PNG与元数据，以及它原先引用的原图、raw50k、共享VAE。
上述文件留在原位置。准备会核对哈希；不要求再次上传权重或5376张小图。
历史016大图不自动替换017原图。新增代码通过Git同步；所有新标签和QA在服务器生成。
如果只剩旧PNG、缺指纹或源快照，不会静默宣称可以严格复用对照。

## 命令

### 6000D / 大显存服务器

使用 `configs/fold03_local5_10k_large_vram.yaml`，复制成同名 `.local.yaml` 后运行相同阶段。
该配置关闭梯度检查点，用更多激活显存换取减少重算；batch仍继承旧A的32，精度和学习率不变。
不预分配或强行占满84GB。prepare读取真实GPU型号、总显存和空闲显存，并运行微型FP16兼容检查；
要求总显存≥64GiB、空闲≥48GiB。这是启动保护，不是精确峰值预测，真实峰值仍记录在训练结果中。
默认配置保留旧执行设置，大显存配置会如实记录checkpointing执行差异。

换GPU/驱动可能使VAE等计算出现逐位差异。即使数学设置相同，也不保证旧指纹匹配；
首次不匹配会停止，不能关闭核对继续宣称严格对照。届时应检查环境，或另立同设备匹配对照，
本脚本不会自动追加另一臂。若旧运行目录来自另一台机器，还需保持原绝对数据/权重路径可用。

```bash
cp configs/fold03_local5_10k_large_vram.yaml configs/fold03_local5_10k_large_vram.local.yaml
python -m scripts.run_descriptor_pilot --config configs/fold03_local5_10k_large_vram.local.yaml --phase prepare
```

后续train/evaluate都使用这个large_vram.local.yaml；使用新的输出目录，不复用已经prepare的旧目录。

在服务器项目根目录、上一轮Python环境中执行：

```bash
git pull --ff-only origin main
cp configs/fold03_local5_10k.yaml configs/fold03_local5_10k.local.yaml
```

只改 local.yaml 的 `baseline_run`（旧完整运行目录）和 `output_root`（新空目录）。
保留原环境和旧数据路径，不升级包；新脚本复用原训练环境已有SciPy/scikit-image。

```bash
python -m scripts.run_descriptor_pilot --config configs/fold03_local5_10k.local.yaml --phase prepare
```

prepare不加载训练GPU任务：检查旧对照、原图/权重身份，CPU生成原尺寸分割QA、
窗口校准表和五维归一化、monitor_targets.json、B配置及源码快照。
查看新目录 `labels/*_segmentation_QA.png` 与 `labels/metadata.json`：
必须排除ID03、分割能分开晶粒、记录无效比例，明确仍为RGB代理。
三个监测合金各选实际观测到的D50第10/50/90百分位附近的**完整五维向量**，不拼造组合。

后台启动（每次复制整块，不依赖上一终端的变量）：

```bash
RUN_DIR="$(python -c 'import yaml; from pathlib import Path; p=yaml.safe_load(Path("configs/fold03_local5_10k.local.yaml").read_text()); print(Path(p["output_root"]).resolve())')"
nohup python -u -m scripts.run_descriptor_pilot \
  --config configs/fold03_local5_10k.local.yaml --phase train \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
tail -f "$RUN_DIR/launcher.log"
```

Ctrl-C退出tail不终止后台训练。nohup可防SSH断开，不防服务器关机/进程被杀。
仅启动B_local5。失败或已有部分输出不会自动重跑/续训/覆盖；先读日志找原因。
指纹不同可能来自环境、数据或RNG行为差异，不能删除旧指纹来绕过。

训练完成，明确执行评价：

```bash
python -m scripts.run_descriptor_pilot --config configs/fold03_local5_10k.local.yaml --phase evaluate
```

只新采B的18张（3合金×3已知条件×2种子），复用旧A的6张，总计24张比较图。
256/DDIM50/CFG1.8/原精度，实际初始噪声与DDIM时间网格必须和旧A匹配。
输出各合金comparison_grid、目标真实窗口、summary.json、paired_descriptor_errors.csv/json、
sampling_inputs.json及带哈希的run_metadata.json。不会调用旧Min-SNR评价或开ID03。

## 换卡后采用同机双臂（6000D）

如果零更新复核显示：当前新旧代码首批各项指纹一致，但旧代码也无法复现历史
4090 指纹，应使用 `configs/fold03_local5_pair_10k_large_vram.yaml`。
这只定位到跨环境不可逐位复现，不能凭合并哈希断言具体是哪一个 CUDA 算子。
旧主表、旧 MSE/Min-SNR 结果保持独立；此前失败的 local5 尚未完成一次更新。

双臂都从原 A raw50k 开始，各新增 10k 步，依次训练 A_mse、B_local5；
batch=32、FP16、LR/日程/随机种子均继承旧 A 配置，两臂都关闭梯度检查点。
唯一科学差异为 B 的局部五维小编码器。B 对照新 A 的前五批和每个日志批指纹，
不是宣称逐批穷尽核验。新 A 完成后冻结权重及指纹哈希，再启动 B。
若 A 已完成并在 manifest 中登记，可复核哈希后跳过 A；部分输出仍不自动续训。

```bash
cp configs/fold03_local5_pair_10k_large_vram.yaml configs/fold03_local5_pair_10k_large_vram.local.yaml
nohup bash -c 'python -u -m scripts.run_descriptor_pilot --config configs/fold03_local5_pair_10k_large_vram.local.yaml --phase prepare && python -u -m scripts.run_descriptor_pilot --config configs/fold03_local5_pair_10k_large_vram.local.yaml --phase train' > local5_pair_6000d.log 2>&1 < /dev/null &
tail -f local5_pair_6000d.log
```

输出使用新的 `08_训练与生成结果/08_Fold03_raw50k_local5_pair_10k_6000D_运行`，
不会覆盖此前的 06、07 目录。prepare 检查完成后训练自动衔接。
断开 Mac 网络或关闭终端不会中断 nohup 进程，服务器须保持开机。
训练完成后单独执行 `--phase evaluate`（使用同一 local YAML）：新采 A 6 张、B 18 张，
只比较同机生成的这 24 张，旧 A 图不作为本次严格对照。额外控制图记录在
`evaluation/control_images.json`，拼图 A 行明确写 fresh same-device。

## 判读

先看同成分/同噪声下三个真实条件是否使组织粗细合理变化，五项误差逐项报告，
同时审阅颜色、边缘和形态。标准化平均误差只是附加摘要，不能靠碎点或黑线增多判成功。
仅两种子、训练内已知描述符、相关窗口；不能宣称新合金成功或真实晶粒/取向物理一致。
若有效，再单独设计成分到描述符的预测流程，保持总体/局部标签语义一致。
本轮不自动追加训练步数、种子、参数搜索或多项损失。
